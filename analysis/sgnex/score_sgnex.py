"""Task 2, deliverable 1: score every m6Anet-processed SG-NEx direct-RNA sample (option A, agreed 10 Oct).

    python analysis/sgnex/score_sgnex.py --line A549          # one cell line
    python analysis/sgnex/score_sgnex.py --all --jobs 3        # every line, 3 at a time
    python analysis/sgnex/score_sgnex.py --line Hct116 --limit 20000   # smoke

The 22 samples in s3://sg-nex-data that are already in the course's input format
(7 cell lines). For each cell line:

  1. each sample, scored on its own - to check that samples of one line agree
  2. all of the line's samples POOLED: a site's reads from every sample combined
     before scoring, so it has more reads (SG-NEx sites mostly have 1-3)

with the shipped model (models/final, numpy). Two scores per site:
  score_rank  what predict.py ships: mean over the six networks of the site's
              rank within this file. Orders sites within a sample; NOT comparable
              between samples (every file has 5% of sites in its own top 5%)
  score_raw   mean over the six networks of their raw outputs (logits): one fixed
              scale, so comparable between samples and cell lines at similar read
              counts. Not a calibrated probability (training up-weighted positives)
plus each network's raw output, the read-only ("own reads") score of the two
constrained designs (their raw output minus it = the neighbour correction), and
the read count. For the platform's model explainer, the pooled step also saves
the reads of a fixed set of showcase transcripts.

Writes data/sgnex_scores/<sample or line>.csv.gz (under the gitignored data/).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.data import iter_sites  # noqa: E402
from m6a.models.site_graph import SiteGraphEnsemble, _chunks, dense, kmer_onehot  # noqa: E402

BUCKET = "https://sg-nex-data.s3.amazonaws.com/"
PREFIX = "data/processed_data/m6Anet/"
DOWNLOADS = ROOT / "data" / "sgnex_runs"
OUT = ROOT / "data" / "sgnex_scores"
LABELS = ROOT / "data0" / "data.info.labelled"     # dataset0: picks the showcase transcripts
SHOWCASE = 300


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def samples() -> dict[str, list[str]]:
    xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={PREFIX}&delimiter=/").read().decode()
    names = [p.split("</Prefix>")[0].rstrip("/").split("/")[-1] for p in xml.split("<Prefix>")[2:]]
    lines: dict[str, list[str]] = {}
    for n in names:
        lines.setdefault(n.split("_")[1], []).append(n)
    return lines


def download(sample: str) -> Path:
    path = DOWNLOADS / sample / "data.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log(f"  downloading {sample}")
        with urllib.request.urlopen(f"{BUCKET}{PREFIX}{sample}/data.json") as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 24):
                f.write(chunk)
        tmp.rename(path)
    return path


def read(path: Path, limit: int | None):
    """Sites of one data.json as (frame of site ids, read values, per-site read counts)."""
    ids, pos, kmers, counts, chunks = [], [], [], [], []
    for site in iter_sites(path, limit=limit):
        ids.append(site.transcript_id)
        pos.append(site.position)
        kmers.append(site.kmer)
        counts.append(len(site.reads))
        chunks.append(np.asarray(site.reads, dtype=np.float32))
    frame = pd.DataFrame({"transcript_id": ids, "transcript_position": pos, "kmer": kmers})
    return frame, np.concatenate(chunks), np.asarray(counts)


def pool(parts):
    """Combine samples: every site's reads from every sample, one site per (transcript, position)."""
    frames = pd.concat([p[0] for p in parts], ignore_index=True)
    key = frames.transcript_id + ":" + frames.transcript_position.astype(str)
    codes, uniques = pd.factorize(key)
    per_read = np.repeat(codes, np.concatenate([p[2] for p in parts]))
    order = np.argsort(per_read, kind="stable")
    values = np.concatenate([p[1] for p in parts])[order]
    counts = np.bincount(codes, weights=np.concatenate([p[2] for p in parts]), minlength=len(uniques)).astype(int)
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    return frames.iloc[first].reset_index(drop=True), values, counts


def score(model: SiteGraphEnsemble, frame: pd.DataFrame, values, counts) -> pd.DataFrame:
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    onehot = kmer_onehot(frame.kmer.tolist())
    pos = frame.transcript_position.to_numpy().astype(np.float32)
    groups = frame.groupby("transcript_id", sort=False).indices
    out = frame.copy()
    out["n_reads"] = counts
    raw, ranks = [], []
    for net in model.networks:
        name = net.meta["model"] + f"_s{net.meta.get('seed', 0)}"
        s = np.concatenate([net.site_vectors(values[offsets[a]:offsets[b]], offsets[a:b + 1] - offsets[a], onehot[a:b])
                            for a, b in _chunks(offsets, 2_000_000)])
        logit = np.empty(len(frame))
        for rows in groups.values():
            logit[rows] = net.logits(s[rows], pos[rows], counts[rows])
        out[f"raw_{name}"] = logit.astype(np.float32)
        if net.v2:
            out[f"own_{name}"] = dense(s, net.w, "aux_head")[:, 0].astype(np.float32)
        raw.append(logit)
        ranks.append(pd.Series(logit).rank(method="average").to_numpy() / len(logit))
    out["score_rank"] = np.mean(ranks, 0).astype(np.float32)
    out["score_raw"] = np.mean(raw, 0).astype(np.float32)
    return out


def showcase_transcripts() -> set[str]:
    """Dataset0 transcripts with a modified site and at least 8 candidate sites: labelled
    examples for the model explainer, the same in every cell line."""
    t = pd.read_csv(LABELS)
    g = t.groupby("transcript_id").label.agg(["size", "sum"])
    keep = g[(g["size"] >= 8) & (g["sum"] >= 1)].sort_values("size", ascending=False).head(SHOWCASE)
    return {s.split(".")[0] for s in keep.index}


def run_line(line: str, names: list[str], limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = SiteGraphEnsemble.load(ROOT / "models" / "final")
    suffix = f"_limit{limit}" if limit else ""
    parts = []
    for name in sorted(names):
        started = time.time()
        frame, values, counts = read(download(name), limit)
        target = OUT / f"{name}{suffix}.csv.gz"
        if not target.exists():
            score(model, frame, values, counts).to_csv(target, index=False, float_format="%.5g")
        log(f"{name}: {len(frame):,} sites, median {np.median(counts):.0f} reads ({time.time() - started:.0f}s)")
        parts.append((frame, values, counts))
    started = time.time()
    frame, values, counts = pool(parts)
    del parts
    score(model, frame, values, counts).to_csv(OUT / f"{line}_pooled{suffix}.csv.gz", index=False,
                                                 float_format="%.5g")
    keep = frame.transcript_id.str.split(".").str[0].isin(showcase_transcripts()).to_numpy()
    offsets = np.r_[0, np.cumsum(counts)]
    rows = np.flatnonzero(keep)
    np.savez_compressed(OUT / f"{line}_pooled_showcase_reads{suffix}.npz",
                        transcript_id=frame.transcript_id.to_numpy()[rows].astype(str),
                        transcript_position=frame.transcript_position.to_numpy()[rows],
                        kmer=frame.kmer.to_numpy()[rows].astype(str), n_reads=counts[rows],
                        reads=np.concatenate([values[offsets[r]:offsets[r + 1]] for r in rows]) if len(rows)
                        else np.empty((0, 9), np.float32))
    log(f"{line} pooled: {len(frame):,} sites from {len(names)} samples, median {np.median(counts):.0f} reads, "
        f"{len(rows):,} showcase sites ({time.time() - started:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", help="one cell line, as named in the bucket (e.g. A549, Hct116)")
    ap.add_argument("--all", action="store_true", help="every line, one process each")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first N sites of each sample (smoke)")
    args = ap.parse_args()
    lines = samples()
    log("samples: " + json.dumps({k: len(v) for k, v in lines.items()}))
    if args.line:
        run_line(args.line, lines[args.line], args.limit)
        return
    if not args.all:
        raise SystemExit("Pass --line NAME or --all.")
    # Biggest lines first, so the pool's tail is short.
    queue = sorted(lines, key=lambda k: -len(lines[k]))
    running: list = []
    logs = ROOT / "analysis" / "sgnex" / "logs"
    logs.mkdir(exist_ok=True)
    while queue or running:
        while queue and len(running) < args.jobs:
            line = queue.pop(0)
            cmd = [sys.executable, "-u", __file__, "--line", line] + (["--limit", str(args.limit)] if args.limit else [])
            running.append((line, subprocess.Popen(cmd, stdout=open(logs / f"score_{line}.log", "w"),
                                                   stderr=subprocess.STDOUT)))
            log(f"start {line}")
        for line, p in list(running):
            if p.poll() is not None:
                log(f"{'done' if p.returncode == 0 else 'FAILED'} {line} (exit {p.returncode})")
                running.remove((line, p))
        time.sleep(20)
    log("all lines finished")


if __name__ == "__main__":
    main()
