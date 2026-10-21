"""One row per labelled site: its type, and the shipped design's held-out rank at each depth.

    python analysis/site_types/build_table.py

Inputs (all gitignored, under .cache/representation/xsrc_nets/):
  site_keys.csv.gz                 row keys of every held-out array (export_site_keys.py)
  <design>_pooled_both_fold<f>[_s<seed>].npy   held-out scores at full depth
  depth_scores/ and depth_scores_draw<N>/      held-out scores at 1, 3, 10 reads
                                               (depth_rescore.py; draw 4262 is the default)
  hct116_runs_sites/<run>.npz      the same sites scored from two other HCT116 runs
and data/annotation/gtf_91.gtf.gz for each site's region and genome position.

"Held-out" means every site was scored by the fold model that never saw its gene.
The model is the shipped ensemble's three designs, three seeds each, trained on
both labelled files: each network's score is ranked within its file, and the nine
ranks are averaged - what predict.py does (report/task2_plan.md, 1.1).

Writes .cache/site_types/sites.csv.gz and hct116_runs.csv.gz. They hold the course
labels, so they stay in the gitignored cache; only aggregates leave it.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NETS = ROOT / ".cache" / "representation" / "xsrc_nets"
OUT = ROOT / ".cache" / "site_types"
GTF = ROOT / "data" / "annotation" / "gtf_91.gtf.gz"
DESIGNS = ["h2gcn_twohead_aux", "res_gate", "scalar_drop"]
SEEDS = [0, 1, 2]
DEPTHS = [1, 3, 10]
DEFAULT_DRAW = 4262
RADIUS = 150          # nt: res_gate's neighbour window, and the corroboration range in findings.md


def log(*a) -> None:
    print(*a, flush=True)


def annotation():
    """transcript -> (strand, exon starts, exon ends, cumulative length, chrom), and CDS span."""
    exons, cds = {}, {}
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if p[2] not in ("exon", "CDS"):
                continue
            t = re.search(r'transcript_id "([^"]+)"', p[8]).group(1)
            (exons if p[2] == "exon" else cds).setdefault(t, []).append((p[0], int(p[3]), int(p[4]), p[6]))
    layout = {}
    for t, ex in exons.items():
        strand = ex[0][3]
        ex = sorted(ex, key=lambda e: e[1], reverse=strand == "-")
        cum = np.concatenate([[0], np.cumsum([e[2] - e[1] + 1 for e in ex])])
        layout[t] = (ex[0][0], strand, np.array([e[1] for e in ex]), np.array([e[2] for e in ex]), cum)
    return layout, cds


def to_genome(layout, t, pos):
    c, strand, st, en, cum = layout[t]
    i = np.clip(np.searchsorted(cum, pos, side="right") - 1, 0, len(st) - 1)
    off = pos - cum[i]
    return c, strand, np.where(strand == "+", st[i] + off, en[i] - off), pos < cum[-1]


def to_transcript(layout, t, g):
    """Genome coordinate -> transcript coordinate (for the CDS ends)."""
    c, strand, st, en, cum = layout[t]
    for i in range(len(st)):
        if st[i] <= g <= en[i]:
            return cum[i] + (g - st[i] if strand == "+" else en[i] - g)
    return None


def add_annotation(sites: pd.DataFrame) -> pd.DataFrame:
    layout, cds = annotation()
    tx = sites.transcript_id.str.split(".").str[0]
    chrom = np.full(len(sites), "", dtype=object)
    gpos = np.full(len(sites), -1, dtype=np.int64)
    region = np.full(len(sites), "unannotated", dtype=object)
    for t, idx in pd.Series(np.arange(len(sites))).groupby(tx.to_numpy()).indices.items():
        if t not in layout:
            continue
        pos = sites.transcript_position.to_numpy()[idx]
        c, strand, g, ok = to_genome(layout, t, pos)
        chrom[idx], gpos[idx] = c, np.where(ok, g, -1)
        if t not in cds:
            region[idx] = "non-coding"
            continue
        lo, hi = min(e[1] for e in cds[t]), max(e[2] for e in cds[t])
        a, b = to_transcript(layout, t, lo), to_transcript(layout, t, hi)
        if a is None or b is None:
            continue
        start, end = min(a, b), max(a, b)
        region[idx] = np.where(pos < start, "5' UTR", np.where(pos > end, "3' UTR", "coding"))
    return sites.assign(chrom=chrom, genome_pos=gpos, region=region)


def neighbours(sites: pd.DataFrame) -> np.ndarray:
    """Other labelled sites on the same transcript of the same file within RADIUS nt."""
    out = np.zeros(len(sites), dtype=np.int64)
    for idx in sites.groupby("graph_id").indices.values():
        p = np.sort(sites.transcript_position.to_numpy()[idx])
        order = np.argsort(sites.transcript_position.to_numpy()[idx])
        n = np.searchsorted(p, p + RADIUS, side="right") - np.searchsorted(p, p - RADIUS, side="left") - 1
        out[idx[order]] = n
    return out


def within_file_rank(score: np.ndarray, file: np.ndarray) -> np.ndarray:
    out = np.empty(len(score))
    for f in np.unique(file):
        m = file == f
        out[m] = pd.Series(score[m]).rank(pct=True).to_numpy()
    return out


def full_depth(n: int) -> list[np.ndarray]:
    nets = []
    for d in DESIGNS:
        for s in SEEDS:
            suffix = "" if s == 0 else f"_s{s}"
            arr = np.nanmax(np.stack([np.load(NETS / f"{d}_pooled_both_fold{f}{suffix}.npy") for f in range(5)]), 0)
            assert len(arr) == n and not np.isnan(arr).any(), f"{d} s{s}: incomplete"
            nets.append(arr)
    return nets


def at_depth(n: int, draw: int, depth: int) -> list[np.ndarray] | None:
    folder = NETS / ("depth_scores" if draw == DEFAULT_DRAW else f"depth_scores_draw{draw}")
    paths = [folder / f"{d}_pooled_both_s{s}.npz" for d in DESIGNS for s in SEEDS]
    if not all(p.exists() for p in paths):
        return None
    nets = [np.load(p)[f"d{depth}"] for p in paths]
    assert all(len(a) == n and not np.isnan(a).any() for a in nets), f"draw {draw} depth {depth}: incomplete"
    return nets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(NETS / "site_keys.csv.gz")
    n = len(sites)
    sites["motif"] = sites.kmer.str[1:6]
    sites["neighbours"] = neighbours(sites)
    sites = add_annotation(sites)
    log(f"{n:,} sites; region: {sites.region.value_counts().to_dict()}")

    ensemble = lambda nets: np.mean([within_file_rank(a, sites.file.to_numpy()) for a in nets], 0)  # noqa: E731
    sites["rank_full"] = ensemble(full_depth(n))
    draws = sorted({DEFAULT_DRAW} | {int(p.name.split("draw")[1]) for p in NETS.glob("depth_scores_draw*")})
    for draw in draws:
        for depth in DEPTHS:
            nets = at_depth(n, draw, depth)
            if nets is None:
                log(f"draw {draw}: missing - skipped")
                break
            sites[f"rank_d{depth}_draw{draw}"] = ensemble(nets)
    log(f"draws at depth: {[c for c in sites if c.startswith('rank_d1_')]}")
    sites.to_csv(OUT / "sites.csv.gz", index=False)

    # Two other HCT116 runs: real molecules at their own depth, dataset0 labels.
    rows = []
    for p in sorted((NETS / "hct116_runs_sites").glob("*.npz")):
        z = np.load(p)
        file = np.zeros(len(z["row0"]))
        nets = [z[f"{d}_s{s}"] for d in DESIGNS for s in SEEDS]
        rows.append(pd.DataFrame({"run": p.stem, "row": z["row0"], "n_reads_run": z["n_reads"],
                                  "rank_run": np.mean([within_file_rank(a, file) for a in nets], 0)}))
    if rows:
        pd.concat(rows).to_csv(OUT / "hct116_runs.csv.gz", index=False)
        log(f"HCT116 runs: {[r.run.iloc[0] for r in rows]}")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
