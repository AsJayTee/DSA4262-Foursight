"""Where on a transcript are modified sites? Exon junctions, stop codons, exon length.

    python analysis/newdata/annotation_context.py --release 91

Known m6A biology to reproduce (or not) in our labels, per cell line:
- depletion near exon-exon junctions (the exon junction complex suppresses
  deposition within roughly 100-200 nt of a junction);
- enrichment around the stop codon / start of the 3' UTR;
- enrichment in long internal exons.
These give position-dependent zones on the hundred-nucleotide scale, one
candidate explanation for why a 100-nt neighbour radius transfers between cell
lines and wider radii do not (decision-free analysis for the report).

Inputs are public: the Ensembl GRCh38 cDNA FASTA and GTF for --release,
downloaded to data/annotation/ (gitignored). Everything is computed in
TRANSCRIPT coordinates - no genome mapping. Two checks run first and the
script refuses to continue if they fail:
- each labelled site's 7-mer matches the cDNA at its position (finds the
  position convention and confirms the release matches the data);
- each annotated stop codon reads TAA, TAG or TGA in the cDNA.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a import external  # noqa: E402
from m6a.data import iter_sites, load_labels, resolve_data_dir  # noqa: E402

ANN = ROOT / "data" / "annotation"
OUT = ROOT / "analysis" / "newdata" / "annotation_context"
JUNCTION_BINS = [0, 25, 50, 100, 200, 400, 10**9]
STOP_BINS = list(range(-1000, 1001, 100))
EXON_LEN_BINS = [0, 200, 400, 800, 1600, 10**9]


def fetch(release: int) -> tuple[Path, Path]:
    ANN.mkdir(parents=True, exist_ok=True)
    base = f"https://ftp.ensembl.org/pub/release-{release}"
    files = {f"{base}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz": ANN / f"cdna_{release}.fa.gz",
             f"{base}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{release}.gtf.gz": ANN / f"gtf_{release}.gtf.gz"}
    for url, path in files.items():
        if not path.exists():
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path)
    return tuple(files.values())


def read_cdna(path: Path, wanted: set[str]) -> dict[str, str]:
    seqs, name, chunks = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name in wanted:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0].split(".")[0], []
            else:
                chunks.append(line.strip())
    if name in wanted:
        seqs[name] = "".join(chunks)
    return seqs


def read_gtf(path: Path, wanted: set[str]) -> pd.DataFrame:
    rows = []
    tid = re.compile(r'transcript_id "([^"]+)"')
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] not in ("exon", "stop_codon", "start_codon"):
                continue
            t = tid.search(f[8])
            if t and t.group(1) in wanted:
                rows.append((t.group(1), f[2], int(f[3]), int(f[4]), f[6]))
    return pd.DataFrame(rows, columns=["transcript_id", "feature", "start", "end", "strand"])


def transcript_layout(gtf: pd.DataFrame) -> dict[str, dict]:
    """Per transcript: exon (start, end) in transcript coords, junction positions,
    and the transcript position (0-based) of the first base of the stop / start codon."""
    out = {}
    for t, g in gtf.groupby("transcript_id"):
        ex = g[g.feature == "exon"]
        strand = ex.strand.iloc[0]
        ex = ex.sort_values("start", ascending=(strand == "+"))
        lengths = (ex.end - ex.start + 1).to_numpy()
        ends = np.cumsum(lengths)
        starts = ends - lengths

        def to_tx(gpos: int):
            for (s, e), t0 in zip(zip(ex.start, ex.end), starts):
                if s <= gpos <= e:
                    return int(t0 + (gpos - s if strand == "+" else e - gpos))
            return None

        def codon(feature: str):
            c = g[g.feature == feature]
            if c.empty:
                return None
            # The codon's 5'-most base in transcript orientation; may span exons.
            gpos = c.start.min() if strand == "+" else c.end.max()
            return to_tx(gpos)

        out[t] = {"exon_starts": starts, "exon_ends": ends, "junctions": ends[:-1],
                  "stop": codon("stop_codon"), "start": codon("start_codon")}
    return out


def sites_frame(data_dir: Path) -> pd.DataFrame:
    keys = ["transcript_id", "transcript_position"]
    d0 = load_labels(data_dir / "data.info.labelled")[keys + ["label"]].assign(cell_line="dataset0")
    d1 = external.load_info(data_dir, "data1").reset_index()[keys + ["label"]].assign(cell_line="data1")
    return pd.concat([d0, d1], ignore_index=True)


def kmers(data_dir: Path) -> dict:
    out = {}
    for path in (data_dir / "dataset0.json.gz", external.paths(data_dir, "data1")[0]):
        for s in iter_sites(path):
            out[(s.transcript_id, s.position)] = s.kmer
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=int, default=91)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir()
    sites = sites_frame(data_dir)
    wanted = set(sites.transcript_id)
    cdna_path, gtf_path = fetch(args.release)
    cdna = read_cdna(cdna_path, wanted)
    km = kmers(data_dir)

    # Check 1: position convention and release match, from the 7-mers.
    sample = sites.drop_duplicates(["transcript_id", "transcript_position"]).sample(20000, random_state=0)
    best = None
    for offset in range(-4, 2):
        ok = n = 0
        for t, p in zip(sample.transcript_id, sample.transcript_position):
            s = cdna.get(t)
            k = km.get((t, p))
            if s is None or k is None:
                continue
            n += 1
            a = p + offset
            ok += a >= 0 and s[a:a + 7] == k
        print(f"  7-mer at position{offset:+d}: {ok / max(n, 1):.1%} of {n:,} sites match", flush=True)
        if best is None or ok > best[1]:
            best = (offset, ok, n)
    offset, ok, n = best
    print(f"release {args.release}: {len(cdna):,}/{len(wanted):,} transcripts in cDNA; 7-mer matches "
          f"{ok / n:.1%} with the window starting {-offset} nt before the site (its A is the 7-mer's centre, index 3)", flush=True)
    if ok / n < 0.95:
        raise SystemExit(f"Only {ok / n:.1%} of 7-mers match release {args.release}: wrong release. "
                         "Try another --release.")

    gtf = read_gtf(gtf_path, wanted)
    layout = transcript_layout(gtf)
    # Check 2: annotated stop codons read as stop codons in the cDNA.
    stops = [(cdna[t][L["stop"]:L["stop"] + 3]) for t, L in layout.items() if L["stop"] is not None and t in cdna]
    good = np.mean([s in ("TAA", "TAG", "TGA") for s in stops])
    print(f"stop codons: {len(stops):,} transcripts, {good:.1%} read TAA/TAG/TGA", flush=True)
    if good < 0.95:
        raise SystemExit("Stop codons do not land on TAA/TAG/TGA: transcript coordinates are wrong.")

    # Per-site context. The modified A is at transcript position p (checked above
    # via the 7-mer: offset tells where the window starts relative to p).
    a_pos = sites.transcript_position.to_numpy() + offset + 3
    rows = []
    for (t, a) in zip(sites.transcript_id, a_pos):
        L = layout.get(t)
        if L is None:
            rows.append((np.nan, None, np.nan, np.nan, None))
            continue
        j = L["junctions"]
        dj = float(np.min(np.abs(j - a))) if len(j) else np.nan
        i = int(np.searchsorted(L["exon_ends"], a, side="right"))
        i = min(i, len(L["exon_ends"]) - 1)
        exon_len = float(L["exon_ends"][i] - L["exon_starts"][i])
        kind = "single" if len(j) == 0 else ("first" if i == 0 else ("last" if i == len(j) else "internal"))
        if L["stop"] is None or L["start"] is None:
            region, ds = "noncoding", np.nan
        else:
            ds = float(a - L["stop"])
            region = "5'UTR" if a < L["start"] else ("CDS" if a < L["stop"] else "3'UTR")
        rows.append((dj, kind, exon_len, ds, region))
    ctx = pd.DataFrame(rows, columns=["dist_junction", "exon_kind", "exon_len", "dist_stop", "region"])
    sites = pd.concat([sites.reset_index(drop=True), ctx], axis=1)
    sites.to_csv(OUT / "site_context.csv.gz", index=False)

    def rate(col, bins=None):
        g = sites.copy()
        if bins is not None:
            g[col] = pd.cut(g[col], bins, right=False)
        t = g.groupby(["cell_line", col], observed=True).label.agg(["mean", "size"]).unstack(0)
        overall = sites.groupby("cell_line").label.mean()
        lift = t["mean"] / overall
        return pd.concat({"positive rate": t["mean"], "lift vs file rate": lift, "sites": t["size"]}, axis=1)

    pd.set_option("display.width", 220)
    for title, table in (
        ("Distance from the modified A to the nearest exon-exon junction (nt)", rate("dist_junction", JUNCTION_BINS)),
        ("Exon the site sits in", rate("exon_kind")),
        ("Length of that exon (nt)", rate("exon_len", EXON_LEN_BINS)),
        ("Transcript region", rate("region")),
        ("Distance to the stop codon (nt; negative = coding sequence)", rate("dist_stop", STOP_BINS)),
    ):
        print(f"\n{title}\n{table.round(3).to_string()}")
        table.to_csv(OUT / f"{re.sub('[^a-z]+', '_', title.lower())[:40]}.csv")


if __name__ == "__main__":
    main()
