"""What are the final-leaderboard files (dataset3-7)? Real cells like dataset0/1, or synthetic like dataset2?

    python analysis/newdata/final_data_profile.py            # profile + SG-NEx source match
    python analysis/newdata/final_data_profile.py --no-sgnex # profile only, no network

For every course file (dataset0-7) this measures what separates the three
known kinds:

- dataset0: HCT116 cells (SG-NEx replicate 3 run 1, exactly; sgnex_source_match.py)
- dataset1: HEK293T cells
- dataset2: one synthetic sequence, in vitro, at known modified fractions:
  anonymised `tx_id_*` names, a site every 10 nt, the same 7-mers in every
  transcript, hundreds of reads per site

and compares the signal itself: for 5-mers shared with dataset0, how far each
file's mean current at the centre base sits from dataset0's (a different
chemistry or in-vitro RNA would shift it).

It then repeats sgnex_source_match.py's test on every SG-NEx m6Anet-processed
run: a file cut from a run has that run's read count at (nearly) every site.

Writes analysis/newdata/final_data_profile/{profile.csv, signal.csv, sgnex_match.csv}.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.data import iter_sites  # noqa: E402

OUT = ROOT / "analysis" / "newdata" / "final_data_profile"
FILES = {
    "dataset0": ROOT / "data0" / "dataset0.json.gz",
    "dataset1": ROOT / "data0" / "data1" / "dataset1.json.gz",
    "dataset2": ROOT / "data0" / "data2" / "dataset2.json.gz",
    **{f"dataset{i}": ROOT / "data" / "final_leaderboard" / f"dataset{i}.json.gz" for i in range(3, 8)},
}
BUCKET = "https://sg-nex-data.s3.amazonaws.com/"
PREFIX = "data/processed_data/m6Anet/"
DRACH = {d + r + "AC" + h for d in "AGT" for r in "AG" for h in "ACT"}


def sites(path: Path) -> pd.DataFrame:
    rows = []
    for s in iter_sites(path):
        m = s.reads.mean(axis=0)
        # Read features are (dwell, sd, mean) at positions -1, 0, +1; index 5 = mean current at the centre.
        rows.append((s.transcript_id, s.position, s.kmer, len(s.reads), m[3], m[4], m[5]))
    return pd.DataFrame(rows, columns=["transcript_id", "transcript_position", "kmer", "n_reads",
                                       "dwell_0", "sd_0", "mean_0"])


def profile(name: str, d: pd.DataFrame, ref_tx: dict[str, set[str]]) -> dict:
    tx = d.transcript_id.str.split(".").str[0]
    per_tx = d.groupby("transcript_id").size()
    return {
        "file": name,
        "sites": len(d),
        "transcripts": d.transcript_id.nunique(),
        "sites per transcript (median)": per_tx.median(),
        "Ensembl IDs %": 100 * d.transcript_id.str.startswith("ENST").mean(),
        "versioned IDs %": 100 * d.transcript_id.str.contains(r"\.").mean(),
        "reads p25": d.n_reads.quantile(0.25),
        "reads median": d.n_reads.median(),
        "reads max": d.n_reads.max(),
        "sites < 20 reads %": 100 * (d.n_reads < 20).mean(),
        "positions multiple of 10 %": 100 * (d.transcript_position % 10 == 0).mean(),
        "distinct 7-mers": d.kmer.nunique(),
        "DRACH centre %": 100 * d.kmer.str[1:6].isin(DRACH).mean(),
        **{f"transcripts shared with {k} %": 100 * tx.isin(v).mean() for k, v in ref_tx.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-sgnex", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, path in FILES.items():
        data[name] = sites(path)
        print(f"{name}: {len(data[name]):,} sites", flush=True)
    ref_tx = {k: set(data[k].transcript_id.str.split(".").str[0]) for k in ("dataset0", "dataset1")}
    prof = pd.DataFrame([profile(n, d, ref_tx) for n, d in data.items()]).set_index("file")
    prof.to_csv(OUT / "profile.csv")
    print(prof.T.round(1).to_string(), flush=True)

    # Signal: per centre 5-mer, read-weighted mean current, against dataset0's.
    def by_motif(d: pd.DataFrame) -> pd.DataFrame:
        d = d.assign(motif=d.kmer.str[1:6], w=d.n_reads)
        g = d.groupby("motif")
        return pd.DataFrame({c: g.apply(lambda x: np.average(x[c], weights=x.w), include_groups=False)
                             for c in ("mean_0", "sd_0", "dwell_0")}).assign(sites=g.size())
    ref = by_motif(data["dataset0"])
    sig = []
    for name, d in data.items():
        m = by_motif(d).join(ref, rsuffix="_ref", how="inner")
        m = m[(m.sites >= 20) & (m.sites_ref >= 20)]
        sig.append({"file": name, "shared 5-mers": len(m),
                    "median |current - dataset0| (pA)": (m.mean_0 - m.mean_0_ref).abs().median(),
                    "median current shift (pA)": (m.mean_0 - m.mean_0_ref).median(),
                    "median sd ratio": (m.sd_0 / m.sd_0_ref).median(),
                    "median dwell ratio": (m.dwell_0 / m.dwell_0_ref).median()})
    sig = pd.DataFrame(sig).set_index("file")
    sig.to_csv(OUT / "signal.csv")
    print(sig.round(3).to_string(), flush=True)

    if args.no_sgnex:
        return
    xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={PREFIX}&delimiter=/").read().decode()
    samples = [p.split("</Prefix>")[0].rstrip("/").split("/")[-1] for p in xml.split("<Prefix>")[2:]]
    keys = {n: d.assign(transcript_id=d.transcript_id.str.split(".").str[0])[
        ["transcript_id", "transcript_position", "n_reads"]] for n, d in data.items() if n != "dataset2"}
    rows = []
    for s in samples:
        rc = pd.read_csv(io.BytesIO(urllib.request.urlopen(f"{BUCKET}{PREFIX}{s}/data.readcount").read()))
        rc.columns = ["transcript_id", "transcript_position", "n_reads"]
        rc["transcript_id"] = rc.transcript_id.str.split(".").str[0]
        for n, k in keys.items():
            m = k.merge(rc, on=["transcript_id", "transcript_position"], how="left", suffixes=("", "_sg"))
            rows.append({"file": n, "sample": s, "present %": 100 * m.n_reads_sg.notna().mean(),
                         "same n_reads %": 100 * (m.n_reads == m.n_reads_sg).mean()})
        print(f"  {s} checked", flush=True)
    match = pd.DataFrame(rows)
    match.to_csv(OUT / "sgnex_match.csv", index=False)
    best = match.sort_values("same n_reads %", ascending=False).groupby("file").head(2)
    print(best.sort_values(["file", "same n_reads %"], ascending=[True, False]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
