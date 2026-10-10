"""Do dataset0's labels map onto the SG-NEx samples' sites? (10 Oct)

    python analysis/sgnex/label_mapping_check.py

Labels are matched by (transcript, position). If two files use the same transcript
coordinates, the 7-mer at a matched site must be identical: a one-nucleotide offset
would change it. For every scored SG-NEx sample (score_sgnex.py output), against
dataset0's 121,838 labelled sites: how many are present, how many have the same
7-mer, and how many reads they have there. Other cell lines are included as a
coordinate check only - dataset0's labels apply to HCT116, not to them.
Writes analysis/sgnex/label_mapping_check.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.data import iter_sites  # noqa: E402


def main() -> None:
    ref = pd.DataFrame([(s.transcript_id.split(".")[0], s.position, s.kmer)
                        for s in iter_sites(ROOT / "data0" / "dataset0.json.gz")], columns=["t", "p", "kmer"])
    lab = pd.read_csv(ROOT / "data0" / "data.info.labelled")
    lab["t"] = lab.transcript_id.str.split(".").str[0]
    ref = ref.merge(lab[["t", "transcript_position", "label"]].rename(columns={"transcript_position": "p"}),
                    on=["t", "p"])
    print(f"dataset0: {len(ref):,} labelled sites")
    rows = []
    for f in sorted((ROOT / "data" / "sgnex_scores").glob("SGNex_*.csv.gz")):
        d = pd.read_csv(f, usecols=["transcript_id", "transcript_position", "kmer", "n_reads"])
        d["t"] = d.transcript_id.str.split(".").str[0]
        m = ref.merge(d.rename(columns={"transcript_position": "p"}), on=["t", "p"], how="left", suffixes=("", "_s"))
        hit = m.kmer_s.notna()
        rows.append({"sample": f.name[:-7], "labelled sites present %": 100 * hit.mean(),
                     "identical 7-mer %": 100 * (m.kmer == m.kmer_s)[hit].mean(),
                     "median reads at labelled sites": m.n_reads[hit].median(),
                     "labelled sites with <20 reads %": 100 * (m.n_reads[hit] < 20).mean(),
                     "labelled sites with <=3 reads": int((m.n_reads[hit] <= 3).sum())})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("label_mapping_check.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
