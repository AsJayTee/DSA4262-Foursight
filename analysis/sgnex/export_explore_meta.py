"""The held-out numbers the Explore page quotes beside a gene's ranking.

    python analysis/site_types/shortlist.py          # first: the results tables
    python analysis/sgnex/export_explore_meta.py

From analysis/site_types/results/ (aggregates only, no labels):
  - gene_best: for genes grouped by where their best site ranks in the sample, the share
    that had a confirmed modified site - at full depth and at one read per site;
  - topk: for a gene's top k sites (k = 1..5), the share of genes where at least one
    was confirmed - the basis for suggesting "test these 3";
  - trust: the labels for the per-site trust note (pairwise accuracy cut-offs, agreed
    with the team: report/task2_plan.md decision 6.4).

Writes dashboard/public/data/explore.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analysis" / "site_types" / "results"
OUT = ROOT / "dashboard" / "public" / "data" / "explore.json"
# Reliability labels from pairwise accuracy. 0.84, not a rounder 0.85: at 0.85 GGACT - the
# textbook m6A motif, 0.840 - would read "moderate", and so would GGACA and GAACC.
TRUST = [{"label": "high", "min": 0.84}, {"label": "moderate", "min": 0.75}, {"label": "low", "min": 0.0}]


def by_line(frame: pd.DataFrame, col: str) -> dict:
    return {r.cell_line.split(" ")[0]: round(float(getattr(r, col)), 3) for r in frame.itertuples()}


def main() -> None:
    g = pd.read_csv(RESULTS / "gene_best.csv").rename(columns={"gene's best site": "band", "share with a modified site": "share"})
    s = pd.read_csv(RESULTS / "shortlist.csv")
    gene_best = {}
    for where, key in [("full reads", "full"), ("1 reads (thinned, 5 draws)", "one_read")]:
        gene_best[key] = [{"band": band, **by_line(x, "share")}
                          for band, x in g[g["evaluated on"] == where].groupby("band", sort=False)]
    full = s[(s["evaluated on"] == "full reads") & (s.method == "model")]
    topk = {int(k): by_line(x, "hit_rate") for k, x in full.groupby("k")}
    OUT.write_text(json.dumps({
        "gene_best": gene_best, "topk": topk, "trust": TRUST,
        "source": "analysis/site_types/results/ (held-out genes; HCT116 and HEK293T)",
    }, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
