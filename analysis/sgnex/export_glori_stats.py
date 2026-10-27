"""Data for the Model stats page's GLORI section.

    python analysis/sgnex/glori_check.py             # first: model vs GLORI
    python analysis/sgnex/m6anet_glori_compare.py    # first: m6Anet on the same sites
    python analysis/sgnex/export_glori_stats.py

Turns those two scripts' result tables into one small JSON. Aggregates only, from public
data (SG-NEx, GLORI on GEO); no per-site values leave analysis/. Definitions behind every
number: report/sgnex_glori_findings.md.

  bands         depth bands (pooled reads per site)
  lines         held-out lines with GLORI: A549, H9, HepG2. HCT116 is left out because
                the model was trained on that cell line.
  by_line       per line, for each band and for "all" (every site ranked together): GLORI positive rate (= a random ranking's PR AUC and
                top-1% hit rate), and PR AUC and top-1% share confirmed for the model, m6Anet
                (pretrained, its 20-read minimum lowered) and the motif rule
  published     the share of graded sites, and of GLORI sites, with >= 20 reads: what m6Anet
                as published scores
  level         Spearman correlation between score and GLORI's modified fraction, GLORI
                sites with >= 20 reads, model and m6Anet

Model and motif-rule values come from glori_check.py; m6Anet's from m6anet_glori_compare.py,
which scores the same sites less the few m6Anet drops (34 of 518,544 in A549, 60 in H9).

Writes dashboard/public/data/glori_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GLORI = ROOT / "analysis" / "sgnex" / "glori" / "metrics.csv"
M6ANET = ROOT / "analysis" / "sgnex" / "m6anet_glori" / "metrics.csv"
OUT = ROOT / "dashboard" / "public" / "data" / "glori_stats.json"
LINES = ["A549", "H9", "HepG2"]
BANDS = {"1-2 reads": "1-2", "3-9 reads": "3-9", "10-19 reads": "10-19", "20+ reads": "20+"}
# GLORI sites per line, from the authors' files (analysis/sgnex/glori/mapping.csv).
SOURCES = {
    "A549": "GSE290408, METTL3-knockdown filtered",
    "H9": "GSE290408, METTL3-knockdown filtered",
    "HepG2": "GSE301419, control cells",
}


def main() -> None:
    for path, cmd in ((GLORI, "glori_check.py"), (M6ANET, "m6anet_glori_compare.py")):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python analysis/sgnex/{cmd}")
    g = pd.read_csv(GLORI)
    m = pd.read_csv(M6ANET)
    mapping = pd.read_csv(ROOT / "analysis" / "sgnex" / "glori" / "mapping.csv").set_index("line")

    by_line, published = {}, {}
    for line in LINES:
        by_line[line] = {}
        for band, key in BANDS.items():
            ours = g[(g.line == line) & (g.band == band) & (g.scorer == "score_raw")].iloc[0]
            motif = g[(g.line == line) & (g.band == band) & (g.scorer == "motif_only")].iloc[0]
            them = m[(m.line == line) & (m.band == band)].iloc[0]
            by_line[line][key] = {
                "sites": int(ours.n),
                "positive_rate": round(float(ours.positive_rate), 4),
                "model": {"pr_auc": round(float(ours.pr_auc), 4), "top1": round(float(ours.precision_top1pct), 4)},
                "m6anet": {"pr_auc": round(float(them.pr_auc_m6anet), 4), "top1": round(float(them.top1pct_m6anet), 4),
                           "diff_lo": round(float(them.diff_lo), 4), "diff_hi": round(float(them.diff_hi), 4)},
                "motif": {"pr_auc": round(float(motif.pr_auc), 4), "top1": round(float(motif.precision_top1pct), 4)},
            }
        bands = dict(by_line[line])
        # Every site of the line ranked together, deep and shallow, as Explore ranks them.
        ours = g[(g.line == line) & (g.band == "all") & (g.scorer == "score_raw")].iloc[0]
        motif = g[(g.line == line) & (g.band == "all") & (g.scorer == "motif_only")].iloc[0]
        them = m[(m.line == line) & (m.band == "all depths")].iloc[0]
        by_line[line]["all"] = {
            "sites": int(ours.n), "positive_rate": round(float(ours.positive_rate), 4),
            "model": {"pr_auc": round(float(ours.pr_auc), 4), "top1": round(float(ours.precision_top1pct), 4)},
            "m6anet": {"pr_auc": round(float(them.pr_auc_m6anet), 4), "top1": round(float(them.top1pct_m6anet), 4),
                       "diff_lo": round(float(them.diff_lo), 4), "diff_hi": round(float(them.diff_hi), 4)},
            "motif": {"pr_auc": round(float(motif.pr_auc), 4), "top1": round(float(motif.precision_top1pct), 4)},
        }
        total = sum(b["sites"] for b in bands.values())
        pos = {k: b["sites"] * b["positive_rate"] for k, b in bands.items()}
        published[line] = {"sites": round(bands["20+"]["sites"] / total, 4),
                           "glori_sites": round(pos["20+"] / sum(pos.values()), 4),
                           "glori_total": int(mapping.loc[line, "glori_sites"]), "source": SOURCES[line]}

    level = {}
    for line_block in (ROOT / "analysis" / "sgnex" / "m6anet_glori" / "summary.md").read_text(encoding="utf8").split(
            "Spearman, GLORI sites with 20+ reads)")[1].splitlines():
        parts = line_block.split()
        if len(parts) == 4 and parts[0] in LINES:
            level[parts[0]] = {"model": float(parts[2]), "m6anet": float(parts[3])}

    OUT.write_text(json.dumps({"bands": list(BANDS.values()), "lines": LINES, "by_line": by_line,
                               "published": published, "level": level}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
