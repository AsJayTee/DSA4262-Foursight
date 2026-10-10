"""Export summary data for the dashboard (dashboard/public/data/summary.json).

    python analysis/sgnex/export_dashboard_data.py

Reads the committed preliminary-analysis tables (analysis/sgnex/explore/, written
by explore_lines.py) and writes the small JSON the dashboard's Overview loads. Only
aggregate numbers from public SG-NEx data: no course labels, no reads. Add an
export here for each new view, so the site's data always comes from a script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPLORE = ROOT / "analysis" / "sgnex" / "explore"
OUT = ROOT / "dashboard" / "public" / "data" / "summary.json"


def main() -> None:
    a1 = pd.read_csv(EXPLORE / "A1_reads_per_line.csv")
    reads = [{"line": r.line, "sites": int(r.sites), "median_reads": float(r["median reads"]),
              "pct_1_2": float(r["1-2 reads %"]), "pct_20_plus": float(r["20+ reads %"])} for _, r in a1.iterrows()]
    b = pd.read_csv(EXPLORE / "B_along_genes.csv")
    stop = b[b.profile == "around stop codon"].copy()
    stop["x"] = stop["bin start (nt)"].astype(float) + 50          # bin centres
    wide = stop.pivot(index="x", columns="line", values="mean score").sort_index()
    profile = {"x": wide.index.tolist(), "lines": {c: wide[c].round(3).tolist() for c in wide.columns}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"reads_per_line": reads, "stop_codon_profile": profile}, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
