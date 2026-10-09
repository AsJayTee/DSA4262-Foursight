"""One-page summary of the 2026-10-08 batch -> results/day_summary.md

    python analysis/representation/summarize_day.py

Per model: mean and SD over seeds of the gain over quantiles + LightGBM, for
the unseen-cell-line test (trained on cell line 1, scored on cell line 2), the
both-cell-lines model on cell line 2 and on cell line 1, and data1's new genes;
plus the difference from h2gcn_aux on the SAME seeds (matched baseline).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common

MODELS = ["h2gcn_aux", "h2gcn_aux_r100", "fk_band", "fk_band_shuffled", "fk_kernel", "res_gate",
          "res_nogate", "res_nodrop", "scalar_msg", "scalar_random", "scalar_drop", "gps", "h2gcn_twohead_aux"]


def read(model: str, seed: int):
    f = common.RESULTS / f"{model}_nn__xsrc{'' if seed == 0 else f'__seed{seed}'}.json"
    if not f.exists():
        return None
    x = json.load(open(f))["cross_source"]
    a = x["arms"]
    row = {"seed": seed}
    if "dataset0" in a:
        row["unseen cell line (d0 -> d1)"] = a["dataset0"]["data1"]["gain"]
        row["new genes"] = x.get("data1_new_genes", {}).get("gain", np.nan)
    if "pooled_both" in a:
        row["both -> cell line 2"] = a["pooled_both"]["data1"]["gain"]
        row["both -> cell line 1"] = a["pooled_both"]["dataset0"]["gain"]
    return row


def main() -> None:
    rows = []
    for m in MODELS:
        for s in (0, 1, 2):
            r = read(m, s)
            if r:
                rows.append({"model": m, **r})
    t = pd.DataFrame(rows)
    cols = ["unseen cell line (d0 -> d1)", "new genes", "both -> cell line 2", "both -> cell line 1"]
    lines = ["# 2026-10-08 batch: constrained corroboration + GraphGPS control", "",
             "Gain in PR AUC over quantiles + LightGBM (mean +- SD over seeds; n = seeds).", "",
             "| model | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for m, g in t.groupby("model", sort=False):
        cells = [f"{g[c].mean():+.4f} +- {g[c].std(ddof=1):.4f}" if g[c].notna().sum() > 1
                 else (f"{g[c].mean():+.4f}" if g[c].notna().any() else "-") for c in cols]
        lines.append(f"| {m} | {len(g)} | " + " | ".join(cells) + " |")
    base = t[t.model == "h2gcn_aux"].set_index("seed")
    lines += ["", "Matched difference from h2gcn_aux on the same seeds (positive = better):", "",
              "| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, g in t[t.model != "h2gcn_aux"].groupby("model", sort=False):
        g = g.set_index("seed")
        common_seeds = g.index.intersection(base.index)
        cells = []
        for c in cols:
            d = (g.loc[common_seeds, c] - base.loc[common_seeds, c]).dropna()
            cells.append(f"{d.mean():+.4f} (n={len(d)})" if len(d) else "-")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    out = common.RESULTS / "day_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
