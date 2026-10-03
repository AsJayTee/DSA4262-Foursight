"""Which models are worth combining? Rank-averaged ensembles, scored for free.

    python analysis/representation/ensemble.py

Every model in xsrc_nets.py and the quantiles + LightGBM baseline was scored
on the same held-out sites of the cross-source split, so an ensemble is just
an average of those scores - nothing is retrained. Scores are converted to
ranks within each file first: the networks output logits and LightGBM
probabilities, and an average of the raw numbers would let the wider scale
dominate. Each ensemble is judged like any model (decision 0032): gain over
the LightGBM baseline under each labelling, the worse one ranks.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.stats import rankdata

import common
import xsrc_nets as X
from m6a import crosssource as xs

MEMBERS = ("h2gcn", "h2gcn_local", "h2gcn_twohead", "h2gcn_local_deep", "h2gcn_aux", "deepset")
# Finalists retrained with seeds 1 and 2 (both files only): their seed average
# is an ensemble too, and the usual cheap gain.
SEEDED = ("h2gcn", "h2gcn_local", "h2gcn_twohead")


def seeded(model: str, sources, seed: int):
    X.SEED = seed
    try:
        return X.candidate(model, sources, 1.0)
    finally:
        X.SEED = 0


def ranks(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


def main() -> None:
    sources, _ = X.load(1.0)
    base = X.baseline_scores(sources, 1.0, print)
    pool = {"lightgbm": base}
    for m in MEMBERS:
        cand = X.candidate(m, sources, 1.0)
        if cand is not None:
            pool[m] = cand
    for m in SEEDED:
        runs = [seeded(m, sources, s) for s in (0, 1, 2)]
        if all(r is not None and "pooled_both" in r for r in runs):
            pool[f"{m} x3 seeds"] = {"pooled_both": {s: np.mean([ranks(r["pooled_both"][s]) for r in runs], 0)
                                                     for s in xs.SOURCES}}
            for i, r in enumerate(runs[1:], 1):
                pool[f"{m} seed {i}"] = {"pooled_both": r["pooled_both"]}
    rows = []
    for arm in X.ARMS:
        names = [n for n in pool if arm in pool[n]]
        # Seeds of one model are compared alone, not mixed into combinations.
        mixable = [n for n in names if " seed " not in n]
        combos = [(n,) for n in names] + [c for k in (2, 3) for c in itertools.combinations(mixable, k)]
        for combo in combos:
            avg = {s: np.mean([ranks(pool[n][arm][s]) for n in combo], 0) for s in xs.SOURCES}
            block = xs.summarise(sources, {arm: avg}, {arm: base[arm]}, n=200, headline_arm=arm)
            r = block["arms"][arm]
            ng = block.get("data1_new_genes", {}).get("gain", np.nan)
            rows.append({"arm": arm, "members": " + ".join(combo), "dataset0": r["dataset0"]["gain"],
                         "data1": r["data1"]["gain"], "worst": r["gain_worst"], "new_genes": ng,
                         "data1_pr_auc": r["data1"]["pr_auc"]})
    rows.sort(key=lambda r: (r["arm"], -r["worst"]))
    for r in rows:
        print(f"{r['arm']:12s} {r['members']:45s} worst {r['worst']:+.4f}  dataset0 {r['dataset0']:+.4f}  "
              f"data1 {r['data1']:+.4f}  new genes {r['new_genes']:+.4f}  data1 PR {r['data1_pr_auc']:.4f}")
    (common.RESULTS / "ensembles.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
