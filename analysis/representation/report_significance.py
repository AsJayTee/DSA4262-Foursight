"""Significance of the report's headline comparisons (Section 4.1, 16 Oct).

    python analysis/representation/report_significance.py

No torch: reads the cached out-of-fold predictions (.cache/representation/xsrc_nets)
and m6Anet's pretrained scores, as report_models.py does. Same paired, gene-resampled
PR AUC difference as day_significance.py (1,000 resamples of whole genes); the
one-sided p-value is 1 - win rate (the share of resamples where the first model
is NOT better).

  both cell lines   the final ensemble against quantiles + LightGBM and the
                    logistic baseline, scored on each cell line
  cell line 1 only  the final ensemble's two constrained parts (the two-head
                    network cannot be trained without cell line 2's labels)
                    against m6Anet and LightGBM, on cell line 2 and its new genes

Every model's seeds are rank-averaged within each file first.

Writes results/report_significance.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import day_significance as D
import report_models as R
from m6a import crosssource as xs
from m6a.config import Config


def seed_avg(key: str, arm: str, n0: int, extra: dict) -> np.ndarray:
    seeds = R.per_seed(key, arm, n0, extra)
    return np.mean([D.rank_within(p, n0) for p in seeds.values()], 0)


def main() -> None:
    cfg = Config.load(D.BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=D.quiet)
    s0, s1 = sources["dataset0"], sources["data1"]
    n0 = len(s0)
    m6 = np.concatenate([R.m6anet_scores(common.ROOT / "data/m6anet/dataset0_pretrained_out/data.site_proba.csv", s0.index),
                         R.m6anet_scores(common.ROOT / "data/m6anet/data1_pretrained_out/data.site_proba.csv", s1.index)])
    extra = {"logistic": R.logistic_scores(sources), "m6anet": np.nan_to_num(m6, nan=-1.0)}
    y = {"dataset0": s0.y, "data1": s1.y}
    genes = {"dataset0": np.asarray(s0.genes), "data1": np.asarray(s1.genes)}
    new = ~np.isin(genes["data1"], genes["dataset0"])

    def cut(v, file, mask=None):
        v = v[:n0] if file == "dataset0" else v[n0:]
        return v if mask is None else v[mask]

    comparisons = [  # (model, baseline, arm, file, mask, target label)
        ("ens_final", "lightgbm", "pooled_both", "dataset0", None, "both -> cell line 1"),
        ("ens_final", "lightgbm", "pooled_both", "data1", None, "both -> cell line 2"),
        ("ens_final", "logistic", "pooled_both", "dataset0", None, "both -> cell line 1"),
        ("ens_final", "logistic", "pooled_both", "data1", None, "both -> cell line 2"),
        ("ens_final", "m6anet", "pooled_both", "data1", None, "both -> cell line 2"),
    ]
    for m in ("res_gate", "scalar_drop", "lightgbm"):
        for base in ("m6anet", "lightgbm"):
            if m == base:
                continue
            comparisons += [(m, base, "dataset0", "data1", None, "unseen cell line"),
                            (m, base, "dataset0", "data1", new, "unseen cell line, new genes")]
    # Added 16 Oct for the abstract. Appended last so the shared random stream
    # leaves every earlier comparison's interval as already quoted. m6Anet was
    # trained on cell line 1's data, so this comparison favours m6Anet.
    comparisons.append(("ens_final", "m6anet", "pooled_both", "dataset0", None, "both -> cell line 1"))

    rng = np.random.default_rng(4262)
    rows = []
    for model, base, arm, file, mask, target in comparisons:
        a, b = seed_avg(model, arm, n0, extra), seed_avg(base, arm, n0, extra)
        yy = y[file] if mask is None else y[file][mask]
        gg = genes[file] if mask is None else genes[file][mask]
        d, lo, hi, win = D.paired(yy, cut(a, file, mask), cut(b, file, mask), gg, rng)
        rows.append({"model": model, "vs": base, "target": target, "diff": d, "ci_low": lo,
                     "ci_high": hi, "one-sided p": 1 - win})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(common.RESULTS / "report_significance.csv", index=False)


if __name__ == "__main__":
    main()
