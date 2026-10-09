"""Is "train on one cell line, predict another" the reason stronger models fail? (9 Oct)

    python analysis/representation/data1_arm_eval.py

h2gcn_aux and gps, seed 0, trained three ways and scored on cell line 2's
held-out genes (identical sites):
  cell line 1 only  (arm dataset0)     - the cross-cell-line test
  cell line 2 only  (arm data1)        - the within-cell-line reference
  both              (arm pooled_both)

  - gps vs h2gcn_aux WITHIN cell line 2: if gps wins there, capacity helps
    inside any one cell line and its cross-line failure is the label shift; if
    not, its cell line 1 gain is specific to cell line 1.
  - train 2 -> test 2 minus train 1 -> test 2: the part of the cross-line drop
    that cell line 2's own labels recover (the rest is not learnable from
    either line's labels by this model).
Paired, gene-resampled intervals (day_significance.paired). Needs the data1-arm
fits (run_data1_arm.py). Writes results/data1_arm.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import common
import day_significance as D
from m6a import crosssource as xs
from m6a.config import Config

ARM_NAMES = {"dataset0": "cell line 1 only", "data1": "cell line 2 only", "pooled_both": "both"}


def main() -> None:
    cfg = Config.load(D.BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=D.quiet)
    n0, s1 = len(sources["dataset0"]), sources["data1"]
    y, genes = s1.y, np.asarray(s1.genes)
    score = {(m, a): D.preds(m, a, 0)[n0:] for m in ("h2gcn_aux", "gps") for a in ARM_NAMES}
    rng = np.random.default_rng(4262)
    rows = [{"comparison": f"{m}, trained on {ARM_NAMES[a]}: PR AUC on cell line 2",
             "value": average_precision_score(y, v), "ci_low": np.nan, "ci_high": np.nan, "win %": np.nan}
            for (m, a), v in score.items()]
    pairs = [(f"gps - h2gcn_aux, both trained on {ARM_NAMES[a]}", ("gps", a), ("h2gcn_aux", a)) for a in ARM_NAMES]
    pairs += [(f"{m}: trained on cell line 2 - trained on cell line 1", (m, "data1"), (m, "dataset0"))
              for m in ("h2gcn_aux", "gps")]
    for name, a, b in pairs:
        d, lo, hi, win = D.paired(y, score[a], score[b], genes, rng)
        rows.append({"comparison": name, "value": d, "ci_low": lo, "ci_high": hi, "win %": 100 * win})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    table.to_csv(common.RESULTS / "data1_arm.csv", index=False)


if __name__ == "__main__":
    main()
