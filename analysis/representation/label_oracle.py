"""How well could a PERFECT cell line 1 model do on cell line 2's labels? (10 Oct)

    python analysis/representation/label_oracle.py

On the 67,320 sites in both files, cell line 1's own labels are used as the
score for cell line 2's labels - the best any model that learned cell line 1's
labelling exactly could transfer. Binary labels tie most sites, so it is also
reported with ties broken by a real model's score (h2gcn_aux trained on cell
line 1, seeds averaged), next to that model alone. No torch. Writes
results/label_oracle.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common
import day_significance as D
from m6a import crosssource as xs
from m6a.config import Config


def main() -> None:
    cfg = Config.load(D.BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=D.quiet)
    s0, s1 = sources["dataset0"], sources["data1"]
    n0 = len(s0)
    shared = s0.index.intersection(s1.index)
    i0 = pd.Series(np.arange(n0), index=s0.index)[shared].to_numpy()
    i1 = pd.Series(np.arange(len(s1)), index=s1.index)[shared].to_numpy()
    y0, y1 = s0.y[i0], s1.y[i1]
    model = D.seed_mean("h2gcn_aux", "dataset0", n0)
    m1, m0 = model[n0:][i1], model[:n0][i0]
    rows = []
    for target, y, label, score in (("cell line 2", y1, y0, m1), ("cell line 1", y0, y1, m0)):
        other = "cell line 1" if target == "cell line 2" else "cell line 2"
        for name, s in ((f"{other}'s label alone", label.astype(float)),
                        (f"{other}'s label, ties broken by the model", label + 1e-3 * score),
                        ("model trained on cell line 1 (h2gcn_aux)", score)):
            rows.append({"scored against": f"{target} labels", "score": name, "sites": len(shared),
                         "positive %": 100 * y.mean(), "pr_auc": average_precision_score(y, s),
                         "roc_auc": roc_auc_score(y, s)})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    table.to_csv(common.RESULTS / "label_oracle.csv", index=False)


if __name__ == "__main__":
    main()
