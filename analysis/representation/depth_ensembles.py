"""The shipped ensemble (and its parts) at 1, 3 and 10 reads per site, no torch (10 Oct).

    python analysis/representation/depth_ensembles.py

Reads the per-site scores depth_rescore.py saved (.cache/representation/xsrc_nets/
depth_scores/, arm pooled_both: the shipped setting) and LightGBM fitted at full
depth scored on the same thinned reads (baseline_lgbm_depths.npz). Every site's
reads - and its neighbours' - are thinned to the depth, keyed as everywhere else.
Ensembles are the evaluated combination: each network seed-averaged by within-file
rank, then the models averaged. Full depth comes from report_models_pooled.csv.
Writes results/depth_ensembles.csv; fig_depth reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common
import day_significance as D
from m6a import crosssource as xs
from m6a.config import Config

DEPTHS = (1, 3, 10)
SCORES = D.NETS / "depth_scores"
MODELS = {"h2gcn_twohead_aux": ["h2gcn_twohead_aux"], "h2gcn_aux": ["h2gcn_aux"], "res_gate": ["res_gate"],
          "scalar_drop": ["scalar_drop"],
          "ens_intermediate": ["h2gcn_twohead_aux", "h2gcn_aux"],
          "ens_final": ["h2gcn_twohead_aux", "res_gate", "scalar_drop"]}


def main() -> None:
    cfg = Config.load(D.BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=D.quiet)
    n0 = len(sources["dataset0"])
    y = np.concatenate([sources["dataset0"].y, sources["data1"].y])
    file = np.r_[np.zeros(n0, int), np.ones(len(y) - n0, int)]
    lgbm = np.load(D.NETS / "baseline_lgbm_depths.npz")

    def seed_avg(model: str, d: int) -> np.ndarray:
        paths = sorted(SCORES.glob(f"{model}_pooled_both_s*.npz"))
        if not paths:
            raise SystemExit(f"no depth scores for {model}: run depth_rescore.py --models {model} --arms pooled_both")
        return np.mean([D.rank_within(np.load(p)[f"d{d}"], n0) for p in paths], 0)

    rows = []
    for d in DEPTHS:
        base = np.concatenate([lgbm[f"d{d}__pooled_both__dataset0"], lgbm[f"d{d}__pooled_both__data1"]])
        scored = {"lightgbm": base}
        for name, parts in MODELS.items():
            scored[name] = np.mean([seed_avg(m, d) for m in parts], 0)
        for name, s in scored.items():
            for f, line in ((0, "cell line 1"), (1, "cell line 2")):
                m = file == f
                rows.append({"model": name, "reads": d, "scored on": line,
                             "seeds": len(sorted(SCORES.glob(f"{name}_pooled_both_s*.npz"))) or None,
                             "pr_auc": average_precision_score(y[m], s[m]), "roc_auc": roc_auc_score(y[m], s[m])})
    table = pd.DataFrame(rows)
    print(table.pivot_table(index="model", columns=["scored on", "reads"], values="pr_auc").round(3).to_string())
    table.to_csv(common.RESULTS / "depth_ensembles.csv", index=False)


if __name__ == "__main__":
    main()
