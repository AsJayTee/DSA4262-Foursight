"""Are the 8 Oct batch's gains real across GENES, and does the model read each cell line's own labels?

    python analysis/representation/day_significance.py

No torch: reads the out-of-fold predictions xsrc_nets.py cached (copy them off
Ronin first, `.cache/representation/xsrc_nets/*.npy`).

1. Paired, gene-resampled differences from h2gcn_aux. day_summary.md gives
   seed spread only, which says training is stable, not that a gain holds
   across genes. Each model's three seeds are rank-averaged first (the same
   for h2gcn_aux), then PR AUC difference on identical sites with a 95%
   interval from resampling whole genes; plus the per-seed differences.
2. Ensembles: res_gate + scalar_msg (rank average), and against a seed-0
   stand-in for the shipped ensemble (h2gcn_twohead_aux + h2gcn_aux).
3. Discordant sites. 67k sites are in both files; where their labels
   disagree, does a model trained on cell line 1 ONLY rank the site higher
   in the file whose label says modified? Scores are percentile ranks within
   each file (the files differ in score level). AUC 0.5 = the model cannot
   tell which cell line carries the modification; it is then applying one
   prior to both. Also: the two files' labels agree on how many sites.

Writes results/day_significance.csv and results/discordant_sites.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import common
from m6a import crosssource as xs
from m6a.config import Config

NETS = common.OUT / "xsrc_nets"
BASELINE = common.ROOT / "configs" / "quantiles.yaml"
DESIGNS = ["h2gcn_aux_r100", "fk_band", "fk_kernel", "res_gate", "scalar_msg", "gps"]
CONTROLS = ["fk_band_shuffled", "res_nogate", "res_nodrop", "scalar_random"]
N_BOOT = 1000


def quiet(*_a, **_k) -> None:
    pass


def preds(model: str, arm: str, seed: int) -> np.ndarray | None:
    s = "" if seed == 0 else f"_s{seed}"
    paths = [NETS / f"{model}_{arm}_fold{f}{s}.npy" for f in range(5)]
    if not all(p.exists() for p in paths):
        return None
    return np.nanmax(np.stack([np.load(p) for p in paths]), 0)


def rank_within(score: np.ndarray, n0: int) -> np.ndarray:
    """Percentile rank inside each file, so the two files' scores are comparable."""
    out = np.empty_like(score, dtype=float)
    out[:n0] = rankdata(score[:n0]) / n0
    out[n0:] = rankdata(score[n0:]) / (len(score) - n0)
    return out


def seed_mean(model: str, arm: str, n0: int, seeds=(0, 1, 2)) -> np.ndarray | None:
    got = [p for s in seeds if (p := preds(model, arm, s)) is not None]
    return np.mean([rank_within(p, n0) for p in got], 0) if got else None


def paired(y, a, b, genes, rng) -> tuple[float, float, float, float]:
    codes = pd.factorize(genes)[0]
    members = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_list()
    d = average_precision_score(y, a) - average_precision_score(y, b)
    boots, wins = [], 0
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if y[idx].sum() == 0:
            continue
        diff = average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx])
        boots.append(diff)
        wins += diff > 0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d, lo, hi, wins / len(boots)


def main() -> None:
    cfg = Config.load(BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=quiet)
    s0, s1 = sources["dataset0"], sources["data1"]
    n0 = len(s0)
    y = {"dataset0": s0.y, "data1": s1.y}
    genes = {"dataset0": np.asarray(s0.genes), "data1": np.asarray(s1.genes)}
    new = ~np.isin(genes["data1"], genes["dataset0"])
    targets = [  # (column, arm, file, row mask within file)
        ("unseen cell line", "dataset0", "data1", None),
        ("new genes", "dataset0", "data1", new),
        ("both -> cell line 2", "pooled_both", "data1", None),
        ("both -> cell line 1", "pooled_both", "dataset0", None),
    ]

    def cut(score, file, mask):
        v = score[:n0] if file == "dataset0" else score[n0:]
        return v if mask is None else v[mask]

    rows = []
    rng = np.random.default_rng(4262)
    for model in DESIGNS + CONTROLS + ["res_gate+scalar_msg"]:
        for col, arm, file, mask in targets:
            if model == "res_gate+scalar_msg":
                parts = [seed_mean(m, arm, n0) for m in ("res_gate", "scalar_msg")]
                cand = None if any(p is None for p in parts) else np.mean(parts, 0)
                base = seed_mean("h2gcn_aux", arm, n0)
                seeds = "0-2"
            else:
                seeds_have = [s for s in (0, 1, 2) if preds(model, arm, s) is not None]
                cand = seed_mean(model, arm, n0, seeds_have)
                base = seed_mean("h2gcn_aux", arm, n0, seeds_have)
                seeds = ",".join(map(str, seeds_have))
            if cand is None or base is None:
                continue
            yy = y[file] if mask is None else y[file][mask]
            gg = genes[file] if mask is None else genes[file][mask]
            d, lo, hi, win = paired(yy, cut(cand, file, mask), cut(base, file, mask), gg, rng)
            rows.append({"model": model, "vs": "h2gcn_aux", "seeds": seeds, "target": col,
                         "diff": d, "ci_low": lo, "ci_high": hi, "gene-resample win %": 100 * win})
            print(rows[-1], flush=True)

    # Seed-0 stand-in for the shipped ensemble (both files only; twohead has seed 0 only here).
    shipped = [preds(m, "pooled_both", 0) for m in ("h2gcn_twohead_aux", "h2gcn_aux")]
    if all(p is not None for p in shipped):
        ship = np.mean([rank_within(p, n0) for p in shipped], 0)
        for name, parts in (("res_gate", ["res_gate"]), ("scalar_msg", ["scalar_msg"]),
                            ("res_gate+scalar_msg", ["res_gate", "scalar_msg"]),
                            ("twohead+res_gate", ["h2gcn_twohead_aux", "res_gate"]),
                            ("twohead+res_gate+scalar_msg", ["h2gcn_twohead_aux", "res_gate", "scalar_msg"]),
                            ("shipped+res_gate+scalar_msg",
                             ["h2gcn_twohead_aux", "h2gcn_aux", "res_gate", "scalar_msg"])):
            cand = np.mean([rank_within(preds(m, "pooled_both", 0), n0) for m in parts], 0)
            for col, file in (("both -> cell line 2", "data1"), ("both -> cell line 1", "dataset0")):
                d, lo, hi, win = paired(y[file], cut(cand, file, None), cut(ship, file, None),
                                        genes[file], rng)
                rows.append({"model": name, "vs": "shipped ensemble (seed 0)", "seeds": "0",
                             "target": col, "diff": d, "ci_low": lo, "ci_high": hi,
                             "gene-resample win %": 100 * win})
                print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(common.RESULTS / "day_significance.csv", index=False)

    # 3. Discordant sites.
    pos0 = pd.Series(np.arange(n0), index=s0.index)
    pos1 = pd.Series(np.arange(len(s1)), index=s1.index)
    shared = s0.index.intersection(s1.index)
    i0, i1 = pos0[shared].to_numpy(), pos1[shared].to_numpy()
    y0, y1 = s0.y[i0], s1.y[i1]
    print(f"\nshared sites {len(shared)}: both 1 {int(((y0 == 1) & (y1 == 1)).sum())}, "
          f"only cell line 1 {int(((y0 == 1) & (y1 == 0)).sum())}, "
          f"only cell line 2 {int(((y0 == 0) & (y1 == 1)).sum())}, both 0 {int(((y0 == 0) & (y1 == 0)).sum())}")
    disc = y0 != y1
    lgbm = np.load(NETS / "baseline_lgbm.npz")
    scorers = {"quantiles + LightGBM": np.concatenate([lgbm["dataset0__dataset0"], lgbm["dataset0__data1"]])}
    for m in ("deepset", "h2gcn_aux", "res_gate", "scalar_msg", "gps"):
        p = preds(m, "dataset0", 0)
        if p is not None:
            scorers[m] = p
    out = []
    for name, score in scorers.items():
        r = rank_within(score, n0)
        delta = r[n0:][i1] - r[:n0][i0]          # rank in cell line 2's file minus cell line 1's
        out.append({
            "model (trained on cell line 1 only, seed 0)": name,
            "discordant sites": int(disc.sum()),
            # Positive class: modified in cell line 2 only. 0.5 = cannot tell which line has it.
            "which-line AUC": roc_auc_score(y1[disc], delta[disc]),
            "mean rank shift, only cell line 2 modified": delta[(y0 == 0) & (y1 == 1)].mean(),
            "mean rank shift, only cell line 1 modified": delta[(y0 == 1) & (y1 == 0)].mean(),
            "mean rank shift, concordant": delta[~disc].mean(),
            "rank corr between files (shared sites)": pd.Series(r[:n0][i0]).corr(pd.Series(r[n0:][i1]), method="spearman"),
        })
    disc_table = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(disc_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    disc_table.to_csv(common.RESULTS / "discordant_sites.csv", index=False)


if __name__ == "__main__":
    main()
