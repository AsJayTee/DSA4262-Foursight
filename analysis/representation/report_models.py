"""The report's model comparison: per-fold distribution and ranking scatter (10 Oct).

    python analysis/representation/report_models.py
    python analysis/representation/report_models.py --plot-only   # redraw from the saved tables

No torch. Reads the cached out-of-fold predictions (.cache/representation/xsrc_nets,
copied off Ronin), every model on the same union gene split (seed 4262):

  two training versions   "cell line 1 only" (arm dataset0: cell line 2 is a cell
                          line the model never saw) and "both cell lines" (arm
                          pooled_both: the shipped setting; cell line 2's held-out
                          GENES are unseen, the cell line is not)
  Fig A  per fold and seed, the PR AUC gain over quantiles + LightGBM on the SAME
         fold (seed-matched) - folds differ more in difficulty than most models
         differ, so raw per-fold PR AUC hides every comparison
  Fig B  PR AUC (and ROC AUC) on cell line 1 against cell line 2, all held-out
         sites, seeds rank-averaged; top right is better

Writes results/report_models_folds.csv, results/report_models_pooled.csv and
report/figures/models_{distribution,scatter}_{cl1,both}.png.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common
import day_significance as D
from m6a import crosssource as xs
from m6a import registry
from m6a.config import Config

sys.path.insert(0, str(common.ROOT / "analysis" / "m6anet"))
from xsrc_m6anet import m6anet_scores  # noqa: E402

FIGURES = common.ROOT / "report" / "figures"
# Three categorical hues (validated all-pairs, light) plus grey for context.
GREY, ORANGE, BLUE, AQUA = "#75746f", "#eb6834", "#2a78d6", "#1baf7a"
GROUPS = {GREY: "no neighbours (references, reads only)", ORANGE: "free use of neighbours",
          BLUE: "constrained corroboration", AQUA: "ensembles"}
# key, label, colour, control (hollow marker)
MODELS = [
    ("logistic", "Logistic regression (baseline)", GREY, False),
    ("lightgbm", "Quantiles + LightGBM", GREY, False),
    ("m6anet", "m6Anet (pretrained on HCT116)", GREY, False),
    ("deepset", "DeepSet (reads only)", GREY, False),
    ("set_transformer", "Set Transformer (reads only)", GREY, False),
    ("gcn", "GCN", ORANGE, False),
    ("gat", "GAT", ORANGE, False),
    ("h2gcn_aux", "H2GCN + own-reads head", ORANGE, False),
    ("h2gcn_aux_r400", "H2GCN, 400 nt context", ORANGE, False),
    ("gps", "Graph transformer (GPS)", ORANGE, False),
    ("h2gcn_twohead_aux", "Two-head H2GCN", ORANGE, False),
    ("scalar_msg", "Scalar messages", BLUE, False),
    ("scalar_random", "  control: random neighbours", BLUE, True),
    ("res_gate", "Residual + dropout", BLUE, False),
    ("res_nodrop", "  control: no dropout", BLUE, True),
    ("scalar_drop", "Scalar messages + dropout", BLUE, False),
    ("ens_intermediate", "Intermediate leaderboard ensemble", AQUA, False),
    ("ens_final", "Final ensemble", AQUA, False),
]
ENSEMBLES = {"ens_intermediate": ["h2gcn_twohead_aux", "h2gcn_aux"],
             "ens_final": ["h2gcn_twohead_aux", "res_gate", "scalar_drop"]}
ARMS = {"cl1": ("dataset0", "trained on cell line 1 only"), "both": ("pooled_both", "trained on both cell lines")}
FILES = (("dataset0", "cell line 1"), ("data1", "cell line 2"))


class WeightedLogistic(registry.get("models", "logistic")):
    """baseline_logistic, taking the per-site weights the both-cell-lines arm uses
    (a shared site's two copies weigh 1/2 each). Here, not in m6a.models, so the
    shared model is untouched; sklearn's pipeline routes the weight to the classifier."""

    def fit(self, X, y, groups=None, sample_weight=None) -> None:
        self.columns = list(X.columns)
        self.pipeline.fit(X.values, y, **({} if sample_weight is None else {"clf__sample_weight": sample_weight}))


def logistic_scores(sources_q) -> dict:
    """baseline_logistic (configs/baseline.yaml) out of fold, cached; rows in the quantile sources' order."""
    path = D.NETS / "baseline_logistic.npz"
    if not path.exists():
        cfg = Config.load(common.ROOT / "configs" / "baseline.yaml")
        src = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=D.quiet)
        oof = xs.out_of_fold(src, WeightedLogistic, cfg.model_params, [None],
                             arms=["dataset0", "pooled_both"], log=D.quiet)
        aligned = {f"{a}__{f}": pd.Series(oof[a][f], index=src[f].index).reindex(sources_q[f].index).to_numpy()
                   for a in ("dataset0", "pooled_both") for f, _ in FILES}
        np.savez(path, **aligned)
    stored = np.load(path)
    return {a: np.concatenate([stored[f"{a}__dataset0"], stored[f"{a}__data1"]]) for a in ("dataset0", "pooled_both")}


def per_seed(key: str, arm: str, n0: int, extra: dict) -> dict[int, np.ndarray]:
    """seed -> scores over both files (dataset0 rows first), or {} if not available."""
    if key == "lightgbm":
        out = {}
        for seed, suffix in ((0, ""), (1, "_s1"), (2, "_s2")):
            p = D.NETS / f"baseline_lgbm{suffix}.npz"
            if p.exists():
                z = np.load(p)
                out[seed] = np.concatenate([z[f"{arm}__dataset0"], z[f"{arm}__data1"]])
        return out
    if key == "logistic":
        return {0: extra["logistic"][arm]}
    if key == "m6anet":
        return {0: extra["m6anet"]}
    if key in ENSEMBLES:
        out = {}
        for seed in (0, 1, 2):
            parts = [D.preds(m, arm, seed) for m in ENSEMBLES[key]]
            if all(p is not None for p in parts):
                out[seed] = np.mean([D.rank_within(p, n0) for p in parts], 0)
        return out
    return {s: p for s in (0, 1, 2) if (p := D.preds(key, arm, s)) is not None}


def collect():
    cfg = Config.load(D.BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=D.quiet)
    s0, s1 = sources["dataset0"], sources["data1"]
    n0 = len(s0)
    y = np.concatenate([s0.y, s1.y])
    folds = np.concatenate([s0.folds, s1.folds])
    file = np.r_[np.zeros(n0, int), np.ones(len(s1), int)]
    m6 = np.concatenate([m6anet_scores(common.ROOT / "data/m6anet/dataset0_pretrained_out/data.site_proba.csv", s0.index),
                         m6anet_scores(common.ROOT / "data/m6anet/data1_pretrained_out/data.site_proba.csv", s1.index)])
    # m6Anet scored all but 10 data1 sites; those rank last rather than vanish.
    extra = {"logistic": logistic_scores(sources), "m6anet": np.nan_to_num(m6, nan=-1.0)}
    fold_rows, pooled_rows = [], []
    for version, (arm, _) in ARMS.items():
        lgbm = per_seed("lightgbm", arm, n0, extra)
        for key, label, _, _ in MODELS:
            seeds = per_seed(key, arm, n0, extra)
            if not seeds:
                continue
            for seed, score in seeds.items():
                base = lgbm.get(seed, lgbm[0])
                for f, (_, fname) in enumerate(FILES):
                    for fold in range(5):
                        m = (file == f) & (folds == fold)
                        ap = average_precision_score(y[m], score[m])
                        fold_rows.append({"version": version, "model": key, "label": label, "seed": seed,
                                          "scored on": fname, "fold": fold, "pr_auc": ap,
                                          "gain over lightgbm": ap - average_precision_score(y[m], base[m])})
            avg = np.mean([D.rank_within(s, n0) for s in seeds.values()], 0)
            for f, (_, fname) in enumerate(FILES):
                m = file == f
                pooled_rows.append({"version": version, "model": key, "label": label, "seeds": len(seeds),
                                    "scored on": fname, "pr_auc": average_precision_score(y[m], avg[m]),
                                    "roc_auc": roc_auc_score(y[m], avg[m])})
            print(f"  {version} {key}: {len(seeds)} seed(s)", flush=True)
    return pd.DataFrame(fold_rows), pd.DataFrame(pooled_rows)


def theme(ax):
    from m6a import figures as F
    ax.set_facecolor(F.SURFACE)
    ax.grid(True, color=F.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(F.GRID)
    ax.tick_params(colors=F.MUTED, labelsize=8, length=0)


def distribution(folds: pd.DataFrame, version: str):
    from m6a import figures as F
    plt = F._plt()
    keys = [m for m in MODELS if m[0] in set(folds[folds.version == version].model)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(keys) + 1.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(F.SURFACE)
    rng = np.random.default_rng(4262)
    for ax, (_, fname) in zip(axes, FILES):
        theme(ax)
        for i, (key, label, colour, control) in enumerate(keys):
            g = folds[(folds.version == version) & (folds.model == key) & (folds["scored on"] == fname)]
            v = g["gain over lightgbm"].to_numpy()
            yy = i + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(v, yy, s=14, facecolors="none" if control else colour, edgecolors=colour,
                       linewidths=0.9, alpha=0.85, zorder=3)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=F.INK, linewidth=1.4, zorder=4)
        ax.axvline(0, color=F.INK_SOFT, linewidth=0.9, linestyle="--", zorder=2)
        ax.set_title(f"scored on {fname}" + (" (unseen cell line)" if version == "cl1" and fname == "cell line 2" else ""),
                     color=F.INK, fontsize=9.5, loc="left")
        ax.set_xlabel("PR AUC gain over quantiles + LightGBM, same fold", color=F.INK_SOFT, fontsize=8.5)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([k[1] for k in keys], fontsize=8, color=F.INK)
    axes[0].invert_yaxis()
    fig.suptitle(f"Per-fold gain over the LightGBM baseline, {ARMS[version][1]}",
                 x=0.012, ha="left", color=F.INK, fontsize=11)
    fig.text(0.012, 0.008, "Each point: one fold (of 5) of one seed (up to 3), held-out genes; bar = median. "
             "Hollow = control. Dashed line = LightGBM.\nColour: grey no neighbours, orange free use of "
             "neighbours, blue constrained corroboration, aqua ensembles.", color=F.MUTED, fontsize=7.2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def place_labels(ax, points, fontsize: float = 7.0) -> None:
    """Number each point, nudging numbers apart where they would overlap (greedy,
    in screen space), with a hairline back to the point when nudged."""
    from m6a import figures as F
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    w_px, h_px = fontsize * 1.25 * fig.dpi / 72, fontsize * 1.3 * fig.dpi / 72
    placed = []
    for x, y, text in sorted(points, key=lambda t: -t[1]):
        px, py = to_px((x, y))
        best = None
        for step in range(0, 40):
            for dx, dy in ((7, 4 - 1.1 * h_px * step), (-7 - w_px * len(text) / 2, 4 - 1.1 * h_px * step),
                           (7, 4 + 1.1 * h_px * step)):
                box = (px + dx, py + dy, px + dx + w_px * len(text) / 2 + 2, py + dy + h_px)
                if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                    best = (dx, dy, box)
                    break
            if best:
                break
        dx, dy, box = best
        placed.append(box)
        ax.annotate(text, (x, y), xytext=(dx * 72 / fig.dpi, dy * 72 / fig.dpi), textcoords="offset points",
                    fontsize=fontsize, color=F.INK, va="bottom",
                    arrowprops=None if abs(dy - 4) < 1 else dict(arrowstyle="-", color=F.GRID, linewidth=0.6,
                                                                  shrinkA=0, shrinkB=3))


def scatter(pooled: pd.DataFrame, version: str):
    from m6a import figures as F
    plt = F._plt()
    p = pooled[pooled.version == version]
    present = [m for m in MODELS if m[0] in set(p.model)]
    number = {m[0]: i + 1 for i, m in enumerate(present)}
    fig = plt.figure(figsize=(13, 5.6), dpi=200)
    fig.patch.set_facecolor(F.SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.55], wspace=0.28)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    style = {k: (c, ctrl) for k, _, c, ctrl in MODELS}
    for ax, metric in zip(axes, ("pr_auc", "roc_auc")):
        theme(ax)
        w = p.pivot_table(index="model", columns="scored on", values=metric)
        for key, r in w.iterrows():
            c, ctrl = style[key]
            ax.scatter(r["cell line 1"], r["cell line 2"], s=40, facecolors="none" if ctrl else c,
                       edgecolors=c if ctrl else F.SURFACE, linewidths=1.2 if ctrl else 1.0, zorder=3)
        name = "PR AUC" if metric == "pr_auc" else "ROC AUC"
        ax.set_xlabel(f"{name}, cell line 1 (held-out genes)", color=F.INK_SOFT, fontsize=8.5)
        ax.set_ylabel(f"{name}, cell line 2" + (" (unseen cell line)" if version == "cl1" else " (held-out genes)"),
                      color=F.INK_SOFT, fontsize=8.5)
        ax.set_title(name, color=F.INK, fontsize=9.5, loc="left")
        place_labels(ax, [(r["cell line 1"], r["cell line 2"], str(number[k])) for k, r in w.iterrows()])
    key_ax = fig.add_subplot(grid[0, 2])
    key_ax.axis("off")
    for i, (k, label, c, ctrl) in enumerate(present):
        yy = 1 - i / max(len(present), 18)
        key_ax.scatter([0.02], [yy], s=30, facecolors="none" if ctrl else c, edgecolors=c, linewidths=1.1,
                       transform=key_ax.transAxes, clip_on=False)
        key_ax.text(0.07, yy, f"{number[k]:>2}  {label.strip()}", transform=key_ax.transAxes, va="center",
                    fontsize=7.6, color=F.INK)
    fig.suptitle(f"Models ranked on both cell lines, {ARMS[version][1]} (top right is better)",
                 x=0.012, ha="left", color=F.INK, fontsize=11)
    fig.text(0.012, 0.01, "All held-out sites, seeds rank-averaged. Hollow = control. Colour: grey no neighbours, "
             "orange free use of neighbours, blue constrained corroboration, aqua ensembles.",
             color=F.MUTED, fontsize=7.2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.13)
    return fig


def main() -> None:
    if "--plot-only" in sys.argv:          # redraw from the saved tables, no recomputation
        folds = pd.read_csv(common.RESULTS / "report_models_folds.csv")
        pooled = pd.read_csv(common.RESULTS / "report_models_pooled.csv")
    else:
        folds, pooled = collect()
        folds.to_csv(common.RESULTS / "report_models_folds.csv", index=False)
        pooled.to_csv(common.RESULTS / "report_models_pooled.csv", index=False)
    print(pooled.pivot_table(index=["version", "label"], columns="scored on", values=["pr_auc", "roc_auc"],
                             sort=False).round(3).to_string())
    FIGURES.mkdir(parents=True, exist_ok=True)
    for version in ARMS:
        distribution(folds, version).savefig(FIGURES / f"models_distribution_{version}.png")
        scatter(pooled, version).savefig(FIGURES / f"models_scatter_{version}.png")
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
