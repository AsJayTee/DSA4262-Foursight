"""UMAP and t-SNE pictures of each representation (PLAN.md E6).

    python analysis/representation/visualise.py --models hand deepset read_ae

Held-out fold 0: every positive plus 5,000 random negatives. Each
representation is drawn three times, coloured by label, by the motif's
positive rate (is the dominant structure just sequence?) and by read depth.

A picture is not evidence of separability: a 2-D map can merge what is
separable in 32 dimensions. So each panel also reports how much a k-nearest-
neighbour classifier loses by working in the 2-D map instead of the full
vector - if the 2-D PR AUC is far below the full one, the picture is hiding
structure rather than showing its absence.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

import common

warnings.filterwarnings("ignore")


def knn_ap(x: np.ndarray, y: np.ndarray) -> float:
    scores = cross_val_predict(KNeighborsClassifier(30), x, y, cv=5, method="predict_proba")[:, 1]
    return float(average_precision_score(y, scores))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--negatives", type=int, default=5000)
    args = ap.parse_args()

    bundle = common.load(with_reads=False)
    roles = common.split_roles(bundle, 0)
    held = np.flatnonzero(roles["heldout"])
    rng = np.random.default_rng(4262)
    positives = held[bundle.y[held] == 1]
    negatives = rng.choice(held[bundle.y[held] == 0], size=args.negatives, replace=False)
    rows = np.concatenate([negatives, positives])      # positives drawn last, on top
    y = bundle.y[rows]
    motif_rate = pd.Series(bundle.y[roles["probe"]]).groupby(
        bundle.motifs[roles["probe"]]).mean()
    colour_motif = motif_rate.reindex(bundle.motifs[rows]).to_numpy()
    colour_depth = np.log10(bundle.n_reads[rows])

    common.FIGURES.mkdir(parents=True, exist_ok=True)
    stats = {}
    for method in ("umap", "tsne"):
        fig, axes = plt.subplots(len(args.models), 3, figsize=(12, 3.6 * len(args.models)),
                                 squeeze=False)
        for i, name in enumerate(args.models):
            z, _, _ = common.load_embedding(name, 0)
            probe = z[roles["probe"]]
            x = (z[rows] - probe.mean(0)) / np.where(probe.std(0) > 0, probe.std(0), 1)
            x = np.nan_to_num(x)
            if method == "umap":
                xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=4262).fit_transform(x)
            else:
                xy = TSNE(2, perplexity=30, init="pca", random_state=4262).fit_transform(x)
            full, flat = knn_ap(x, y), knn_ap(xy, y)
            stats.setdefault(name, {})[method] = {"knn_ap_full": full, "knn_ap_2d": flat}
            panels = [(y, "label (positives in red)", "coolwarm"),
                      (colour_motif, "motif's positive rate", "viridis"),
                      (colour_depth, "log10 reads", "magma")]
            for j, (c, title, cmap) in enumerate(panels):
                ax = axes[i, j]
                ax.scatter(xy[:, 0], xy[:, 1], c=c, s=3 if j else np.where(y == 1, 6, 2),
                           cmap=cmap, alpha=0.7, linewidths=0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{name}: {title}" + (f"\nkNN AP full {full:.3f} / 2-D {flat:.3f}"
                                                   if j == 0 else ""), fontsize=9)
        fig.tight_layout()
        fig.savefig(common.FIGURES / f"{method}.png", dpi=110)
        plt.close(fig)
        print(f"wrote {common.FIGURES / f'{method}.png'}", flush=True)
    (common.RESULTS / "projection_stats.json").write_text(json.dumps(stats, indent=1))
    for name, s in stats.items():
        print(f"{name:>14}: " + "  ".join(f"{m} kNN AP full {v['knn_ap_full']:.3f} 2-D {v['knn_ap_2d']:.3f}"
                                          for m, v in s.items()))


if __name__ == "__main__":
    main()
