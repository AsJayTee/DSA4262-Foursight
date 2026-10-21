"""The report's UMAP figures of site representations.

    python analysis/representation/report_umap.py reads    # -> report/figures/fig_read_umap.png
    python analysis/representation/report_umap.py pooling  # -> report/figures/fig_pooling_umap.png

A compact version of visualise.py (same sites, same UMAP settings, same seed):

- `reads`: four ways of turning reads into a site vector - hand-made
  summaries, DeepSet, a per-read autoencoder with a prediction head, and an
  LSTM autoencoder trained without labels;
- `pooling`: why DeepSet's specific form - against a wider DeepSet,
  gated-attention MIL, and DeepSet without the 7-mer.

Top row coloured by label, bottom row by the motif's positive rate: whether
the map is organised by modification or by sequence. Each column's title
quotes a 30-NN classifier's PR AUC on the full vector and on the 2-D map, so
a map that hides structure can be told from one that lacks it.

Needs each model's site vectors in .cache/representation/<name>/fold0.npz.
attn_mil and deepset_large were first embedded on Ronin and only their scores
came back; regenerate their fold-0 vectors (on Ronin, then copy
.cache/representation/<name>/fold0.npz back) with:

    python analysis/representation/learn.py --model attn_mil --folds 0 --minutes 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common
from visualise import knn_ap

FIGURES = {
    "reads": ("fig_read_umap.png", [
        # `hand` is quantiles_all_v1: per-site quantiles plus sequence flanks,
        # within-read coupling and cross-site (neighbour, transcript-position)
        # columns - not the per-site quantiles the LightGBM baseline uses.
        ("hand", "Hand-made features\n(incl. neighbour columns)"),
        ("deepset", "DeepSet\n(trained on labels)"),
        ("read_ae_pred", "Read autoencoder\n+ prediction head"),
        ("pos_lstm_ae", "LSTM autoencoder\n(no labels)"),
    ]),
    "pooling": ("fig_pooling_umap.png", [
        ("deepset", "DeepSet\n(shipped: 32-d)"),
        ("deepset_large", "DeepSet, wider\n(64-d, 128 units)"),
        ("attn_mil", "Gated-attention MIL\n(weights per read)"),
        ("deepset_nokmer", "DeepSet\nwithout the 7-mer"),
    ]),
}
OUT_DIR = Path(__file__).resolve().parents[2] / "report" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("figure", choices=sorted(FIGURES))
    args = ap.parse_args()
    filename, models = FIGURES[args.figure]

    missing = [n for n, _ in models if not common.embedding_path(n, 0).exists()]
    if missing:
        sys.exit("no site vectors for " + ", ".join(missing) + " - regenerate with\n"
                 + "\n".join(f"    python analysis/representation/learn.py --model {n} "
                             "--folds 0 --minutes 25" for n in missing))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import umap

    # Identical sampling to visualise.py, so the numbers match projection_stats.json.
    bundle = common.load(with_reads=False)
    roles = common.split_roles(bundle, 0)
    held = np.flatnonzero(roles["heldout"])
    rng = np.random.default_rng(4262)
    positives = held[bundle.y[held] == 1]
    negatives = rng.choice(held[bundle.y[held] == 0], size=5000, replace=False)
    rows = np.concatenate([negatives, positives])      # positives drawn last, on top
    y = bundle.y[rows]
    motif_rate = pd.Series(bundle.y[roles["probe"]]).groupby(
        bundle.motifs[roles["probe"]]).mean()
    colour_motif = motif_rate.reindex(bundle.motifs[rows]).to_numpy()

    n = len(models)
    fig = plt.figure(figsize=(2.9 * n + 2.4, 6.4))
    # Two narrow columns after the maps hold the keys - the label key for the
    # top row, then the motif colour bar for the bottom row - each spanning
    # both rows, so the maps stay aligned as one grid.
    # Column n + 1 is an empty spacer that keeps the label bar's tick labels
    # clear of the motif colour bar.
    grid = GridSpec(2, n + 3, figure=fig, width_ratios=[1] * n + [0.06, 0.2, 0.06],
                    wspace=0.12, hspace=0.12)
    label_colours = np.where(y == 1, "#c0392b", "#3b6fb6")
    for i, (name, title) in enumerate(models):
        z, _, _ = common.load_embedding(name, 0)
        probe = z[roles["probe"]]
        x = np.nan_to_num((z[rows] - probe.mean(0)) / np.where(probe.std(0) > 0, probe.std(0), 1))
        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=4262).fit_transform(x)
        full, flat = knn_ap(x, y), knn_ap(xy, y)
        top, bottom = fig.add_subplot(grid[0, i]), fig.add_subplot(grid[1, i])
        top.scatter(xy[:, 0], xy[:, 1], c=label_colours, s=np.where(y == 1, 5, 2),
                    alpha=0.7, linewidths=0)
        top.set_title(f"{title}\n30-NN PR AUC {full:.2f} (2-D: {flat:.2f})", fontsize=9)
        sc = bottom.scatter(xy[:, 0], xy[:, 1], c=colour_motif, s=2, cmap="viridis",
                            alpha=0.7, linewidths=0)
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
        if i == 0:
            top.set_ylabel("coloured by label", fontsize=9)
            bottom.set_ylabel("coloured by the motif's\npositive rate", fontsize=9)
        print(f"{name}: kNN AP full {full:.3f}, 2-D {flat:.3f}", flush=True)

    # The label key is drawn as a two-colour bar, so it is as thin as the motif
    # colour bar beside it and reads the same way.
    two = ListedColormap(["#3b6fb6", "#c0392b"])
    label_bar = fig.colorbar(ScalarMappable(BoundaryNorm([0, 1, 2], 2), two),
                             cax=fig.add_subplot(grid[:, n]), ticks=[0.5, 1.5])
    label_bar.ax.set_yticklabels(["unmodified", "modified"], rotation=90, va="center")
    label_bar.set_label("label (top row)")
    fig.colorbar(sc, cax=fig.add_subplot(grid[:, n + 2]), label="motif's positive rate (bottom row)")
    out = OUT_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
