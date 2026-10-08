"""Which neighbour distances does the 400-nt model actually use, on each cell line?

    python analysis/representation/band_ablation.py

The radius test (2026-10-06) found a 100-nt radius transfers best between cell
lines while wider radii help only the training cell line. This asks the
trained 400-nt network (h2gcn_aux_r400, seed 0, both arms) directly: score
every held-out site again with its neighbours restricted, without retraining -

  all          every neighbour within 400 nt (as trained)
  none         no neighbours (the site's own reads only)
  only a-b     neighbours a < d <= b nt only
  drop a-b     every neighbour except those a < d <= b nt

- and report PR AUC on each cell line's held-out sites. "only" shows what a
band carries by itself; "drop" what removing it costs from the full model (the
smaller departure from what the network was trained on, so the fairer of the
two). Both are input changes the network never saw in training: read them as
contributions, not as models one could ship.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import common
import graph
import xsrc_nets as X

MODEL = "h2gcn_aux_r400"
# 0-10 and 10-20 split out (2026-10-08): nanopore signal from one m6A spreads ~+-10 nt,
# so a very close neighbour may partly re-measure the scored site itself.
BANDS = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 400)]


def configs():
    yield "all", 0, 400
    yield "none", 400, 400            # min > max: no neighbour qualifies
    for lo, hi in BANDS:
        yield f"only {lo}-{hi}", lo, hi
    for lo, hi in BANDS:
        yield f"drop {lo}-{hi}", (lo, hi), None


def main() -> None:
    torch.set_num_threads(32)
    sources, bundle = X.load(1.0)
    n0 = len(sources["dataset0"])
    rows = []
    for arm in X.ARMS:
        scores: dict[str, np.ndarray] = {}
        for fold in range(5):
            saved = torch.load(X.OUT / "weights" / f"{MODEL}_{arm}_fold{fold}.pt", weights_only=False)
            values = common.Standardiser(saved["read_mean"], saved["read_scale"])(bundle.reads.values)
            held = np.flatnonzero(bundle.folds == fold)
            graphs = graph.graphs_of(held, bundle.graph_id, bundle.position)
            started = time.time()
            for name, lo, hi in configs():
                net = graph.GraphNet(**X.GRAPH_MODELS[MODEL])
                net.load_state_dict(saved["state_dict"])
                if isinstance(lo, tuple):
                    net.drop_band = lo          # every neighbour within 400 nt except a < d <= b
                else:
                    net.window, net.min_window = hi, lo
                got = graph.score(net, graphs, values, bundle.reads.offsets, bundle.kmer_onehot, bundle.position)
                scores.setdefault(name, np.full(len(bundle.genes), np.nan))[held] = [got[r] for r in held]
            print(f"  {arm} fold {fold} ({time.time() - started:.0f}s)", flush=True)
        for name, v in scores.items():
            for src, sl in (("dataset0", slice(0, n0)), ("data1", slice(n0, None))):
                rows.append({"trained on": arm, "neighbours": name, "scored on": src,
                             "pr_auc": average_precision_score(sources[src].y, v[sl])})
    table = pd.DataFrame(rows).pivot_table(index=["trained on", "neighbours"], columns="scored on",
                                           values="pr_auc", sort=False)
    print(table.round(4).to_string())
    table.to_csv(common.RESULTS / "band_ablation.csv")


if __name__ == "__main__":
    main()
