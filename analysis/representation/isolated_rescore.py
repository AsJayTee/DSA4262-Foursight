"""What do the shipped designs score when a site has no neighbours?

    python analysis/representation/isolated_rescore.py

The graph networks use the other candidate sites on a site's transcript. If a
test file held partial transcripts, they would have fewer or none. This takes
the saved cross-validation networks of the two shipped designs (both files,
seed 0), scores every held-out site twice - with its transcript as usual, and
alone, as a one-site graph - and compares, beside deepset (no graph at all)
and quantiles + LightGBM. Nothing is retrained.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

import common
import graph
import xsrc_nets as X

MODELS = ("h2gcn_twohead_aux", "h2gcn_aux")
WEIGHTS = re.compile(r"(.+)_pooled_both_fold(\d)\.pt$")


def main() -> None:
    torch.set_num_threads(16)
    sources, bundle = X.load(1.0)
    n0 = len(sources["dataset0"])
    alone = np.arange(len(bundle.genes))          # every site its own graph
    scores = {}
    for model in MODELS:
        for mode in ("with transcript", "alone"):
            scores[(model, mode)] = np.full(len(bundle.genes), np.nan)
        for fold in range(5):
            saved = torch.load(X.OUT / "weights" / f"{model}_pooled_both_fold{fold}.pt", weights_only=False)
            net = graph.GraphNet(**X.GRAPH_MODELS[model])
            net.load_state_dict(saved["state_dict"])
            values = common.Standardiser(saved["read_mean"], saved["read_scale"])(bundle.reads.values)
            held = np.flatnonzero(bundle.folds == fold)
            for mode, gid in (("with transcript", bundle.graph_id), ("alone", alone)):
                got = graph.score(net, graph.graphs_of(held, gid, bundle.position), values,
                                  bundle.reads.offsets, bundle.kmer_onehot, bundle.position)
                scores[(model, mode)][held] = [got[r] for r in held]
            print(f"  {model} fold {fold}", flush=True)
    rank = lambda v, sl: rankdata(v[sl]) / len(v[sl])  # noqa: E731
    rows = []
    for name, src, sl in (("dataset0", "dataset0", slice(0, n0)), ("data1", "data1", slice(n0, None))):
        y = sources[src].y
        for mode in ("with transcript", "alone"):
            for model in MODELS:
                rows.append({"labels": name, "model": model, "mode": mode,
                             "pr_auc": average_precision_score(y, scores[(model, mode)][sl])})
            ens = np.mean([rank(scores[(m, mode)], sl) for m in MODELS], 0)
            rows.append({"labels": name, "model": "ensemble (both)", "mode": mode,
                         "pr_auc": average_precision_score(y, ens)})
        deepset = X.candidate("deepset", sources, 1.0)["pooled_both"][src]
        base = X.baseline_scores(sources, 1.0, print)["pooled_both"][src]
        rows += [{"labels": name, "model": "deepset (no graph)", "mode": "-",
                  "pr_auc": average_precision_score(y, deepset)},
                 {"labels": name, "model": "quantiles + LightGBM", "mode": "-",
                  "pr_auc": average_precision_score(y, base)}]
    table = pd.DataFrame(rows).pivot_table(index=["labels", "model"], columns="mode", values="pr_auc")
    print(table.round(4).to_string())
    table.to_csv(common.RESULTS / "isolated_rescore.csv")


if __name__ == "__main__":
    main()
