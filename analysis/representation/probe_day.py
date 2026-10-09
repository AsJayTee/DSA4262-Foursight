"""Scoring-only probes of the 8 Oct designs (job 4 of the 9 Oct batch). No training.

    python analysis/representation/probe_day.py

Seed-0 fold models, every held-out site, both arms:

  gps         why does it fail to transfer? Score again with its global
              attention cut to |d| <= 150 nt and to |d| <= 50 nt (the local
              channels are untouched), and record where its attention goes by
              distance. Prediction (findings 12C): cutting the long range costs
              cell line 1 more than cell line 2.
  res_gate    what did the gate learn? Per site: gate g, read count, |own
              score|; and the learned (a, b, c) of every fold and seed.
  scalar_msg  the corroboration curve: the correction it adds to a site's
              own score as the neighbours' mean and best scores vary (partial
              dependence over held-out sites that have neighbours).

Writes results/probe_gps.csv, probe_gps_attention.csv, probe_gate_params.csv,
probe_scalar_curve.csv; per-site gate values (with labels) to .cache/representation/.
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

GPS_RADII = {"as trained (whole transcript)": None, "attention cut to 150 nt": 150, "attention cut to 50 nt": 50}
DEPTH_BINS = [0, 30, 50, 100, 10**6]
GRID = np.linspace(0.02, 0.98, 25)        # quantiles of the observed neighbour feature


def load_net(model: str, arm: str, fold: int, seed: int = 0):
    path = X.OUT / "weights" / f"{model}_{arm}_fold{fold}{'' if seed == 0 else f'_s{seed}'}.pt"
    saved = torch.load(path, weights_only=False)
    net = graph.GraphNet(**X.GRAPH_MODELS[model])
    net.load_state_dict(saved["state_dict"])
    net.eval()
    return net, saved


def scored(net, saved, bundle, held):
    values = common.Standardiser(saved["read_mean"], saved["read_scale"])(bundle.reads.values)
    graphs = graph.graphs_of(held, bundle.graph_id, bundle.position)
    out = graph.score(net, graphs, values, bundle.reads.offsets, bundle.kmer_onehot, bundle.position)
    return out           # row -> logit, in scoring order (capture entries follow the same order)


def main() -> None:
    torch.set_num_threads(8)
    sources, bundle = X.load(1.0)
    n0 = len(sources["dataset0"])
    y = np.concatenate([sources["dataset0"].y, sources["data1"].y])
    file = np.r_[np.zeros(n0, int), np.ones(len(y) - n0, int)]
    counts = np.diff(bundle.reads.offsets)

    # --- gps -------------------------------------------------------------
    rows, att_rows = [], []
    for arm in X.ARMS:
        scores = {k: np.full(len(y), np.nan) for k in [*GPS_RADII, "h2gcn_aux (reference)"]}
        att = np.zeros((2, len(graph.ATT_BANDS) + 1))
        for fold in range(5):
            started = time.time()
            held = np.flatnonzero(bundle.folds == fold)
            for name, r in GPS_RADII.items():
                net, saved = load_net("gps", arm, fold)
                net.gps_radius = r
                if r is None:
                    net.att_stats = att
                out = scored(net, saved, bundle, held)
                scores[name][list(out)] = list(out.values())
            net, saved = load_net("h2gcn_aux", arm, fold)
            out = scored(net, saved, bundle, held)
            scores["h2gcn_aux (reference)"][list(out)] = list(out.values())
            print(f"  gps {arm} fold {fold} ({time.time() - started:.0f}s)", flush=True)
        for name, v in scores.items():
            for f, label in ((0, "cell line 1"), (1, "cell line 2")):
                m = file == f
                rows.append({"trained on": arm, "scoring": name, "scored on": label,
                             "pr_auc": average_precision_score(y[m], v[m])})
        for layer in range(2):
            mass = att[layer, :-1] / att[layer, -1]
            att_rows += [{"trained on": arm, "layer": layer + 1,
                          "distance band": "self" if lo < 0 else f"{lo}-{hi if hi < 10**9 else 'end'} nt",
                          "attention share": m} for (lo, hi), m in zip(graph.ATT_BANDS, mass)]
    gps = pd.DataFrame(rows).pivot_table(index=["trained on", "scoring"], columns="scored on",
                                         values="pr_auc", sort=False)
    print(gps.round(4).to_string())
    gps.to_csv(common.RESULTS / "probe_gps.csv")
    att_table = pd.DataFrame(att_rows)
    print(att_table.round(3).to_string(index=False))
    att_table.to_csv(common.RESULTS / "probe_gps_attention.csv", index=False)

    # --- res_gate ----------------------------------------------------------
    params = []
    for arm in X.ARMS:
        for seed in (0, 1, 2):
            for fold in range(5):
                net, _ = load_net("res_gate", arm, fold, seed)
                a, b, c = (float(x) for x in net.gate_params)
                params.append({"trained on": arm, "seed": seed, "fold": fold, "a (log reads)": a,
                               "b (|own score|)": b, "c": c})
    pd.DataFrame(params).to_csv(common.RESULTS / "probe_gate_params.csv", index=False)
    print(pd.DataFrame(params).groupby("trained on")[["a (log reads)", "b (|own score|)", "c"]]
          .agg(["mean", "std"]).round(3).to_string())
    gate_rows = []
    for arm in X.ARMS:
        for fold in range(5):
            held = np.flatnonzero(bundle.folds == fold)
            net, saved = load_net("res_gate", arm, fold)
            net.capture = []
            out = scored(net, saved, bundle, held)
            cap = {k: np.concatenate([c[k] for c in net.capture]) for k in net.capture[0]}
            r = np.array(list(out))
            gate_rows.append(pd.DataFrame({"trained on": arm, "row": r, "file": file[r], "label": y[r],
                                           "reads": counts[r], "own_score": cap["s"], "gate": cap["g"],
                                           "correction": cap["correction"], "has_neighbour": cap["has"]}))
    gate = pd.concat(gate_rows)
    # Per-site labels: kept out of results/ (git), as all labelled rows are.
    gate.drop(columns="row").to_csv(common.OUT / "probe_gate.csv.gz", index=False)
    g1 = gate[gate.has_neighbour == 1]
    print(g1.groupby(["trained on", pd.cut(g1.reads, DEPTH_BINS)], observed=True).gate.mean().round(3).to_string())
    print(g1.groupby(["trained on", pd.qcut(g1.own_score.abs(), 5)], observed=True).gate.mean().round(3).to_string())

    # --- scalar_msg: partial dependence ------------------------------------
    curve = []
    for arm in X.ARMS:
        for fold in range(5):
            held = np.flatnonzero(bundle.folds == fold)
            net, saved = load_net("scalar_msg", arm, fold)
            net.capture = []
            scored(net, saved, bundle, held)
            feats = np.concatenate([c["feats"] for c in net.capture])
            feats = feats[feats[:, 5] == 1]                  # sites with a neighbour
            for col, what in ((2, "neighbour mean score"), (3, "best neighbour score within 75 nt")):
                for q in GRID:
                    v = np.quantile(feats[:, col], q)
                    f = feats.copy()
                    f[:, col] = v
                    if col == 2:                          # keep the best >= the mean, as in real data
                        f[:, 3] = np.maximum(f[:, 3], v)
                    with torch.no_grad():
                        corr = net.c_mlp(torch.from_numpy(f)).squeeze(-1).numpy()
                    curve.append({"trained on": arm, "fold": fold, "varied": what, "quantile": q,
                                  "value (logit)": v, "mean correction (logit)": corr.mean()})
    curve = pd.DataFrame(curve).groupby(["trained on", "varied", "quantile"], as_index=False).mean()
    curve.drop(columns="fold").to_csv(common.RESULTS / "probe_scalar_curve.csv", index=False)
    print(curve[curve["quantile"].isin(GRID[[0, 6, 12, 18, 24]])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
