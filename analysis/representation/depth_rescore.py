"""The finalist networks scored from 1, 3 and 10 reads per site (Task 2).

    python analysis/representation/depth_rescore.py --models h2gcn,h2gcn_local,h2gcn_twohead

SG-NEx sites have a median of 3 reads; every training site has at least 20.
This loads each saved network (xsrc_nets.py writes weights since 2026-10-02),
thins every held-out site to `depth` of its reads - the keyed subset
`m6a.data.subsample_blocks` picks, the same reads the LightGBM baseline's
depth features are built from - and scores it. Nothing is retrained: the
networks and the baseline are both fitted at full depth, as they would ship.

Graph networks only (the finalists). A transcript's other sites are thinned
too, so a neighbour cannot lend a sparse site evidence it would not have.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch

import common
import graph
import xsrc_nets as X
from m6a import crosssource as xs
from m6a import registry
from m6a.config import Config
from m6a.data import subsample_blocks

DEPTHS = (1, 3, 10)
WEIGHTS = re.compile(r"(.+)_(dataset0|pooled_both)_fold(\d)(?:_s(\d+))?\.pt$")


def lgbm_at_depths(sources, log) -> dict:
    """{depth: {arm: {source: scores}}} for quantiles + LightGBM, fitted at full depth."""
    path = X.OUT / "baseline_lgbm_depths.npz"
    if path.exists():
        z = np.load(path)
        return {d: {a: {s: z[f"d{d}__{a}__{s}"] for s in xs.SOURCES} for a in X.ARMS} for d in DEPTHS}
    cfg = Config.load(X.BASELINE)
    deep = xs.load(cfg.features, [None, *DEPTHS], common.DATA_DIR, seed=cfg.split.seed,
                   n_folds=cfg.split.n_folds, log=lambda *a: None)
    srcs = {}
    for name, src in sources.items():
        at = deep[name].index.get_indexer(src.index)
        srcs[name] = xs.Source(name, src.index, {d: Xd.iloc[at] for d, Xd in deep[name].X.items()},
                               src.y, src.genes, src.folds)
    model_class = registry.get("models", cfg.model)
    out = {d: {a: {s: np.full(len(srcs[s]), np.nan) for s in xs.SOURCES} for a in X.ARMS} for d in DEPTHS}
    for arm in X.ARMS:
        for fold in sorted(np.unique(srcs["dataset0"].folds)):
            Xt, y, w = xs.training_rows(srcs, arm, fold, [None])
            model = xs._fit(model_class, cfg.model_params, Xt, y, w)
            for s in xs.SOURCES:
                held = srcs[s].folds == fold
                for d in DEPTHS:
                    out[d][arm][s][held] = model.predict_proba(srcs[s].X[d][held])
            log(f"  LightGBM at depth: {arm} fold {fold}")
    np.savez(path, **{f"d{d}__{a}__{s}": out[d][a][s] for d in DEPTHS for a in X.ARMS for s in xs.SOURCES})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="h2gcn,h2gcn_local,h2gcn_twohead")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    sources, bundle = X.load(1.0)
    n0 = len(sources["dataset0"])
    keys = list(sources["dataset0"].index) + list(sources["data1"].index)
    started = time.time()
    thinned = {d: subsample_blocks(bundle.reads, keys, d) for d in DEPTHS}
    log(f"thinned reads to {DEPTHS} ({time.time() - started:.0f}s)")
    base = lgbm_at_depths(sources, log)

    # (model, seed, arm) -> {depth: scores over every row}
    scores: dict = {}
    for model in args.models.split(","):
        for path in sorted((X.OUT / "weights").glob(f"{model}_*.pt")):
            m = WEIGHTS.match(path.name)
            if not m or m[1] != model:
                continue
            arm, fold, seed = m[2], int(m[3]), int(m[4] or 0)
            saved = torch.load(path, weights_only=False)
            net = graph.GraphNet(**X.GRAPH_MODELS[model])
            net.load_state_dict(saved["state_dict"])
            std = common.Standardiser(saved["read_mean"], saved["read_scale"])
            held = np.flatnonzero(bundle.folds == fold)
            graphs = graph.graphs_of(held, bundle.graph_id, bundle.position)
            slot = scores.setdefault((model, seed, arm), {d: np.full(len(bundle.genes), np.nan) for d in DEPTHS})
            for d in DEPTHS:
                got = graph.score(net, graphs, std(thinned[d].values), thinned[d].offsets,
                                  bundle.kmer_onehot, bundle.position)
                slot[d][held] = [got[r] for r in held]
            log(f"  scored {path.name}")

    rows = []
    for (model, seed, arm), by_depth in sorted(scores.items()):
        if any(np.isnan(v).any() for v in by_depth.values()):
            log(f"{model} seed {seed} {arm}: not every fold has weights - skipped")
            continue
        for d in DEPTHS:
            for s, sl in (("dataset0", slice(0, n0)), ("data1", slice(n0, None))):
                g = xs.paired_gain(sources[s].y, by_depth[d][sl], base[d][arm][s], sources[s].genes)
                rows.append({"model": model, "seed": seed, "arm": arm, "depth": d, "source": s,
                             "pr_auc": g["pr_auc"], "lgbm_pr_auc": g["baseline_pr_auc"], "gain": g["gain"],
                             "ci_low": g["ci_low"], "ci_high": g["ci_high"]})
    for r in rows:
        log(f"{r['model']:20s} s{r['seed']} {r['arm']:12s} {r['depth']:>2d} reads  {r['source']:8s} "
            f"{r['pr_auc']:.4f} vs LightGBM {r['lgbm_pr_auc']:.4f}  {r['gain']:+.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    (common.RESULTS / "depth_rescore.json").write_text(json.dumps(rows, indent=2))
    log(f"-> {common.RESULTS / 'depth_rescore.json'}")


if __name__ == "__main__":
    main()
