"""Fully neural models under cross-source evaluation (decisions 0029, 0032).

The slate agreed on 2026-10-02. Each network makes the prediction itself - no
LightGBM on top - and is scored against `configs/quantiles.yaml` (LightGBM) on
the cross-source union split, arms dataset0 (as shipped) and pooled_both (both
files; a shared site's target is the mean of its two labels):

  deepset          the network behind today's +0.022: per-read MLP, mean + sd
  deepset_hand     the same, with quantiles_v1 features joining after pooling -
                   one end-to-end model instead of the two-model stack
  attn_mil         gated attention over reads (which reads look modified)
  set_transformer  reads attend to each other - a GNN over one site's reads
  gcn / h2gcn / gat  reads -> site -> transcript graphs (graph.py)
  knn_static / knn_dynamic / knn_random  mutual k-NN graph over a site's reads
                   (readgraph.py, k = 4); knn_random is the shuffled-graph control

Every model: one seed, learn.py's recipe v3 - trained to convergence (warm-up,
halve the rate on an 8-epoch validation plateau, stop after 25 without a gain;
the time cap is a safety net only) on 90% of training genes, early-stopped on
the other 10%. `convergence.py` checks every network stopped on patience. Each model is also compared against `deepset`.

    python analysis/representation/xsrc_nets.py all --jobs 14           # Ronin
    python analysis/representation/xsrc_nets.py fit --model gat --fold 0
    python analysis/representation/xsrc_nets.py combine --model gat
    # smoke: 5% of genes, tiny budgets, nothing logged
    python analysis/representation/xsrc_nets.py all --genes 0.05 --minutes 0.2 --epochs 2 --no-wandb --jobs 1

Scores are cached per (model, arm, fold) under .cache/representation/xsrc_nets/,
so a killed run resumes and finished models are not refitted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import common
import encoders as E
import graph
import learn
import readgraph
import xsrc_deepset
from m6a import crosssource as xs
from m6a import registry, tracking
from m6a import report as reporting
from m6a.config import Config
from m6a.data import SUBSAMPLE_SEED
from m6a.env import load_env

ARMS = ("dataset0", "pooled_both")
SET_MODELS = ("deepset", "deepset_hand", "attn_mil", "set_transformer")
# k-NN read graphs (readgraph.py); k = 4 from knn_k_analysis.py, agreed 2026-10-02.
KNN_MODELS = tuple(f"knn_{m}" for m in readgraph.MODES)
KNN_K = 4
MODELS = SET_MODELS + graph.KINDS + KNN_MODELS
# Slowest first, so the pool's tail is short.
ORDER = ("set_transformer", "gat", "h2gcn", "gcn", "knn_dynamic", "knn_static", "knn_random",
         "deepset_hand", "attn_mil", "deepset")
OUT = common.OUT / "xsrc_nets"
BASELINE = xsrc_deepset.BASELINE


class DeepSetHand(nn.Module):
    """DeepSet whose site head also sees the hand features. learn.train passes
    one site-level array, so the k-mer one-hot and the standardised hand
    features travel together in it and are split here."""

    def __init__(self, n_hand: int, width: int = 64, z: int = 32):
        super().__init__()
        self.phi = E.mlp([E.N_VALUES + E.KMER, width, width])
        self.hand = E.mlp([n_hand, width])
        self.rho = E.mlp([3 * width + 1 + E.KMER, width, z])
        self.head = nn.Linear(z, 1)

    def forward(self, reads, mask, site, windows=None):
        kmer, hand = site[:, :E.KMER], site[:, E.KMER:]
        mean, sd = E.masked_mean_sd(self.phi(E.with_kmer(reads, kmer)), mask)
        z = self.rho(torch.cat([mean, sd, E.log_count(mask), kmer, self.hand(hand)], -1))
        return self.head(z).squeeze(-1)


def suffix(genes: float) -> str:
    return "" if genes >= 1 else f"_genes{genes:g}"


def cache_path(model: str, arm: str, fold: int, genes: float):
    return OUT / f"{model}_{arm}_fold{fold}{suffix(genes)}.npy"


def load(genes: float):
    sources, bundle = xsrc_deepset.load(genes)
    s0, s1 = sources["dataset0"], sources["data1"]
    transcripts = np.concatenate([s0.index.get_level_values(0), s1.index.get_level_values(0)])
    # One graph per (file, transcript): the two files are separate runs.
    _, bundle.graph_id = np.unique(np.char.add(bundle.file.astype(str),
                                               transcripts.astype(str)), return_inverse=True)
    bundle.position = np.concatenate([s0.index.get_level_values(1),
                                      s1.index.get_level_values(1)]).astype(np.int64)
    bundle.hand = np.concatenate([s0.X[None].to_numpy(), s1.X[None].to_numpy()]).astype(np.float32)
    return sources, bundle


def fit_one(model_name: str, bundle, arm: str, fold: int, args, log) -> np.ndarray:
    held = bundle.folds == fold
    train = ~held & ((bundle.file == 0) if arm == "dataset0" else True)
    v = np.array([common._unit(f"4262:val:{g}") for g in bundle.genes])
    fit_rows, val_rows = train & (v >= 0.1), train & (v < 0.1)
    target = bundle.y_own if arm == "dataset0" else bundle.y_both
    std = common.Standardiser.fit(bundle.reads, fit_rows)
    values = std(bundle.reads.values)
    torch.manual_seed(4262 + fold)
    out = np.full(len(bundle.genes), np.nan, dtype=np.float32)
    held_rows = np.flatnonzero(held)

    if model_name in graph.KINDS:
        graph_args = (values, bundle.reads.offsets, bundle.kmer_onehot, bundle.position)
        model = graph.GraphNet(model_name)
        graph.train(model, graph.graphs_of(np.flatnonzero(fit_rows), bundle.graph_id, bundle.position),
                    graph.graphs_of(np.flatnonzero(val_rows), bundle.graph_id, bundle.position),
                    *graph_args, target, bundle.y_own, args.minutes, args.epochs, 4262 + fold, log)
        scores = graph.score(model, graph.graphs_of(held_rows, bundle.graph_id, bundle.position),
                             *graph_args)
        out[held_rows] = [scores[r] for r in held_rows]
        return out

    site = bundle.kmer_onehot
    if model_name == "deepset_hand":
        mean = np.nanmean(bundle.hand[fit_rows], 0)
        sd = np.nanstd(bundle.hand[fit_rows], 0)
        hand = np.nan_to_num((bundle.hand - mean) / np.where(sd > 0, sd, 1)).astype(np.float32)
        site = np.concatenate([site, hand], 1)
        model = DeepSetHand(hand.shape[1])
    elif model_name in KNN_MODELS:
        model = readgraph.KNNDeepSet(model_name.removeprefix("knn_"), k=KNN_K)
    else:
        model = learn.build(model_name, 0)
    view = SimpleNamespace(reads=bundle.reads, kmer_onehot=site, window_onehot=bundle.window_onehot,
                           y=target, y_val=bundle.y_own)
    # learn.train branches on the name only to pick the loss; every model here
    # is a supervised set encoder, which "deepset" selects.
    learn.train(model, "deepset", "set", True, view, {"encoder": fit_rows, "encoder_val": val_rows},
                values, args.minutes, args.epochs, fold, log, recipe="v3")
    out[held_rows] = learn.site_logits(model, values, bundle.reads.offsets, held_rows, site,
                                       bundle.window_onehot)
    return out


def baseline_scores(sources, genes: float, log) -> dict:
    path = OUT / f"baseline_lgbm{suffix(genes)}.npz"
    if path.exists():
        stored = np.load(path)
        return {a: {s: stored[f"{a}__{s}"] for s in xs.SOURCES} for a in ARMS}
    cfg = Config.load(BASELINE)
    oof = xs.out_of_fold(sources, registry.get("models", cfg.model), cfg.model_params, [None],
                         arms=list(ARMS), log=log)
    np.savez(path, **{f"{a}__{s}": oof[a][s] for a in ARMS for s in xs.SOURCES})
    return oof


def candidate(model: str, sources, genes: float) -> dict | None:
    n0 = len(sources["dataset0"])
    folds = sorted(np.unique(sources["dataset0"].folds))
    paths = [cache_path(model, a, f, genes) for a in ARMS for f in folds]
    if not all(p.exists() for p in paths):
        return None
    out = {}
    for arm in ARMS:
        total = np.nanmax(np.stack([np.load(cache_path(model, arm, f, genes)) for f in folds]), 0)
        out[arm] = {"dataset0": total[:n0], "data1": total[n0:]}
    return out


def combine(model: str, sources, args, log) -> None:
    cand = candidate(model, sources, args.genes)
    if cand is None:
        raise SystemExit(f"{model}: not every fold is fitted - run `xsrc_nets.py fit --model {model}`.")
    base = baseline_scores(sources, args.genes, log)
    block = xs.summarise(sources, cand, base)
    report = reporting.new("standard", subsample_seed=SUBSAMPLE_SEED, log=log)
    reporting.cross_source(report, block, f"{model} (network)", Config.load(BASELINE).name)
    extra = {}
    deepset = candidate("deepset", sources, args.genes) if model != "deepset" else None
    if deepset is not None:
        vs = xs.summarise(sources, cand, deepset)
        report.data["vs_deepset"] = vs
        log(f"\nvs deepset alone (same arms, same genes):")
        for arm, r in vs["arms"].items():
            log(f"  {arm:12s} dataset0 {r['dataset0']['gain']:+.4f}  data1 {r['data1']['gain']:+.4f}  "
                f"worst {r['gain_worst']:+.4f}")
            extra.update({f"vs_deepset/{arm}/gain_worst": r["gain_worst"],
                          f"vs_deepset/{arm}/dataset0/gain": r["dataset0"]["gain"],
                          f"vs_deepset/{arm}/data1/gain": r["data1"]["gain"]})
    report.data["network"] = {"model": model, "minutes": args.minutes, "epochs": args.epochs,
                              "genes": args.genes, "recipe": "v3", "seeds": 1}
    smoke = args.genes < 1
    name = f"{model}_nn__xsrc"
    path = reporting.write(report, common.RESULTS, name + ("__smoke" if smoke else ""))
    tracker = tracking.start(
        name, enabled=not (args.no_wandb or smoke), job_type="evaluate", tags=[tracking.EVAL_TAG],
        config={"eval_schema": xs.EVAL_SCHEMA, "baseline": Config.load(BASELINE).name,
                "arms": ",".join(ARMS), "features": "reads (network)", "model": model,
                "train_depths": "full (random read subsets)", "network_minutes": args.minutes,
                "script": "analysis/representation/xsrc_nets.py"})
    try:
        reporting.publish(report, tracker, path, name=name)
        if extra:
            tracker.log(extra)
            tracker.summary(extra)
    finally:
        tracker.finish()
    r = block["arms"]["dataset0"]
    log(f"{model}: worst {r['gain_worst']:+.4f}  mean {r['gain_mean']:+.4f}  -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fit", "baseline", "combine", "all"))
    ap.add_argument("--model", help="one model or a comma-separated list (default: all)")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--minutes", type=float, default=240.0, help="safety cap per network")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=14, help="all: concurrent fit processes")
    ap.add_argument("--genes", type=float, default=1.0, help="fraction of genes (smoke only)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    load_env(common.ROOT / ".env")
    torch.set_num_threads(args.threads)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    models = args.model.split(",") if args.model else [m for m in ORDER]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {', '.join(MODELS)}.")
    n_folds = Config.load(BASELINE).split.n_folds

    if args.stage == "all":
        logs = common.ROOT / "analysis" / "representation" / "logs"
        logs.mkdir(exist_ok=True)
        common_args = ["--minutes", str(args.minutes), "--epochs", str(args.epochs),
                       "--threads", str(args.threads), "--genes", str(args.genes)]
        # One job per network, so seventy networks spread over the cores.
        queue = [["baseline"]] + [["fit", "--model", m, "--fold", str(f), "--arm", a]
                                  for m in models for f in range(n_folds) for a in ARMS
                                  if not cache_path(m, a, f, args.genes).exists()]
        running, failed = [], []
        while queue or running:
            while queue and len(running) < args.jobs:
                job = queue.pop(0)
                tag = "_".join(job[::2] if job[0] == "fit" else job)
                handle = open(logs / f"xsrc_nets_{tag}{suffix(args.genes)}.log", "w")
                running.append((job, subprocess.Popen([sys.executable, "-u", __file__, *job, *common_args],
                                                      stdout=handle, stderr=subprocess.STDOUT)))
                log(f"start {' '.join(job)}")
            time.sleep(5)
            for item in list(running):
                code = item[1].poll()
                if code is not None:
                    running.remove(item)
                    log(f"{'done' if code == 0 else 'FAILED'} {' '.join(item[0])}")
                    if code:
                        failed.append(item[0])
        if failed:
            log(f"{len(failed)} job(s) failed - see analysis/representation/logs/xsrc_nets_*.log; "
                "rerun `all` to retry only those. Combining what finished.")

    sources, bundle = load(args.genes)
    log(f"loaded: dataset0 {len(sources['dataset0']):,} sites, data1 {len(sources['data1']):,}")
    if args.stage == "baseline":
        baseline_scores(sources, args.genes, log)
        return
    if args.stage == "fit":
        folds = [args.fold] if args.fold is not None else range(n_folds)
        for model in models:
            for fold in folds:
                for arm in ([args.arm] if args.arm else ARMS):
                    path = cache_path(model, arm, fold, args.genes)
                    if path.exists():
                        continue
                    started = time.time()
                    log(f"[{model} {arm} fold {fold}]")
                    np.save(path, fit_one(model, bundle, arm, fold, args, log))
                    log(f"[{model} {arm} fold {fold}] done ({(time.time() - started) / 60:.1f} min)")
        return
    # Deepset first: every other model is also compared against it.
    for model in sorted(models, key=lambda m: m != "deepset"):
        if candidate(model, sources, args.genes) is not None:
            combine(model, sources, args, log)
        else:
            log(f"{model}: incomplete, skipped")


if __name__ == "__main__":
    main()
