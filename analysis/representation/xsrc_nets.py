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
import pandas as pd
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
# h2gcn variants (graph.py), discussed 2026-10-02.
GRAPH_VARIANTS = {
    "h2gcn_local": {"window": 50, "transcript": False},      # local edges within 50 nt only
    "h2gcn_transcript": {"local": False},                    # transcript summary only
    "h2gcn_twohead": {"output": "twohead"},                  # one head per labelling
    "h2gcn_noisy": {"output": "noisy"},                      # two noisy annotators of one truth
    "h2gcn_noisy2": {"output": "noisy"},                     # the same, positives not up-weighted
    "h2gcn_knn": {"reader": "knn"},                          # k-NN read graph inside h2gcn
    # local edges only + one head per labelling: the two best findings together (2026-10-03)
    "h2gcn_local_twohead": {"window": 50, "transcript": False, "output": "twohead"},
    # The last pre-leaderboard batch (2026-10-03), both on h2gcn_local:
    "h2gcn_local_deep": {"window": 50, "transcript": False, "deep": 3},   # residual read + site encoders
    "h2gcn_aux": {"window": 50, "transcript": False, "aux": 0.5},         # + own-reads-only auxiliary head
    # The two winners together, on twohead's base (full h2gcn): 2026-10-03
    "h2gcn_twohead_aux": {"output": "twohead", "aux": 0.5},
    # Richer read pooling on h2gcn_aux (2026-10-04): quantiles; quantiles + modified fraction
    "h2gcn_aux_q": {"window": 50, "transcript": False, "aux": 0.5, "pool": "quantile"},
    "h2gcn_aux_qf": {"window": 50, "transcript": False, "aux": 0.5, "pool": "quantile", "frac": True},
    # Radius test (2026-10-06): h2gcn_aux with only the neighbour radius changed,
    # to separate "a wider radius helps" from the transcript node's effect.
    "h2gcn_aux_r100": {"window": 100, "transcript": False, "aux": 0.5},
    "h2gcn_aux_r200": {"window": 200, "transcript": False, "aux": 0.5},
    "h2gcn_aux_r400": {"window": 400, "transcript": False, "aux": 0.5},
    # Constrained corroboration and a GraphGPS control (2026-10-08), all on h2gcn_aux.
    # A: neighbour bands weighted by cell line 1's co-modification curve, frozen;
    #    its control reverses the weights across distances.
    "fk_band": {"transcript": False, "aux": 0.5, "nbr": "bands"},
    "fk_band_shuffled": {"transcript": False, "aux": 0.5, "nbr": "bands", "band_shuffle": True},
    # A2: one exponential distance kernel (90 nt), up to 150 nt.
    "fk_kernel": {"transcript": False, "aux": 0.5, "nbr": "kernel"},
    # B: logit = read-only score + depth/confidence gate x neighbour correction (exactly
    #    the read-only score when isolated), trained with edge and site dropout.
    "res_gate": {"transcript": False, "aux": 0.5, "nbr": "kernel", "residual": True,
                 "edge_drop": 0.3, "node_drop": 0.15},
    "res_nogate": {"transcript": False, "aux": 0.5, "nbr": "kernel", "residual": True, "gate": False,
                   "edge_drop": 0.3, "node_drop": 0.15},
    "res_nodrop": {"transcript": False, "aux": 0.5, "nbr": "kernel", "residual": True},
    # C: neighbours pass only (read-only score, log depth); control: messages from other sites.
    "scalar_msg": {"transcript": False, "aux": 0.5, "scalar": True},
    "scalar_random": {"transcript": False, "aux": 0.5, "scalar": True, "scalar_shuffle": True},
    # GraphGPS-style control: local channels + global attention over the whole transcript.
    "gps": {"window": 50, "transcript": False, "aux": 0.5, "gps": True},
}
GRAPH_MODELS = {**{k: {"kind": k} for k in graph.KINDS},
                **{k: {"kind": "h2gcn", **v} for k, v in GRAPH_VARIANTS.items()}}
# Two labellings in one training set exist only when both files are trained on.
MODEL_ARMS = {"h2gcn_twohead": ("pooled_both",), "h2gcn_noisy": ("pooled_both",),
              "h2gcn_noisy2": ("pooled_both",), "h2gcn_local_twohead": ("pooled_both",),
              "h2gcn_twohead_aux": ("pooled_both",), "h2gcn_aux_q": ("pooled_both",),
              "h2gcn_aux_qf": ("pooled_both",)}
# The radius variants run both arms: dataset0-only is the unseen-cell-line test
# (dataset0 and data1 are different cell lines - course staff, 2026-10-05).
# --arms: restrict every model to some arms (the seed runs train only the arm that ships).
ARMS_OVERRIDE: tuple | None = None
# --seed: 0 is the run every W&B row comes from; other seeds change only the
# initialisation, batch order and read subsets - never the split or the folds.
SEED = 0
MODELS = SET_MODELS + KNN_MODELS + tuple(GRAPH_MODELS)


def arms_of(model: str) -> tuple:
    arms = MODEL_ARMS.get(model, ARMS)
    return tuple(a for a in arms if a in ARMS_OVERRIDE) if ARMS_OVERRIDE else arms
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
    return ("" if genes >= 1 else f"_genes{genes:g}") + (f"_s{SEED}" if SEED else "")


def cache_path(model: str, arm: str, fold: int, genes: float):
    return OUT / f"{model}_{arm}_fold{fold}{suffix(genes)}.npy"


def weights_path(model: str, arm: str, fold: int, genes: float):
    return OUT / "weights" / f"{model}_{arm}_fold{fold}{suffix(genes)}.pt"


def save_weights(model, model_name, arm, fold, genes, std, extra=None) -> None:
    """Everything needed to rescore without retraining - other depths, data2,
    SG-NEx: the weights, and the read standardisation the network was trained
    with (and, for deepset_hand, the hand-feature standardisation)."""
    path = weights_path(model_name, arm, fold, genes)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_name, "state_dict": model.state_dict(),
                "read_mean": std.mean, "read_scale": std.scale, **(extra or {})}, path)


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
    # The other file's label of the same site, NaN where the site is in one file.
    bundle.y_other = np.concatenate([
        pd.Series(s1.y, index=s1.index).reindex(s0.index).to_numpy(),
        pd.Series(s0.y, index=s0.index).reindex(s1.index).to_numpy()]).astype(np.float32)
    return sources, bundle


def fit_one(model_name: str, bundle, arm: str, fold: int, args, log) -> np.ndarray:
    held = bundle.folds == fold
    train = ~held & ((bundle.file == 0) if arm == "dataset0" else True)
    v = np.array([common._unit(f"4262:val:{g}") for g in bundle.genes])
    fit_rows, val_rows = train & (v >= 0.1), train & (v < 0.1)
    target = bundle.y_own if arm == "dataset0" else bundle.y_both
    std = common.Standardiser.fit(bundle.reads, fit_rows)
    values = std(bundle.reads.values)
    seed = 4262 + fold + 1000 * SEED
    torch.manual_seed(seed)
    out = np.full(len(bundle.genes), np.nan, dtype=np.float32)
    held_rows = np.flatnonzero(held)

    if model_name in GRAPH_MODELS:
        graph_args = (values, bundle.reads.offsets, bundle.kmer_onehot, bundle.position)
        model = graph.GraphNet(**GRAPH_MODELS[model_name])
        model.obs = (bundle.y_own, bundle.y_other, bundle.file)
        model.unweighted = model_name == "h2gcn_noisy2"
        if GRAPH_MODELS[model_name].get("nbr") == "bands":
            # Frozen from cell line 1's TRAINING labels only - never the held-out fold.
            ref = fit_rows & (bundle.file == 0)
            w = graph.comod_band_weights(bundle.position[ref], bundle.graph_id[ref], bundle.y_own[ref])
            model.band_w = torch.tensor(w, dtype=torch.float32)
            log(f"    band weights {np.round(w, 3).tolist()} (bands {graph.FK_BANDS})")
        if GRAPH_MODELS[model_name].get("nbr") == "kernel" or GRAPH_MODELS[model_name].get("scalar"):
            # The kernel's decay scale, fitted to cell line 1's TRAINING labels only.
            ref = fit_rows & (bundle.file == 0)
            lam = graph.comod_decay_scale(bundle.position[ref], bundle.graph_id[ref], bundle.y_own[ref])
            model.kernel_nt = torch.tensor(lam, dtype=torch.float32)
            log(f"    kernel decay scale {lam:.1f} nt (fitted to cell line 1 training labels)")
        graph.train(model, graph.graphs_of(np.flatnonzero(fit_rows), bundle.graph_id, bundle.position),
                    graph.graphs_of(np.flatnonzero(val_rows), bundle.graph_id, bundle.position),
                    *graph_args, target, bundle.y_own, args.minutes, args.epochs, seed, log)
        save_weights(model, model_name, arm, fold, args.genes, std)
        scores = graph.score(model, graph.graphs_of(held_rows, bundle.graph_id, bundle.position),
                             *graph_args)
        out[held_rows] = [scores[r] for r in held_rows]
        return out

    site, extra = bundle.kmer_onehot, None
    if model_name == "deepset_hand":
        mean = np.nanmean(bundle.hand[fit_rows], 0)
        sd = np.nanstd(bundle.hand[fit_rows], 0)
        hand = np.nan_to_num((bundle.hand - mean) / np.where(sd > 0, sd, 1)).astype(np.float32)
        site = np.concatenate([site, hand], 1)
        model = DeepSetHand(hand.shape[1])
        extra = {"hand_mean": mean, "hand_sd": sd}
    elif model_name in KNN_MODELS:
        model = readgraph.KNNDeepSet(model_name.removeprefix("knn_"), k=KNN_K)
    else:
        model = learn.build(model_name, 0)
    view = SimpleNamespace(reads=bundle.reads, kmer_onehot=site, window_onehot=bundle.window_onehot,
                           y=target, y_val=bundle.y_own)
    # learn.train branches on the name only to pick the loss; every model here
    # is a supervised set encoder, which "deepset" selects.
    learn.train(model, "deepset", "set", True, view, {"encoder": fit_rows, "encoder_val": val_rows},
                # learn.train seeds its batch order and read subsets from this argument.
                values, args.minutes, args.epochs, fold + 1000 * SEED, log, recipe="v3")
    save_weights(model, model_name, arm, fold, args.genes, std, extra)
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
    paths = [cache_path(model, a, f, genes) for a in arms_of(model) for f in folds]
    if not all(p.exists() for p in paths):
        return None
    out = {}
    for arm in arms_of(model):
        total = np.nanmax(np.stack([np.load(cache_path(model, arm, f, genes)) for f in folds]), 0)
        out[arm] = {"dataset0": total[:n0], "data1": total[n0:]}
    return out


def combine(model: str, sources, args, log) -> None:
    cand = candidate(model, sources, args.genes)
    if cand is None:
        raise SystemExit(f"{model}: not every fold is fitted - run `xsrc_nets.py fit --model {model}`.")
    base = baseline_scores(sources, args.genes, log)
    # The selection rule is read off the arm that ships; a both-files-only model ships that one.
    headline = "dataset0" if "dataset0" in cand else "pooled_both"
    block = xs.summarise(sources, cand, base, headline_arm=headline)
    report = reporting.new("standard", subsample_seed=SUBSAMPLE_SEED, log=log)
    reporting.cross_source(report, block, f"{model} (network)", Config.load(BASELINE).name)
    extra = {}
    deepset = candidate("deepset", sources, args.genes) if model != "deepset" else None
    if deepset is not None:
        vs = xs.summarise(sources, cand, deepset, headline_arm=headline)
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
    name = f"{model}_nn__xsrc" + (f"__seed{SEED}" if SEED else "")
    path = reporting.write(report, common.RESULTS, name + ("__smoke" if smoke else ""))
    tracker = tracking.start(
        # Only seed 0 is a Decisions row; other seeds are read from their reports.
        name, enabled=not (args.no_wandb or smoke or SEED), job_type="evaluate", tags=[tracking.EVAL_TAG],
        config={"eval_schema": xs.EVAL_SCHEMA, "baseline": Config.load(BASELINE).name,
                "arms": ",".join(arms_of(model)), "features": "reads (network)", "model": model,
                "train_depths": "full (random read subsets)", "network_minutes": args.minutes,
                "script": "analysis/representation/xsrc_nets.py"})
    try:
        reporting.publish(report, tracker, path, name=name)
        if extra:
            tracker.log(extra)
            tracker.summary(extra)
    finally:
        tracker.finish()
    r = block["arms"][headline]
    log(f"{model}: worst {r['gain_worst']:+.4f}  mean {r['gain_mean']:+.4f}  -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fit", "baseline", "combine", "all"))
    ap.add_argument("--model", help="one model or a comma-separated list (default: all)")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--arms", help="comma-separated: restrict every model to these arms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=240.0, help="safety cap per network")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=14, help="all: concurrent fit processes")
    ap.add_argument("--genes", type=float, default=1.0, help="fraction of genes (smoke only)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    global SEED, ARMS_OVERRIDE
    SEED = args.seed
    ARMS_OVERRIDE = tuple(args.arms.split(",")) if args.arms else None
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
                       "--threads", str(args.threads), "--genes", str(args.genes),
                       "--seed", str(args.seed)] + (["--arms", args.arms] if args.arms else [])
        # One job per network, so seventy networks spread over the cores.
        queue = [["baseline"]] + [["fit", "--model", m, "--fold", str(f), "--arm", a]
                                  for m in models for f in range(n_folds) for a in arms_of(m)
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
                for arm in ([args.arm] if args.arm else arms_of(model)):
                    path = cache_path(model, arm, fold, args.genes)
                    if path.exists():
                        continue
                    started = time.time()
                    tag = f"[{model} {arm} fold {fold}" + (f" seed {SEED}]" if SEED else "]")
                    log(tag)
                    np.save(path, fit_one(model, bundle, arm, fold, args, log))
                    log(f"{tag} done ({(time.time() - started) / 60:.1f} min)")
        return
    # Deepset first: every other model is also compared against it.
    for model in sorted(models, key=lambda m: m != "deepset"):
        if candidate(model, sources, args.genes) is not None:
            combine(model, sources, args, log)
        else:
            log(f"{model}: incomplete, skipped")


if __name__ == "__main__":
    main()
