"""Does deepset's gain survive cross-source evaluation (docs/decisions/0029)?

On dataset0 cross-validation, quantiles + a cross-fitted deepset score beat
quantiles by +0.022 PR AUC (results/r2.md, 50/50 wins). `everything` beat
quantiles by +0.065 on the same test and kept +0.016 on data1, so that number
is not evidence until this has run.

The candidate is `configs/quantiles.yaml` plus one column - the deepset site
logit - and the baseline is `configs/quantiles.yaml` itself, both LightGBM,
both trained at full depth, both on the cross-source union gene split. Arms:

  dataset0     everything trained on dataset0 (as shipped)
  pooled_both  everything trained on both files; a shared site's deepset target
               is the mean of its two labels, as LightGBM's four 1/4-weight rows
               are (m6a.crosssource.training_rows)

The deepset score is cross-fitted as in crossfit.py: per outer fold, the
training genes split in two halves; a network trained on each half scores the
other, so every training row carries an out-of-fold score; held-out rows (both
files) get the mean of the two networks' scores. No score a LightGBM row trains
on came from a network that saw that row's label.

    python analysis/representation/xsrc_deepset.py fit --fold 0   # one outer fold, both arms
    python analysis/representation/xsrc_deepset.py combine        # LightGBM, gains, W&B
    python analysis/representation/xsrc_deepset.py all            # every fold, then combine
    # smoke: a tenth of the genes, short networks, nothing logged
    python analysis/representation/xsrc_deepset.py all --genes 0.1 --minutes 0.3 --no-wandb

Fits are cached per (arm, fold) under .cache/representation/xsrc_deepset/, so
folds can run as parallel processes and a killed run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import common
import learn
from m6a import crosssource as xs
from m6a import external, feature_cache, registry, tracking
from m6a import report as reporting
from m6a.config import Config
from m6a.crossval import _reads_for
from m6a.data import SUBSAMPLE_SEED, ReadBlocks

ARMS = ("dataset0", "pooled_both")
MODEL = "deepset"
BASELINE = common.ROOT / "configs" / "quantiles.yaml"
OUT = common.OUT / "xsrc_deepset"
COLUMN = "deepset_logit"
NAME = "quantiles_deepset__xsrc"


def quiet(*_a, **_k) -> None:
    pass


# ------------------------------------------------------------------ data

def load(gene_fraction: float):
    """Both files on the union split, plus one deepset bundle over their rows
    (dataset0's first, then data1's)."""
    cfg = Config.load(BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=quiet)
    json_paths = {"dataset0": common.DATA, "data1": external.paths(common.DATA_DIR, "data1")[0]}
    reads, kmers = {}, {}
    for name, src in sources.items():
        if gene_fraction < 1:
            keep = np.array([common._unit(f"4262:xsrc_smoke:{g}") < gene_fraction for g in src.genes])
            src = sources[name] = xs.Source(name, src.index[keep], {d: X.iloc[keep] for d, X in src.X.items()},
                                            src.y[keep], src.genes[keep], src.folds[keep])
        ext = feature_cache.extract(json_paths[name], cfg.features, [None], log=quiet)[None]
        blocks = feature_cache.extract_reads(json_paths[name], log=quiet)
        reads[name] = _reads_for(blocks, ext, src.index, None, SUBSAMPLE_SEED)
        kmers[name] = ext.sites.loc[src.index, "kmer"].to_numpy()

    s0, s1 = sources["dataset0"], sources["data1"]
    counts = np.concatenate([reads["dataset0"].counts, reads["data1"].counts])
    all_kmers = np.concatenate([kmers["dataset0"], kmers["data1"]])
    own = np.concatenate([s0.y, s1.y]).astype(np.float32)
    other = np.concatenate([pd.Series(s1.y, index=s1.index).reindex(s0.index).to_numpy(),
                            pd.Series(s0.y, index=s0.index).reindex(s1.index).to_numpy()])
    bundle = SimpleNamespace(
        reads=ReadBlocks(np.concatenate([reads["dataset0"].values, reads["data1"].values]),
                         np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)),
        kmer_onehot=common.onehot(all_kmers, 7),
        window_onehot=np.stack([common.onehot([k[i:i + 5] for k in all_kmers], 5) for i in range(3)], 1),
        y_own=own,
        # A shared site's two labels averaged: 0.5 where the labellings disagree.
        y_both=np.where(np.isnan(other), own, (own + np.nan_to_num(other)) / 2).astype(np.float32),
        file=np.repeat([0, 1], [len(s0), len(s1)]),
        genes=np.concatenate([s0.genes, s1.genes]),
        folds=np.concatenate([s0.folds, s1.folds]),
    )
    return sources, bundle


# ------------------------------------------------------------------ deepset

def train_and_score(bundle, train, val, score_rows, target, fold, args, log) -> np.ndarray:
    view = SimpleNamespace(reads=bundle.reads, kmer_onehot=bundle.kmer_onehot,
                           window_onehot=bundle.window_onehot, y=target, y_val=bundle.y_own)
    roles = {"encoder": train, "encoder_val": val}
    std = common.Standardiser.fit(bundle.reads, train)
    values = std(bundle.reads.values)
    model = learn.build(MODEL, 0)
    learn.train(model, MODEL, "set", True, view, roles, values, args.minutes, args.epochs, fold, log)
    return learn.site_logits(model, values, bundle.reads.offsets, score_rows,
                             bundle.kmer_onehot, bundle.window_onehot)


def fit_fold(bundle, arm: str, fold: int, args, log) -> np.ndarray:
    """One outer fold's cross-fitted deepset column over every row (NaN where unused)."""
    held = bundle.folds == fold
    train = ~held & ((bundle.file == 0) if arm == "dataset0" else True)
    target = bundle.y_own if arm == "dataset0" else bundle.y_both
    # The same gene hashes as crossfit.py, so halves never split a gene.
    u = np.array([common._unit(f"4262:roles:{g}") for g in bundle.genes])
    v = np.array([common._unit(f"4262:val:{g}") for g in bundle.genes])
    out = np.full(len(bundle.genes), np.nan, dtype=np.float32)
    held_scores = []
    for half, other in ((u < 0.5, u >= 0.5), (u >= 0.5, u < 0.5)):
        fit_rows = train & half
        score_rows = np.flatnonzero((train & other) | held)
        log(f"  [{arm} fold {fold}] network on {int(fit_rows.sum()):,} sites, "
            f"scoring {len(score_rows):,}")
        s = train_and_score(bundle, fit_rows & (v >= 0.1), fit_rows & (v < 0.1),
                            score_rows, target, fold, args, log)
        scored = np.zeros(len(out), dtype=bool)
        scored[score_rows] = True
        out[scored & ~held] = s[(train & other)[score_rows]]
        held_scores.append(s[held[score_rows]])
    out[held] = np.mean(held_scores, axis=0)
    return out


def cache_path(arm: str, fold: int, gene_fraction: float):
    tag = "" if gene_fraction >= 1 else f"_genes{gene_fraction:g}"
    return OUT / f"{arm}_fold{fold}{tag}.npy"


# ------------------------------------------------------------------ LightGBM + readouts

def combine(sources, bundle, args, log) -> dict:
    cfg = Config.load(BASELINE)
    model_class = registry.get("models", cfg.model)
    n0 = len(sources["dataset0"])
    folds = sorted(np.unique(sources["dataset0"].folds))
    cand = {a: {s: np.full(len(sources[s]), np.nan) for s in xs.SOURCES} for a in ARMS}
    base = {a: {s: np.full(len(sources[s]), np.nan) for s in xs.SOURCES} for a in ARMS}
    for arm in ARMS:
        for fold in folds:
            column = np.load(cache_path(arm, fold, args.genes))
            with_score = {
                name: replace(src, X={None: src.X[None].assign(**{COLUMN: part})})
                for (name, src), part in zip(sources.items(), (column[:n0], column[n0:]))
            }
            for oof, srcs in ((cand, with_score), (base, sources)):
                X, y, w = xs.training_rows(srcs, arm, fold, [None])
                if X.isna().any().any():
                    raise RuntimeError(f"{arm} fold {fold}: a training row has no deepset "
                                       "score - the cross-fit left a hole.")
                model = xs._fit(model_class, cfg.model_params, X, y, w)
                for s in xs.SOURCES:
                    held = srcs[s].folds == fold
                    oof[arm][s][held] = model.predict_proba(srcs[s].X[None][held])
            log(f"  {arm} fold {fold}: LightGBM with and without the deepset column")
    return xs.summarise(sources, cand, base)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fit", "combine", "all"))
    ap.add_argument("--fold", type=int, help="fit: one outer fold (default: every fold)")
    ap.add_argument("--minutes", type=float, default=8.0, help="per network, as crossfit.py")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--genes", type=float, default=1.0, help="fraction of genes (smoke only)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--parallel", action="store_true",
                    help="all: fit each fold in its own process (a machine with the cores and "
                         "~3 GB of memory per fold), then combine")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    started = time.time()

    if args.stage == "all" and args.parallel:
        import subprocess
        import sys
        common.ROOT.joinpath("analysis", "representation", "logs").mkdir(exist_ok=True)
        passthrough = ["--minutes", str(args.minutes), "--epochs", str(args.epochs),
                       "--threads", str(args.threads), "--genes", str(args.genes)]
        jobs = []
        for fold in range(Config.load(BASELINE).split.n_folds):
            log_file = open(common.ROOT / "analysis" / "representation" / "logs"
                            / f"xsrc_deepset_fold{fold}.log", "w")
            jobs.append(subprocess.Popen([sys.executable, "-u", __file__, "fit", "--fold", str(fold),
                                          *passthrough], stdout=log_file, stderr=subprocess.STDOUT))
        failed = [fold for fold, job in enumerate(jobs) if job.wait() != 0]
        if failed:
            raise SystemExit(f"fold(s) {failed} failed - see analysis/representation/logs/"
                             "xsrc_deepset_fold<k>.log, then rerun; finished folds are cached.")
        args.stage = "combine"

    sources, bundle = load(args.genes)
    log(f"loaded: dataset0 {len(sources['dataset0']):,} sites, data1 {len(sources['data1']):,}, "
        f"{bundle.reads.total_reads:,} reads ({time.time() - started:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("fit", "all"):
        folds = [args.fold] if args.fold is not None else sorted(np.unique(bundle.folds))
        for fold in folds:
            for arm in ARMS:
                path = cache_path(arm, fold, args.genes)
                if path.exists():
                    log(f"  [{arm} fold {fold}] cached")
                    continue
                t0 = time.time()
                np.save(path, fit_fold(bundle, arm, fold, args, log))
                log(f"  [{arm} fold {fold}] done ({(time.time() - t0) / 60:.1f} min)")
    if args.stage == "fit":
        return

    block = combine(sources, bundle, args, log)
    report = reporting.new("standard", subsample_seed=SUBSAMPLE_SEED, log=log)
    reporting.cross_source(report, block, "quantiles + deepset", Config.load(BASELINE).name)
    report.data["deepset"] = {"minutes": args.minutes, "epochs": args.epochs, "genes": args.genes,
                              "cross_fit": "two gene halves per outer fold (crossfit.py)"}
    smoke = args.genes < 1
    stem = NAME + ("__smoke" if smoke else "")
    path = reporting.write(report, common.RESULTS, stem)
    tracker = tracking.start(
        NAME, enabled=not (args.no_wandb or smoke), job_type="evaluate", tags=[tracking.EVAL_TAG],
        config={"eval_schema": xs.EVAL_SCHEMA, "baseline": Config.load(BASELINE).name,
                "arms": ",".join(ARMS), "features": "quantiles_v1 + deepset logit (cross-fitted)",
                "model": "lightgbm", "train_depths": "full", "deepset_minutes": args.minutes,
                "script": "analysis/representation/xsrc_deepset.py"})
    try:
        reporting.publish(report, tracker, path, name=stem)
    finally:
        tracker.finish()
    log(f"\nreport -> {path}   ({(time.time() - started) / 60:.1f} min)")
    log(json.dumps({a: {k: round(r[k], 4) for k in ("gain_mean", "gain_worst")}
                    for a, r in block["arms"].items()}))


if __name__ == "__main__":
    main()
