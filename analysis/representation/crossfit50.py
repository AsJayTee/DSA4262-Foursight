"""R2 worker: cross-fitted network scores over 10 repetitions x 5 folds.

    python crossfit50.py --jobs deepset:0:0 deepset:0:1 mlp_hand:3:4 --threads 4

A job is (model, repetition, fold). Repetition r uses the harness's split seed
`m6a.crossval.repetition_seeds(10)[r]` - repetition 0 is the canonical 4262
split - so the 50 observations pair with the harness's own. Within the job, as
in crossfit.py: the training genes are halved, an encoder trained on each half
scores the other, held-out sites get the mean of the two. Scores are saved at
full depth AND at 1 / 3 / 5 / 10 reads (the harness's keyed subsample), so
combine50.py can train depth-augmented trees and test at low depth.

One process loads the data once and runs its jobs in order; run_r2.py starts
many of them. A job whose output exists is skipped, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import pandas as pd
import torch

import common
import learn
from crossfit import half_roles
from m6a.crossval import repetition_seeds
from m6a.data import assign_folds

DEPTHS = (1, 3, 5, 10)
# The round-3 architectures train with recipe v2 (see learn.train), exactly as
# they did in diagnose2.py: (recipe, epochs, minutes per half-encoder).
RECIPES = {"res_deepset": ("v2", 60, 150), "res_attn_mil": ("v2", 60, 150),
           "set_transformer": ("v2", 60, 150)}
N_REPEATS = 10


def job_path(model: str, rep: int, fold: int):
    return common.OUT / "cf50" / model / f"r{rep}f{fold}.npz"


def repetition_folds(bundle, rep: int) -> np.ndarray:
    seed = repetition_seeds(N_REPEATS)[rep]
    return assign_folds(pd.DataFrame({"gene_id": bundle.genes}), seed, 5, "gene_id").to_numpy()


def run_job(bundle, model_name: str, rep: int, fold: int, minutes: float, epochs: int, log) -> None:
    path = job_path(model_name, rep, fold)
    if path.exists():
        log(f"skip {model_name} r{rep}f{fold}: done")
        return
    started = time.time()
    b = dataclasses.replace(bundle, folds=repetition_folds(bundle, rep))
    roles = common.split_roles(b, fold)
    held = np.flatnonzero(roles["heldout"])
    half = {"A": np.flatnonzero(roles["encoder"] | roles["encoder_val"]),
            "B": np.flatnonzero(roles["probe"])}
    kind, _ = learn.SPECS[model_name]
    keys = ["full", *[f"d{d}" for d in DEPTHS]]
    out = {k: np.full(len(b.y), np.nan, dtype=np.float32) for k in keys}
    held_parts = {k: [] for k in keys}
    info = {"model": model_name, "rep": rep, "fold": fold, "halves": []}

    for i, (trained_on, scored) in enumerate((("A", "B"), ("B", "A"))):
        torch.manual_seed(4262 + 1000 * rep + 10 * fold + i)
        r = half_roles(roles, trained_on, b)
        mean = b.X[r["encoder"]].mean().to_numpy()
        sd = b.X[r["encoder"]].std().replace(0, 1).to_numpy()
        hand_std = lambda X: ((X.to_numpy() - mean) / sd).astype(np.float32)  # noqa: E731
        b._hand_std = hand_std(b.X)
        std = common.Standardiser.fit(b.reads, r["encoder"])
        values = std(b.reads.values)
        model = learn.build(model_name, b.X.shape[1])
        recipe, ep, box = RECIPES.get(model_name, ("v1", epochs, minutes))
        history = learn.train(model, model_name, kind, True, b, r, values, box, ep,
                              fold, lambda *a: None, recipe=recipe)
        info["halves"].append({"epochs": len(history),
                               "best_val_ap": max(h.get("val_ap", -1) for h in history)})
        model.eval()
        rows = np.concatenate([half[scored], held])
        for key in keys:
            if kind == "hand":
                X = b.X if key == "full" else b.depth_X[int(key[1:])]
                with torch.no_grad():
                    s = model(learn.t(hand_std(X)[rows])).numpy()
            else:
                blocks = b.reads if key == "full" else b.depth_reads[int(key[1:])]
                v = values if key == "full" else std(blocks.values)
                s = learn.site_logits(model, v, blocks.offsets, rows, b.kmer_onehot, b.window_onehot)
            out[key][half[scored]] = s[:len(half[scored])]
            held_parts[key].append(s[len(half[scored]):])
    for key in keys:
        out[key][held] = np.mean(held_parts[key], axis=0)
    info["minutes"] = round((time.time() - started) / 60, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, info=json.dumps(info), **out)
    log(f"done {model_name} r{rep}f{fold} in {info['minutes']} min, epochs "
        f"{[h['epochs'] for h in info['halves']]}, val AP "
        f"{[round(h['best_val_ap'], 4) for h in info['halves']]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True, help="model:rep:fold")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
    bundle = common.load(args.limit, extra_depths=DEPTHS)
    log(f"loaded; {len(args.jobs)} jobs")
    for spec in args.jobs:
        name, rep, fold = spec.split(":")
        try:
            run_job(bundle, name, int(rep), int(fold), args.minutes, args.epochs, log)
        except Exception as exc:  # one bad job must not take the worker's other jobs down
            log(f"FAILED {spec}: {exc!r}")


if __name__ == "__main__":
    main()
