"""The control the E3 numbers need for SUPERVISED encoders.

In the screen, a supervised encoder learns from half A's labels and the probes
from half B's, so "hand + z" indirectly uses twice the labels that "hand" does.
Some of any gain could be that. This removes it:

  per outer fold, the training genes are split in the same two halves;
  encoder 1 trains on A and scores B, encoder 2 trains on B and scores A,
  so every training site gets an out-of-fold network score; held-out sites get
  the mean of the two encoders' scores;
  LightGBM is trained on ALL training rows, hand features vs hand + that score.

Both arms see every training label, as in the real harness. A network's latent
vectors from two separately trained encoders are not aligned, so the stacked
quantity is the scalar logit, which is.

`mlp_hand` run through the identical procedure is the stacking control: a
network over the SAME hand features. What `deepset` gains beyond it comes from
the raw reads rather than from stacking a second learner.

    python analysis/representation/crossfit.py --model deepset
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import common
import learn
from probe import LGBM, scores
from m6a import registry


def half_roles(roles: dict, which: str, bundle) -> dict:
    """Roles for an encoder trained on one half, validated on 10% of its genes."""
    if which == "A":
        return {"encoder": roles["encoder"], "encoder_val": roles["encoder_val"]}
    v = np.array([common._unit(f"4262:val:{g}") for g in bundle.genes])
    return {"encoder": roles["probe"] & (v >= 0.1), "encoder_val": roles["probe"] & (v < 0.1)}


def encoder_scores(name, bundle, roles, rows_to_score, fold, minutes, epochs, log):
    kind, _ = learn.SPECS[name]
    model = learn.build(name, bundle.X.shape[1])
    fit_rows = roles["encoder"] | roles["encoder_val"]
    mean = bundle.X[roles["encoder"]].mean().to_numpy()
    sd = bundle.X[roles["encoder"]].std().replace(0, 1).to_numpy()
    bundle._hand_std = ((bundle.X.to_numpy() - mean) / sd).astype(np.float32)
    std = common.Standardiser.fit(bundle.reads, roles["encoder"])
    values = std(bundle.reads.values)
    learn.train(model, name, kind, True, bundle, roles, values, minutes, epochs, fold, log)
    model.eval()
    if kind == "hand":
        with torch.no_grad():
            return model(learn.t(bundle._hand_std[rows_to_score])).numpy()
    return learn.site_logits(model, values, bundle.reads.offsets, rows_to_score,
                             bundle.kmer_onehot, bundle.window_onehot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    bundle = common.load(args.limit)
    records = []
    saved = {}
    for fold in sorted(np.unique(bundle.folds)):
        started = time.time()
        torch.manual_seed(4262 + fold)
        roles = common.split_roles(bundle, fold)
        held = np.flatnonzero(roles["heldout"])
        half = {"A": np.flatnonzero(roles["encoder"] | roles["encoder_val"]),
                "B": np.flatnonzero(roles["probe"])}
        score = np.full(len(bundle.y), np.nan)
        held_scores = []
        for trained_on, scored in (("A", "B"), ("B", "A")):
            log(f"[{args.model} fold {fold}] encoder on half {trained_on}")
            s = encoder_scores(args.model, bundle, half_roles(roles, trained_on, bundle),
                               np.concatenate([half[scored], held]), fold,
                               args.minutes, args.epochs, log)
            score[half[scored]] = s[:len(half[scored])]
            held_scores.append(s[len(half[scored]):])
        score[held] = np.mean(held_scores, axis=0)
        saved[f"fold{fold}"] = score.astype(np.float32)   # for combine.py

        train = np.flatnonzero(~roles["heldout"])
        X = bundle.X.copy()
        X["stacked_score"] = score
        y = bundle.y
        out = {"fold": int(fold), "repetition": 0,
               "encoder_alone": scores(y[held], score[held])}
        for arm, cols in (("hand", list(bundle.X.columns)),
                          ("hand_plus_score", list(bundle.X.columns) + ["stacked_score"])):
            model = registry.get("models", "lightgbm")(**LGBM)
            model.fit(X.iloc[train][cols], y[train])
            out[arm] = scores(y[held], model.predict_proba(X.iloc[held][cols]))
        records.append(out)
        log(f"  fold {fold}: hand {out['hand']['pr_auc']:.4f}  hand+score "
            f"{out['hand_plus_score']['pr_auc']:.4f}  encoder alone "
            f"{out['encoder_alone']['pr_auc']:.4f}  ({(time.time() - started) / 60:.1f} min)")
    suffix = f"__limit{args.limit}" if args.limit else ""
    path = common.RESULTS / f"crossfit_{args.model}{suffix}.json"
    path.write_text(json.dumps(records, indent=1))
    common.OUT.mkdir(parents=True, exist_ok=True)
    np.savez(common.OUT / f"crossfit_scores_{args.model}{suffix}.npz", **saved)
    d = np.array([r["hand_plus_score"]["pr_auc"] - r["hand"]["pr_auc"] for r in records])
    log(f"[{args.model}] hand+score minus hand: {d.mean():+.4f}, wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
