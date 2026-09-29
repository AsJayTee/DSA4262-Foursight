"""Fit diagnostic, round 2: does better optimisation lift the networks?

    python diagnose2.py --jobs deepset@v1:0 res_deepset@v2:3 ...   # workers
    python diagnose2.py --aggregate                                # table

A job is model@recipe:fold. Same protocol as the screen - the encoder learns
from half A of the fold's training genes and is scored on the held-out fold -
except the validation slice is 20% of half A (not 10%, which made early
stopping chase noise), and every epoch also scores 5,000 of the network's own
training sites. Reported per job: average precision on the network's own
training sites at the chosen epoch, on validation, and on the held-out fold.

The baseline is deepset@v1, the network in the 50-observation result. Each
variant is paired against it over the 5 folds (held-out AP).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score

import common
import learn

OUT = common.RESULTS / "diagnose2"
# Heavier models get fewer epochs so the cosine schedule completes in the time box.
EPOCHS = {"res_deepset": 60, "res_attn_mil": 60, "set_transformer": 60}
MINUTES = 150


def roles_with_bigger_val(bundle, fold):
    roles = common.split_roles(bundle, fold)
    a = roles["encoder"] | roles["encoder_val"]
    v = np.array([common._unit(f"4262:val:{g}") for g in bundle.genes])
    return {**roles, "encoder": a & (v >= 0.2), "encoder_val": a & (v < 0.2)}


def run(bundle, spec: str, log) -> None:
    path = OUT / f"{spec.replace('@', '_').replace(':', '_f')}.json"
    if path.exists():
        return
    model_recipe, fold = spec.split(":")
    name, recipe = model_recipe.split("@")
    fold = int(fold)
    torch.manual_seed(4262 + fold)
    started = time.time()
    roles = roles_with_bigger_val(bundle, fold)
    std = common.Standardiser.fit(bundle.reads, roles["encoder"])
    values = std(bundle.reads.values)
    model = learn.build(name, bundle.X.shape[1])
    epochs = EPOCHS.get(name, 200) if recipe == "v2" else 400
    history = learn.train(model, name, "set", True, bundle, roles, values, MINUTES, epochs, fold,
                          lambda *a: None, track_train=5000, recipe=recipe)
    held = np.flatnonzero(roles["heldout"])
    s = learn.site_logits(model, values, bundle.reads.offsets, held, bundle.kmer_onehot,
                          bundle.window_onehot)
    best = max(history, key=lambda h: h["val_ap"])
    record = {"model": name, "recipe": recipe, "fold": fold,
              "parameters": int(sum(p.numel() for p in model.parameters())),
              "epochs": len(history), "best_epoch": best["epoch"],
              "train_ap_at_best": best["train_ap"], "val_ap_at_best": best["val_ap"],
              "train_ap_last": history[-1]["train_ap"],
              "heldout_ap": float(average_precision_score(bundle.y[held], s)),
              "minutes": round((time.time() - started) / 60, 1), "history": history}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    log(f"{spec}: best epoch {record['best_epoch']}/{record['epochs']}  train {record['train_ap_at_best']:.4f}"
        f"  val {record['val_ap_at_best']:.4f}  held-out {record['heldout_ap']:.4f}  ({record['minutes']} min)")


def aggregate() -> None:
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    by = {}
    for r in recs:
        by.setdefault(f"{r['model']}@{r['recipe']}", {})[r["fold"]] = r
    base = by.get("deepset@v1", {})
    lines = ["| model | recipe | params | epochs (best/ran) | train AP | val AP | held-out AP | vs deepset@v1 | wins | p (corrected) |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key, folds in sorted(by.items()):
        f = sorted(folds)
        m = lambda k: np.mean([folds[i][k] for i in f])  # noqa: E731
        row = (f"| {key.split('@')[0]} | {key.split('@')[1]} | {folds[f[0]]['parameters']:,} | "
               f"{m('best_epoch'):.0f}/{m('epochs'):.0f} | {m('train_ap_at_best'):.4f} | "
               f"{m('val_ap_at_best'):.4f} | {m('heldout_ap'):.4f} |")
        shared = [i for i in f if i in base]
        if key != "deepset@v1" and len(shared) >= 2:
            d = np.array([folds[i]["heldout_ap"] - base[i]["heldout_ap"] for i in shared])
            n = len(d)
            # 5-fold CV correction (test/train = 1/4); the encoder trains on
            # fewer rows here, which makes this, if anything, optimistic.
            t_stat = d.mean() / np.sqrt((1 / n + 0.25) * d.var(ddof=1)) if d.std() > 0 else np.nan
            p = 2 * stats.t.sf(abs(t_stat), n - 1) if np.isfinite(t_stat) else np.nan
            row += f" {d.mean():+.4f} | {(d > 0).sum()}/{n} | {p:.3f} |"
        else:
            row += " - | - | - |"
        lines.append(row)
    text = "\n".join(lines)
    (common.RESULTS / "diagnose2.md").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    torch.set_num_threads(args.threads)
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
    bundle = common.load(args.limit, extra_depths=())
    for spec in args.jobs:
        try:
            run(bundle, spec, log)
        except Exception as exc:
            log(f"FAILED {spec}: {exc!r}")


if __name__ == "__main__":
    main()
