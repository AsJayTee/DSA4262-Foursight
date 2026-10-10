"""Fit the shipped ensemble on every labelled site of both files (agreed 2026-10-04).

    python analysis/representation/final_fit.py --model h2gcn_twohead_aux --seed 0
    python analysis/representation/final_fit.py --all        # the four, one process each
    # smoke: 5% of genes, one epoch
    python analysis/representation/final_fit.py --model h2gcn_aux --seed 0 --genes 0.05 --epochs 1

The ensemble chosen on cross-source evaluation (decisions 0032, 0034; analysis in
day_significance.py): h2gcn_twohead_aux + res_gate + scalar_drop, two seeds each,
trained on both files ("pooled_both"). Until 2026-10-10 it was h2gcn_twohead_aux
+ h2gcn_aux (0033).

Training matches every evaluated network except that no fold is held out:
the same model definitions (xsrc_nets.GRAPH_MODELS), recipe v3, and early
stopping on the same 10% of genes (hash "4262:val:<gene>") - so the stopping
rule is the one that was evaluated, not a fixed epoch count chosen afterwards.

Writes .cache/representation/final/<model>_s<seed>.pt: weights, the read
standardisation, the model's constructor arguments and the training history.
export_final.py turns these into the numpy files predict.py loads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import torch

import common
import graph
import xsrc_nets as X

# Decision 0034 (2026-10-10): h2gcn_twohead_aux + res_gate + scalar_drop, two seeds each, equal
# weight per model - the ensemble that tied the previous one on cell line 1 and beat it on cell
# line 2 (+0.006) and data1's new genes (+0.012), every component seed-averaged.
ENSEMBLE = [("h2gcn_twohead_aux", 0), ("h2gcn_twohead_aux", 1), ("res_gate", 0), ("res_gate", 1),
            ("scalar_drop", 0), ("scalar_drop", 1)]
OUT = common.OUT / "final"


def fit(model_name: str, seed: int, args, log) -> None:
    sources, bundle = X.load(args.genes)
    v = np.array([common._unit(f"4262:val:{g}") for g in bundle.genes])
    fit_rows, val_rows = v >= 0.1, v < 0.1
    std = common.Standardiser.fit(bundle.reads, fit_rows)
    values = std(bundle.reads.values)
    torch.manual_seed(4262 + 1000 * seed)
    kwargs = X.GRAPH_MODELS[model_name]
    model = graph.GraphNet(**kwargs)
    model.obs = (bundle.y_own, bundle.y_other, bundle.file)
    if kwargs.get("nbr") == "kernel" or kwargs.get("scalar"):
        # As in xsrc_nets.fit_one: the kernel's decay scale from cell line 1's training labels.
        ref = fit_rows & (bundle.file == 0)
        lam = graph.comod_decay_scale(bundle.position[ref], bundle.graph_id[ref], bundle.y_own[ref])
        model.kernel_nt = torch.tensor(lam, dtype=torch.float32)
        log(f"    kernel decay scale {lam:.1f} nt (fitted to cell line 1 training labels)")
    log(f"[{model_name} seed {seed}] {int(fit_rows.sum()):,} training sites, {int(val_rows.sum()):,} for early stopping")
    started = time.time()
    history = graph.train(model, graph.graphs_of(np.flatnonzero(fit_rows), bundle.graph_id, bundle.position),
                          graph.graphs_of(np.flatnonzero(val_rows), bundle.graph_id, bundle.position),
                          values, bundle.reads.offsets, bundle.kmer_onehot, bundle.position,
                          bundle.y_both, bundle.y_own, args.minutes, args.epochs, 4262 + 1000 * seed, log)
    best = max(history, key=lambda h: h["val_ap"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model_name}_s{seed}{X.suffix(args.genes)}.pt"
    torch.save({"model": model_name, "seed": seed, "kwargs": kwargs, "state_dict": model.state_dict(),
                "read_mean": std.mean, "read_scale": std.scale, "history": history,
                "best_epoch": best["epoch"], "best_val_ap": best["val_ap"],
                "stopped": "patience" if len(history) - best["epoch"] >= 25 else "cap"}, path)
    log(f"[{model_name} seed {seed}] done ({(time.time() - started) / 60:.1f} min): best epoch "
        f"{best['epoch']} of {len(history)}, validation AP {best['val_ap']:.4f} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted({m for m, _ in ENSEMBLE}))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="fit the whole ensemble, one process per network")
    ap.add_argument("--minutes", type=float, default=720.0, help="safety cap per network")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--genes", type=float, default=1.0, help="fraction of genes (smoke only)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    if args.all:
        logs = common.ROOT / "analysis" / "representation" / "logs"
        # Networks already fitted (a file exists) are kept, so adding a model refits only it.
        todo = [(m, s) for m, s in ENSEMBLE if not (OUT / f"{m}_s{s}{X.suffix(args.genes)}.pt").exists()]
        log(f"fitting {todo}")
        jobs = [(m, s, subprocess.Popen(
            [sys.executable, "-u", __file__, "--model", m, "--seed", str(s), "--minutes", str(args.minutes),
             "--epochs", str(args.epochs), "--threads", str(args.threads), "--genes", str(args.genes)],
            stdout=open(logs / f"final_fit_{m}_s{s}.log", "w"), stderr=subprocess.STDOUT)) for m, s in todo]
        failed = [(m, s) for m, s, job in jobs if job.wait() != 0]
        log("all fitted" if not failed else f"FAILED: {failed} - see logs/final_fit_*.log")
        summary = {f"{m}_s{s}": {k: v for k, v in torch.load(OUT / f"{m}_s{s}{X.suffix(args.genes)}.pt",
                                                             weights_only=False).items()
                                 if k in ("best_epoch", "best_val_ap", "stopped")}
                   for m, s in ENSEMBLE if (OUT / f"{m}_s{s}{X.suffix(args.genes)}.pt").exists()}
        log(json.dumps(summary, indent=1))
        return
    if not args.model:
        raise SystemExit("Pass --model and --seed, or --all.")
    fit(args.model, args.seed, args, log)


if __name__ == "__main__":
    main()
