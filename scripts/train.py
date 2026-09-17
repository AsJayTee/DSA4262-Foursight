#!/usr/bin/env python
"""Train a model and evaluate it, in one run. One config YAML in, one model out.

    python scripts/train.py --config configs/lightgbm.yaml
    python scripts/train.py --config configs/lightgbm.yaml --quick
    python scripts/train.py --config configs/lightgbm.yaml --smoke

--smoke runs the whole pipeline on 5,000 sites in about a second and skips W&B.
Run it before handing anyone a command: it catches the errors that would
otherwise surface only after a VM spin-up.

**Training evaluates.** The full standard-profile evaluation - per fold, per
stratum, calibration, the depth sweep and every figure - runs as part of this
command and goes to the same W&B run. On a machine that gets terminated with
nothing pulled off it, a training run nobody evaluated before the instance died
has lost everything, and re-running it costs far more than the sweep would have.
Gather everything while the machine exists. `--quick` skips the sweep when you
are iterating; a result you intend to record should not use it.
See docs/decisions/0007.

Each run writes into the model directory:

  model.txt / columns.json   the model itself, refitted on all the data
  meta.json                  what produced it, its per-fold metrics, and the
                             W&B run id so an evaluation can attach to it

There is deliberately no per-site score table. What survives a run is the metric
*vector* - five PR AUCs, not 121,838 scores - which is everything a paired test
needs at a fraction of the size. See docs/decisions/0009.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m6a import crossval, registry, report as reporting, tracking
from m6a.config import Config
from m6a.data import SUBSAMPLE_SEED, resolve_data_dir
from m6a.env import load_env

SMOKE_SITES = 5000
DEFAULT_DEPTHS = [1, 3, 5, 10, 20, None]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="Path to a config YAML")
    ap.add_argument("--data-dir", default=None, help="Override the data directory")
    ap.add_argument("--json", default=None, help="Override the signal JSON path")
    ap.add_argument("--labels", default=None, help="Override the labels CSV path")
    ap.add_argument("--out", default=None, help="Model output dir (default models/<name>)")
    ap.add_argument(
        "--profile",
        default=reporting.DEFAULT_PROFILE,
        choices=sorted(reporting.PROFILES),
        help="How much to evaluate (default: %(default)s). quick = folds, pooled "
             "and calibration; standard = + strata, depth sweep and figures; "
             "full = + ablations.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Shorthand for --profile quick: skip the depth sweep and the "
             "figures. For iterating, not for a result you intend to record.",
    )
    ap.add_argument("--smoke", action="store_true", help=f"Use {SMOKE_SITES} sites, skip W&B")
    ap.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    ap.add_argument(
        "--subsample-seed",
        type=int,
        default=SUBSAMPLE_SEED,
        help=f"Seed for the depth sweep's read subsampling (default {SUBSAMPLE_SEED}). "
             "NOT the split seed; changing it only redraws which reads are kept.",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the extracted-feature cache (see src/m6a/feature_cache.py)",
    )
    args = ap.parse_args()
    if args.quick:
        args.profile = "quick"
    return args


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir()
    json_path = Path(args.json) if args.json else data_dir / "dataset0.json.gz"
    labels_path = Path(args.labels) if args.labels else data_dir / "data.info.labelled"

    for path, what in ((json_path, "signal JSON"), (labels_path, "labels file")):
        if not path.exists():
            raise SystemExit(
                f"Cannot find the {what} at {path}.\n"
                "Either run `python scripts/download_data.py`, or point at a local "
                "copy with --data-dir (or set M6A_DATA_DIR in .env)."
            )
    return json_path, labels_path


def main() -> int:
    args = parse_args()
    load_env()

    config = Config.load(args.config)
    json_path, labels_path = resolve_inputs(args)
    limit = SMOKE_SITES if args.smoke else None

    report = reporting.new(args.profile, subsample_seed=args.subsample_seed)
    print(f"[{config.name}] features={config.features} model={config.model}")
    print(f"  profile {args.profile}")
    if args.smoke:
        print(f"  SMOKE: first {SMOKE_SITES:,} sites only, no W&B")

    # Started before the compute, not after it: a broken W&B key should surface
    # now rather than at minute 40 of a run on an instance about to be
    # terminated. The run id also has to exist before meta.json is written.
    run = tracking.start(
        config.name,
        enabled=not (args.smoke or args.no_wandb),
        config={**config.as_dict(), "profile": args.profile,
                "subsample_seed": args.subsample_seed},
        notes=config.notes,
        tags=[config.features, config.model],
        job_type="train",
    )

    started = time.time()
    depths = list(DEFAULT_DEPTHS) if report.profile.depth_sweep else [None]
    datasets = crossval.build_datasets(
        json_path,
        labels_path,
        config.features,
        depths,
        seed=config.split.seed,
        n_folds=config.split.n_folds,
        group_by=config.split.group_by,
        subsample_seed=args.subsample_seed,
        limit=limit,
        use_cache=not args.no_cache,
    )
    dataset = datasets[None]
    feature_columns = dataset.columns
    X, y = dataset.X, dataset.y
    print(
        f"  features: {len(dataset):,} sites x {len(feature_columns)} columns "
        f"({time.time() - started:.1f}s)"
    )
    print(
        f"  labels:   {len(y):,} sites, {int(y.sum()):,} positive "
        f"({100 * y.mean():.2f}%), {config.split.n_folds} folds "
        f"grouped by {config.split.group_by}"
    )

    model_class = registry.get("models", config.model)

    # Out-of-fold predictions: every site is scored by a model that never saw
    # any transcript of its gene. This is the only number worth comparing.
    # Per-fold metrics are kept rather than collapsed - one pooled figure cannot
    # tell a real improvement from fold noise, and the folds differ enough to
    # matter (positive rates here run from 4.10% to 5.15%).
    result = crossval.cross_validate(
        dataset, model_class, config.model_params, label=config.name
    )
    scores = result.pooled

    report.data["source"] = {
        "kind": "config",
        "path": str(args.config),
        "name": config.name,
        "features": config.features,
        "model": config.model,
        "model_params": config.model_params,
        "split": {"seed": config.split.seed, "n_folds": config.split.n_folds,
                  "group_by": config.split.group_by},
        "data": str(json_path),
        "limit": limit,
    }
    reporting.per_fold(report, result.oof, name=config.name)
    if report.profile.strata:
        reporting.strata(report, result.oof, ["depth", "motif"], min_positive=10)
    reporting.calibration(report, result.oof)
    reporting.curves(
        report,
        {config.name: (result.oof["label"].to_numpy(), result.oof["score"].to_numpy())},
    )
    if report.profile.depth_sweep:
        reporting.depth_sweep(
            report, datasets, result.models, result.columns,
            result.oof, args.subsample_seed,
        )

    # Refit on everything for the model we actually ship.
    final = model_class(**config.model_params)
    final.fit(X, y)

    out_dir = Path(args.out) if args.out else Path("models") / config.name
    final.save(out_dir)

    meta = {
        "name": config.name,
        "features": config.features,
        "model": config.model,
        "model_params": config.model_params,
        "columns": feature_columns,
        "split": {
            "seed": config.split.seed,
            "n_folds": config.split.n_folds,
            "group_by": config.split.group_by,
        },
        "metrics_oof": scores,
        # Added by the evaluation harness. Everything above this line is what
        # predict.py reads and must not change shape; everything below is
        # additive, so a model trained before the harness existed still loads.
        "metrics_per_fold": result.per_fold,
        "profile": args.profile,
        "wandb_run_id": run.id or None,
        "n_sites": int(len(y)),
        "smoke": bool(args.smoke),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if report.profile.plots and hasattr(final, "feature_importance"):
        from m6a import figures

        report.figure("fig/feature_importance", figures.feature_importance(
            final.feature_importance()
        ))

    report.data["elapsed_seconds"] = round(time.time() - started, 1)
    out_path = reporting.write(report, reporting.report_dir(), config.name)
    reporting.publish(report, run, out_path, name=config.name)

    if run.enabled:
        run.log_artifact(out_dir, config.name, kind="model", metadata=meta)
    run.finish()

    print(f"\n  saved -> {out_dir}/   ({time.time() - started:.1f}s total)")
    print(f"  report -> {out_path}")
    if args.profile == "quick":
        print(
            "  profile was 'quick', so this run has no depth numbers and no figures.\n"
            "  It is not comparable with a standard run on the dimension that matters\n"
            "  most. Re-run without --quick before recording it anywhere."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
