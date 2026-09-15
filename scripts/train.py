#!/usr/bin/env python
"""Train a model. One config YAML in, one model directory out.

    python scripts/train.py --config configs/lightgbm.yaml
    python scripts/train.py --config configs/lightgbm.yaml --smoke

--smoke runs the whole pipeline on 5,000 sites in about a second and skips
W&B. Run it before handing anyone a command: it catches the errors that would
otherwise surface only after a VM spin-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m6a import registry
from m6a.config import Config
from m6a.data import (
    align_to_features,
    assign_folds,
    iter_sites,
    load_labels,
    resolve_data_dir,
)
from m6a.env import load_env
from m6a.evaluation import metrics

SMOKE_SITES = 5000


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="Path to a config YAML")
    ap.add_argument("--data-dir", default=None, help="Override the data directory")
    ap.add_argument("--json", default=None, help="Override the signal JSON path")
    ap.add_argument("--labels", default=None, help="Override the labels CSV path")
    ap.add_argument("--out", default=None, help="Model output dir (default models/<name>)")
    ap.add_argument("--smoke", action="store_true", help=f"Use {SMOKE_SITES} sites, skip W&B")
    ap.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    return ap.parse_args()


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


def start_wandb(config: Config, enabled: bool):
    """Returns a W&B run, or None if tracking is off or unavailable.

    Never fatal: a missing key or package degrades to a local run rather than
    losing the experiment.
    """
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("  wandb not installed (pip install -e '.[train]') - continuing without it")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("  WANDB_API_KEY not set in .env - continuing without W&B")
        return None

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dsa4262-project"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=config.name,
        config=config.as_dict(),
        notes=config.notes,
        tags=[config.features, config.model],
    )


def main() -> int:
    args = parse_args()
    load_env()

    config = Config.load(args.config)
    json_path, labels_path = resolve_inputs(args)
    limit = SMOKE_SITES if args.smoke else None

    print(f"[{config.name}] features={config.features} model={config.model}")
    if args.smoke:
        print(f"  SMOKE: first {SMOKE_SITES:,} sites only, no W&B")

    started = time.time()
    extractor = registry.get("features", config.features)()
    features = extractor.transform(iter_sites(json_path, limit=limit))
    print(
        f"  features: {features.shape[0]:,} sites x {features.shape[1]} columns "
        f"({time.time() - started:.1f}s)"
    )

    labels = load_labels(labels_path)
    joined = align_to_features(features, labels)

    feature_columns = list(features.columns)
    X = joined[feature_columns]
    y = joined["label"].to_numpy()
    folds = assign_folds(
        joined.reset_index(),
        seed=config.split.seed,
        n_folds=config.split.n_folds,
        group_by=config.split.group_by,
    ).to_numpy()

    print(
        f"  labels:   {len(y):,} sites, {int(y.sum()):,} positive "
        f"({100 * y.mean():.2f}%), {len(np.unique(folds))} folds "
        f"grouped by {config.split.group_by}"
    )

    model_class = registry.get("models", config.model)
    run = start_wandb(config, enabled=not (args.smoke or args.no_wandb))

    # Out-of-fold predictions: every site is scored by a model that never saw
    # any transcript of its gene. This is the only number worth comparing.
    oof = np.zeros(len(y), dtype=float)
    for fold in sorted(np.unique(folds)):
        holdout = folds == fold
        model = model_class(**config.model_params)
        model.fit(X[~holdout], y[~holdout])
        oof[holdout] = model.predict_proba(X[holdout])
        print(f"  fold {fold}: trained on {int((~holdout).sum()):,}, scored {int(holdout.sum()):,}")

    scores = metrics(y, oof)
    print(
        f"\n  ROC AUC {scores['roc_auc']:.4f}   PR AUC {scores['pr_auc']:.4f}"
        f"   ({scores['pr_auc_lift']:.1f}x random)\n"
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
        "n_sites": int(len(y)),
        "smoke": bool(args.smoke),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved -> {out_dir}/   ({time.time() - started:.1f}s total)")

    if run is not None:
        import wandb

        run.log({f"oof/{key}": value for key, value in scores.items()})
        if hasattr(final, "feature_importance"):
            top = sorted(final.feature_importance().items(), key=lambda kv: -kv[1])[:30]
            run.log(
                {
                    "feature_importance": wandb.Table(
                        columns=["feature", "gain"], data=[list(row) for row in top]
                    )
                }
            )
        artifact = wandb.Artifact(config.name, type="model", metadata=meta)
        artifact.add_dir(str(out_dir))
        run.log_artifact(artifact)
        run.finish()
        print("  logged to W&B")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
