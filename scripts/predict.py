#!/usr/bin/env python
"""Predict m6A probabilities for every site in a data.json.

    python scripts/predict.py --model models/final \
                              --input data/sample/sample.json.gz \
                              --output predictions.csv

This is the script other students run on their own machine. It must therefore
work from a clean `git clone` plus `pip install -e .` with:

  * no .env and no credentials
  * no network access
  * no wandb, no boto3, no torch

If you are about to add an import here, check it is in the base dependency list
in pyproject.toml. If it is not, the code belongs somewhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m6a import registry
from m6a.data import iter_sites
from m6a.evaluation import SUBMISSION_COLUMNS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="models/final", help="Model directory")
    ap.add_argument("--input", required=True, help="data.json or data.json.gz")
    ap.add_argument("--output", default="predictions.csv", help="Where to write the CSV")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    model_dir = Path(args.model)
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"No model at {model_dir} (expected {meta_path}).\n"
            "Train one with: python scripts/train.py --config configs/lightgbm.yaml\n"
            "Or check the repo was cloned with models/final/ intact."
        )

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    meta = json.loads(meta_path.read_text())
    started = time.time()

    # The model records which feature extractor produced its training columns,
    # so prediction cannot silently use a different one.
    extractor = registry.get("features", meta["features"])()
    features = extractor.transform(iter_sites(input_path))

    missing = [column for column in meta["columns"] if column not in features.columns]
    if missing:
        raise SystemExit(
            f"Feature extractor {meta['features']} produced a table missing "
            f"{len(missing)} column(s) the model expects, e.g. {missing[:5]}. "
            "The model and the code have drifted apart."
        )

    model = registry.get("models", meta["model"]).load(model_dir)
    scores = model.predict_proba(features[meta["columns"]])

    out = features.index.to_frame(index=False)
    out["score"] = scores
    out = out[SUBMISSION_COLUMNS]

    output_path = Path(args.output)
    if str(output_path.parent) not in ("", "."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(
        f"{len(out):,} sites scored -> {output_path}  "
        f"({time.time() - started:.1f}s, model={meta['name']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
