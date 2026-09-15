#!/usr/bin/env python
"""Build the small committed test dataset under data/sample/.

    python scripts/make_sample.py            (or: make sample)

The handout requires a small test dataset so other people can run the
prediction script immediately after cloning. This produces it, deterministically
from a fixed seed, so regenerating it does not churn the repo.

Writes:
  data/sample/sample.json.gz        input for predict.py
  data/sample/sample.info.labelled  the matching labels (ours, for regression tests)

The labels are not needed by an evaluator, but they let tests/ check that the
pipeline still scores this file the way it used to.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a.config import DEFAULT_SEED
from m6a.data import load_labels, resolve_data_dir
from m6a.env import load_env

DEFAULT_N = 1000


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-n", type=int, default=DEFAULT_N, help=f"Sites to sample (default {DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="data/sample")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    load_env(ROOT / ".env")

    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir()
    json_path = data_dir / "dataset0.json.gz"
    labels_path = data_dir / "data.info.labelled"

    for path in (json_path, labels_path):
        if not path.exists():
            raise SystemExit(
                f"Cannot find {path}.\n"
                "Run `python scripts/download_data.py`, or pass --data-dir."
            )

    # Pass 1: count lines. Cheaper than holding 625 MB of decompressed JSON.
    with gzip.open(json_path, "rt") as handle:
        total = sum(1 for _ in handle)

    if args.n >= total:
        raise SystemExit(f"--n {args.n} is not smaller than the {total:,} available sites.")

    rng = np.random.default_rng(args.seed)
    chosen = set(rng.choice(total, size=args.n, replace=False).tolist())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_json = out_dir / "sample.json.gz"

    keys: list[tuple[str, int]] = []
    # mtime=0 so regenerating an identical sample produces an identical file
    # and does not show up as a spurious diff.
    with gzip.open(json_path, "rt") as source, gzip.GzipFile(
        sample_json, "wb", compresslevel=9, mtime=0
    ) as raw_out:
        for index, line in enumerate(source):
            if index not in chosen:
                continue
            raw_out.write(line.encode("utf-8"))
            record = json.loads(line)
            transcript_id = next(iter(record))
            position = next(iter(record[transcript_id]))
            keys.append((transcript_id, int(position)))

    labels = load_labels(labels_path)
    wanted = labels.set_index(["transcript_id", "transcript_position"])
    subset = wanted.loc[keys].reset_index()
    subset = subset[["gene_id", "transcript_id", "transcript_position", "label"]]
    subset.to_csv(out_dir / "sample.info.labelled", index=False)

    size_kb = sample_json.stat().st_size / 1024
    positives = int(subset["label"].sum())
    print(
        f"{len(keys):,} sites -> {sample_json} ({size_kb:.0f} KB), "
        f"{positives} positive ({100 * positives / len(keys):.1f}%)"
    )
    print(f"labels -> {out_dir / 'sample.info.labelled'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
