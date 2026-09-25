"""Lay out our own cross-validation folds in the form m6Anet's trainer expects.

## Why retrain m6Anet at all

The shipped `HCT116_RNA002` model is almost certainly trained on HCT116, and our
training set is SG-NEx HCT116 `replicate3_run1`. Scoring it on our data is
therefore a test it may have seen the answers to. That comparison is still worth
running - it is biased *in m6Anet's favour*, so if it loses anyway that is
conclusive - but it cannot support "our architecture is better".

Retraining m6Anet on **our** folds can. Same 121,838 sites, same gene-grouped
seed-4262 split, same five folds, out-of-fold predictions throughout. That turns
"their pretrained model against our trained model" into an architecture
comparison, which is what the handout is asking for.

## What m6Anet needs, and how this differs from the inference layout

Its training loader reads `data.info.labelled` rather than `data.info`, filters
on a `set_type` column, and takes the label from `modification_status`
(`m6anet/utils/data_utils.py`). So each fold needs its own index over the *same*
`data.json`:

    transcript_id, transcript_position, n_reads, start, end,
    modification_status, set_type

`set_type` is one of `Train` / `Val` / `Test`.

`root_dir` must *contain* `data.json`, so each fold directory gets a **symlink**
to the single 631 MB copy rather than its own. Five real copies would be 3.2 GB
for no reason. Symlinks are free on Linux; on Windows they need either developer
mode or an admin shell, which is one more reason this is a Ronin job.

## The validation split, and why it is carved out of training rather than taken from a fold

The obvious layout - test on fold k, validate on fold k+1, train on the other
three - would train m6Anet on 3/5 of the data while our models train on 4/5.
That is a handicap, and a benchmark we win because we gave the opponent less
data is not a benchmark.

So validation is carved out of the four *training* folds, by gene, at
`VAL_FRACTION`. m6Anet then trains on ~90% of the same four folds our models
train on, and its test fold is identical to ours. Gene grouping is preserved at
every level, so nothing leaks between train, val and test.

Usage:

    python analysis/m6anet/prepare_cv.py --out data/m6anet/cv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from m6a.data import assign_folds, load_labels, resolve_data_dir  # noqa: E402

VAL_FRACTION = 0.10
VAL_SEED = 4262  # a different draw from the split, but deterministic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/m6anet/input",
                        help="directory holding data.json and data.info from make_m6anet_index.py")
    parser.add_argument("--labels", default=None,
                        help="default: $M6A_DATA_DIR/data.info.labelled")
    parser.add_argument("--out", required=True, help="directory to write the per-fold layout into")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4262, help="DO NOT CHANGE - see AGENTS.md section 3")
    args = parser.parse_args()
    if args.labels is None:
        args.labels = resolve_data_dir() / "data.info.labelled"

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out).resolve()
    data_json = input_dir / "data.json"
    if not data_json.exists():
        raise SystemExit(
            f"{data_json} not found. Run first:\n"
            f"  python analysis/m6anet/make_m6anet_index.py --out {input_dir}"
        )

    index = pd.read_csv(input_dir / "data.info")
    labels = load_labels(args.labels)
    merged = index.merge(labels, on=["transcript_id", "transcript_position"], how="inner")
    if len(merged) != len(index):
        print(f"  note: {len(index) - len(merged):,} indexed sites had no label and are dropped")

    # The identical assignment our own runs use. Same function, same seed.
    merged["fold"] = assign_folds(merged, args.seed, args.n_folds, "gene_id").to_numpy()
    merged = merged.rename(columns={"label": "modification_status"})
    print(f"{len(merged):,} sites, {merged.modification_status.sum():,} positive "
          f"({merged.modification_status.mean():.2%}), {merged.gene_id.nunique():,} genes")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(VAL_SEED)
    columns = ["transcript_id", "transcript_position", "n_reads", "start", "end",
               "modification_status", "set_type"]

    for fold in range(args.n_folds):
        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)

        link = fold_dir / "data.json"
        if not link.exists():
            try:
                os.symlink(data_json, link)
            except (OSError, NotImplementedError) as exc:
                raise SystemExit(
                    f"Could not symlink data.json into {fold_dir}: {exc}\n"
                    "On Linux this just works. On Windows it needs developer mode "
                    "or an admin shell - which is one reason this is a Ronin job."
                ) from exc

        is_test = merged["fold"] == fold
        # Carve validation out of the TRAINING genes, so the test fold is
        # untouched and m6Anet still trains on ~90% of what our models train on.
        train_genes = merged.loc[~is_test, "gene_id"].unique()
        n_val = max(1, int(round(len(train_genes) * VAL_FRACTION)))
        val_genes = set(rng.choice(train_genes, size=n_val, replace=False).tolist())

        set_type = np.where(is_test, "Test",
                            np.where(merged["gene_id"].isin(val_genes), "Val", "Train"))
        table = merged.assign(set_type=set_type)[columns]
        table.to_csv(fold_dir / "data.info.labelled", index=False)

        # The inference-mode index for this fold: its test sites only, so the
        # trained model scores exactly the rows our models score out of fold.
        test_rows = table[table.set_type == "Test"].drop(
            columns=["modification_status", "set_type"])
        test_dir = out_dir / f"fold{fold}_test"
        test_dir.mkdir(exist_ok=True)
        test_link = test_dir / "data.json"
        if not test_link.exists():
            os.symlink(data_json, test_link)
        test_rows.to_csv(test_dir / "data.info", index=False)

        counts = table.set_type.value_counts()
        positives = table.groupby("set_type").modification_status.sum()
        print(f"  fold {fold}: "
              + "  ".join(f"{k} {counts.get(k, 0):,} ({positives.get(k, 0):,}+)"
                          for k in ("Train", "Val", "Test")))

        config = fold_dir / "train_config.toml"
        config.write_text(
            "# Generated by analysis/m6anet/prepare_cv.py - do not hand-edit.\n"
            "[dataset]\n"
            f'root_dir = "{fold_dir.as_posix()}"\n'
            "min_reads = 20\n"
            "num_neighboring_features = 1\n\n"
            "[dataloader.train]\n"
            "batch_size = 256\n"
            'sampler = "ImbalanceOverSampler"\n\n'
            "[dataloader.val]\n"
            "batch_size = 256\n\n"
            "[dataloader.test]\n"
            "batch_size = 256\n\n"
            "[loss_function]\n"
            'loss_function_type = "binary_cross_entropy_loss"\n',
            encoding="utf-8",
        )

    print(f"\nWrote {args.n_folds} fold(s) to {out_dir}")
    print("data.json is symlinked, not copied - the whole layout is a few MB.")


if __name__ == "__main__":
    main()
