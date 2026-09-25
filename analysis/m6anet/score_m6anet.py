"""Score m6Anet's predictions on the same footing as ours, and pair the two.

Takes m6Anet site probabilities - either the pretrained model's single
`data.site_proba.csv`, or the five per-fold files from a retrained
cross-validation - joins them to our labels, and reports the same metrics our
own harness reports. Uses `m6a.evaluation` and `m6a.compare` rather than
reimplementing anything, so the numbers are computed by the same code that
produced every number in GAPS.md.

    # the pretrained model, one file over all sites
    python analysis/m6anet/score_m6anet.py --pretrained data/m6anet/pretrained_out/data.site_proba.csv

    # retrained on our folds, five files
    python analysis/m6anet/score_m6anet.py --cv-dir data/m6anet/cv

    # and pair it fold-by-fold against one of our runs
    python analysis/m6anet/score_m6anet.py --cv-dir data/m6anet/cv \\
           --compare analysis/evaluation/reports/final_candidate.json

**On comparing against the pretrained model.** It is very likely trained on
HCT116, which is our cell line, so it may have seen these sites' answers. That
biases the comparison *in its favour*, which makes the result one-directional:
if m6Anet loses anyway the conclusion is safe, and if it wins the number cannot
separate a better architecture from a memorised training set. This script prints
that caveat with the pretrained numbers rather than leaving it to a reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from m6a.data import assign_folds, load_labels, resolve_data_dir  # noqa: E402
from m6a.evaluation import calibration_summary, metrics  # noqa: E402

KEY = ["transcript_id", "transcript_position"]


def load_predictions(pretrained: Path | None, cv_dir: Path | None) -> tuple[pd.DataFrame, str]:
    if pretrained is not None:
        frame = pd.read_csv(pretrained)[KEY + ["probability_modified"]]
        return frame, "pretrained HCT116_RNA002"

    parts = []
    for fold_out in sorted(cv_dir.glob("fold*_test/out/data.site_proba.csv")):
        part = pd.read_csv(fold_out)[KEY + ["probability_modified"]]
        part["fold"] = int(fold_out.parts[-3].replace("fold", "").replace("_test", ""))
        parts.append(part)
    if not parts:
        raise SystemExit(
            f"No per-fold predictions under {cv_dir}/fold*_test/out/. "
            "Run the training and inference loop first."
        )
    return pd.concat(parts, ignore_index=True), f"retrained on our folds ({len(parts)} folds)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", type=Path, help="data.site_proba.csv from the shipped model")
    group.add_argument("--cv-dir", type=Path, help="directory holding fold*_test/out/")
    parser.add_argument("--labels", default=None,
                        help="default: $M6A_DATA_DIR/data.info.labelled")
    parser.add_argument("--compare", type=Path, default=None,
                        help="one of our report JSONs, to pair against fold by fold")
    parser.add_argument("--seed", type=int, default=4262)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--wandb-name", default=None,
                        help="log the result to W&B under this run name. Ronin instances are "
                             "terminated with nothing pulled off them (docs/decisions/0007), "
                             "and m6Anet writes CSVs that know nothing about W&B - so without "
                             "this the benchmark dies with the machine.")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.labels is None:
        args.labels = resolve_data_dir() / "data.info.labelled"

    predictions, label = load_predictions(args.pretrained, args.cv_dir)
    labels = load_labels(args.labels)
    joined = labels.merge(predictions, on=KEY, how="inner")
    if joined.empty:
        raise SystemExit("No sites matched between the predictions and the labels.")

    missing = len(labels) - len(joined)
    y = joined["label"].to_numpy()
    scores = joined["probability_modified"].to_numpy()

    print(f"m6Anet, {label}")
    print(f"  {len(joined):,} sites scored, {int(y.sum()):,} positive ({y.mean():.2%})"
          + (f"   [{missing:,} of our sites unscored]" if missing else ""))
    print()

    overall = metrics(y, scores)
    print("  pooled      " + "   ".join(f"{k} {v:.4f}" for k, v in overall.items()
                                        if isinstance(v, float)))
    calibration = calibration_summary(y, scores)
    print(f"  calibration count_ratio {calibration['count_ratio']:.3f}x   "
          f"ece {calibration['ece']:.4f}   mean predicted {calibration['mean_predicted']:.4f}")

    if "fold" not in joined.columns:
        joined["fold"] = assign_folds(joined, args.seed, args.n_folds, "gene_id").to_numpy()

    print()
    print("  per fold (our split, so these pair against any of our runs):")
    per_fold = []
    for fold in sorted(joined["fold"].unique()):
        here = (joined["fold"] == fold).to_numpy()
        # PR AUC is undefined without a positive, and `metrics` divides by the
        # positive rate to get lift. Cannot happen on the full set - the
        # thinnest fold has ~1,000 positives - but it does on a small sample,
        # and a scoring script that dies on the last fold is a bad command to
        # hand someone.
        if y[here].sum() == 0:
            print(f"    fold {fold}: n {int(here.sum()):>6,}  no positives, not scored")
            continue
        fold_metrics = metrics(y[here], scores[here])
        per_fold.append(fold_metrics["pr_auc"])
        print(f"    fold {fold}: n {int(here.sum()):>6,}  pr_auc {fold_metrics['pr_auc']:.4f}"
              f"  roc_auc {fold_metrics['roc_auc']:.4f}")
    if len(per_fold) < 2:
        raise SystemExit("\n  Fewer than two scorable folds - nothing to summarise. "
                         "This is a sample, not a result.")
    print(f"    mean {np.mean(per_fold):.4f}  sd {np.std(per_fold, ddof=1):.4f}")

    if args.compare is not None:
        from m6a.compare import paired_comparison

        ours = json.load(open(args.compare))
        theirs = [{"repetition": 0, "fold": i, "pr_auc": v} for i, v in enumerate(per_fold)]
        mine = [{"repetition": 0, "fold": i, "pr_auc": f["pr_auc"]}
                for i, f in enumerate(ours["per_fold"])]
        if len(mine) != len(theirs):
            raise SystemExit(f"fold count mismatch: {len(mine)} vs {len(theirs)}")
        result = paired_comparison(theirs, mine, "m6anet", ours.get("source", {}).get("name", "ours"))
        print()
        print(f"  Paired against {args.compare.name}, five folds:")
        for field in ("mean_difference", "wins", "p_value", "p_value_corrected"):
            if field in result:
                value = result[field]
                print(f"    {field:<20} {value:.6f}" if isinstance(value, float)
                      else f"    {field:<20} {value}")
        print("    NOTE five folds is five correlated observations. Our own runs use")
        print("    fifty; this is the strongest pairing available against an external")
        print("    tool and it is underpowered by comparison (docs/decisions/0013).")

    if args.wandb_name and not args.no_wandb:
        from m6a import tracking

        tracker = tracking.start(
            args.wandb_name,
            job_type="benchmark",
            tags=["m6anet", "benchmark"],
            notes=(
                "m6Anet, " + label + ". Scored on our labels with m6a.evaluation, "
                "so these numbers are computed by the same code as every other run "
                "in this project. NOT one of our models."
            ),
            config={
                "external_tool": "m6anet==2.1.0",
                "m6anet_variant": label,
                "split_seed": args.seed,
                "split_n_folds": args.n_folds,
                "split_group_by": "gene_id",
                "n_sites_scored": int(len(joined)),
            },
        )
        flat = {f"oof/{k}": v for k, v in overall.items() if isinstance(v, float)}
        flat.update({f"calib/{k}": v for k, v in calibration.items() if isinstance(v, float)})
        flat.update({f"fold/{i}/pr_auc": v for i, v in enumerate(per_fold)})
        flat["fold/pr_auc_mean"] = float(np.mean(per_fold))
        flat["fold/pr_auc_sd"] = float(np.std(per_fold, ddof=1))
        tracker.log(flat)
        tracker.summary(flat)
        tracker.finish()
        print()
        print(f"  logged to W&B as '{args.wandb_name}' - this is what survives the instance.")

    if args.pretrained is not None:
        print()
        print("  CAVEAT, and it is not a small one. The shipped HCT116_RNA002 weights are")
        print("  very likely trained on HCT116, which is our cell line - our training set")
        print("  is SG-NEx HCT116 replicate3_run1. m6Anet may have seen these sites'")
        print("  labels. The bias runs in ITS favour, so: if it loses here the conclusion")
        print("  is safe, and if it wins the number cannot separate a better architecture")
        print("  from a memorised training set. Use --cv-dir for the fair comparison.")


if __name__ == "__main__":
    main()
