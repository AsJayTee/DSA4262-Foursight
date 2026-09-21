#!/usr/bin/env python
"""Evaluate a model properly: per fold, per stratum, per depth, and calibrated.

    # the standard profile: folds, strata, depth sweep, calibration, figures
    python scripts/evaluate.py --config configs/quantiles.yaml

    # is quantiles really better than pooled, or is that fold noise?
    python scripts/evaluate.py --config configs/quantiles.yaml --compare-features pooled_v1

    # too close to call on five folds? get fifty paired observations
    python scripts/evaluate.py --config configs/quantiles.yaml
                               --compare-features pooled_v1 --repeats 10

    # an error bar on the headline number, over the sites rather than the split
    python scripts/evaluate.py --config configs/quantiles.yaml --bootstrap 2000

    # what happens at SG-NEx's read depths?
    python scripts/evaluate.py --config configs/quantiles.yaml --depth-sweep

    # how much of the score is the DRACH motif alone, with no signal at all?
    python scripts/evaluate.py --config configs/quantiles.yaml --ablate

    # score a shipped model on some other labelled dataset
    python scripts/evaluate.py --model models/final --json data/other.json.gz
                               --labels data/other.info.labelled

Two input modes, one report:

  --config  run the gene-grouped cross-validation here. The only mode that can
            do a depth sweep, because that needs live fold models.
  --model   score an already-fitted model on a labelled dataset it did not train
            on. No folds, so no per-fold metrics and no paired test.

A comparison evaluates **both** arms in full - strata, calibration, depth sweep,
figures - and then pairs them overall and again inside each read-depth band and
motif. Three numbers come out of every comparison: unpaired (wrong), paired
(optimistic), and the Nadeau & Bengio corrected test, which is the one to quote.

`--profile` decides how much is computed; `standard` is the default and includes
the depth sweep. Everything durable goes to W&B - on a Ronin instance the local
JSON under analysis/evaluation/reports/ dies with the machine, so treat it as a
convenience rather than the record. See docs/decisions/0007.

**Training already runs this.** `scripts/train.py` evaluates inline at the same
profile and logs to the same W&B run, so this script is for comparisons and for
scoring against a second dataset, not for routine use after a training run.

This is a thin CLI; the work is in m6a.report, m6a.crossval, m6a.evaluation,
m6a.compare and m6a.tracking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m6a import crossval, registry, report as reporting, tracking
from m6a.config import Config
from m6a.data import SUBSAMPLE_SEED, resolve_data_dir
from m6a.env import load_env
from m6a import evaluation
from m6a.evaluation import depth_bands, metrics

SMOKE_SITES = 5000
# See scripts/train.py for why this stops at 25.
DEFAULT_DEPTHS = "1,2,3,4,5,7,10,15,20,25,full"
# Mirrors m6a.compare.BOOTSTRAP_SEED rather than importing it: m6a.compare
# pulls in scipy, and this script should not do that just to print a default.
BOOTSTRAP_SEED = 4262


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_argument_group("what to evaluate (pick one)")
    source.add_argument("--config", help="Config YAML: run the cross-validation here")
    source.add_argument("--model", help="A trained model directory to score a dataset with")
    source.add_argument(
        "--features",
        metavar="NAME",
        help="Run --config with this feature set instead of the one it names, "
             "holding the model and every parameter fixed. Use it to sweep or "
             "ablate a second feature set under identical conditions, which is "
             "what makes the two sets of numbers comparable.",
    )

    compare = ap.add_argument_group("comparisons")
    compare.add_argument(
        "--compare-features",
        metavar="NAME",
        help="Re-run --config with this feature set instead, holding the model "
             "and its parameters fixed. The controlled comparison: only the "
             "feature set differs, so the paired difference is attributable.",
    )
    compare.add_argument(
        "--compare-with",
        metavar="CONFIG",
        help="Paired comparison against another config. Varies everything the "
             "two configs differ in, so read the difference with that in mind.",
    )
    compare.add_argument(
        "--compare-run",
        metavar="RUN",
        help="Paired comparison against a finished W&B run, by id or name, "
             "WITHOUT refitting it. Pulls its per-(repetition, fold) metric "
             "vector straight out of W&B, so the other arm can be a run from an "
             "instance that no longer exists. Refuses unless both runs record "
             "the same dataset fingerprint and split. Gets the corrected paired "
             "test; cannot get the bootstrap, which needs per-site scores that "
             "are deliberately not stored.",
    )
    compare.add_argument(
        "--ablate",
        action="store_true",
        help="Also score signal-only and motif-only column subsets, paired "
             "against the full feature set. Motif-only is the sequence floor. "
             "On by default at --profile full.",
    )

    sweep = ap.add_argument_group("read depth")
    sweep.add_argument(
        "--depth-sweep",
        action="store_true",
        help="Score the fold models on read-subsampled copies of their own "
             "held-out data. On by default at --profile standard and above.",
    )
    sweep.add_argument("--depths", default=DEFAULT_DEPTHS, help=f"Default: {DEFAULT_DEPTHS}")

    boot = ap.add_argument_group("bootstrap")
    boot.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Resample the sites with replacement N times (2000 is the usual "
             "number) for a confidence interval on the headline PR AUC, and on "
             "the gap when there is a comparison. Off by default. This is the "
             "only thing here that asks 'would this hold on a different sample "
             "of sites' rather than 'a different split'. Runs in-process while "
             "the out-of-fold vector is in memory; only the interval is kept.",
    )
    boot.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Seed for the resampling (default 4262). A third knob, separate "
             "from the split seed and the subsample seed; changing it is safe.",
    )

    repeat = ap.add_argument_group("repeated cross-validation")
    repeat.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Cross-validate this many times over independently seeded splits. "
             "Defaults to whatever the profile says: 10 for standard and full, "
             "1 for quick. Repetition 0 is always the canonical seed-4262 "
             "split, so nothing already recorded moves. 10 turns 5 paired "
             "observations into 50, which is what makes a comparison between "
             "two close models mean anything. Costs one set of fits per "
             "repetition; features are extracted once.",
    )
    sweep.add_argument(
        "--subsample-seed",
        type=int,
        default=SUBSAMPLE_SEED,
        help=f"Seed for read subsampling (default {SUBSAMPLE_SEED}). This is NOT "
             "the split seed and changing it is safe - it only redraws which "
             "reads are kept.",
    )

    data = ap.add_argument_group("data")
    data.add_argument("--data-dir", default=None, help="Override the data directory")
    data.add_argument("--json", default=None, help="Override the signal JSON path")
    data.add_argument("--labels", default=None, help="Override the labels CSV path")
    data.add_argument("--smoke", action="store_true", help=f"Use {SMOKE_SITES} sites")
    data.add_argument("--limit", type=int, default=None, help="Use the first N sites")
    data.add_argument("--no-cache", action="store_true", help="Ignore the feature cache")

    out = ap.add_argument_group("output")
    out.add_argument(
        "--profile",
        default=reporting.DEFAULT_PROFILE,
        choices=sorted(reporting.PROFILES),
        help="How much to compute (default: %(default)s). quick = folds, pooled "
             "and calibration; standard = + strata, depth sweep and figures; "
             "full = + ablations. A recorded result should be standard or better.",
    )
    out.add_argument(
        "--out", default=None,
        help=f"Local JSON report directory (default: {reporting.DEFAULT_REPORT_DIR}, or $M6A_REPORT_DIR)",
    )
    out.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    out.add_argument(
        "--min-positive",
        type=int,
        default=10,
        help="Strata with fewer positives than this get no metrics (default 10)",
    )
    out.add_argument("--by", default="depth,motif", help="Strata to report: depth,motif,fold")

    args = ap.parse_args()

    chosen = [name for name in ("config", "model") if getattr(args, name)]
    if len(chosen) != 1:
        ap.error(
            "Pick exactly one of --config or --model.\n"
            "  --config configs/quantiles.yaml                (runs the folds)\n"
            "  --model models/final --json ... --labels ...   (scores a dataset)"
        )
    for flag in ("compare_features", "compare_with", "compare_run", "ablate"):
        if getattr(args, flag) and not args.config:
            ap.error(f"--{flag.replace('_', '-')} needs --config.")
    if args.features and not args.config:
        ap.error("--features overrides the feature set named in --config, so it needs one.")
    if args.repeats is not None and args.repeats < 1:
        ap.error("--repeats must be at least 1.")
    if args.bootstrap < 0:
        ap.error("--bootstrap must be 0 (off) or a positive number of resamples.")
    if args.bootstrap and args.bootstrap < 200:
        ap.error(
            f"--bootstrap {args.bootstrap} is too few resamples for a 95% "
            "interval - the 2.5th percentile would rest on a handful of draws. "
            "Use at least 200, and 2000 for anything quoted."
        )
    if args.repeats is not None and args.repeats > 1 and not args.config:
        ap.error(
            "--repeats needs --config: repeated cross-validation refits the "
            "folds, which scoring a saved model cannot do."
        )
    if args.model and not (args.json or args.data_dir):
        ap.error(
            "--model needs a dataset to score: --json <data.json.gz> "
            "--labels <data.info.labelled>"
        )
    return args


def parse_depths(text: str) -> list[int | None]:
    depths: list[int | None] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in ("full", "none", "all"):
            depths.append(None)
            continue
        try:
            value = int(token)
        except ValueError:
            raise SystemExit(
                f"Cannot read {token!r} as a read depth. Use whole numbers and "
                f"'full', e.g. --depths {DEFAULT_DEPTHS}"
            ) from None
        if value < 1:
            raise SystemExit(f"Read depth must be >= 1, got {value}.")
        depths.append(value)
    if not depths:
        raise SystemExit("--depths selected nothing.")
    return depths


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


def find_training_run(name: str) -> str | None:
    """The W&B run id `train.py` recorded for this experiment, if there is one.

    Resuming it keeps one experiment to one row in the run table instead of
    splitting it across a training row and an evaluation row, which is the thing
    that makes filtering across everyone's runs work at all.
    """
    meta_path = Path("models") / name / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text()).get("wandb_run_id") or None
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def compare_arm(
    report: reporting.Report,
    args: argparse.Namespace,
    key: str,
    title: str,
    primary,
    primary_features: str,
    baseline,
    baseline_name: str,
    n_folds: int,
    y,
    sites,
    sweep: dict | None,
) -> None:
    """Report a comparison arm in full, pair it overall, then pair it per stratum.

    Three things, in the order they should be read: the arm's own numbers (so
    nobody has to re-run it standalone to find its calibration), the overall
    paired test (confirmatory), and the per-stratum tests (descriptive - where a
    difference concentrates). See docs/decisions/0008.
    """
    from m6a.compare import paired_comparison

    reporting.arm(
        report, baseline_name, baseline.canonical.oof,
        by=[b for b in args.by.split(",") if b != "fold"],
        min_positive=args.min_positive,
        sweep=sweep,
    )

    reporting.comparison(
        report,
        paired_comparison(
            baseline.observations, primary.observations,
            baseline_name, primary_features, n_folds=n_folds,
        ),
        title,
        key,
    )

    if args.bootstrap:
        reporting.bootstrap(
            report,
            baseline.canonical.oof["label"].to_numpy(),
            {baseline_name: baseline.canonical.oof["score"].to_numpy(),
             primary_features: primary.canonical.oof["score"].to_numpy()},
            args.bootstrap,
            args.bootstrap_seed if args.bootstrap_seed is not None else BOOTSTRAP_SEED,
        )

    if not report.profile.strata:
        return

    # "Is it better *where we are currently weak*" is the question the next round
    # of work asks, and low read depth is where the weakness is. Motifs are
    # included because per-motif lift spans 2.8x to 19.5x and a change can move
    # one without moving the average.
    wanted = {
        "read depth": ("depth", depth_bands(sites["n_reads"].to_numpy()),
                       list(evaluation.DEPTH_BAND_LABELS)),
        "DRACH motif": ("motif", sites["motif"].to_numpy(), None),
    }
    for label, (which, groups, order) in wanted.items():
        if which not in args.by.split(","):
            continue
        reporting.stratified_comparison(
            report,
            reporting.stratified_observations(
                baseline, y, groups, min_positive=args.min_positive
            ),
            reporting.stratified_observations(
                primary, y, groups, min_positive=args.min_positive
            ),
            baseline_name, primary_features, label, f"{key}_{which}",
            n_folds=n_folds,
            order=order,
        )


def run_from_model(args: argparse.Namespace, report: reporting.Report) -> None:
    from m6a import feature_cache
    from m6a.data import align_to_features, load_labels

    model_dir = Path(args.model)
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"No model at {model_dir} (expected {meta_path}).")
    meta = json.loads(meta_path.read_text())
    json_path, labels_path = resolve_inputs(args)

    report.log(f"[{meta['name']}] scoring {json_path.name} with {model_dir}")
    report.log(
        "  NOTE: this model was fitted on all of its own training data. If this\n"
        "  dataset overlaps that, these numbers are in-fold and flattering.\n"
        "  For an honest number on the training set, use --config instead."
    )
    report.data["source"] = {
        "kind": "model", "path": str(model_dir), "model": meta["name"],
        "data": str(json_path),
    }

    sweep = args.depth_sweep or report.profile.depth_sweep
    depths = parse_depths(args.depths) if sweep else [None]
    if None not in depths:
        depths.append(None)  # full depth is the row every other one is read against
    limit = SMOKE_SITES if args.smoke else args.limit
    extractions = feature_cache.extract(
        json_path, meta["features"], depths,
        seed=args.subsample_seed, limit=limit, use_cache=not args.no_cache,
        log=report.log,
    )
    missing = [c for c in meta["columns"] if c not in extractions[None].columns]
    if missing:
        raise SystemExit(
            f"Feature extractor {meta['features']} produced a table missing "
            f"{len(missing)} column(s) the model expects, e.g. {missing[:5]}. "
            "The model and the code have drifted apart."
        )
    model = registry.get("models", meta["model"]).load(model_dir)
    labels = load_labels(labels_path)

    scored: dict = {}
    for depth in depths:
        joined = align_to_features(extractions[depth].features, labels)
        y = joined["label"].to_numpy()
        scores = model.predict_proba(joined[meta["columns"]])
        scored[depth] = (joined, y, scores, extractions[depth])

    joined, y, scores, extraction = scored[None]
    sites = extraction.sites.loc[joined.index]
    oof = pd.DataFrame(
        {
            "transcript_id": joined.index.get_level_values(0),
            "transcript_position": joined.index.get_level_values(1),
            "gene_id": joined["gene_id"].to_numpy(),
            "motif": sites["motif"].to_numpy(),
            "n_reads": sites["n_reads"].to_numpy(),
            "fold": -1,  # no folds: one model scored everything
            "label": y,
            "score": scores,
        }
    )

    report.heading("Pooled")
    reporting.show_pooled(report.log, metrics(y, scores), prefix="held-out")
    report.data["pooled"] = metrics(y, scores)

    if report.profile.strata:
        reporting.strata(
            report, oof, [b for b in args.by.split(",") if b != "fold"], args.min_positive
        )
    reporting.calibration(report, oof)
    reporting.curves(report, {meta["name"]: (y, scores)})

    if sweep:
        rows = []
        for depth in depths:
            _, y_d, scores_d, _ = scored[depth]
            rows.append(
                {"depth": "full" if depth is None else str(depth),
                 **metrics(y_d, scores_d)}
            )
        frame = pd.DataFrame(rows).set_index("depth")[["roc_auc", "pr_auc", "pr_auc_lift"]]
        frame["retained"] = frame["pr_auc"] / frame.loc["full", "pr_auc"]
        report.heading("Depth sweep")
        report.show(frame)
        report.data["depth_sweep"] = {
            "subsample_seed": args.subsample_seed,
            "note": "one model fitted on all its training data, scored at the stated depth",
            "rows": reporting.records(frame),
        }
        if report.profile.plots:
            from m6a import figures

            report.figure(
                "fig/depth_sweep",
                figures.depth_sweep(report.data["depth_sweep"]["rows"], args.subsample_seed),
            )


def run_from_config(args: argparse.Namespace, report: reporting.Report) -> None:
    config = Config.load(args.config)
    features = args.features or config.features
    json_path, labels_path = resolve_inputs(args)
    limit = SMOKE_SITES if args.smoke else args.limit

    sweep = args.depth_sweep or report.profile.depth_sweep
    ablate = args.ablate or report.profile.ablations
    repeats = args.repeats if args.repeats is not None else report.profile.repeats
    depths = parse_depths(args.depths) if sweep else [None]
    if None not in depths:
        depths.append(None)  # the full-depth models are what the sweep scores with

    report.log(f"[{config.name}] features={features} model={config.model}")
    report.log(
        f"  profile {report.profile.name}"
        + (f", {repeats} repetitions x {config.split.n_folds} folds "
           f"= {repeats * config.split.n_folds} observations" if repeats > 1 else "")
    )
    if args.features:
        report.log(
            f"  feature set overridden: {config.features} -> {features}. Everything "
            "else is exactly as the config says, so these numbers are comparable "
            "with a plain run of it."
        )
    if limit:
        report.log(f"  first {limit:,} sites only")
    report.data["source"] = {
        "kind": "config",
        "path": str(args.config),
        "name": config.name,
        "features": features,
        "features_from_config": config.features,
        "model": config.model,
        "model_params": config.model_params,
        "split": {"seed": config.split.seed, "n_folds": config.split.n_folds,
                  "group_by": config.split.group_by},
        "data": str(json_path),
        "limit": limit,
    }

    build = dict(
        seed=config.split.seed, n_folds=config.split.n_folds,
        group_by=config.split.group_by, subsample_seed=args.subsample_seed,
        limit=limit, use_cache=not args.no_cache, log=report.log,
    )
    datasets = crossval.build_datasets(json_path, labels_path, features, depths, **build)
    dataset = datasets[None]
    model_class = registry.get("models", config.model)

    report.log(
        f"  {len(dataset):,} sites, {int(dataset.y.sum()):,} positive "
        f"({100 * dataset.y.mean():.2f}%), {config.split.n_folds} folds grouped by "
        f"{config.split.group_by} (seed {config.split.seed})"
    )
    report.heading("Fitting folds")
    repeated = crossval.repeated_cross_validate(
        dataset, model_class, config.model_params,
        n_repeats=repeats, seed=config.split.seed,
        n_folds=config.split.n_folds, group_by=config.split.group_by,
        label=config.name, log=report.log,
    )
    # Everything except the comparison is computed from repetition 0, the
    # canonical split, so a repeated run reports the same headline as a plain one.
    result = repeated.canonical

    reporting.per_fold(report, result.oof, name=features)
    if repeats > 1:
        reporting.repeated(report, repeated, name=features)
    if report.profile.strata:
        reporting.strata(report, result.oof, args.by.split(","), args.min_positive)
    reporting.calibration(report, result.oof)
    reporting.curves(
        report,
        {config.name: (result.oof["label"].to_numpy(), result.oof["score"].to_numpy())},
    )

    if sweep:
        reporting.depth_sweep(
            report, {d: datasets[d] for d in depths}, result.models, result.columns,
            result.oof, args.subsample_seed,
        )

    if ablate:
        from m6a.compare import paired_comparison

        report.heading("Ablations")
        report.log(
            "Same model, same folds, fewer columns. motif-only is the sequence\n"
            "floor: an 18-way one-hot over the DRACH motif with no signal data at\n"
            "all. A model that does not clear it has learned nothing from the pore."
        )
        ablations = {}
        for which in ("signal", "motif"):
            columns = crossval.column_subset(dataset.columns, which)
            report.log(f"\n  {which}-only ({len(columns)} columns)")
            ablations[which] = crossval.cross_validate(
                dataset, model_class, config.model_params,
                columns=columns, label=f"{config.name}:{which}", keep_models=False,
                log=report.log,
            )
            reporting.show_pooled(
                report.log, ablations[which].pooled, prefix=f"  {which}-only OOF"
            )
        report.data["ablations"] = {
            which: {"n_columns": len(run.columns), "pooled": run.pooled,
                    "per_fold": run.per_fold}
            for which, run in ablations.items()
        }
        for which in ("signal", "motif"):
            reporting.comparison(
                report,
                paired_comparison(
                    ablations[which].per_fold, result.per_fold, f"{which}-only",
                    config.name, n_folds=config.split.n_folds,
                ),
                f"Paired: {config.name} vs {which}-only",
                f"ablation_{which}",
            )

    if args.compare_features:
        # Every depth the sweep wants, because the arm is evaluated in full now.
        # Features are cached per (feature set, depth), so a second sweep of a
        # set someone has already swept costs a cache read.
        other_sets = crossval.build_datasets(
            json_path, labels_path, args.compare_features, depths, **build
        )
        crossval.assert_same_folds(
            crossval.oof_table(dataset, np.zeros(len(dataset))),
            crossval.oof_table(other_sets[None], np.zeros(len(other_sets[None]))),
            features, args.compare_features,
        )
        report.heading(f"Fitting folds: {args.compare_features}")
        baseline = crossval.repeated_cross_validate(
            other_sets[None], model_class, config.model_params,
            n_repeats=repeats, seed=config.split.seed,
            n_folds=config.split.n_folds, group_by=config.split.group_by,
            label=args.compare_features, log=report.log,
        )
        reporting.show_pooled(
            report.log, baseline.canonical.pooled, prefix=f"{args.compare_features} OOF"
        )
        report.data["compare_features"] = {
            "features": args.compare_features,
            "pooled": baseline.canonical.pooled,
            "per_fold": baseline.canonical.per_fold,
        }
        compare_arm(
            report, args, "features",
            f"Paired: {features} vs {args.compare_features} "
            f"(same model, same parameters, same folds)",
            repeated, features, baseline, args.compare_features,
            config.split.n_folds, dataset.y, dataset.sites,
            sweep=(
                {"datasets": {d: other_sets[d] for d in depths},
                 "models": baseline.canonical.models,
                 "columns": baseline.canonical.columns,
                 "subsample_seed": args.subsample_seed}
                if sweep else None
            ),
        )

    if args.compare_run:
        from m6a.compare import paired_comparison

        report.heading(f"Paired against W&B run: {args.compare_run}")
        observations, fingerprint = tracking.fetch_observations(args.compare_run)
        tracking.require_same_dataset(
            tracking.dataset_fingerprint(json_path, labels_path, config, limit),
            fingerprint,
            args.compare_run,
        )
        report.log(
            f"  {fingerprint['run_name']} ({fingerprint['run_id']}): "
            f"{len(observations)} observations, pooled PR AUC "
            f"{fingerprint['pooled']:.4f}"
        )
        report.log(f"  {fingerprint['url']}")
        report.log(
            "  Nothing was refitted - this arm's numbers came out of W&B. That\n"
            "  buys the corrected paired test and costs the bootstrap on the\n"
            "  difference, which needs both arms' per-site scores at once."
        )
        report.data["compare_run"] = {
            "reference": args.compare_run,
            "n_observations": len(observations),
            **{k: v for k, v in fingerprint.items() if k != "url"},
        }
        reporting.comparison(
            report,
            paired_comparison(
                observations, repeated.observations,
                fingerprint["run_name"], features, n_folds=config.split.n_folds,
            ),
            f"Paired: {features} vs {fingerprint['run_name']} (pulled from W&B)",
            "run",
        )

    if args.bootstrap and not (args.compare_features or args.compare_with):
        reporting.bootstrap(
            report,
            result.oof["label"].to_numpy(),
            {features: result.oof["score"].to_numpy()},
            args.bootstrap,
            args.bootstrap_seed if args.bootstrap_seed is not None else BOOTSTRAP_SEED,
        )

    if args.compare_with:
        other_config = Config.load(args.compare_with)
        if (other_config.split.seed, other_config.split.group_by) != (
            config.split.seed, config.split.group_by
        ):
            raise SystemExit(
                f"{args.config} and {args.compare_with} use different splits "
                f"(seed {config.split.seed}/{config.split.group_by} vs "
                f"{other_config.split.seed}/{other_config.split.group_by}). "
                "A paired comparison needs the same folds - see AGENTS.md section 3."
            )
        other_sets = crossval.build_datasets(
            json_path, labels_path, other_config.features, depths, **build
        )
        report.heading(f"Fitting folds: {other_config.name}")
        baseline = crossval.repeated_cross_validate(
            other_sets[None], registry.get("models", other_config.model),
            other_config.model_params,
            n_repeats=repeats, seed=config.split.seed,
            n_folds=config.split.n_folds, group_by=config.split.group_by,
            label=other_config.name, log=report.log,
        )
        reporting.show_pooled(
            report.log, baseline.canonical.pooled, prefix=f"{other_config.name} OOF"
        )
        report.data["compare_with"] = {
            "config": str(args.compare_with),
            "name": other_config.name,
            "pooled": baseline.canonical.pooled,
            "per_fold": baseline.canonical.per_fold,
        }
        compare_arm(
            report, args, "config",
            f"Paired: {config.name} vs {other_config.name} "
            f"(these configs differ in more than one thing)",
            repeated, config.name, baseline, other_config.name,
            config.split.n_folds, dataset.y, dataset.sites,
            sweep=(
                {"datasets": {d: other_sets[d] for d in depths},
                 "models": baseline.canonical.models,
                 "columns": baseline.canonical.columns,
                 "subsample_seed": args.subsample_seed}
                if sweep else None
            ),
        )


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    load_env()
    started = time.time()

    report = reporting.new(args.profile, subsample_seed=args.subsample_seed)

    fingerprint: dict = {}
    if args.model:
        stem = Path(args.model).name
        run_name = stem
    else:
        config = Config.load(args.config)
        stem = config.name
        if args.features:
            stem = f"{stem}__{args.features}"
        run_name = stem
        # Recorded on the run so a later --compare-run can check this one rather
        # than trust it. Hashing the input costs about a second and is memoised.
        json_path, labels_path = resolve_inputs(args)
        fingerprint = tracking.dataset_fingerprint(
            json_path, labels_path, config,
            SMOKE_SITES if args.smoke else args.limit,
        )

    tracker = tracking.start(
        run_name,
        enabled=not (args.no_wandb or args.smoke),
        config={"profile": args.profile, "subsample_seed": args.subsample_seed,
                # The resolved value, not args.repeats, which is None whenever
                # the profile's default is being used - and a run config that
                # says "repeats: null" tells a reader nothing.
                "repeats": (args.repeats if args.repeats is not None
                            else reporting.profile(args.profile).repeats),
                "bootstrap": args.bootstrap,
                **fingerprint},
        tags=[tracking.EVAL_TAG],
        job_type="evaluate",
        resume_id=find_training_run(run_name),
    )

    try:
        if args.model:
            run_from_model(args, report)
        else:
            run_from_config(args, report)

        report.data["elapsed_seconds"] = round(time.time() - started, 1)
        out_path = reporting.write(report, args.out or reporting.report_dir(), stem)
        reporting.publish(report, tracker, out_path, name=stem)
        print(f"\nreport -> {out_path}   ({report.data['elapsed_seconds']:.1f}s)")
    finally:
        tracker.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
