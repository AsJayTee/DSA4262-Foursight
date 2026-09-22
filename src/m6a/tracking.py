"""W&B logging, and the flat metric schema every run is filtered by.

**Why this module exists.** Ronin instances are created, used and terminated
with nothing pulled off them first. Anything on local disk is destroyed with the
machine, so W&B is not a convenience here - it is the only place a result exists
afterwards. See docs/decisions/0007.

Two things live here:

1. `Tracker`, a thin wrapper around a W&B run that **degrades to a no-op**
   rather than failing. No key, no package, no network: print a line and carry
   on. Call sites never branch on whether tracking is on.
2. `flat_metrics`, which turns a nested report into the flat scalar keys W&B's
   run table can sort and filter on. Those key names are an interface: renaming
   one orphans every historical run, so they change only via a decision record.

**What is small enough to upload.** Metric vectors and figures, not per-site
scores. Fifty floats carry everything a paired test needs; a 9 MB score table
per experiment per person is the wrong thing to push at W&B repeatedly
(docs/decisions/0009). The model artifact train.py uploads is the one large
thing, and it is deliberate.

wandb is imported inside the functions that need it, not at module level, so
importing this module is safe on a machine without it. Not reachable from
predict.py - nothing on the prediction path imports it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Runs that only ever evaluate are tagged so they can be told apart from
# training runs in the W&B table at a glance.
EVAL_TAG = "eval"


class Tracker:
    """A W&B run, or a convincing impression of one that does nothing.

    Everything is best-effort: a logging failure must never take down a run that
    has already spent VM time computing the numbers. Failures are printed once
    and then swallowed, because the alternative - dying at minute 40 of a 45
    minute run - is strictly worse than losing the upload.
    """

    def __init__(self, run=None, log=print) -> None:
        self.run = run
        self._log = log
        self._broken = False

    @property
    def enabled(self) -> bool:
        return self.run is not None and not self._broken

    @property
    def url(self) -> str:
        return getattr(self.run, "url", "") if self.enabled else ""

    @property
    def id(self) -> str:
        return getattr(self.run, "id", "") if self.run is not None else ""

    def _guard(self, what: str, action) -> None:
        if not self.enabled:
            return
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._broken = True
            self._log(
                f"  W&B {what} failed ({type(exc).__name__}: {exc}) - continuing without it"
            )

    def log(self, values: dict[str, Any]) -> None:
        """Log flat scalars. Non-finite values are dropped rather than sent as NaN.

        A NaN in the run table reads as "this run scored nothing" rather than
        "this stratum was too thin to score", and those are not the same claim.
        """
        clean = {
            key: value
            for key, value in values.items()
            if not isinstance(value, float) or np.isfinite(value)
        }
        self._guard("log", lambda: self.run.log(clean))

    def log_curve_series(self, series: dict[str, Any], pairs: list[tuple[str, str]]) -> None:
        """Log curves point-by-point so W&B can overlay them across runs.

        An image cannot be overlaid, filtered, or put on a shared axis - so the
        ROC and PR curves are logged a second way, as a *step series*: one
        `log()` per point, x and y together at the same step. A W&B line panel
        with y = `curve/roc/tpr` and x = `curve/roc/fpr` then draws one line per
        run, and the run selector filters them. That only works because every
        run interpolates onto the same grid (`report.CURVE_GRID`).

        `pairs` names which key is the x-axis for which y, via `define_metric`.
        Without it the panel defaults to `_step` and every reader has to fix the
        axis by hand. `summary="none"` keeps the last point of each curve out of
        the run summary, where it would sit as a meaningless `fpr = 1.0`.
        """
        if not series:
            return

        def action():
            for x_key, y_key in pairs:
                self.run.define_metric(x_key, summary="none")
                self.run.define_metric(y_key, step_metric=x_key, summary="none")
            # Series have different lengths - 201 points for a ROC curve, 10 for
            # a depth sweep. Log by step across the longest, emitting only the
            # keys that have a value there. W&B tolerates a missing key at a
            # step; it does not tolerate steps going backwards, which is what
            # logging each series over its own range would do.
            longest = max(len(values) for values in series.values())
            for i in range(longest):
                row = {key: float(values[i]) for key, values in series.items()
                       if i < len(values)}
                if row:
                    self.run.log(row, step=i)

        self._guard("curve series", action)

    def log_table(self, name: str, frame: pd.DataFrame) -> None:
        """Log a DataFrame as a W&B table. The index becomes the first column."""

        def action():
            import wandb

            flat = frame.reset_index()
            flat.columns = [str(c) for c in flat.columns]
            self.run.log({name: wandb.Table(dataframe=flat)})

        self._guard(f"table {name}", action)

    def log_figures(self, figures: dict[str, Any]) -> None:
        """Log matplotlib figures as images, then close them.

        Closed here rather than by the caller so a run producing thirty figures
        does not leave thirty open, which matplotlib warns about and which costs
        memory on a VM that has better uses for it.
        """
        if not figures:
            return

        def action():
            import wandb

            self.run.log({name: wandb.Image(figure) for name, figure in figures.items()})

        self._guard("figures", action)
        close_figures(figures)

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        kind: str = "evaluation",
        metadata: dict | None = None,
    ) -> None:
        """Upload a file or directory as a named artifact."""

        def action():
            import wandb

            artifact = wandb.Artifact(name, type=kind, metadata=metadata or {})
            target = Path(path)
            if target.is_dir():
                artifact.add_dir(str(target))
            else:
                artifact.add_file(str(target))
            self.run.log_artifact(artifact)

        self._guard(f"artifact {name}", action)

    def summary(self, values: dict[str, Any]) -> None:
        """Set run summary fields - what shows in the run table without charting."""
        self._guard("summary", lambda: self.run.summary.update(values))

    def finish(self) -> None:
        if self.run is None:
            return
        url = self.url
        try:
            self.run.finish()
        except Exception:  # noqa: BLE001
            return
        if url:
            self._log(f"  logged to W&B: {url}")


def close_figures(figures: dict[str, Any]) -> None:
    """Close matplotlib figures. Safe to call when matplotlib is not installed."""
    if not figures:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for figure in figures.values():
        plt.close(figure)


def start(
    name: str,
    *,
    enabled: bool = True,
    config: dict | None = None,
    notes: str = "",
    tags: list[str] | None = None,
    job_type: str = "train",
    resume_id: str | None = None,
    log=print,
) -> Tracker:
    """Start (or resume) a W&B run. Returns a disabled Tracker if it cannot.

    `resume_id` attaches to a run a previous script started, so one experiment is
    one row in the table rather than two - a training run and its evaluation
    belong together (docs/decisions/0007 section 6).
    """
    if not enabled:
        return Tracker(None, log)
    try:
        import wandb
    except ImportError:
        log("  wandb not installed (pip install -e '.[train]') - continuing without it")
        return Tracker(None, log)
    if not os.environ.get("WANDB_API_KEY"):
        log("  WANDB_API_KEY not set in .env - continuing without W&B")
        return Tracker(None, log)

    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "dsa4262-project"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=name,
            config=config or {},
            notes=notes,
            # Passing tags on a resume *replaces* them, which silently threw away
            # the feature-set and model tags a training run set and left the row
            # tagged only `eval`. Those tags are how the run table is filtered by
            # what a run actually is, so they are unioned below instead.
            tags=None if resume_id else (tags or []),
            job_type=job_type,
            id=resume_id,
            resume="allow" if resume_id else None,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  W&B init failed ({type(exc).__name__}: {exc}) - continuing without it")
        return Tracker(None, log)

    if resume_id and tags:
        try:
            run.tags = tuple(dict.fromkeys(list(run.tags) + list(tags)))
        except Exception:  # noqa: BLE001 - never fatal, it is a label
            pass
    return Tracker(run, log)


def check(timeout: float = 20.0) -> tuple[bool, str]:
    """Does the configured W&B key actually work? Returns (ok, detail).

    `doctor` used to report "W&B configured" from the *presence* of a key, which
    a placeholder passes. On a disposable instance that costs a whole run: it
    trains for 40 minutes, logs nothing anywhere durable, and is then terminated.
    Ask the server instead.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return False, "WANDB_API_KEY not set"
    try:
        import wandb
    except ImportError:
        return False, "wandb not installed"

    entity = os.environ.get("WANDB_ENTITY") or None
    project = os.environ.get("WANDB_PROJECT", "dsa4262-project")
    try:
        api = wandb.Api(timeout=timeout)
        viewer = api.viewer
        who = getattr(viewer, "username", None) or getattr(viewer, "entity", "?")
        if entity:
            # Resolving the entity proves the key can reach the place runs land,
            # not merely that it authenticates somewhere.
            api.projects(entity=entity)
        return True, f"key valid for {who}, target {entity or who}/{project}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# identifying what a run was computed on
# --------------------------------------------------------------------------

def dataset_fingerprint(json_path, labels_path, config, limit=None) -> dict:
    """What a run has to record so another run can refuse to pair with it.

    `crossval.assert_same_folds` protects a local comparison: it refuses to pair
    two arms that did not see the same sites in the same folds. Pulling an arm
    out of W&B (`--compare-run`) there is nothing to check - just a vector of
    numbers with no memory of where it came from. Compare against a run from
    before a data refresh and you get a perfectly plausible p-value for a
    comparison nobody made, which is the failure AGENTS.md section 3 exists to
    prevent.

    So every run records the content hash of its input, the number of sites, and
    the split. A cross-run comparison checks all of them and refuses on any
    mismatch rather than trusting that they line up.
    """
    from m6a import feature_cache

    return {
        "data_digest": feature_cache.file_digest(json_path),
        "labels_digest": feature_cache.file_digest(labels_path),
        # 0 rather than None for "all of it". W&B drops None from a run config,
        # so a legitimately-unlimited run would come back indistinguishable from
        # one that never recorded the field at all - and the check below treats
        # those two very differently.
        "data_limit": int(limit or 0),
        "split_seed": config.split.seed,
        "split_n_folds": config.split.n_folds,
        "split_group_by": config.split.group_by,
    }


# Keys that must agree before two runs may be paired. `data_limit` is in the
# list because a 5,000-site smoke run and a full run are not the same
# experiment, however similar their key sets look.
FINGERPRINT_KEYS = (
    "data_digest", "labels_digest", "data_limit",
    "split_seed", "split_n_folds", "split_group_by",
)

def resolve_run(reference: str):
    """A W&B run from an id, an entity/project/id path, or a run name.

    Names are not unique in W&B, so a name matching more than one run is an
    error that lists the candidates rather than a silent pick of the newest.
    Reaching for the wrong arm produces a plausible number for a comparison
    nobody made, which is the whole class of failure this module guards.
    """
    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or None
    project = os.environ.get("WANDB_PROJECT", "dsa4262-project")

    if reference.count("/") == 2:
        return api.run(reference)
    try:
        return api.run(f"{entity}/{project}/{reference}")
    except Exception:  # noqa: BLE001 - not an id; fall through and try the name
        pass

    matches = list(api.runs(f"{entity}/{project}", {"displayName": reference}))
    if not matches:
        raise SystemExit(
            f"No W&B run in {entity}/{project} with id or name {reference!r}. "
            "Pass a run id (the short code at the end of the run URL) or an "
            "exact run name."
        )
    if len(matches) > 1:
        listed = chr(10).join(
            f"    {r.id}  {r.name}  {r.created_at}" for r in matches[:10]
        )
        raise SystemExit(
            f"{len(matches)} runs in {entity}/{project} are named {reference!r}. "
            f"Pass the id instead:{chr(10)}{listed}"
        )
    return matches[0]


def observations_from_summary(summary: dict, metric: str = "pr_auc") -> list[dict]:
    """The per-(repetition, fold) metric vector hiding in a run's flat summary keys.

    `rep/{r}/fold/{f}/{metric}` is the full vector; `fold/{f}/{metric}` is the
    fallback for a run logged before repeated CV shipped, read as repetition 0 -
    which is what it is, since repetition 0 *is* the canonical split
    (docs/decisions/0012).

    Extracted so `fetch_observations` and `fetch_run` cannot drift into parsing
    the same keys two different ways. Returns [] rather than raising; the callers
    differ on whether an empty vector is fatal.
    """
    observations: list[dict] = []
    for key, value in summary.items():
        parts = key.split("/")
        if len(parts) == 5 and parts[0] == "rep" and parts[2] == "fold" and parts[4] == metric:
            observations.append(
                {"repetition": int(parts[1]), "fold": int(parts[3]), metric: float(value)}
            )
    if not observations:
        for key, value in summary.items():
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == "fold" and parts[2] == metric:
                observations.append(
                    {"repetition": 0, "fold": int(parts[1]), metric: float(value)}
                )
    return sorted(observations, key=lambda o: (o["repetition"], o["fold"]))


def fetch_run(reference: str, metric: str = "pr_auc") -> dict:
    """Everything `scripts/compare_runs.py` needs about one finished run, in one call.

    `fetch_observations` answers "give me the vector or fail", which is right for
    `--compare-run`: a comparison that cannot pair is not a comparison. Drawing N
    runs on one figure is a different job - a run with no per-fold vector can
    still be a point on the scatter - so this returns whatever is there and lets
    the caller decide what is missing.

    One `resolve_run` per run, because each one is a network round trip.
    """
    run = resolve_run(reference)
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    config = dict(run.config)
    fingerprint = {key: config.get(key) for key in FINGERPRINT_KEYS}
    return {
        "reference": reference,
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "summary": summary,
        "config": config,
        "fingerprint": fingerprint,
        "observations": observations_from_summary(summary, metric),
    }


def fetch_observations(reference: str, metric: str = "pr_auc") -> tuple[list[dict], dict]:
    """Pull a finished run's per-(repetition, fold) metric vector out of W&B.

    This is what lets a comparison happen without refitting the other arm, and
    it is why observations are keyed on (repetition, fold) in the first place
    (docs/decisions/0012). A run from an instance that was terminated last week
    can still be the baseline for one running now.

    Falls back to the `fold/{f}/{metric}` keys when a run carries no `rep/*`, so
    a single-split run from before repeated CV shipped is still usable - as five
    observations, which the corrected t-test will weigh accordingly.

    Returns (observations, fingerprint). The caller checks the fingerprint before
    pairing; `require_same_dataset` is what does that.
    """
    run = resolve_run(reference)
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    observations = observations_from_summary(summary, metric)
    if not observations:
        raise SystemExit(
            f"W&B run {run.id} ({run.name}) has no per-fold {metric} values, so "
            "there is nothing to pair against. It was probably logged before the "
            "evaluation harness existed - re-run it with scripts/train.py."
        )

    config = dict(run.config)
    fingerprint = {key: config.get(key) for key in FINGERPRINT_KEYS}
    fingerprint.update(
        run_id=run.id,
        run_name=run.name,
        url=run.url,
        pooled=summary.get(f"oof/{metric}"),
        n_sites=summary.get("oof/n"),
    )
    return observations, fingerprint  # already sorted by the helper above


# Must match m6a.report.STRATA_TABLE. Not imported from there because report.py
# imports pandas and the whole evaluation stack, and this module is deliberately
# importable on a machine with neither.
STRATA_TABLE = "strata_observations"


def fetch_strata(reference: str, metric: str = "pr_auc") -> dict[str, dict[str, list[dict]]]:
    """Pull a finished run's per-stratum metric vectors out of W&B.

    Returns {kind: {stratum: [observation, ...]}} shaped exactly like
    `m6a.report.stratified_observations` produces locally, so a pulled arm and a
    refitted one go through the same comparison code.

    Returns an empty dict when the run has no such table - a run from before
    docs/decisions/0018, or one evaluated at `--profile quick`. The caller says
    so and falls back to the overall comparison rather than failing: an overall
    test is still worth having.
    """
    run = resolve_run(reference)
    table = None
    for name in (f"{run.entity}/{run.project}/run-{run.id}-{STRATA_TABLE}:latest",):
        try:
            import wandb

            table = wandb.Api().artifact(name, type="run_table").get(STRATA_TABLE)
            break
        except Exception:  # noqa: BLE001 - absent is a normal answer here
            continue
    if table is None:
        return {}

    columns = list(table.columns)
    out: dict[str, dict[str, list[dict]]] = {}
    for row in table.data:
        record = dict(zip(columns, row))
        if metric not in record:
            continue
        out.setdefault(str(record["kind"]), {}).setdefault(str(record["stratum"]), []).append(
            {
                "repetition": int(record["repetition"]),
                "fold": int(record["fold"]),
                "n": int(record.get("n", 0)),
                "n_positive": int(record.get("n_positive", 0)),
                metric: float(record[metric]),
            }
        )
    return out


def require_same_dataset(local: dict, remote: dict, name: str) -> None:
    """Refuse to pair two runs that were not computed on the same thing.

    The cross-machine counterpart of `crossval.assert_same_folds`. A mismatch is
    not a warning: two runs over different data, or different splits, produce two
    plausible numbers whose difference means nothing, and nothing downstream
    would ever notice.
    """
    missing = [key for key in FINGERPRINT_KEYS if remote.get(key) is None]
    if missing:
        raise SystemExit(
            f"W&B run {remote['run_id']} ({name}) does not record "
            f"{', '.join(missing)}, so there is no way to check it saw the same "
            "data as this run. Runs logged before the fingerprint shipped cannot "
            "be compared across machines - re-run it, or compare locally with "
            "--compare-with, which refits and checks the folds directly."
        )
    differences = [
        f"    {key}: this run {local[key]!r}, {name} {remote[key]!r}"
        for key in FINGERPRINT_KEYS
        if local.get(key) != remote.get(key)
    ]
    if differences:
        raise SystemExit(
            f"This run and {name} were not computed on the same data or split, so "
            "a paired comparison is not valid:"
            + chr(10)
            + chr(10).join(differences)
            + chr(10)
            + "Pairing them anyway would produce a plausible p-value for a "
            "comparison nobody made. See AGENTS.md section 3."
        )


# --------------------------------------------------------------------------
# the flat metric namespace
# --------------------------------------------------------------------------
#
# These key names are a schema. Every run in the project is sorted, filtered and
# charted on them, so renaming one silently orphans every historical run that
# used the old name. Add keys freely; rename one only with a decision record.
#
#   oof/*        pooled out-of-fold, the headline
#   fold/*       the per-fold vector and its spread
#   rep/*        repeated-CV summary, when there is one
#   depth/*      the sweep: the same sites scored at reduced read depth
#   band/*       true depth: how the model does on sites that really are shallow
#   motif/*      per-DRACH-motif lift
#   calib/*      whether a score may be read as a probability
#   threshold/*  the operating points, and how far a site count swings between them
#   ablation/*   the signal-only and motif-only floors
#   compare/*    a paired difference against another run


def flat_metrics(report: dict) -> dict[str, float]:
    """The nested report as flat scalars, for the W&B run table.

    Nested JSON is right for the report artifact and useless for filtering. This
    is the other half: one row per run, the same keys for every config, so "what
    do I compare against when I have no baseline" is answered by every run
    anyone has already made.
    """
    flat: dict[str, float] = {}

    pooled = report.get("pooled")
    if pooled:
        for key in ("pr_auc", "roc_auc", "pr_auc_lift", "positive_rate"):
            if key in pooled:
                flat[f"oof/{key}"] = float(pooled[key])
        flat["oof/n"] = float(pooled.get("n", 0))
        flat["oof/n_positive"] = float(pooled.get("n_positive", 0))

    for fold in report.get("per_fold", []):
        for key in ("pr_auc", "roc_auc"):
            flat[f"fold/{int(fold['fold'])}/{key}"] = float(fold[key])

    summary = report.get("per_fold_summary")
    if summary:
        flat["fold/pr_auc_mean"] = float(summary["mean"])
        flat["fold/pr_auc_sd"] = float(summary["sd"])
        flat["fold/pr_auc_min"] = float(summary["min"])
        flat["fold/pr_auc_max"] = float(summary["max"])

    repeated = report.get("repeated_cv")
    if repeated:
        flat["rep/n_repeats"] = float(repeated["n_repeats"])
        flat["rep/n_observations"] = float(repeated["n_observations"])
        flat["rep/pr_auc_mean"] = float(repeated["pr_auc_mean"])
        flat["rep/pr_auc_sd"] = float(repeated["pr_auc_sd"])
        flat["rep/pr_auc_min"] = float(repeated["pr_auc_min"])
        flat["rep/pr_auc_max"] = float(repeated["pr_auc_max"])
        # Every observation individually, so two runs can be paired from the run
        # table alone without either still existing on disk. fold/{f}/pr_auc
        # above stays repetition 0 and keeps meaning exactly what it always did.
        for row in repeated["observations"]:
            key = f"rep/{int(row['repetition'])}/fold/{int(row['fold'])}"
            flat[f"{key}/pr_auc"] = float(row["pr_auc"])
            flat[f"{key}/roc_auc"] = float(row["roc_auc"])

    sweep = report.get("depth_sweep")
    if sweep:
        for row in sweep["rows"]:
            depth = row["depth"]
            flat[f"depth/{depth}/pr_auc"] = float(row["pr_auc"])
            flat[f"depth/{depth}/roc_auc"] = float(row["roc_auc"])
            flat[f"depth/{depth}/retained"] = float(row["retained"])

    for row in report.get("by_depth", []):
        if row.get("pr_auc_lift") is not None:
            flat[f"band/{row['group']}/pr_auc_lift"] = float(row["pr_auc_lift"])
            flat[f"band/{row['group']}/pr_auc"] = float(row["pr_auc"])

    for row in report.get("by_motif", []):
        if row.get("pr_auc_lift") is not None:
            flat[f"motif/{row['group']}/pr_auc_lift"] = float(row["pr_auc_lift"])

    calibration = report.get("calibration")
    if calibration:
        for key in ("ece", "calibrated", "brier", "count_ratio",
                    "mean_predicted", "actual_rate"):
            value = calibration["summary"].get(key)
            if value is not None:
                flat[f"calib/{key}"] = float(value)

    # The operating points, and how far a site count moves between two of them.
    # `threshold/count_swing` is the one to sort on: it is how arbitrary a Task 2
    # site count is for this model, in one number.
    thresholds = report.get("thresholds")
    if thresholds:
        for name, row in thresholds["operating_points"].items():
            for key in ("threshold", "precision", "recall", "f1",
                        "predicted_positives", "predicted_rate"):
                value = row.get(key)
                if value is not None:
                    flat[f"threshold/{name}/{key}"] = float(value)
        swing = thresholds["count_swing"]
        flat["threshold/count_swing"] = float(swing["ratio"])
        flat["threshold/count_low"] = float(swing["low_count"])
        flat["threshold/count_high"] = float(swing["high_count"])

    for which, ablation in (report.get("ablations") or {}).items():
        flat[f"ablation/{which}/pr_auc"] = float(ablation["pooled"]["pr_auc"])

    boot = report.get("bootstrap")
    if boot:
        flat["boot/n_resamples"] = float(boot["n_resamples"])
        for name, stats in boot["arms"].items():
            flat[f"boot/{name}/ci_low"] = float(stats["ci_low"])
            flat[f"boot/{name}/ci_high"] = float(stats["ci_high"])
        if "difference" in boot:
            flat["boot/difference/mean"] = float(boot["difference"]["mean"])
            flat["boot/difference/ci_low"] = float(boot["difference"]["ci_low"])
            flat["boot/difference/ci_high"] = float(boot["difference"]["ci_high"])

    # A comparison arm gets the identical key structure under its own prefix, so
    # the baseline's calibration and depth numbers are recorded rather than
    # thrown away - and the run's own headline keeps the unprefixed names.
    for name, arm in (report.get("arms") or {}).items():
        for key, value in flat_metrics(arm).items():
            flat[f"arm/{name}/{key}"] = value

    for key, result in (report.get("comparisons") or {}).items():
        flat[f"compare/{key}/mean_difference"] = float(result["mean_difference"])
        flat[f"compare/{key}/p_value"] = float(result["p_value"])
        flat[f"compare/{key}/wins"] = float(result["wins"])
        # `wins` alone cannot be read: 5 might be 5 out of 5 or 5 out of 50, and
        # those are very different claims. A run row has to say which, because
        # evaluate.py resumes a run - so a later 5-fold comparison overwrites an
        # earlier --repeats 10 one, and nothing else in the row would show it.
        flat[f"compare/{key}/n_observations"] = float(result["n_folds"])
        flat[f"compare/{key}/n_repeats"] = float(result.get("n_repeats", 1))
        if result["n_folds"]:
            flat[f"compare/{key}/win_rate"] = float(result["wins"]) / float(result["n_folds"])
        if result.get("p_value_corrected") is not None:
            flat[f"compare/{key}/p_value_corrected"] = float(result["p_value_corrected"])

    # Per-stratum differences. Sortable, and labelled `descriptive` in the report
    # for the reason stated there: ten correlated tests, uncorrected by design.
    for key, block in (report.get("stratified_comparisons") or {}).items():
        for row in block["rows"]:
            stratum = row["stratum"]
            flat[f"compare/{key}/{stratum}/mean_difference"] = float(row["mean_difference"])
            flat[f"compare/{key}/{stratum}/p_value_corrected"] = float(
                row["p_value_corrected"]
            )
            flat[f"compare/{key}/{stratum}/wins"] = float(row["wins"])

    return flat
