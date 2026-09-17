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
            tags=tags or [],
            job_type=job_type,
            id=resume_id,
            resume="allow" if resume_id else None,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  W&B init failed ({type(exc).__name__}: {exc}) - continuing without it")
        return Tracker(None, log)
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
        for key in ("ece", "brier", "count_ratio", "mean_predicted", "actual_rate"):
            value = calibration["summary"].get(key)
            if value is not None:
                flat[f"calib/{key}"] = float(value)

    for which, ablation in (report.get("ablations") or {}).items():
        flat[f"ablation/{which}/pr_auc"] = float(ablation["pooled"]["pr_auc"])

    for key, result in (report.get("comparisons") or {}).items():
        flat[f"compare/{key}/mean_difference"] = float(result["mean_difference"])
        flat[f"compare/{key}/p_value"] = float(result["p_value"])
        flat[f"compare/{key}/wins"] = float(result["wins"])
        if result.get("p_value_corrected") is not None:
            flat[f"compare/{key}/p_value_corrected"] = float(result["p_value_corrected"])

    return flat
