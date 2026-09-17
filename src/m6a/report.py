"""One evaluation report: its sections, its profiles, and where it ends up.

This used to live inside `scripts/evaluate.py`. It moved here because
`scripts/train.py` now runs the evaluation inline
([0007](../../docs/decisions/0007-evaluating-without-a-baseline.md)): on an
instance that gets terminated with nothing pulled off it, a training run that
finishes before anyone evaluates it has lost everything, and re-running it costs
far more than the evaluation would have. Two scripts needing the same report
means the report cannot live in one of them.

**Profiles exist so runs are comparable by construction.** If one experiment was
run with a depth sweep and the next was not, the two are not comparable on the
dimension that matters most, and nobody finds out until they try. `standard` is
the default and includes the sweep deliberately - optional evaluation is
evaluation that does not happen.

| profile | what it computes |
|---|---|
| `quick` | per fold, pooled, calibration. Iteration only; not a recorded result |
| `standard` | + strata by depth and motif, + the depth sweep, + every figure |
| `full` | + the signal-only and motif-only ablations |

**Nothing here is the record.** The JSON under `analysis/evaluation/reports/` is
a convenience for laptop work; W&B holds the copy that outlives the machine.

Imports pandas, numpy and m6a.evaluation. scipy arrives only through
`m6a.compare`, and only when a comparison is actually asked for - see
docs/decisions/0004. Not reachable from predict.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from m6a import crossval
from m6a.evaluation import (
    calibration_summary,
    calibration_table,
    depth_bands,
    metrics,
    metrics_by,
)

DEFAULT_REPORT_DIR = "analysis/evaluation/reports"
DEFAULT_PROFILE = "standard"


def report_dir() -> str:
    """Where local JSON reports go. Override with M6A_REPORT_DIR.

    Same shape as M6A_CACHE_DIR and M6A_DATA_DIR. The test suite points it
    somewhere temporary so a test run cannot overwrite a recorded report.
    """
    return os.environ.get("M6A_REPORT_DIR", DEFAULT_REPORT_DIR)


@dataclass(frozen=True)
class Profile:
    """What a named profile computes. See the table in the module docstring."""

    name: str
    strata: bool
    depth_sweep: bool
    plots: bool
    ablations: bool


PROFILES: dict[str, Profile] = {
    "quick": Profile("quick", strata=False, depth_sweep=False, plots=False, ablations=False),
    "standard": Profile("standard", strata=True, depth_sweep=True, plots=True, ablations=False),
    "full": Profile("full", strata=True, depth_sweep=True, plots=True, ablations=True),
}


def profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit(
            f"Unknown profile {name!r}. Pick one of: {', '.join(PROFILES)}.\n"
            "  quick     per fold, pooled, calibration - for iterating\n"
            "  standard  + strata, depth sweep, figures - the default, and what a\n"
            "            recorded result should be\n"
            "  full      + signal-only and motif-only ablations"
        )
    return PROFILES[name]


@dataclass
class Report:
    """The nested report, the figures that go with it, and how to print it.

    `data` is what lands in the JSON artifact; `figures` is what lands in W&B as
    images. Sections append to both.
    """

    profile: Profile
    log: Callable[[str], None] = print
    data: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)

    def heading(self, text: str) -> None:
        self.log(f"\n{text}\n{'-' * len(text)}")

    def show(self, frame: pd.DataFrame, floats: str = "%.4f") -> None:
        self.log(frame.to_string(float_format=lambda v: floats % v, na_rep="-"))

    def figure(self, name: str, figure) -> None:
        """Register a figure under a W&B key. `None` (nothing to draw) is ignored."""
        if figure is not None:
            self.figures[name] = figure


def new(profile_name: str = DEFAULT_PROFILE, *, subsample_seed: int, log=print) -> Report:
    report = Report(profile=profile(profile_name), log=log)
    report.data.update(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile=profile_name,
        subsample_seed=subsample_seed,
    )
    return report


# --------------------------------------------------------------------------
# printing helpers shared with the scripts
# --------------------------------------------------------------------------

def show_pooled(log, scores: dict, prefix: str = "pooled OOF") -> None:
    log(
        f"{prefix}: ROC AUC {scores['roc_auc']:.4f}   PR AUC {scores['pr_auc']:.4f}"
        f"   ({scores['pr_auc_lift']:.2f}x random)   "
        f"n={scores['n']:,}  positives={scores['n_positive']:,} "
        f"({100 * scores['positive_rate']:.2f}%)"
    )


def records(frame: pd.DataFrame) -> list[dict]:
    """A frame as JSON-safe records, keeping the index as a column."""
    out = frame.reset_index()
    out.columns = [str(c) for c in out.columns]
    return json.loads(out.to_json(orient="records"))


def per_fold_metrics(oof: pd.DataFrame) -> list[dict]:
    return [
        {"fold": int(fold), **metrics(part["label"].to_numpy(), part["score"].to_numpy())}
        for fold, part in oof.groupby("fold", sort=True)
    ]


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def per_fold(report: Report, oof: pd.DataFrame, name: str = "") -> list[dict]:
    """Per-fold and pooled metrics, plus the distribution figure over the folds."""
    folds = per_fold_metrics(oof)
    frame = crossval.per_fold_frame(folds)
    pr = frame["pr_auc"].to_numpy()

    report.heading("Per fold")
    report.show(frame)
    report.log(
        f"\nper-fold PR AUC: mean {pr.mean():.4f}  sd {pr.std(ddof=1):.4f}  "
        f"range [{pr.min():.4f}, {pr.max():.4f}]"
    )
    report.log(
        "That sd is fold-to-fold variation, not the uncertainty on a comparison.\n"
        "Folds differ in size and positive rate, and PR AUC depends on the base\n"
        "rate, so a model scores highest on the richest fold whatever its merit.\n"
        "To compare two models, pair them fold by fold: --compare-features."
    )
    pooled = metrics(oof["label"].to_numpy(), oof["score"].to_numpy())
    show_pooled(report.log, pooled)

    report.data["per_fold"] = folds
    report.data["per_fold_summary"] = {
        "mean": float(pr.mean()),
        "sd": float(pr.std(ddof=1)),
        "min": float(pr.min()),
        "max": float(pr.max()),
    }
    report.data["pooled"] = pooled

    if report.profile.plots:
        from m6a import figures

        report.figure(
            "fig/pr_auc_distribution",
            figures.metric_distribution({name or "this run": pr.tolist()}),
        )
    return folds


def strata(report: Report, oof: pd.DataFrame, which: list[str], min_positive: int) -> None:
    """Metrics inside read-depth bands, DRACH motifs and folds - where it fails."""
    y = oof["label"].to_numpy()
    scores = oof["score"].to_numpy()
    wanted = {}
    if "depth" in which:
        wanted["read depth"] = ("by_depth", depth_bands(oof["n_reads"].to_numpy()))
    if "motif" in which:
        wanted["DRACH motif"] = ("by_motif", oof["motif"].to_numpy())
    if "fold" in which:
        wanted["fold"] = ("by_fold", oof["fold"].to_numpy())

    for title, (key, groups) in wanted.items():
        frame = metrics_by(y, scores, groups, min_positive=min_positive)
        report.heading(f"By {title}")
        report.show(frame)
        if key == "by_depth":
            report.log(
                "Depth is the number of distinct RNA molecules measured at the site,\n"
                "not repeated readings of one molecule - docs/data.md#read-depth.\n"
                "Bands below 20 reads are empty on this training set by construction."
            )
        report.log(
            "\nRead pr_auc_lift across rows, not pr_auc: PR AUC is bounded below by\n"
            "each stratum's own positive rate, and those differ by a lot here."
        )
        report.data[key] = records(frame)

    if report.profile.plots and "by_depth" in report.data:
        from m6a import figures

        report.figure("fig/depth_bands", figures.depth_bands(report.data["by_depth"]))


def calibration(report: Report, oof: pd.DataFrame) -> None:
    """Whether a score may be read as a probability. Rank metrics cannot see this."""
    y = oof["label"].to_numpy()
    scores = oof["score"].to_numpy()
    table = calibration_table(y, scores)
    summary = calibration_summary(y, scores)

    report.heading("Calibration")
    report.show(table)
    report.log(
        f"\nmean predicted {summary['mean_predicted']:.4f} vs actual positive rate "
        f"{summary['actual_rate']:.4f}"
    )
    report.log(
        f"expected positives {summary['expected_positives']:,.0f} vs actual "
        f"{summary['actual_positives']:,.0f}  ->  "
        f"{summary['count_ratio']:.2f}x overcount"
    )
    report.log(
        "ROC AUC and PR AUC cannot see this: both are rank-based, so the ordering\n"
        "can be good while the magnitude is wrong by this factor. It matters the\n"
        "moment a score is read as a probability - counting modified sites per\n"
        "cell line in Task 2 does exactly that."
    )
    report.data["calibration"] = {"summary": summary, "deciles": records(table)}

    if report.profile.plots:
        from m6a import figures

        report.figure("fig/reliability", figures.reliability(table, summary))


def curves(report: Report, arms: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """ROC and precision-recall over the pooled out-of-fold scores."""
    if not report.profile.plots or not arms:
        return
    from m6a import figures

    base_rate = float(np.asarray(next(iter(arms.values()))[0]).mean())
    report.figure("fig/pr_curve", figures.pr_curve(arms, base_rate))
    report.figure("fig/roc_curve", figures.roc_curve(arms))


def depth_sweep(
    report: Report,
    datasets: dict,
    models: list,
    columns: list[str],
    oof_full: pd.DataFrame,
    subsample_seed: int,
) -> None:
    """Score the fold models on read-subsampled copies of their own held-out fold.

    The largest known risk in the project: every training site has at least 20
    reads and SG-NEx has a median of 3.
    """
    rows = []
    for depth, dataset in datasets.items():
        if depth is None:
            scores = oof_full["score"].to_numpy()
        else:
            scores = crossval.score_folds(models, dataset, columns)
        rows.append(
            {"depth": "full" if depth is None else str(depth), **metrics(dataset.y, scores)}
        )

    frame = pd.DataFrame(rows).set_index("depth")[["roc_auc", "pr_auc", "pr_auc_lift"]]
    frame["retained"] = frame["pr_auc"] / frame.loc["full", "pr_auc"]

    report.heading("Depth sweep")
    report.log(
        "Models are fitted on FULL-depth training folds and scored on read-\n"
        "subsampled copies of their own held-out fold. That is the Task 2\n"
        "situation: we train on depth >= 20 data and predict on SG-NEx, whose\n"
        "median depth is 3. Labels come from m6ACE-Seq rather than from the\n"
        "nanopore reads, so they stay valid when reads are dropped.\n"
        f"Subsample seed {subsample_seed} (not the split seed; safe to vary)."
    )
    report.show(frame)
    report.data["depth_sweep"] = {
        "subsample_seed": subsample_seed,
        "note": "trained at full depth, scored at the stated depth",
        "rows": records(frame),
    }

    if report.profile.plots:
        from m6a import figures

        report.figure(
            "fig/depth_sweep",
            figures.depth_sweep(report.data["depth_sweep"]["rows"], subsample_seed),
        )


def comparison(report: Report, result: dict, title: str, key: str) -> None:
    """A paired fold-by-fold comparison between two runs, printed and plotted."""
    from m6a.compare import comparison_frame, verdict

    report.heading(title)
    report.show(comparison_frame(result))
    report.log(
        f"\n{result['baseline']:<22} mean {result['baseline_mean']:.4f}  "
        f"sd {result['baseline_sd']:.4f}"
    )
    report.log(
        f"{result['candidate']:<22} mean {result['candidate_mean']:.4f}  "
        f"sd {result['candidate_sd']:.4f}"
    )
    report.log(
        f"\nunpaired (wrong)  gap {result['mean_difference']:+.4f} against a fold sd "
        f"of ~{result['baseline_sd']:.4f}, Welch p = {result['unpaired_p_value']:.4f}"
    )
    report.log(
        f"paired   (right)  mean difference {result['mean_difference']:+.4f}  "
        f"sd {result['sd_difference']:.4f}  "
        f"95% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
        f"t = {result['t_statistic']:.2f}  p = {result['p_value']:.4f}  "
        f"wins {result['wins']}/{result['n_folds']}"
    )
    report.log(f"\n{verdict(result)}")
    report.data.setdefault("comparisons", {})[key] = result

    if report.profile.plots:
        from m6a import figures

        report.figure(f"fig/compare/{key}/distribution", figures.comparison_distribution(result))
        report.figure(f"fig/compare/{key}/difference", figures.difference_strip(result))


# --------------------------------------------------------------------------
# where it goes
# --------------------------------------------------------------------------

def write(report: Report, out_dir: str | Path, stem: str) -> Path:
    """Write the nested JSON report locally.

    Kept after [0007](../../docs/decisions/0007-evaluating-without-a-baseline.md)
    made W&B the record, because a laptop run with no key still has to leave
    something behind, and because a number quoted in GAPS.md wants a file to
    point at. It is a convenience, not the record: on a Ronin instance this file
    dies with the machine.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # A run over a subset of the sites is not the run whose name it shares, and
    # overwriting a recorded report with a 5,000-site smoke result is a silent
    # way to make a quoted number untraceable. Give it its own file.
    if report.data.get("source", {}).get("limit"):
        stem = f"{stem}__limit{report.data['source']['limit']}"
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(report.data, indent=2, default=str))
    return path


def publish(report: Report, tracker, report_path: Path | None = None, name: str = "report") -> None:
    """Send everything durable to W&B: flat scalars, tables, figures, the JSON.

    Order matters slightly - figures are logged last because `log_figures` closes
    them, and a failure earlier should not leave a hundred open figures behind on
    a long run.
    """
    from m6a import tracking

    if not tracker.enabled:
        tracking.close_figures(report.figures)
        return

    flat = tracking.flat_metrics(report.data)
    tracker.log(flat)
    tracker.summary(flat)

    if report.data.get("per_fold"):
        tracker.log_table(
            "per_fold", crossval.per_fold_frame(report.data["per_fold"])
        )
    for key, title in (("by_depth", "by_depth"), ("by_motif", "by_motif")):
        if report.data.get(key):
            tracker.log_table(title, pd.DataFrame(report.data[key]))
    if report.data.get("depth_sweep"):
        tracker.log_table("depth_sweep", pd.DataFrame(report.data["depth_sweep"]["rows"]))

    if report_path is not None and report_path.exists():
        tracker.log_artifact(
            report_path, f"{name}-report", kind="evaluation",
            metadata={"profile": report.data.get("profile")},
        )

    tracker.log_figures(report.figures)
