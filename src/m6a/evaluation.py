"""Metrics and submission validation.

At 4.49% positives, PR AUC is the metric that discriminates between models;
ROC AUC flatters everything. Both are reported because the course evaluates on
both, but rank your own experiments on PR AUC.

`metrics` scores a whole set of predictions. `metrics_by` scores it inside
strata (read depth, DRACH motif), which is how you find out *where* a model
fails rather than only how well it does on average. `calibration_table` and
`calibration_summary` ask a different question again: not whether the ranking is
good, but whether a score of 0.6 means 60%.

**Nothing here may grow a module-level import.** predict.py imports this module
and runs on an evaluator's machine with only the base dependencies installed.
That specifically rules out scipy, which is present today only as a transitive
dependency of scikit-learn - the paired significance test lives in
`m6a.compare`, which predict.py never touches. See AGENTS.md section 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from m6a.data import iter_sites

SUBMISSION_COLUMNS = ["transcript_id", "transcript_position", "score"]


def metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC AUC and PR AUC, plus the positive rate they should be read against."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positive_rate = float(y_true.mean())
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "positive_rate": positive_rate,
        # A random classifier scores pr_auc == positive_rate. Anything at or
        # below this line has learned nothing, whatever the ROC AUC says.
        "pr_auc_lift": float(average_precision_score(y_true, y_score) / positive_rate),
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
    }


# --------------------------------------------------------------------------
# stratified metrics
# --------------------------------------------------------------------------

# Read-depth bands. The upper boundaries are this training set's own depth
# quartiles and p95 (32 / 47 / 84 / 304, see docs/data.md); the lower five cover
# the regime SG-NEx actually lives in (median depth 3) and the depths the sweep
# scores at. Fixed rather than computed per dataset, so a band means the same
# thing in two reports and the numbers can be read side by side.
#
# 600 splits what used to be a single open-ended 304+ band. Every model drops
# sharply in that band and nobody knows why (GAPS.md), so the question "is it a
# cliff or a slide" was unanswerable while it was one bucket. The split gives
# 3,258 sites / 134 positives below 600 and 2,837 / 129 above - both clear
# `min_positive`, though PR AUC on ~130 positives is noisy and should be read as
# such. True depth here tops out at 991, so there is nothing above 600+ to add.
# Changing these edges changes what every `band/*` key means: see
# docs/decisions/0017.
#
# Depth is the number of distinct RNA molecules measured at one site, not
# repeated readings of one molecule: docs/data.md#read-depth.
DEPTH_BAND_EDGES = [1, 2, 3, 5, 10, 20, 32, 47, 84, 304, 600]
DEPTH_BAND_LABELS = [
    "1", "2", "3-4", "5-9", "10-19", "20-31", "32-46", "47-83", "84-303",
    "304-599", "600+",
]


def depth_bands(n_reads) -> pd.Categorical:
    """Bucket read depths into DEPTH_BAND_LABELS, ordered low to high."""
    edges = DEPTH_BAND_EDGES + [np.inf]
    binned = pd.cut(
        np.asarray(n_reads, dtype=float),
        bins=edges,
        labels=DEPTH_BAND_LABELS,
        right=False,
        include_lowest=True,
    )
    return pd.Categorical(binned, categories=DEPTH_BAND_LABELS, ordered=True)


def band_spans(max_depth: float | None = None) -> dict[str, tuple[int, int]]:
    """The (first, last) read count each depth band actually covers.

    A band is a *range*, and its metric is one number constant across that whole
    range. Plotting it as a single point invites a reader to draw a line to the
    next point and see a trend - but nothing was measured in between, and the
    bands are unevenly spaced (84-303 is 220 reads wide, 20-31 is 12). Knowing
    each band's extent is what lets a figure say "constant here, measured once".

    The top band is open-ended, so its upper bound is whatever the data actually
    reaches. Pass `max_depth` to get the honest edge; without it the last band is
    given a nominal width so the figure still draws.
    """
    spans: dict[str, tuple[int, int]] = {}
    for index, label in enumerate(DEPTH_BAND_LABELS):
        lower = DEPTH_BAND_EDGES[index]
        if index + 1 < len(DEPTH_BAND_EDGES):
            upper = DEPTH_BAND_EDGES[index + 1] - 1
        elif max_depth is not None:
            upper = int(max_depth)
        else:
            upper = int(lower * 1.5)
        spans[label] = (int(lower), max(int(upper), int(lower)))
    return spans


def metrics_by(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups,
    min_positive: int = 10,
) -> pd.DataFrame:
    """`metrics` computed within each stratum of `groups`.

    Strata with fewer than `min_positive` positives, or with no negatives, get
    NaN metrics and a reason in `note` rather than a number: PR AUC over three
    positives is noise with a decimal point on it.

    **Do not compare pr_auc between strata.** PR AUC is bounded below by the
    stratum's own positive rate, and the per-motif base rate here spans three
    orders of magnitude (GGACT 22.6%, TAACA 0.02%). `pr_auc_lift` divides that
    out and is the column to read across rows; `pr_auc` is the column to read
    down a column, comparing two models on the same stratum.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    frame = pd.DataFrame({"y": y_true, "score": y_score, "group": pd.Series(groups)})

    rows = []
    for name, part in frame.groupby("group", observed=True, sort=True):
        n_positive = int(part["y"].sum())
        row = {
            "group": name,
            "n": int(len(part)),
            "n_positive": n_positive,
            "positive_rate": float(part["y"].mean()),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "pr_auc_lift": np.nan,
            "note": "",
        }
        if n_positive < min_positive:
            row["note"] = f"only {n_positive} positive(s)"
        elif n_positive == len(part):
            row["note"] = "no negatives"
        else:
            row.update(
                {k: v for k, v in metrics(part["y"].to_numpy(), part["score"].to_numpy()).items()
                 if k in ("roc_auc", "pr_auc", "pr_auc_lift")}
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("group")


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibration_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability by score quantile: what we predicted against what happened.

    A calibrated model's `mean_predicted` tracks `observed_rate` down every row.
    ROC AUC and PR AUC cannot see this at all - both are rank-based, so a model
    can be perfectly ordered and still wrong about magnitude by a factor of two.
    That only matters when the scores are used as probabilities, which Task 2
    does the moment it counts modified sites in a cell line.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    frame = pd.DataFrame({"y": y_true, "score": y_score})

    binned = pd.qcut(frame["score"], n_bins, duplicates="drop")
    if binned.isna().any():
        # qcut drops bin boundaries when scores tie heavily, and drops *every*
        # boundary when they are all identical - which leaves the whole table
        # empty and makes a broken model look like it has no calibration error
        # at all. A degenerate score distribution is exactly when this number is
        # worth having, so fall back to one bin covering everything.
        binned = binned.cat.add_categories(["all"]).fillna("all")
    frame["bin"] = binned

    table = frame.groupby("bin", observed=True).agg(
        n=("y", "size"),
        n_positive=("y", "sum"),
        mean_predicted=("score", "mean"),
        observed_rate=("y", "mean"),
    )
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    table.index = [f"{i}" for i in range(1, len(table) + 1)]
    table.index.name = "decile"
    return table


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Size-weighted mean gap between predicted and observed rate, across bins.

    The single number to read for "are these probabilities honest". Unlike
    `count_ratio`, errors in opposite directions cannot cancel: a model that
    over-predicts in the middle and under-predicts at the top scores badly here
    and perfectly on the ratio.

    0 is perfect. It says nothing at all about whether the model is any good -
    predicting the base rate for every site scores 0.0 and is useless. Rank
    models on PR AUC; read this only to decide whether a score may be used as a
    probability.
    """
    table = calibration_table(y_true, y_score, n_bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def calibration_summary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """How far the scores are from being probabilities, in one line.

    Read `ece` first - it is the honest summary. `count_ratio` is the one that
    makes the consequence concrete: it is how many times too many sites you would
    count by summing the scores, which is what breaks a Task 2 claim of the form
    "cell line X has N modified sites". It is a weak diagnostic on its own,
    because over- and under-prediction in different bins cancel in it.

    `brier` is a proper scoring rule and moves with both calibration and
    discrimination. Under a 4.49% positive rate its floor is dominated by the
    easy negatives - predicting the base rate everywhere scores 0.0429 - so read
    changes in it, not its level.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    expected = float(y_score.sum())
    actual = float(y_true.sum())
    return {
        "ece": expected_calibration_error(y_true, y_score),
        "mean_predicted": float(y_score.mean()),
        "actual_rate": float(y_true.mean()),
        "expected_positives": expected,
        "actual_positives": actual,
        "count_ratio": float(expected / actual) if actual else float("nan"),
        "brier": float(np.mean((y_score - y_true) ** 2)),
        # What a model that learned nothing but the base rate would score on
        # brier. Anything above this line is worse than predicting the average.
        "brier_baseline": float(y_true.mean() * (1 - y_true.mean())),
        # ECE flipped so that higher is better, for a scatter axis where "up and
        # to the right" means "better" on both dimensions. Derived, not new
        # information - `ece` stays the number to quote. Logged rather than
        # computed in the panel because a W&B axis cannot transform a metric.
        "calibrated": 1.0 - expected_calibration_error(y_true, y_score),
    }


def write_submission(scores: pd.DataFrame, path: str | Path) -> Path:
    """Write the required CSV: transcript_id,transcript_position,score."""
    path = Path(path)
    out = scores.reset_index()[SUBMISSION_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def validate_submission(csv_path: str | Path, json_path: str | Path) -> list[str]:
    """Check a predictions CSV against the JSON it was produced from.

    The handout requires that transcript_id and transcript_position match the
    input data.json exactly, and that score is a float in [0, 1]. Cheap
    insurance against a silent format failure at 23:58 on a deadline.

    Returns a list of problems; empty means the file is good.
    """
    problems: list[str] = []

    try:
        sub = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        return [f"Could not read {csv_path} as CSV: {exc}"]

    if list(sub.columns) != SUBMISSION_COLUMNS:
        problems.append(
            f"Columns are {list(sub.columns)}, expected exactly {SUBMISSION_COLUMNS}"
        )
        return problems

    scores = pd.to_numeric(sub["score"], errors="coerce")
    if scores.isna().any():
        problems.append(f"{int(scores.isna().sum())} score(s) are not numeric")
    else:
        out_of_range = ((scores < 0) | (scores > 1)).sum()
        if out_of_range:
            problems.append(f"{int(out_of_range)} score(s) outside [0, 1]")

    expected = {site.key for site in iter_sites(json_path)}
    got = set(zip(sub["transcript_id"], sub["transcript_position"]))

    if len(got) != len(sub):
        problems.append(f"{len(sub) - len(got)} duplicate (transcript, position) row(s)")

    missing = expected - got
    extra = got - expected
    if missing:
        example = sorted(missing)[:3]
        problems.append(f"{len(missing)} site(s) in the JSON have no prediction, e.g. {example}")
    if extra:
        example = sorted(extra)[:3]
        problems.append(f"{len(extra)} predicted site(s) are not in the JSON, e.g. {example}")

    return problems
