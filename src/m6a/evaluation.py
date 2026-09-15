"""Metrics and submission validation.

At 4.49% positives, PR AUC is the metric that discriminates between models;
ROC AUC flatters everything. Both are reported because the course evaluates on
both, but rank your own experiments on PR AUC.
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
