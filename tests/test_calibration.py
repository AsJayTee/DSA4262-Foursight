"""Guardrails for `calibrated_lightgbm` (src/m6a/models/calibrated.py).

Calibration's whole promise is **"the ordering is untouched, only the magnitude
moves"**. If that promise breaks, the model is not a calibrated version of the
booster - it is a different model, and comparing its `calib/*` against an
uncalibrated run stops being an apples-to-apples statement about calibration.

The promise holds **within a fold** and, measured, it holds exactly: per-fold
average precision is bit-identical. It does *not* hold for the pooled
out-of-fold number, because each fold fits its own calibration and the pooled
vector concatenates five differently-shifted score scales. That is a real
property of the harness's headline metric rather than a defect here, and the
last test pins the distinction so nobody "fixes" the wrong half of it.

The other thing worth a test: the Platt path must never fit a negative slope,
which would silently invert the ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402

PARAMS = dict(n_estimators=40, num_leaves=15, min_child_samples=5, learning_rate=0.1)


def a_dataset(n=1200, seed=4262):
    """A separable-ish imbalanced problem with the index shape the model expects."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    signal = rng.normal(size=(n, 4)) + y[:, None] * 1.4
    index = pd.MultiIndex.from_arrays(
        [[f"ENST{i % 60:05d}" for i in range(n)], np.arange(n)],
        names=["transcript_id", "transcript_position"],
    )
    X = pd.DataFrame(signal, columns=list("abcd"), index=index).astype(np.float32)
    return X, y


def test_calibration_does_not_change_the_ranking_within_a_fold():
    X, y = a_dataset()
    train, test = slice(0, 800), slice(800, None)

    plain = registry.get("models", "lightgbm")(**PARAMS)
    plain.fit(X.iloc[train], y[train])
    calibrated = registry.get("models", "calibrated_lightgbm")(
        method="prior_shift", **PARAMS
    )
    calibrated.fit(X.iloc[train], y[train])

    raw = plain.predict_proba(X.iloc[test])
    fixed = calibrated.predict_proba(X.iloc[test])

    # Bit-identical, not merely close. Anything else and this is a different model.
    assert average_precision_score(y[test], raw) == average_precision_score(
        y[test], fixed
    )
    assert np.array_equal(np.argsort(raw), np.argsort(fixed))


def test_prior_shift_moves_the_scores_down_by_the_class_weight():
    """The derived correction, checked against its own arithmetic."""
    X, y = a_dataset()
    model = registry.get("models", "calibrated_lightgbm")(
        method="prior_shift", **PARAMS
    )
    model.fit(X, y)

    positives, negatives = float(y.sum()), float(len(y) - y.sum())
    assert model.slope == 1.0
    assert model.intercept == pytest.approx(-np.log(negatives / positives))
    # It must lower the scores - that is the entire point.
    assert model.predict_proba(X).mean() < model.base.predict_proba(X).mean()


def test_platt_fits_a_map_and_refuses_to_invert_the_ranking():
    X, y = a_dataset(n=2000)
    model = registry.get("models", "calibrated_lightgbm")(
        method="platt", inner_folds=3, **PARAMS
    )
    model.fit(X, y)

    assert model.slope > 0, "a non-positive slope would reverse the ranking"
    raw = model.base.predict_proba(X)
    fixed = model.predict_proba(X)
    assert np.array_equal(np.argsort(raw), np.argsort(fixed))


def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(ValueError, match="prior_shift"):
        registry.get("models", "calibrated_lightgbm")(method="isotonic", **PARAMS)


def test_the_calibration_survives_a_save_and_load_round_trip():
    import tempfile

    X, y = a_dataset()
    model = registry.get("models", "calibrated_lightgbm")(
        method="prior_shift", **PARAMS
    )
    model.fit(X, y)
    before = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        reloaded = type(model).load(directory)

    assert reloaded.slope == model.slope
    assert reloaded.intercept == model.intercept
    assert np.allclose(reloaded.predict_proba(X), before)


def test_pooled_scores_can_reorder_across_folds_even_though_each_fold_cannot():
    """The subtlety that makes `oof/pr_auc` not invariant to per-fold rescaling.

    Two folds, each calibrated with its own intercept. Every within-fold ordering
    is preserved; the concatenated ordering need not be. This is documented
    behaviour, not a bug, and the test exists so that a future change which makes
    the pooled vector identical is recognised as a *change* rather than a fix.
    """
    # The two folds have to interleave closely enough that a 0.10 difference in
    # intercept straddles a gap: 0.500 (fold 0) and 0.505 (fold 1) are 0.02
    # apart on the logit scale, so a 0.10 shift reverses them.
    raw = np.array([0.400, 0.500, 0.450, 0.505])
    fold = np.array([0, 0, 1, 1])
    logit = np.log(raw / (1 - raw))
    intercepts = {0: -3.00, 1: -3.10}
    shifted = 1 / (1 + np.exp(-(logit + np.array([intercepts[f] for f in fold]))))

    for f in (0, 1):
        here = fold == f
        assert np.array_equal(
            np.argsort(raw[here]), np.argsort(shifted[here])
        ), "within-fold order must never change"

    assert not np.array_equal(np.argsort(raw), np.argsort(shifted)), (
        "this example exists to show the pooled order CAN change; if it no "
        "longer does, the per-fold intercepts stopped differing"
    )
