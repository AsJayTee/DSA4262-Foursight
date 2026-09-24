"""LightGBM with its scores turned back into probabilities.

Every model in this repo overcounts positives - 1.80x for `lightgbm_quantiles`,
2.77x for `lightgbm_pooled`, 6.10x for `baseline_logistic` - and GAPS.md
recorded the cause as unexplained for a long time. It is not unexplained now,
and the explanation says exactly what the fix should be.

## The textbook explanation, and why it is NOT the whole story here

`is_unbalance=True` makes LightGBM weight the positive class by
`w = n_negative / n_positive`, which is `(121838 - 5475) / 5475 = 21.25` here. A
weighted logistic loss is minimised by the odds of the *reweighted* population,
so at the optimum

    logit(p_weighted) = logit(p_true) + log(w),   log(21.25) = 3.06

That arithmetic is correct and it is what GAPS.md and the literature review
record as the explanation for the overcount. **Measured, it is the wrong size.**
Subtracting 3.06 from the logit turns a 1.76x *over*count into a 0.36x
*under*count - it overshoots by a factor of five.

The reason is that a fitted booster does not sit at the optimum of its loss. 600
trees at 63 leaves produce logits that are **too steep** as well as offset.
Fitting the map instead of deriving it, on the full training set with
`quantiles_flank_v1`:

| method | count ratio | ECE | fitted map |
|---|---:|---:|---|
| none | 1.7581x | 0.0356 | - |
| `prior_shift` | **0.3561x** | 0.0289 | `logit - 3.06` (derived) |
| `platt` | **1.2584x** | **0.0119** | `0.69 * logit - 1.00` (fitted) |

**The slope is 0.69, not 1.0.** So most of the miscalibration is
over-confidence, not the class-weight offset, and the class-weight story
explains less of it than the repo currently claims. `prior_shift` is kept
because it is the falsifiable version of that story and the number above is
what falsifies it.

## What this does not do

**It does not improve ranking, and it is not supposed to.** Both methods are
monotone in the model's logit, so **per-fold** PR AUC and ROC AUC come out
bit-identical - measured, `diff 0.0e+00` on all five folds.

**But the pooled out-of-fold number moves slightly, and that is worth
understanding rather than dismissing.** `oof/pr_auc` concatenates scores from
five *different* models, and each fold fits its own calibration (intercepts
-3.034 to -3.099, a spread of 0.065). Sites in different folds are therefore
shifted by different amounts, so the pooled ordering changes across fold
boundaries even though every within-fold ordering is exactly preserved. Measured
effect: 5.68e-04 on pooled AP, an order of magnitude below the ~0.006 the
harness resolves, so it threatens no conclusion - but `oof/pr_auc` is not
invariant to per-fold rescaling and nothing else in the repo says so.

What actually moves is `calib/*` - the count ratio, the ECE, the reliability
curve - and the threshold sweep, because a threshold is an absolute cut on a
score distribution that has just moved.

That is the point: no Task 2 claim of the form "cell line X has N modified
sites" is defensible until the scores are probabilities
([0019](../../../docs/decisions/0019-a-threshold-sweep-because-a-ranking-cannot-count.md)).

## The inner split is grouped by transcript, not by gene

`crossval` does not pass `groups` to `fit`, so the gene is not available here;
the feature frame's index carries the transcript. Grouping the inner folds by
transcript is weaker than grouping by gene - two transcripts of one gene can
land on opposite sides - but it is the outer split that decides every reported
number, and that one is gene-grouped as always. The consequence of the weaker
inner split is that the inner predictions are slightly optimistic, which makes
the fitted calibrator slightly conservative. Recorded rather than hidden.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m6a.models.base import BaseModel
from m6a.models.lightgbm import LightGBMModel
from m6a.registry import register

EPSILON = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@register("models", "calibrated_lightgbm")
class CalibratedLightGBMModel(BaseModel):
    """LightGBM plus a one-dimensional map from its logit to a probability."""

    METHODS = ("prior_shift", "platt")

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.method = str(params.pop("method", "prior_shift"))
        if self.method not in self.METHODS:
            raise ValueError(
                f"calibrated_lightgbm: method must be one of {self.METHODS}, "
                f"got {self.method!r}. Set it in the config's model_params."
            )
        self.inner_folds = int(params.pop("inner_folds", 3))
        self.base_params = params
        self.base = LightGBMModel(**params)
        # a * logit(p) + b. prior_shift fixes a = 1 and solves for b in closed
        # form; platt fits both.
        self.slope = 1.0
        self.intercept = 0.0

    # ----------------------------------------------------------------- fitting

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups=None) -> None:
        y = np.asarray(y)
        if self.method == "prior_shift":
            self.base.fit(X, y)
            positives = float(y.sum())
            negatives = float(len(y) - positives)
            if positives <= 0 or negatives <= 0:
                self.slope, self.intercept = 1.0, 0.0
                return
            # The same w LightGBM's is_unbalance uses internally.
            weight = negatives / positives
            self.slope, self.intercept = 1.0, -float(np.log(weight))
            return

        # platt: cross-fit so the calibrator never sees a score from a model
        # that trained on the row it is calibrating.
        held_out_logit = np.full(len(y), np.nan)
        for inner_train, inner_test in self._inner_splits(X, y):
            if inner_train.sum() == 0 or inner_test.sum() == 0:
                continue
            fold_model = LightGBMModel(**self.base_params)
            fold_model.fit(X.loc[inner_train], y[inner_train])
            frame = X.loc[inner_test]
            held_out_logit[inner_test] = np.asarray(
                fold_model.booster.predict(
                    frame[fold_model.columns].values, raw_score=True
                ),
                dtype=np.float64,
            )

        self.base.fit(X, y)
        usable = np.isfinite(held_out_logit)
        if usable.sum() < 100 or len(np.unique(y[usable])) < 2:
            # Not enough cross-fitted scores to fit a map; fall back to the
            # closed-form correction rather than to an unfitted identity.
            positives = float(y.sum())
            negatives = float(len(y) - positives)
            self.slope = 1.0
            self.intercept = (
                -float(np.log(negatives / positives))
                if positives > 0 and negatives > 0
                else 0.0
            )
            return

        from sklearn.linear_model import LogisticRegression

        scaler = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        scaler.fit(held_out_logit[usable].reshape(-1, 1), y[usable])
        self.slope = float(scaler.coef_[0][0])
        self.intercept = float(scaler.intercept_[0])
        if self.slope <= 0:
            # A negative slope would invert the ranking, which would make this a
            # different model rather than a calibrated one. Refuse it.
            raise ValueError(
                "calibrated_lightgbm: Platt scaling fitted a non-positive slope "
                f"({self.slope:.4f}), which would reverse the model's ranking. "
                "Use model_params: {method: prior_shift} instead."
            )

    def _inner_splits(self, X: pd.DataFrame, y: np.ndarray):
        """Grouped inner folds, keyed on the transcript in the frame's index."""
        level = 0
        transcripts = np.asarray(X.index.get_level_values(level))
        unique = np.unique(transcripts)
        # Deterministic and independent of row order, like m6a.data.assign_folds.
        rng = np.random.default_rng(4262)
        assignment = {
            name: int(i % self.inner_folds)
            for i, name in zip(rng.permutation(len(unique)), unique)
        }
        fold_of_row = np.array([assignment[t] for t in transcripts])
        for fold in range(self.inner_folds):
            test = fold_of_row == fold
            yield ~test, test

    # --------------------------------------------------------------- scoring

    def _base_logit(self, X: pd.DataFrame) -> np.ndarray:
        """The booster's log-odds, without a probability round-trip.

        `predict_proba` returns a probability, and going back to a logit clips
        at EPSILON. On the full training set some out-of-fold predictions fall
        below 1e-7, so the clip creates ties among rows that were distinct -
        which silently changes the ranking of a transform whose entire promise
        is that it does not. LightGBM will hand over the raw score instead;
        asking for it removes the hazard rather than tuning EPSILON against it.
        """
        booster = getattr(self.base, "booster", None)
        if booster is not None:
            columns = getattr(self.base, "columns", None)
            frame = X[columns] if columns else X
            return np.asarray(
                booster.predict(frame.values, raw_score=True), dtype=np.float64
            )
        return _logit(self.base.predict_proba(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.slope * self._base_logit(X) + self.intercept)

    # ------------------------------------------------------------ persistence

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.base.save(path)
        (path / "calibration.json").write_text(
            json.dumps(
                {
                    "method": self.method,
                    "slope": self.slope,
                    "intercept": self.intercept,
                    "inner_folds": self.inner_folds,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        stored = json.loads((path / "calibration.json").read_text())
        model = cls(method=stored["method"], inner_folds=stored["inner_folds"])
        model.base = LightGBMModel.load(path)
        model.slope = float(stored["slope"])
        model.intercept = float(stored["intercept"])
        return model
