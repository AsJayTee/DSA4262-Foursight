"""Gradient-boosted trees. The intended final model.

Saved in LightGBM's native text format rather than as a pickle: evaluators run
predict.py with whatever versions pip resolves for them, and a pickled sklearn
wrapper is far more likely to break across a version boundary than a text
booster is.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from m6a.models.base import BaseModel
from m6a.registry import register

# **No training curve here, deliberately.** LightGBM does train iteratively —
# one boosting round per tree — and it used to fill `curve/train/*` and `fit/*`
# (docs/decisions/0021). It no longer does: the training-curve panel is for
# models that learn by gradient descent, and a boosting round is not an epoch.
# 600 rounds beside 40 epochs on one x axis is a picture nobody can read, and
# the units are not comparable even when both are honestly called iterations.
# See docs/decisions/0024, which amends 0021 on this point.
#
# Do not re-enable it by setting REPORTS_TRAINING_CURVE here. For a one-off
# diagnostic — "is n_estimators far wrong?" — set the flag locally, run, and do
# not commit it; the numbers that came from doing exactly that are in GAPS.md.
#
# The side effect of removing it is that fits are faster again: passing a
# valid_set made LightGBM score the held-out fold every round, which cost 1.65x
# on the one repetition that recorded it.


@register("models", "lightgbm")
class LightGBMModel(BaseModel):
    DEFAULTS = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 400,
        "verbosity": -1,
        # 4.49% positives. Rebalancing rather than reweighting keeps the output
        # probabilities usable as scores without a calibration step.
        "is_unbalance": True,
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.params = {**self.DEFAULTS, **params}
        self.booster: lgb.Booster | None = None
        self.columns: list[str] = []

    def fit(self, X, y, groups=None) -> None:
        self.columns = list(X.columns)
        n_estimators = self.params.get("n_estimators", 400)
        train_params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        dataset = lgb.Dataset(X.values, label=np.asarray(y), feature_name=self.columns)
        self.booster = lgb.train(train_params, dataset, num_boost_round=n_estimators)

    def predict_proba(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted and no booster was loaded.")
        X = X[self.columns]
        scores = self.booster.predict(X.values)
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            return {}
        gains = self.booster.feature_importance(importance_type="gain")
        return {c: float(g) for c, g in zip(self.columns, gains)}

    def save(self, path) -> None:
        if self.booster is None:
            raise RuntimeError("Nothing to save — fit the model first.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "model.txt"))
        (path / "columns.json").write_text(json.dumps(self.columns, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        model = cls()
        model.booster = lgb.Booster(model_file=str(path / "model.txt"))
        model.columns = json.loads((path / "columns.json").read_text())
        return model
