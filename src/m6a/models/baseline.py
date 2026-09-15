"""Logistic regression on standardised features.

The handout requires a comparison against a simple baseline and a random
classifier. Having this from day one means every experiment is measured against
it automatically, instead of re-running things in week five to fill a table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from m6a.models.base import BaseModel
from m6a.registry import register


@register("models", "logistic")
class LogisticModel(BaseModel):
    def __init__(self, **params) -> None:
        super().__init__(**params)
        params.setdefault("max_iter", 1000)
        # 4.49% positives: without this the model predicts "unmodified" always.
        params.setdefault("class_weight", "balanced")
        self.pipeline = Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(**params))]
        )
        self.columns: list[str] = []

    def fit(self, X, y, groups=None) -> None:
        self.columns = list(X.columns)
        self.pipeline.fit(X.values, y)

    def predict_proba(self, X) -> np.ndarray:
        X = X[self.columns]
        return self.pipeline.predict_proba(X.values)[:, 1]

    def save(self, path) -> None:
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "columns": self.columns}, path / "model.joblib")

    @classmethod
    def load(cls, path):
        import joblib

        blob = joblib.load(Path(path) / "model.joblib")
        model = cls()
        model.pipeline = blob["pipeline"]
        model.columns = blob["columns"]
        return model
