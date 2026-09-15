"""The Model contract.

To add a model: create one new module in this package, subclass BaseModel,
decorate it with @register("models", "your_name"), and reference that name from
a config YAML. Do not edit any existing file.

Two rules that are not obvious:

1. `save` writes into a *directory*, not a file. Models own their own format —
   LightGBM writes a native text booster, which survives library upgrades far
   better than a pickle. That matters because evaluators load our final model
   with whatever versions pip resolves on their machine.

2. If your model needs a heavy dependency (torch), import it inside the methods
   that use it, never at module level. The registry imports every module in this
   package to discover it, and predict.py must not pull in torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel:
    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(m6A) per row, shape (len(X),), every value in [0, 1]."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        raise NotImplementedError
