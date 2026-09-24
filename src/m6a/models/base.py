"""The Model contract.

To add a model: create one new module in this package, subclass BaseModel,
decorate it with @register("models", "your_name"), and reference that name from
a config YAML. Do not edit any existing file.

Three rules that are not obvious:

1. `save` writes into a *directory*, not a file. Models own their own format —
   LightGBM writes a native text booster, which survives library upgrades far
   better than a pickle. That matters because evaluators load our final model
   with whatever versions pip resolves on their machine.

2. If your model needs a heavy dependency (torch), import it inside the methods
   that use it, never at module level. The registry imports every module in this
   package to discover it, and predict.py must not pull in torch.

3. If your model learns by gradient descent over epochs, set
   `REPORTS_TRAINING_CURVE = True` and fill `self.history`. Cross-validation then
   hands `fit` the fold's held-out rows as `validation`, and the run reports a
   training curve. A model that leaves the flag alone is never handed a
   validation set — see the note on the flag below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel:
    name: str = "base"

    # Opt-in. When it is True the fold's held-out rows arrive as
    # `fit(..., validation=(X, y))` and the run logs `curve/train/*` and
    # `fit/*`; when it is False `validation` is never passed at all.
    #
    # **This is a scope rule, not a fact about the model.** The training curve
    # is for models that learn by gradient descent - a neural net whose loss is
    # never plotted is a net nobody can debug. LightGBM trains iteratively too,
    # in boosting rounds, and deliberately does **not** set this: a boosting
    # round and an epoch are different units of work, so putting them on one
    # x axis produces a panel that cannot be read. See docs/decisions/0024,
    # which amends 0021 on exactly this point. Do not "fix" LightGBM by setting
    # the flag; if you want a booster's curve for a one-off diagnostic, set it
    # locally and do not commit it.
    #
    # **The flag is what stops the silent failure**, not the parameter. A model
    # that accepted an evaluation set and quietly ignored it would report no
    # curve and no error, and the missing panel would read as "this model has no
    # interesting curve" rather than "nobody wired it up". A model that does not
    # set it is free to declare `fit(self, X, y, groups=None)` with no
    # `validation` at all: setting the flag without accepting the argument
    # raises a TypeError naming the model, which is the loud version of the
    # same mistake.
    REPORTS_TRAINING_CURVE: bool = False

    # The second opt-in capability, and the same bargain as the first. A model
    # that sets this is handed the raw reads behind each row as a
    # `m6a.data.ReadBlocks` - `fit(..., reads=)`, `predict_proba(X, reads=)`,
    # and the held-out reads as a third element of `validation`.
    #
    # It exists because the pipeline is otherwise **structurally site-level**:
    # a FeatureExtractor turns one Site into one row of numbers and nothing
    # after that can see an individual read. Multiple Instance Learning needs
    # the reads, so they travel beside the feature table rather than through it.
    # See docs/decisions/0023.
    #
    # Setting it is expensive: the dataset has to be built with
    # `with_reads=True`, which is ~397 MB on the full training set, and
    # cross-validation raises rather than quietly scoring without them.
    CONSUMES_READS: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params
        # One dict per step: {"iteration": int, "train_*": float, "valid_*": ...}.
        # Named `iteration` and not `epoch` because nothing here has epochs:
        # LightGBM has boosting rounds and a linear model has solver iterations.
        # A booster and a neural net can both fill this field honestly.
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        validation: tuple | None = None,
        reads: Any | None = None,
    ) -> None:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame, reads: Any | None = None) -> np.ndarray:
        """P(m6A) per row, shape (len(X),), every value in [0, 1]."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        raise NotImplementedError
