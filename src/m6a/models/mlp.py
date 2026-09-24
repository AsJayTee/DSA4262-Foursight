"""A small feed-forward network on the same site-level features everything else uses.

**Boring on purpose.** It is not expected to beat LightGBM on 101 summary
statistics, and that is not why it exists. It exists because it is the first
model in this repo with real epochs, so it is what proves the training-curve
machinery (docs/decisions/0021) works end to end on something that is not a
booster — and because the read-level work that follows needs a torch path that
has already been through cross-validation, W&B and `predict.py` once.

Adding it is additive (AGENTS.md section 1): one new module, one new config, no
decision record.

**`import torch` lives inside the methods that use it, never at module level.**
The registry imports every module in this package to discover the models in it,
so a module-level `import torch` would make torch a hard requirement of
`scripts/predict.py` — the script other students run from a clean `git clone`
with only the base dependencies installed. torch is in the `mil` extra and stays
there; the extra is named for the work it was added for, and this model borrows
it. `tests/test_smoke.py::test_predict_path_has_no_heavy_imports` enforces this,
and if it fails the fix is to move the import, not to change the test.

Features are standardised inside the model rather than by a scikit-learn
pipeline, because the statistics then travel with the weights in one file and a
network is far less forgiving than a tree of a column measured in seconds
(dwell, ~0.008) sitting beside one measured in picoamps (current, ~110). A tree
splits on rank and does not care.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from m6a.models.base import BaseModel
from m6a.registry import register

TORCH_MISSING = (
    "m6a.models.mlp needs torch. Install the MIL extras:  pip install -e '.[mil]'\n"
    "It is deliberately not a base dependency: predict.py runs on an evaluator's "
    "machine with only the base list installed, so torch is imported inside the "
    "methods that use it and never at module level (AGENTS.md section 4)."
)


def _torch():
    """torch, with an error that names the fix instead of a bare ImportError."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(TORCH_MISSING) from exc
    return torch


@register("models", "mlp")
class MLPModel(BaseModel):
    REPORTS_TRAINING_CURVE = True

    DEFAULTS = {
        "hidden": [128, 64],
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "epochs": 40,
        # Not the split seed. It fixes weight initialisation and batch order so
        # two runs of the same config agree; changing it redraws the network's
        # starting point and nothing about which sites are in which fold. Same
        # separation as SUBSAMPLE_SEED (docs/decisions/0003).
        "seed": 4262,
        # 4.49% positives. The analogue of LightGBM's `is_unbalance`: positives
        # are weighted up by the class ratio inside the loss. It buys the same
        # thing and costs the same thing - the outputs are calibrated to a
        # rebalanced world, so this model will overcount like every other model
        # in the repo (see the calibration entry in GAPS.md).
        "balance": True,
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.params = {**self.DEFAULTS, **params}
        self.columns: list[str] = []
        self.network = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    # ----------------------------------------------------------------- fitting

    def _build(self, n_features: int):
        torch = _torch()
        from torch import nn

        layers: list = []
        width = n_features
        for size in self.params["hidden"]:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.params["dropout"])]
            width = size
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def _standardise(self, X: pd.DataFrame):
        """Centre and scale with the statistics learned at fit time.

        A zero-variance column is left alone rather than divided by zero. That
        is not hypothetical here: `quantiles_v1` at depth 1 has 27 columns that
        are exactly constant, which is the measurement recorded under Modelling
        in GAPS.md.
        """
        values = np.asarray(X[self.columns].to_numpy(), dtype=np.float32)
        return (values - self.mean) / self.scale

    def fit(self, X, y, groups=None, validation=None) -> None:
        torch = _torch()
        from torch import nn

        torch.manual_seed(int(self.params["seed"]))
        self.columns = list(X.columns)
        raw = np.asarray(X.to_numpy(), dtype=np.float32)
        self.mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        self.scale = np.where(scale > 0, scale, 1.0).astype(np.float32)

        features = torch.from_numpy((raw - self.mean) / self.scale)
        labels = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)

        self.network = self._build(features.shape[1])
        optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        positives = float(labels.sum())
        pos_weight = None
        if self.params["balance"] and positives > 0:
            pos_weight = torch.tensor([(len(labels) - positives) / positives])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n = len(labels)
        batch_size = int(self.params["batch_size"])
        generator = torch.Generator().manual_seed(int(self.params["seed"]))
        self.history = []

        for epoch in range(int(self.params["epochs"])):
            self.network.train()
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self.network(features[batch]), labels[batch])
                loss.backward()
                optimiser.step()

            # Scoring every epoch is the expensive half, so it happens only when
            # a validation set was handed over - which cross-validation does for
            # repetition 0 and nothing else (docs/decisions/0021). The final
            # refit in scripts/train.py holds nothing out and records no curve.
            if validation is None:
                continue
            self.history.append(
                {"iteration": float(epoch + 1),
                 **_scored("train", np.asarray(y), self._scores(features)),
                 **_scored("valid", np.asarray(validation[1]),
                           self.predict_proba(validation[0]))}
            )

    def _scores(self, features) -> np.ndarray:
        torch = _torch()

        self.network.eval()
        with torch.no_grad():
            logits = self.network(features)
            return torch.sigmoid(logits).squeeze(1).numpy().astype(float)

    def predict_proba(self, X) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Model is not fitted and no weights were loaded.")
        torch = _torch()

        features = torch.from_numpy(self._standardise(X))
        return np.clip(self._scores(features), 0.0, 1.0)

    # ------------------------------------------------------------- persistence

    def save(self, path) -> None:
        if self.network is None:
            raise RuntimeError("Nothing to save — fit the model first.")
        torch = _torch()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path / "model.pt")
        # Beside the weights, everything needed to rebuild the same architecture
        # and reproduce the same inputs. A state_dict on its own is not a model.
        (path / "columns.json").write_text(json.dumps(self.columns, indent=2))
        (path / "mlp.json").write_text(json.dumps({
            "params": self.params,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path):
        torch = _torch()

        path = Path(path)
        blob = json.loads((path / "mlp.json").read_text())
        model = cls(**blob["params"])
        model.columns = json.loads((path / "columns.json").read_text())
        model.mean = np.asarray(blob["mean"], dtype=np.float32)
        model.scale = np.asarray(blob["scale"], dtype=np.float32)
        model.network = model._build(len(model.columns))
        model.network.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        model.network.eval()
        return model


def _scored(split: str, y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """One epoch's metrics, named the way `BaseModel.history` expects.

    The same two quantities LightGBM reports, computed the same way, so the two
    models' curves are the same object and land on one W&B panel: `pr_auc` is
    `average_precision_score`, which is what `m6a.evaluation.metrics` reports
    under that name.
    """
    return {
        f"{split}_logloss": float(log_loss(y, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1])),
        f"{split}_pr_auc": float(average_precision_score(y, scores)),
    }
