"""Attention-based Multiple Instance Learning over the reads at a site.

The briefing frames this problem as MIL and nothing in the repo treated it that
way: the site carries the label, the individual reads do not, and only a
fraction of the reads at a modified site actually carry the modification
(docs/data.md#read-depth). Every other model here sees summary statistics over
the reads - a mean, a quantile - and a summary is exactly where "some of these
reads are unusual" goes to die.

This one sees the reads. Each read is encoded independently, an attention head
scores how much each one matters, and the site embedding is the attention-
weighted sum. That is the standard Ilse et al. (2018) gated-attention MIL
pooling, and it is the right shape for this problem: the network can learn to
put its weight on the handful of reads that look modified instead of averaging
them away.

**Two capability flags, both opt-in** (see `m6a.models.base`):

- `CONSUMES_READS` - cross-validation hands `fit` and `predict_proba` the raw
  reads as a `m6a.data.ReadBlocks`, and refuses rather than quietly scoring
  without them. The dataset has to be built with `with_reads=True`.
- `REPORTS_TRAINING_CURVE` - it reports a per-epoch training curve, as every
  gradient-descent model here does (docs/decisions/0021, scoped by 0024).

The site-level feature row is **not** discarded. The motif one-hot lives there
and is a property of the site rather than of any read - a motif-only classifier
scores 0.1537 on its own, so throwing it away would start from behind. The site
embedding and the site features are concatenated before the classifier head.

**`import torch` lives inside the methods that use it, never at module level.**
The registry imports every module in this package to discover the models in it,
and `scripts/predict.py` runs on an evaluator's machine with only the base
dependencies. torch is in the `mil` extra. See AGENTS.md section 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m6a.data import N_READ_FEATURES
from m6a.models.base import BaseModel
from m6a.models.mlp import TORCH_MISSING, _scored
from m6a.registry import register

# Reads per site handed to the network in one training step. Sites here have 20
# to 991 reads, so the batch is padded to the longest site in it and masked -
# capping the tail keeps a single 991-read site from setting the pad width for a
# whole batch of 20-read ones. The cap applies to *training* only; scoring uses
# every read, because throwing away evidence at prediction time is the thing
# this model exists to avoid.
MAX_TRAIN_READS = 128


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(TORCH_MISSING.replace("m6a.models.mlp", "m6a.models.mil")) from exc
    return torch


@register("models", "attention_mil")
class AttentionMILModel(BaseModel):
    CONSUMES_READS = True
    REPORTS_TRAINING_CURVE = True

    DEFAULTS = {
        "read_hidden": 64,       # per-read encoder width
        "attention_hidden": 32,  # the head that decides which reads matter
        "site_hidden": 64,       # the classifier over [pooled reads | site features]
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,       # sites, not reads
        "epochs": 20,
        "max_train_reads": MAX_TRAIN_READS,
        # Weight init and batch order. NOT the split seed and NOT the subsample
        # seed; changing it redraws where the network starts and nothing about
        # which sites or which reads are used.
        "seed": 4262,
        "balance": True,
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.params = {**self.DEFAULTS, **params}
        self.columns: list[str] = []
        self.network = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.read_mean: np.ndarray | None = None
        self.read_scale: np.ndarray | None = None

    # ------------------------------------------------------------------ shape

    def _build(self, n_site_features: int):
        torch = _torch()
        from torch import nn

        params = self.params

        class GatedAttentionMIL(nn.Module):
            """Encode each read, weight it, sum, then classify with the site row."""

            def __init__(self) -> None:
                super().__init__()
                width = int(params["read_hidden"])
                self.encoder = nn.Sequential(
                    nn.Linear(N_READ_FEATURES, width), nn.ReLU(),
                    nn.Linear(width, width), nn.ReLU(),
                )
                # Gated attention (Ilse et al. 2018): one branch decides how
                # interesting a read is, the other gates it. A plain softmax
                # over a single linear score cannot suppress a read it finds
                # uninformative-but-large, which is most of what varies here.
                gate = int(params["attention_hidden"])
                self.attend = nn.Sequential(nn.Linear(width, gate), nn.Tanh())
                self.gate = nn.Sequential(nn.Linear(width, gate), nn.Sigmoid())
                self.weigh = nn.Linear(gate, 1)
                self.head = nn.Sequential(
                    nn.Linear(width + n_site_features, int(params["site_hidden"])),
                    nn.ReLU(),
                    nn.Dropout(float(params["dropout"])),
                    nn.Linear(int(params["site_hidden"]), 1),
                )

            def forward(self, reads, mask, site_features):
                # reads (B, R, 9); mask (B, R) with True where a read is real.
                encoded = self.encoder(reads)
                scores = self.weigh(self.attend(encoded) * self.gate(encoded)).squeeze(-1)
                # Padding must not compete for attention. -inf before the softmax
                # rather than zeroing after it: a zero weight still contributes
                # to the normaliser, which would shrink every real read's weight
                # by however much padding the batch happened to need.
                scores = scores.masked_fill(~mask, float("-inf"))
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                pooled = (encoded * weights).sum(dim=1)
                return self.head(torch.cat([pooled, site_features], dim=1))

        return GatedAttentionMIL()

    # ---------------------------------------------------------------- batching

    def _standardise_reads(self, blocks) -> np.ndarray:
        """Every read centred and scaled, once per fit or per scoring pass.

        Done here and not inside `_pad` because `_pad` runs once per batch per
        epoch: standardising there re-did the identical arithmetic on the same
        rows twenty times over, and on the full training set that is 220 million
        redundant row-scalings.
        """
        return ((blocks.values - self.read_mean) / self.read_scale).astype(np.float32)

    def _pad(self, blocks, standardised, rows, cap: int | None):
        """A (B, R, 9) padded tensor and its (B, R) mask for the given site rows.

        `cap` limits reads per site during training only. The first `cap` reads
        are taken rather than a random draw, because the draw that matters is
        already made and keyed upstream (`m6a.data.subsample_blocks`,
        docs/decisions/0003) and adding a second unkeyed one here would make a
        fit unreproducible for no gain.

        Gathered with one fancy-index rather than a loop over sites. The loop
        version was correct and cost 365 seconds on a 5,000-site smoke run,
        which extrapolates to about a day for one full-data run at the standard
        profile - so it was not a model anyone could use.
        """
        torch = _torch()

        counts = np.minimum(blocks.counts[rows], cap or np.iinfo(np.int64).max)
        width = max(int(counts.max()) if counts.size else 1, 1)
        columns = np.arange(width)

        # True where a read is real; everything else is padding that must not
        # reach the attention softmax.
        mask = columns[None, :] < counts[:, None]
        # Row r's read c lives at offsets[r] + c in the flat array. Out-of-range
        # entries are clamped to 0 and then zeroed, so the gather never reads
        # another site's memory.
        flat = blocks.offsets[rows][:, None] + columns[None, :]
        padded = standardised[np.where(mask, flat, 0)]
        padded[~mask] = 0.0
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _site_tensor(self, X: pd.DataFrame):
        torch = _torch()

        values = np.asarray(X[self.columns].to_numpy(), dtype=np.float32)
        return torch.from_numpy((values - self.mean) / self.scale)

    # ---------------------------------------------------------------- fitting

    def fit(self, X, y, groups=None, validation=None, reads=None) -> None:
        torch = _torch()
        from torch import nn

        if reads is None:
            raise ValueError(
                "attention_mil needs the reads. Build the dataset with "
                "crossval.build_datasets(..., with_reads=True) - the model "
                "declares CONSUMES_READS so the harness normally does this for "
                "you."
            )
        if len(reads) != len(X):
            raise ValueError(
                f"Got {len(reads):,} sites of reads for {len(X):,} feature rows. "
                "The reads and the feature table are out of step."
            )

        torch.manual_seed(int(self.params["seed"]))
        self.columns = list(X.columns)

        site_values = np.asarray(X.to_numpy(), dtype=np.float32)
        self.mean = site_values.mean(axis=0)
        scale = site_values.std(axis=0)
        self.scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
        # Read features are standardised on their own statistics over every read
        # in the training rows, not per site: a site's reads are the thing being
        # compared to each other, so centring them per site would erase exactly
        # the signal this model is looking for.
        self.read_mean = reads.values.mean(axis=0)
        read_scale = reads.values.std(axis=0)
        self.read_scale = np.where(read_scale > 0, read_scale, 1.0).astype(np.float32)

        self.network = self._build(site_values.shape[1])
        optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        labels = np.asarray(y, dtype=np.float32)
        positives = float(labels.sum())
        pos_weight = None
        if self.params["balance"] and positives > 0:
            pos_weight = torch.tensor([(len(labels) - positives) / positives])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        site_features = self._site_tensor(X)
        targets = torch.from_numpy(labels).unsqueeze(1)
        standardised = self._standardise_reads(reads)
        n = len(labels)
        batch_size = int(self.params["batch_size"])
        cap = int(self.params["max_train_reads"])
        generator = torch.Generator().manual_seed(int(self.params["seed"]))
        self.history = []

        for epoch in range(int(self.params["epochs"])):
            self.network.train()
            order = torch.randperm(n, generator=generator).numpy()
            for start in range(0, n, batch_size):
                rows = order[start:start + batch_size]
                padded, mask = self._pad(reads, standardised, rows, cap)
                optimiser.zero_grad()
                logits = self.network(padded, mask, site_features[rows])
                loss = loss_fn(logits, targets[rows])
                loss.backward()
                optimiser.step()

            # Scoring every epoch is the expensive half, so it happens only when
            # a validation set was handed over - repetition 0 and nothing else
            # (docs/decisions/0021).
            if validation is None:
                continue
            valid_X, valid_y, valid_reads = validation
            self.history.append(
                {"iteration": float(epoch + 1),
                 **_scored("train", labels, self._score(X, reads)),
                 **_scored("valid", np.asarray(valid_y),
                           self._score(valid_X, valid_reads))}
            )

    # --------------------------------------------------------------- scoring

    def _score(self, X: pd.DataFrame, reads, batch_size: int = 256) -> np.ndarray:
        torch = _torch()

        self.network.eval()
        site_features = self._site_tensor(X)
        standardised = self._standardise_reads(reads)
        out = np.zeros(len(X), dtype=float)
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                rows = np.arange(start, min(start + batch_size, len(X)))
                # No read cap here: scoring uses every read a site has. A model
                # whose whole claim is that it reads the evidence should not
                # throw evidence away at the moment it is judged.
                padded, mask = self._pad(reads, standardised, rows, cap=None)
                logits = self.network(padded, mask, site_features[rows])
                out[rows] = torch.sigmoid(logits).squeeze(1).numpy()
        return out

    def predict_proba(self, X, reads=None) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Model is not fitted and no weights were loaded.")
        if reads is None:
            raise ValueError(
                "attention_mil scores from the reads, so predict_proba needs a "
                "ReadBlocks. Cross-validation passes one because the model "
                "declares CONSUMES_READS.\n"
                "scripts/predict.py does NOT: it streams sites through the "
                "feature extractor and never builds read blocks, so this model "
                "cannot be shipped as models/final/ yet. Recorded in GAPS.md; "
                "see docs/decisions/0023."
            )
        return np.clip(self._score(X, reads), 0.0, 1.0)

    # ----------------------------------------------------------- persistence

    def save(self, path) -> None:
        if self.network is None:
            raise RuntimeError("Nothing to save — fit the model first.")
        torch = _torch()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path / "model.pt")
        (path / "columns.json").write_text(json.dumps(self.columns, indent=2))
        (path / "mil.json").write_text(json.dumps({
            "params": self.params,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "read_mean": self.read_mean.tolist(),
            "read_scale": self.read_scale.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path):
        torch = _torch()

        path = Path(path)
        blob = json.loads((path / "mil.json").read_text())
        model = cls(**blob["params"])
        model.columns = json.loads((path / "columns.json").read_text())
        for field in ("mean", "scale", "read_mean", "read_scale"):
            setattr(model, field, np.asarray(blob[field], dtype=np.float32))
        model.network = model._build(len(model.columns))
        model.network.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        model.network.eval()
        return model
