"""Experiment configuration.

One YAML file is one experiment is one W&B run. Adding an experiment means
adding a YAML here and (usually) one new module under features/ or models/ —
never editing an existing file. See AGENTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The gene-level split seed. DO NOT VARY THIS. If two people use different
# seeds their metrics are not comparable, and nothing will warn them.
DEFAULT_SEED = 4262


@dataclass
class SplitConfig:
    seed: int = DEFAULT_SEED
    n_folds: int = 5
    group_by: str = "gene_id"


@dataclass
class Config:
    name: str
    features: str
    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    split: SplitConfig = field(default_factory=SplitConfig)
    notes: str = ""
    # Which read depths the *training* rows are drawn from. `None` is full
    # depth, and `[None]` - the default - is what every config did before this
    # existed. A list of several stacks one copy of each training site per
    # depth; a list without `None` trains at reduced depth only.
    #
    # It is a property of the experiment rather than of the model or the feature
    # set, which is why it is a config key and not a model param: the same model
    # and the same features trained at different depths is exactly the
    # comparison it exists for. See docs/decisions/0022.
    train_depths: list[int | None] = field(default_factory=lambda: [None])

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}

        missing = [k for k in ("name", "features", "model") if k not in raw]
        if missing:
            raise ValueError(
                f"{path} is missing required key(s): {', '.join(missing)}. "
                "Every config needs at least: name, features, model."
            )

        return cls(
            name=raw["name"],
            features=raw["features"],
            model=raw["model"],
            model_params=raw.get("model_params") or {},
            split=SplitConfig(**(raw.get("split") or {})),
            notes=raw.get("notes", ""),
            train_depths=parse_train_depths(raw.get("train_depths"), path),
        )

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for W&B config logging."""
        return {
            "name": self.name,
            "features": self.features,
            "model": self.model,
            "model_params": self.model_params,
            "split_seed": self.split.seed,
            "split_n_folds": self.split.n_folds,
            "split_group_by": self.split.group_by,
            # A string, because W&B drops None from a run config - the same trap
            # `data_limit` hit in docs/decisions/0015. "full" and "1,3,full" are
            # both filterable; [None] and a missing key are indistinguishable.
            "train_depths": describe_train_depths(self.train_depths),
        }


def parse_train_depths(raw: Any, path: Any = "config") -> list[int | None]:
    """Validate `train_depths`, keeping order and dropping duplicates.

    `null` / `"full"` both mean full depth. Anything else has to be a positive
    integer, because it is handed to `m6a.data.subsample_reads` as a read count.
    """
    if raw is None:
        return [None]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            f"{path}: train_depths must be a non-empty list, e.g. "
            "[1, 3, 5, 10, null]. `null` means full depth."
        )
    out: list[int | None] = []
    for item in raw:
        if item is None or (isinstance(item, str) and item.lower() == "full"):
            depth: int | None = None
        else:
            try:
                depth = int(item)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}: train_depths entry {item!r} is not a read count. "
                    "Use positive integers, or null for full depth."
                ) from None
            if depth < 1:
                raise ValueError(
                    f"{path}: train_depths entry {item!r} is below 1. A site "
                    "cannot be subsampled to fewer than one read."
                )
        if depth not in out:
            out.append(depth)
    return out


def describe_train_depths(depths: list[int | None]) -> str:
    return ",".join("full" if d is None else str(d) for d in depths)
