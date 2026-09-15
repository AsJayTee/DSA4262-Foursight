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
        }
