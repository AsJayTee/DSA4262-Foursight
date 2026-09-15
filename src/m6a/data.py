"""Reading the direct RNA-seq signal data and the m6A labels.

Data layout of dataset0.json.gz: one JSON object per line, shaped

    {"ENST00000000233": {"244": {"AAGACCA": [[9 floats], [9 floats], ...]}}}

one line per candidate site, one inner list per read aligned to that site. The
nine features per read are (dwell, sd, mean) for the -1, 0 and +1 positions of
the 7-mer window. The middle 5-mer is always one of the 18 DRACH motifs.

Sites are streamed rather than loaded at once: the training set holds 11M reads,
which is ~400 MB as float32 if you materialise it. Feature extractors consume
an iterable of Site and emit one row each, so memory stays flat.

This module must not import boto3 or wandb — predict.py imports it, and runs on
an evaluator's machine with neither installed. R2 access lives in m6a.r2.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

N_READ_FEATURES = 9
READ_FEATURE_NAMES = [
    "dwell_m1", "sd_m1", "mean_m1",
    "dwell_0",  "sd_0",  "mean_0",
    "dwell_p1", "sd_p1", "mean_p1",
]

LABEL_COLUMNS = ["gene_id", "transcript_id", "transcript_position", "label"]


@dataclass(slots=True)
class Site:
    """One transcript position and every read aligned to it."""

    transcript_id: str
    position: int
    kmer: str
    reads: np.ndarray  # (n_reads, 9) float32

    @property
    def n_reads(self) -> int:
        return int(self.reads.shape[0])

    @property
    def motif(self) -> str:
        """The central 5-mer — always one of the 18 DRACH motifs."""
        return self.kmer[1:6]

    @property
    def key(self) -> tuple[str, int]:
        return (self.transcript_id, self.position)


def resolve_data_dir() -> Path:
    """Where the raw data lives. Override with M6A_DATA_DIR for local work."""
    return Path(os.environ.get("M6A_DATA_DIR", "data/raw"))


def _open(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def iter_sites(path: str | Path, limit: int | None = None) -> Iterator[Site]:
    """Stream sites from a data.json (optionally gzipped).

    `limit` reads only the first N lines — used by `make smoke` so an agent can
    verify a new experiment runs in about a second instead of 20.
    """
    with _open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            transcript_id = next(iter(record))
            position_str = next(iter(record[transcript_id]))
            kmer = next(iter(record[transcript_id][position_str]))
            reads = np.asarray(
                record[transcript_id][position_str][kmer], dtype=np.float32
            )

            if reads.ndim != 2 or reads.shape[1] != N_READ_FEATURES:
                raise ValueError(
                    f"{path} line {i + 1}: expected (n_reads, {N_READ_FEATURES}) "
                    f"features, got shape {reads.shape}"
                )

            yield Site(transcript_id, int(position_str), kmer, reads)


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read data.info.labelled.

    Note this is the *labels* file. It is not m6Anet's data.info, which is a
    byte-offset index with the same name and a different format entirely.
    """
    labels = pd.read_csv(path)
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        raise ValueError(
            f"{path} is missing column(s): {', '.join(missing)}. "
            f"Expected header: {','.join(LABEL_COLUMNS)}"
        )
    return labels[LABEL_COLUMNS]


def assign_folds(
    labels: pd.DataFrame,
    seed: int = 4262,
    n_folds: int = 5,
    group_by: str = "gene_id",
) -> pd.Series:
    """Fold index per row, grouped so a gene never straddles a split.

    Transcripts of one gene share sequence and positions, so a random split
    leaks and inflates AUC. Implemented directly rather than via GroupKFold so
    the assignment depends only on (sorted group ids, seed) and not on a
    library version — the same seed gives the same folds on every machine, for
    everyone, forever.
    """
    if group_by not in labels.columns:
        raise ValueError(f"Cannot group by {group_by!r}; not a column in labels.")

    groups = np.sort(labels[group_by].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    fold_of_group = {group: int(order[i] % n_folds) for i, group in enumerate(groups)}
    return labels[group_by].map(fold_of_group).astype(int)


def align_to_features(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels onto a feature frame indexed by (transcript_id, position).

    Never relies on row order. The training files happen to be line-aligned,
    but evaluation data arrives without labels and in its own order.
    """
    keyed = labels.set_index(["transcript_id", "transcript_position"])
    joined = features.join(keyed, how="inner")
    if len(joined) == 0:
        raise ValueError(
            "No sites matched between the feature table and the labels. "
            "Check that the JSON and the label file describe the same dataset."
        )
    return joined
