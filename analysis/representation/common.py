"""Data, splits and batching shared by every representation experiment.

Everything here reuses the harness rather than re-deriving it: the feature
table, the labels, the gene-grouped folds and the keyed read subsampling all
come from `m6a.crossval.build_datasets`, so a site is in the same fold and has
the same depth-3 reads here as in every recorded run. See PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# The harness's cache path is relative to the working directory, so scripts run
# from this folder silently built a second 700 MB cache here. Pin it.
os.environ.setdefault("M6A_CACHE_DIR", str(ROOT / ".cache" / "features"))

from m6a.crossval import build_datasets  # noqa: E402
from m6a.data import ReadBlocks  # noqa: E402
from m6a.feature_cache import extract  # noqa: E402

DATA_DIR = Path(os.environ.get("M6A_DATA_DIR", ROOT / "data0"))
DATA = DATA_DIR / "dataset0.json.gz"
LABELS = DATA_DIR / "data.info.labelled"
HAND_FEATURES = "quantiles_all_v1"
EVAL_DEPTHS = (1, 3, 10)
OUT = ROOT / ".cache" / "representation"       # embeddings: big, gitignored
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"

# dwell and signal sd at each position: positive, heavy-tailed, so logged.
LOG_COLUMNS = [0, 1, 3, 4, 6, 7]
BASES = "ACGT"
TRAIN_READS = 48     # reads per site per training step, at most


@dataclass
class Bundle:
    X: pd.DataFrame                 # hand features, full depth
    y: np.ndarray
    folds: np.ndarray
    genes: np.ndarray
    motifs: np.ndarray
    n_reads: np.ndarray
    kmer_onehot: np.ndarray         # (n_sites, 28) float32
    window_onehot: np.ndarray       # (n_sites, 3, 20) - the 5-mer in the pore at -1, 0, +1
    reads: ReadBlocks               # raw, full depth
    depth_reads: dict               # depth -> ReadBlocks, same rows
    depth_X: dict                   # depth -> hand features at that depth


def load(limit: int | None = None, with_reads: bool = True,
         extra_depths: tuple = ()) -> Bundle:
    wanted = tuple(dict.fromkeys([*EVAL_DEPTHS, *extra_depths]))
    depths = [None, *wanted]
    datasets = build_datasets(DATA, LABELS, HAND_FEATURES, depths, limit=limit,
                              with_reads=with_reads, log=lambda *a: None)
    full = datasets[None]
    kmers = extract(DATA, HAND_FEATURES, [None], limit=limit,
                    log=lambda *a: None)[None].sites.loc[full.X.index, "kmer"].to_numpy()
    return Bundle(
        X=full.X, y=full.y.astype(np.float32), folds=full.folds,
        genes=full.sites["gene_id"].to_numpy(), motifs=full.sites["motif"].to_numpy(),
        n_reads=full.sites["n_reads"].to_numpy(),
        kmer_onehot=onehot(kmers, 7),
        window_onehot=np.stack([onehot([k[i:i + 5] for k in kmers], 5) for i in range(3)], 1),
        reads=full.reads,
        depth_reads={d: datasets[d].reads for d in wanted},
        depth_X={d: datasets[d].X for d in wanted},
    )


def onehot(strings, length: int) -> np.ndarray:
    lookup = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(strings), length * 4), dtype=np.float32)
    for row, s in enumerate(strings):
        for pos, base in enumerate(s):
            out[row, pos * 4 + lookup[base]] = 1.0
    return out


def _unit(key: str) -> float:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") / 2.0**64


def split_roles(bundle: Bundle, fold: int) -> dict[str, np.ndarray]:
    """Boolean masks: encoder-training (A), its validation slice, probe-training (B), held out.

    Split by GENE, like the outer folds, so no transcript of a gene the encoder
    learned from is in the probe's training rows. The hash is fixed, so every
    encoder and every probe sees the same partition.
    """
    heldout = bundle.folds == fold
    u = np.array([_unit(f"4262:roles:{g}") for g in bundle.genes])
    v = np.array([_unit(f"4262:val:{g}") for g in bundle.genes])
    a = ~heldout & (u < 0.5)
    return {
        "encoder": a & (v >= 0.1),
        "encoder_val": a & (v < 0.1),
        "probe": ~heldout & (u >= 0.5),
        "heldout": heldout,
    }


# ---------------------------------------------------------------- reads

def transform_values(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    out[:, LOG_COLUMNS] = np.log(np.maximum(out[:, LOG_COLUMNS], 1e-6))
    return out


@dataclass
class Standardiser:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, blocks: ReadBlocks, rows: np.ndarray, max_reads: int = 2_000_000,
            seed: int = 4262) -> "Standardiser":
        """Fitted on the encoder's training reads only - never the held-out fold."""
        index = np.concatenate([np.arange(blocks.offsets[i], blocks.offsets[i + 1])
                                for i in np.flatnonzero(rows)])
        if index.size > max_reads:
            index = np.random.default_rng(seed).choice(index, max_reads, replace=False)
        values = transform_values(blocks.values[index])
        scale = values.std(axis=0)
        return cls(values.mean(axis=0), np.where(scale > 0, scale, 1.0).astype(np.float32))

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return ((transform_values(values) - self.mean) / self.scale).astype(np.float32)


class SetBatcher:
    """Draws (sites x reads) training batches, padded and masked.

    Each site shows a random subset of its reads: min(n, 48) half the time, a
    uniform 1..min(n, 48) otherwise (PLAN.md section 1). Drawn without
    replacement - duplicated reads would fake a correlation.
    """

    def __init__(self, standardised: np.ndarray, offsets: np.ndarray, seed: int):
        self.values = standardised
        self.offsets = offsets
        self.rng = np.random.default_rng(seed)

    def _subset(self, site: int, cap: int = TRAIN_READS) -> np.ndarray:
        start, stop = self.offsets[site], self.offsets[site + 1]
        n = stop - start
        top = min(n, cap)
        k = top if self.rng.random() < 0.5 else int(self.rng.integers(1, top + 1))
        return start + self.rng.choice(n, size=k, replace=False)

    def pad(self, picks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        width = max(len(p) for p in picks)
        out = np.zeros((len(picks), width, self.values.shape[1]), dtype=np.float32)
        mask = np.zeros((len(picks), width), dtype=bool)
        for row, pick in enumerate(picks):
            out[row, :len(pick)] = self.values[pick]
            mask[row, :len(pick)] = True
        return out, mask

    def batch(self, sites: np.ndarray):
        return self.pad([self._subset(s) for s in sites])

    def pair(self, sites: np.ndarray):
        """Two DISJOINT random read subsets per site, for the contrastive encoder."""
        first, second = [], []
        for s in sites:
            start, stop = self.offsets[s], self.offsets[s + 1]
            order = start + self.rng.permutation(stop - start)
            half = len(order) // 2
            for part, bucket in ((order[:half], first), (order[half:], second)):
                top = min(len(part), TRAIN_READS // 2)
                k = int(self.rng.integers(max(1, top // 4), top + 1))
                bucket.append(part[:k])
        return self.pad(first), self.pad(second)


def site_batches(offsets: np.ndarray, rows: np.ndarray, quadratic: bool = False,
                 max_sites: int = 512):
    """Batches of every read of the given sites, grouped by read count.

    For embedding, not training: every read is used. Sites are sorted by read
    count, so the site being added is always the widest in its batch, and the
    padded cost is (sites in batch) x width - or x width^2 for attention, whose
    cost grows with the square of the reads.
    """
    budget = 6_000_000 if quadratic else 60_000
    rows = np.asarray(rows)
    counts = np.diff(offsets)[rows]
    batch: list[int] = []
    for site in rows[np.argsort(counts, kind="stable")]:
        n = int(offsets[site + 1] - offsets[site])
        cost = (len(batch) + 1) * (n * n if quadratic else n)
        if batch and (cost > budget or len(batch) >= max_sites):
            yield np.array(batch)
            batch = []
        batch.append(site)
    if batch:
        yield np.array(batch)


def pad_all(standardised: np.ndarray, offsets: np.ndarray, sites: np.ndarray):
    picks = [np.arange(offsets[s], offsets[s + 1]) for s in sites]
    width = max(len(p) for p in picks)
    out = np.zeros((len(sites), width, standardised.shape[1]), dtype=np.float32)
    mask = np.zeros((len(sites), width), dtype=bool)
    for row, pick in enumerate(picks):
        out[row, :len(pick)] = standardised[pick]
        mask[row, :len(pick)] = True
    return out, mask


# ---------------------------------------------------------------- per-read codes -> site vector

def pool_codes(codes: np.ndarray, errors: np.ndarray | None, offsets: np.ndarray,
               threshold: float | None, n_reads: np.ndarray) -> np.ndarray:
    """Site vector from per-read codes: mean, sd, q10/q50/q90 of each code
    dimension, then reconstruction-error summaries, then log n. PLAN.md s2."""
    rows = []
    for i in range(len(offsets) - 1):
        c = codes[offsets[i]:offsets[i + 1]]
        parts = [c.mean(0), c.std(0), *np.quantile(c, [0.1, 0.5, 0.9], axis=0)]
        if errors is not None:
            e = errors[offsets[i]:offsets[i + 1]]
            parts.append(np.array([e.mean(), *np.quantile(e, [0.5, 0.9, 0.95]),
                                   (e > threshold).mean()]))
        parts.append(np.array([np.log1p(n_reads[i])]))
        rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------- outputs

def embedding_path(name: str, fold: int) -> Path:
    return OUT / name / f"fold{fold}.npz"


def save_embedding(name: str, fold: int, z: np.ndarray, depth_z: dict, info: dict) -> None:
    path = embedding_path(name, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, z=z, **{f"z_d{d}": v for d, v in depth_z.items()},
             info=json.dumps(info))


def load_embedding(name: str, fold: int) -> tuple[np.ndarray, dict, dict]:
    with np.load(embedding_path(name, fold), allow_pickle=False) as stored:
        depth = {d: stored[f"z_d{d}"] for d in EVAL_DEPTHS if f"z_d{d}" in stored}
        return stored["z"], depth, json.loads(str(stored["info"]))
