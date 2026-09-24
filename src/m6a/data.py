"""Reading the direct RNA-seq signal data and the m6A labels.

Data layout of dataset0.json.gz: one JSON object per line, shaped

    {"ENST00000000233": {"244": {"AAGACCA": [[9 floats], [9 floats], ...]}}}

one line per candidate site, one inner list per read aligned to that site. The
nine features per read are (dwell, sd, mean) for the -1, 0 and +1 positions of
the 7-mer window. The middle 5-mer is always one of the 18 DRACH motifs.

Sites are streamed rather than loaded at once: the training set holds 11M reads,
which is ~400 MB as float32 if you materialise it. Feature extractors consume
an iterable of Site and emit one row each, so memory stays flat.

Read subsampling lives here rather than in the evaluation code because it is a
property of the data, not of how we score it: the depth sweep needs it to build
low-depth *test* sets, and depth-augmented training will need the same function
to build low-depth *training* rows. One implementation, one seed, one meaning.

This module must not import boto3 or wandb — predict.py imports it, and runs on
an evaluator's machine with neither installed. R2 access lives in m6a.r2.
"""

from __future__ import annotations

import gzip
import hashlib
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

# Seed for read subsampling. This is NOT the split seed (4262, in config.py) and
# changing it does not invalidate stored fold assignments - it only redraws which
# reads a depth-sweep keeps. It is fixed and recorded in every report so a depth
# number can be re-derived exactly; vary it deliberately to check a result is not
# an artefact of one draw.
SUBSAMPLE_SEED = 4262


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


@dataclass(slots=True)
class ReadBlocks:
    """Every read of every site, ragged, in one pair of arrays.

    The pipeline is otherwise site-level: a `FeatureExtractor` turns one `Site`
    into one row of numbers, and nothing downstream of that can see an
    individual read. A Multiple Instance Learning model needs the reads
    themselves - the site carries the label, the reads do not, and only a
    fraction of the reads at a modified site actually carry the modification
    (docs/data.md#read-depth). This is how they travel.

    Sites have wildly different read counts (20 to 991 here), so a rectangular
    array would be mostly padding. Instead: one flat `values` array with every
    site's reads concatenated, and `offsets` saying where each site starts, the
    way a CSR matrix stores its rows. Site `i` is
    `values[offsets[i]:offsets[i + 1]]`.

    **The row order is the contract.** A ReadBlocks is only meaningful next to
    the feature table it was built with - site `i` here must be row `i` there.
    `m6a.crossval` reindexes it to match after the label join and checks the
    counts, because a silent misalignment would train a model on one site's
    reads under another site's label and report a plausible number.

    numpy only, so this stays importable on the frozen prediction path
    (AGENTS.md section 4).
    """

    values: np.ndarray    # (total_reads, N_READ_FEATURES) float32
    offsets: np.ndarray   # (n_sites + 1,) int64, ascending, offsets[0] == 0

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    @property
    def counts(self) -> np.ndarray:
        """Reads per site, in row order."""
        return np.diff(self.offsets)

    @property
    def total_reads(self) -> int:
        return int(self.values.shape[0])

    def site(self, i: int) -> np.ndarray:
        """One site's reads, as an (n_reads, 9) view - no copy."""
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    def take(self, rows) -> "ReadBlocks":
        """A new ReadBlocks holding only `rows`, in the order given.

        Accepts a boolean mask or an array of positions. Used for the fold
        split, and to reorder after the label join. Copies, because the reads of
        a subset are not contiguous in the original.
        """
        positions = np.asarray(rows)
        if positions.dtype == bool:
            positions = np.flatnonzero(positions)
        if positions.size and (positions.min() < 0 or positions.max() >= len(self)):
            raise IndexError(
                f"ReadBlocks.take was given a row outside 0..{len(self) - 1}. "
                "The reads and the feature table have gone out of step."
            )
        counts = self.counts[positions]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        if positions.size == 0:
            return ReadBlocks(
                np.empty((0, self.values.shape[1]), dtype=self.values.dtype), offsets
            )
        values = np.concatenate([self.site(int(i)) for i in positions])
        return ReadBlocks(values, offsets)


def read_blocks(sites: Iterator[Site] | list[Site]) -> ReadBlocks:
    """Collect an iterable of Site into one ReadBlocks, in the order given.

    Peaks at roughly twice the final size during the concatenation - about
    800 MB on the full training set's 11,027,106 reads. That is the price of a
    single pass; counting first would mean parsing the 625 MB stream twice.
    """
    chunks: list[np.ndarray] = []
    counts: list[int] = []
    for site in sites:
        chunks.append(np.asarray(site.reads, dtype=np.float32))
        counts.append(site.n_reads)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if not chunks:
        return ReadBlocks(np.empty((0, N_READ_FEATURES), dtype=np.float32), offsets)
    return ReadBlocks(np.concatenate(chunks), offsets)


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


def subsample_reads(site: Site, depth: int, seed: int = SUBSAMPLE_SEED) -> Site:
    """Return `site` with only `depth` of its reads, drawn without replacement.

    Depth is a count of distinct RNA molecules measured at this position, not
    repeated readings of one molecule - see docs/data.md#read-depth. Dropping
    reads therefore simulates a shallower sequencing run honestly: what is lost
    is evidence, which is exactly what makes low depth hard.

    A site with `depth` reads or fewer is returned unchanged. You cannot
    subsample upwards, so the sweep's low-depth rows are a mixture of genuinely
    subsampled sites and already-shallow ones; on this training set nothing is
    below 20 reads, so that only bites on external data.

    The draw is keyed on (seed, depth, transcript, position) rather than taken
    from one long RNG stream. That makes it independent of iteration order and
    of which depths you happen to ask for: `subsample_reads(s, 3)` is the same
    three reads whether you ran the sweep over [1, 3] or [1, 3, 5, 10, 20], and
    the same on every machine. A shared stream is reproducible only if nobody
    ever edits the depth list, which is not a property worth relying on.
    """
    keep = kept_reads(site.transcript_id, site.position, site.n_reads, depth, seed)
    if keep is None:
        return site
    return Site(site.transcript_id, site.position, site.kmer, site.reads[keep])


def kept_reads(
    transcript_id: str,
    position: int,
    n_reads: int,
    depth: int,
    seed: int = SUBSAMPLE_SEED,
) -> np.ndarray | None:
    """Which read indices survive subsampling to `depth`. `None` means all of them.

    The keyed draw itself, factored out so that subsampling a `Site` and
    subsampling a `ReadBlocks` cannot drift apart. docs/decisions/0003 asks for
    one implementation, one seed, one meaning; two call sites is exactly how a
    repo ends up with two.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if n_reads <= depth:
        return None

    key = f"{seed}:{depth}:{transcript_id}:{position}".encode()
    stream = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    rng = np.random.default_rng(stream)
    # Sorted so the kept reads stay in file order. Every feature we compute is
    # order-invariant, but a diffable subsample is worth the free sort.
    return np.sort(rng.choice(n_reads, size=depth, replace=False))


def subsample_blocks(
    blocks: ReadBlocks,
    keys: "pd.MultiIndex | list[tuple[str, int]]",
    depth: int,
    seed: int = SUBSAMPLE_SEED,
) -> ReadBlocks:
    """`blocks` thinned to `depth` reads per site, keyed exactly as `subsample_reads`.

    `keys` supplies each row's (transcript_id, transcript_position), in row
    order, because the draw is keyed on the site's identity rather than on its
    position in the file - that is what makes it independent of iteration order
    and of which depths were asked for (docs/decisions/0003).

    **This is why a read-level cache needs only one entry.** The site-level
    feature cache stores one file per depth because the features differ per
    depth and are expensive to recompute. Reads do not: every depth is a subset
    of the full-depth reads, selected by a hash, so the full-depth blocks are
    stored once and any depth is derived from them in memory.
    """
    rows = list(keys)
    if len(rows) != len(blocks):
        raise ValueError(
            f"subsample_blocks got {len(rows):,} keys for {len(blocks):,} sites. "
            "The reads and the site index have gone out of step."
        )
    chunks: list[np.ndarray] = []
    counts: list[int] = []
    for i, (transcript_id, position) in enumerate(rows):
        reads = blocks.site(i)
        keep = kept_reads(str(transcript_id), int(position), reads.shape[0], depth, seed)
        chosen = reads if keep is None else reads[keep]
        chunks.append(chosen)
        counts.append(chosen.shape[0])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if not chunks:
        return ReadBlocks(
            np.empty((0, blocks.values.shape[1]), dtype=blocks.values.dtype), offsets
        )
    return ReadBlocks(np.concatenate(chunks), offsets)


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
