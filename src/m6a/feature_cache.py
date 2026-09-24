"""On-disk cache for extracted site features.

One pass of `quantiles_v1` over the 121,838 training sites costs about 90
seconds, almost all of it gunzip and JSON parsing. A depth sweep needs one pass
per depth and a paired comparison needs one per feature set, so an uncached
evaluation run spends 10-15 minutes recomputing the identical thing. Cached, the
second run costs about a second.

**Where it lives:** `.cache/features/` in the repo root, or `$M6A_CACHE_DIR`.
It is gitignored. Delete the directory to reset it; nothing else depends on it.
Budget ~50 MB per (feature set x depth) on the full training set, so a full
depth sweep of one feature set is ~300 MB.

**What invalidates an entry:** the cache key is the feature set name, the
*content* hash of the input file, the depth, the subsample seed, the site limit,
and a format version. Content hashing costs about a second on the 180 MB file
and is memoised per process, which buys two things a size/mtime check does not:
a regenerated input can never hit a stale entry, and the directory can be copied
between machines. Editing a feature extractor is the one change the key cannot
see - bump CACHE_VERSION below, or pass --no-cache, when you do that.

Not imported by predict.py. numpy only: .npz keeps the cache free of a parquet
engine dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from m6a.data import (
    N_READ_FEATURES,
    SUBSAMPLE_SEED,
    ReadBlocks,
    iter_sites,
    subsample_reads,
)

# Bump when a feature extractor changes meaning, or when the storage layout
# below changes. The key cannot notice either on its own.
CACHE_VERSION = 1

INDEX_NAMES = ["transcript_id", "transcript_position"]
_BLOCK_SITES = 8192

_digest_memo: dict[tuple[str, int, int], str] = {}


@dataclass(slots=True)
class Extraction:
    """Features for every site in one file, at one depth.

    `sites` is indexed identically to `features` and always describes each site
    at its *true* depth, even in a subsampled entry. Stratifying a depth-1 score
    by the depth the site really had is the whole point of the sweep.
    """

    features: pd.DataFrame
    sites: pd.DataFrame
    depth: int | None
    from_cache: bool

    @property
    def columns(self) -> list[str]:
        return list(self.features.columns)


def cache_dir() -> Path:
    return Path(os.environ.get("M6A_CACHE_DIR", ".cache/features"))


def file_digest(path: str | Path) -> str:
    """Content hash of an input file, memoised on (path, size, mtime)."""
    path = Path(path)
    stat = path.stat()
    memo = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if memo in _digest_memo:
        return _digest_memo[memo]

    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    _digest_memo[memo] = digest.hexdigest()
    return _digest_memo[memo]


def _key(features: str, source: str, depth, seed: int, limit) -> dict:
    return {
        "version": CACHE_VERSION,
        "features": features,
        "source": source,
        "depth": depth,
        # The seed only bites when reads were actually dropped, so a full-depth
        # entry records None and stays valid when someone varies the seed.
        "subsample_seed": None if depth is None else seed,
        "limit": limit,
    }


def _path(key: dict) -> Path:
    stamp = hashlib.blake2b(
        json.dumps(key, sort_keys=True).encode(), digest_size=8
    ).hexdigest()
    depth = "full" if key["depth"] is None else "d%d" % key["depth"]
    return cache_dir() / f"{key['features']}_{depth}_{stamp}.npz"


def _load(key: dict) -> Extraction | None:
    path = _path(key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            if json.loads(str(stored["key"])) != key:
                return None  # hash collision, or a layout we no longer understand
            index = pd.MultiIndex.from_arrays(
                [stored["transcript_id"], stored["transcript_position"].astype(int)],
                names=INDEX_NAMES,
            )
            features = pd.DataFrame(
                stored["values"],
                columns=[str(c) for c in stored["columns"]],
                index=index,
            )
            kmers = [str(k) for k in stored["kmer"]]
            sites = pd.DataFrame(
                {
                    "kmer": kmers,
                    "motif": [k[1:6] for k in kmers],
                    "n_reads": stored["n_reads"].astype(int),
                },
                index=index,
            )
    except (OSError, ValueError, KeyError):
        # A truncated or half-written entry is a cache miss, never an error. The
        # only cost of ignoring one is recomputing what it held.
        return None
    return Extraction(features, sites, key["depth"], from_cache=True)


def _store(key: dict, extraction: Extraction) -> None:
    path = _path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = extraction.features.index
    temporary = path.with_suffix(".tmp")
    np.savez(
        temporary,
        key=json.dumps(key, sort_keys=True),
        values=extraction.features.to_numpy(dtype=np.float32),
        columns=np.asarray([str(c) for c in extraction.features.columns]),
        transcript_id=np.asarray(index.get_level_values(0), dtype=str),
        transcript_position=np.asarray(index.get_level_values(1), dtype=np.int64),
        kmer=np.asarray(extraction.sites["kmer"], dtype=str),
        n_reads=np.asarray(extraction.sites["n_reads"], dtype=np.int64),
    )
    # np.savez appends .npz to a path that lacks it; rename last so a killed run
    # leaves no half-written entry for the next run to read back.
    Path(str(temporary) + ".npz").replace(path)


class _Blocks:
    """Collects site_features() dicts into float32 blocks.

    FeatureExtractor.transform holds one dict per site and converts at the end,
    which peaks near a gigabyte on the full set: tolerable for the single pass
    predict.py makes, not for six depths at once. Blocking keeps the peak flat.
    """

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.rows: list[dict[str, float]] = []
        self.columns: list[str] | None = None

    def add(self, row: dict[str, float]) -> None:
        self.rows.append(row)
        if len(self.rows) >= _BLOCK_SITES:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        block = pd.DataFrame(self.rows, dtype=np.float32)
        if self.columns is None:
            self.columns = list(block.columns)
        elif list(block.columns) != self.columns:
            raise ValueError(
                "A feature extractor returned different keys for different sites. "
                "site_features() must return the same keys in the same order every "
                "time, or training and prediction silently misalign columns."
            )
        self.blocks.append(block.to_numpy(dtype=np.float32))
        self.rows = []

    def frame(self, index: pd.MultiIndex) -> pd.DataFrame:
        self.flush()
        if not self.blocks:
            raise ValueError("No sites to transform - is the input file empty?")
        return pd.DataFrame(np.vstack(self.blocks), columns=self.columns, index=index)


def extract(
    json_path: str | Path,
    features: str,
    depths: Sequence[int | None] = (None,),
    *,
    seed: int = SUBSAMPLE_SEED,
    limit: int | None = None,
    use_cache: bool = True,
    log=print,
) -> dict[int | None, Extraction]:
    """Extract `features` at each requested depth, reading the file at most once.

    Entries of `depths` are read counts; `None` means full depth. Whatever is not
    already cached is computed in a single streaming pass, because the gunzip and
    the JSON parse dominate: doing them once for six depths is six times cheaper
    than once per depth.
    """
    from m6a import registry  # local, so this module imports without the registry

    depths = list(dict.fromkeys(depths))
    source = file_digest(json_path) if use_cache else "uncached"
    keys = {depth: _key(features, source, depth, seed, limit) for depth in depths}

    out: dict[int | None, Extraction] = {}
    if use_cache:
        for depth in depths:
            hit = _load(keys[depth])
            if hit is not None:
                out[depth] = hit

    missing = [depth for depth in depths if depth not in out]
    if not missing:
        return out

    extractor = registry.get("features", features)()
    blocks = {depth: _Blocks() for depth in missing}
    kmers: list[str] = []
    n_reads: list[int] = []
    index: list[tuple[str, int]] = []

    shown = ", ".join("full" if d is None else str(d) for d in missing)
    log(f"  extracting {features} at depth [{shown}] from {Path(json_path).name} ...")

    for site in iter_sites(json_path, limit=limit):
        index.append(site.key)
        kmers.append(site.kmer)
        n_reads.append(site.n_reads)
        for depth in missing:
            view = site if depth is None else subsample_reads(site, depth, seed)
            blocks[depth].add(extractor.site_features(view))

    site_index = pd.MultiIndex.from_tuples(index, names=INDEX_NAMES)
    sites = pd.DataFrame(
        {"kmer": kmers, "motif": [k[1:6] for k in kmers], "n_reads": n_reads},
        index=site_index,
    )

    for depth in missing:
        # The cross-site pass (docs/decisions/0025). A no-op unless the extractor
        # overrides it, and it must be called here as well as in
        # FeatureExtractor.transform - those are the two paths that build a
        # feature table, and an extractor wired into only one of them would
        # behave differently in predict.py than in training, silently.
        extraction = Extraction(
            extractor.finalise(blocks[depth].frame(site_index), sites),
            sites, depth, from_cache=False,
        )
        if use_cache:
            _store(keys[depth], extraction)
        out[depth] = extraction

    return {depth: out[depth] for depth in depths}


# --------------------------------------------------------------------------
# the read-level cache
# --------------------------------------------------------------------------
#
# Separate from the feature cache, and keyed **without a depth**, because reads
# do not need one: every depth is a subset of the full-depth reads chosen by a
# hash (`m6a.data.subsample_blocks`), so one entry serves the whole sweep. The
# feature cache cannot do that - its rows are computed *from* the reads, and
# recomputing them is the 90 seconds it exists to avoid.
#
# It is big. 11,027,106 reads x 9 float32 is ~397 MB on the full training set,
# against ~50 MB for a feature entry. That is the cost of read-level modelling
# and it is why `with_reads` is opt-in rather than always on.

def _reads_key(source: str, limit) -> dict:
    return {"version": CACHE_VERSION, "kind": "reads", "source": source, "limit": limit}


def _reads_path(key: dict) -> Path:
    stamp = hashlib.blake2b(
        json.dumps(key, sort_keys=True).encode(), digest_size=8
    ).hexdigest()
    return cache_dir() / f"reads_{stamp}.npz"


def _load_reads(key: dict) -> ReadBlocks | None:
    path = _reads_path(key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            if json.loads(str(stored["key"])) != key:
                return None
            return ReadBlocks(stored["values"], stored["offsets"].astype(np.int64))
    except (OSError, ValueError, KeyError):
        return None  # a half-written entry is a miss, never an error


def _store_reads(key: dict, blocks: ReadBlocks) -> None:
    path = _reads_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    np.savez(
        temporary,
        key=json.dumps(key, sort_keys=True),
        values=blocks.values,
        offsets=blocks.offsets,
    )
    Path(str(temporary) + ".npz").replace(path)


def extract_reads(
    json_path: str | Path,
    *,
    limit: int | None = None,
    use_cache: bool = True,
    log=print,
) -> ReadBlocks:
    """Every read of every site, in file order, at full depth.

    Returned in the order `iter_sites` yields, which is the order `extract`
    builds its index in - so the two line up row for row before the label join,
    and `m6a.crossval` reindexes from there.
    """
    from m6a.data import read_blocks

    source = file_digest(json_path) if use_cache else "uncached"
    key = _reads_key(source, limit)
    if use_cache:
        hit = _load_reads(key)
        if hit is not None:
            return hit

    log(f"  reading every read from {Path(json_path).name} ...")
    blocks = read_blocks(iter_sites(json_path, limit=limit))
    megabytes = blocks.values.nbytes / (1 << 20)
    log(
        f"  {blocks.total_reads:,} reads over {len(blocks):,} sites "
        f"({megabytes:,.0f} MB, {N_READ_FEATURES} features each)"
    )
    if use_cache:
        _store_reads(key, blocks)
    return blocks


def clear(features: str | None = None) -> int:
    """Delete cached entries - all of them, or one feature set's. Returns the count."""
    directory = cache_dir()
    if not directory.exists():
        return 0
    pattern = "*.npz" if features is None else f"{features}_*.npz"
    removed = 0
    for path in directory.glob(pattern):
        path.unlink()
        removed += 1
    return removed
