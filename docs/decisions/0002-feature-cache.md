# 0002. Cache extracted features on disk, keyed on file contents

- **Date:** 2026-09-16
- **Status:** Accepted
- **Affects:** `src/m6a/feature_cache.py`, `.cache/features/`, `--no-cache` on train and evaluate

## Context

Extracting `quantiles_v1` over the 121,838 training sites costs about 90
seconds, almost all of it gunzip and JSON parsing rather than arithmetic.

A depth sweep needs one pass per depth, and a paired comparison needs one per
feature set. An uncached evaluation run therefore spends 10-15 minutes
recomputing the identical numbers. That is the difference between an evaluation
you run on every experiment and one you run when you remember to.

## Decision

Extracted features are cached in `.cache/features/` (gitignored, override with
`$M6A_CACHE_DIR`) as `.npz`.

The cache key is: **feature set name + content hash of the input file + depth +
subsample seed + site limit + a format version**.

Multi-depth extraction makes **one** streaming pass over the file and computes
every uncached depth during it, because the parse dominates and doing it once
for six depths is six times cheaper than once per depth.

## Why this and not the alternatives

**Key on path + size + mtime.** Cheaper (no read) but weaker in two ways that
matter. A regenerated input with a preserved mtime would hit a stale entry, and
the directory could not be copied or rsynced between machines. Content hashing
costs about a second on the 180 MB file and is memoised per process, which is
nothing against 90.

**Parquet instead of .npz.** Rejected: parquet needs pyarrow, which is not a
base dependency, and adding one to make a cache work is a bad trade. `.npz` is
numpy, which is already there.

**Cache in memory only, per process.** Does not help, because the expensive case
is running `evaluate.py` repeatedly from the shell.

**No cache; just make extraction faster.** The bottleneck is gzip and JSON
parsing of a 625 MB stream. There is no cheap win there.

## Consequences

- **The key cannot see you editing a feature extractor.** This is the one real
  hazard: change `quantiles.py` without bumping `CACHE_VERSION` and you silently
  get stale features and plausible wrong numbers. Bump `CACHE_VERSION`, run
  `make clean-cache`, or pass `--no-cache` when you touch an extractor.
- Budget ~50 MB per (feature set x depth) on the full training set; a full depth
  sweep of one feature set is ~300 MB. `make clean-cache` drops it. Nothing
  depends on the directory existing.
- Extraction here uses its own blocked accumulator rather than
  `FeatureExtractor.transform`, because `transform` holds one dict per site and
  peaks near a gigabyte — tolerable for the single pass predict.py makes, not
  for six depths at once. A test asserts the two produce identical output.
- A half-written entry is treated as a cache miss, never an error. Entries are
  written to a temporary name and renamed, so a killed run leaves nothing
  poisonous behind.

## How to check it still holds

`tests/test_evaluation.py::test_cache_returns_exactly_what_recomputing_would_have`
compares cached output against both a second uncached run and
`FeatureExtractor.transform`, which is the path `predict.py` uses and the one
that actually has to agree.

`::test_changing_the_input_file_invalidates_the_cache` covers the key.
