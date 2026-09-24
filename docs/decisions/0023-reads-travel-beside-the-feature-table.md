# 0023. Reads travel beside the feature table, ragged, and a model opts in to seeing them

- **Date:** 2026-09-22
- **Status:** Accepted
- **Affects:** `src/m6a/data.py` (`ReadBlocks`, `read_blocks`, `kept_reads`, `subsample_blocks`), `src/m6a/feature_cache.py` (`extract_reads`), `src/m6a/crossval.py` (`Dataset.reads`, `build_datasets(with_reads=)`, `stack_rows`, `score_folds`), `src/m6a/models/base.py` (`CONSUMES_READS`), `scripts/train.py`, `scripts/evaluate.py`

## Context

The briefing frames m6A calling as Multiple Instance Learning: the **site**
carries the label, the individual reads do not, and only a fraction of the
reads at a modified site actually carry the modification
(docs/data.md#read-depth). GAPS.md records that no model in this repo treats
the reads as a set, and that a throwaway read-level probe
(`analysis/evaluation/scratch/mil_lite.py`) beats `quantiles_v1` badly at low
depth.

**The obstacle was never the network.** The pipeline is structurally
site-level, end to end:

```
FeatureExtractor.site_features(site) -> dict[str, float]     one row per site
Dataset(X: site x feature, y, folds, sites, columns)         one row per site
model.fit(X.loc[~holdout], y[~holdout])                      one row per site
```

`Site.reads` exists in `data.py` and is consumed by feature extraction and then
discarded. Nothing downstream of extraction can see an individual read, so
"add a MIL model" is not the additive change that adding a model usually is
(AGENTS.md section 1) — it is a change to shared infrastructure, which is why
this record exists.

Three constraints shaped it. The reads are **ragged** (20 to 991 per site here,
median 47). They are **big** — 11,027,106 reads x 9 float32 is ~397 MB against
49 MB for the whole feature table. And the fold split, the label join, the depth
sweep and depth-augmented training
([0022](0022-training-rows-may-come-from-several-depths.md)) all reorder or
subset rows, so whatever carries the reads has to survive all of that without
ever silently going out of step.

## Decision

### 1. `ReadBlocks`: one flat array plus offsets, CSR-style

```python
@dataclass(slots=True)
class ReadBlocks:
    values: np.ndarray    # (total_reads, 9) float32, sites concatenated in row order
    offsets: np.ndarray   # (n_sites + 1,) int64; site i is values[offsets[i]:offsets[i+1]]
```

A rectangular `(n_sites, max_reads, 9)` array would be 97% padding at the tail
(991 wide for a median of 47) and 4.3 GB. Two arrays and an index is how every
ragged structure worth using is stored.

It lives in `data.py`, on the frozen import path, because it needs nothing
beyond numpy and because a shipped MIL model would need `predict.py` to build
one. It carries **no site ids**: its whole contract is "row *i* here is row *i*
in the feature table".

### 2. That contract is established once, by position, and then checked

`ReadBlocks` cannot self-identify, so the one place the correspondence is made
is `crossval._reads_for`, after the label join. The join is an **inner** join —
it can drop rows and reorder them — so the reads are reindexed by
`index.get_indexer`, never by assuming the two stayed parallel.

Then it is verified: the resulting per-site read counts must equal `n_reads` in
the extracted site table, which was recorded during the streaming pass. A
mismatch raises. The failure this prevents is training a model on one site's
reads under another site's label, which produces a perfectly plausible number
and no warning — the same class of silent failure
[0015](0015-comparing-against-a-run-that-no-longer-exists.md) and
[0022](0022-training-rows-may-come-from-several-depths.md) guard against.

### 3. The read cache has no depth in its key, and needs only one entry

The feature cache stores one entry per (feature set, depth) because features are
computed *from* the reads and recomputing them is the 90 seconds
[0002](0002-feature-cache.md) exists to avoid. Reads are different: **every
depth is a subset of the full-depth reads, chosen by a hash**
([0003](0003-read-subsampling-in-data.md)). So the full-depth blocks are stored
once and any depth is derived in memory by `subsample_blocks`.

One ~397 MB entry serves the whole depth sweep. Storing per depth would have
been ~1.5 GB for the ten default depths, for nothing.

`subsample_reads` (a `Site`) and `subsample_blocks` (a `ReadBlocks`) now share
their draw through `kept_reads`, so the two cannot drift. 0003 asked for one
implementation, one seed, one meaning; two call sites is exactly how a repo ends
up with two.

### 4. `CONSUMES_READS`, the second capability flag

Same bargain as `REPORTS_TRAINING_CURVE`
([0021](0021-models-report-how-the-fit-progressed.md), renamed and scoped by
[0024](0024-the-training-curve-is-for-gradient-descent-models.md)), for the
same reason. A
model that sets it gets `fit(..., reads=)`, `predict_proba(X, reads=)`, and the
held-out reads as a third element of `validation`. A model that does not is
never handed them and need not mention them in its signature.

Cross-validation **raises** rather than scoring without reads when the flag is
set and the dataset has none. A MIL model silently falling back to site features
would look exactly like a MIL model that does not work, and the whole question
is which of those is true.

`build_datasets(with_reads=True)` is what loads them, and `train.py` and
`evaluate.py` decide by asking the model class before the data is built. Nothing
that does not need 397 MB pays for it.

### 5. Read-level data flows through everything row-based, not around it

- **The fold split**: `ReadBlocks.take(mask)`.
- **Depth-augmented training** ([0022](0022-training-rows-may-come-from-several-depths.md)):
  `stack_rows` returns stacked reads alongside the stacked frame, and refuses a
  source set where only some depths carry them.
- **The depth sweep**: `score_folds` passes the *subsampled* dataset's reads, so
  a MIL model feels fewer reads directly rather than through a summary computed
  over fewer of them. That is the sweep finally asking a read-level model the
  question it was built to answer.

## Why this and not the alternatives

**One row per read, with a site id column.** What the throwaway probe did, and
the reason it had to be a throwaway. It breaks the invariant everything
downstream rests on — `Dataset` rows are sites, and `folds`, `sites`, the
out-of-fold table, the strata and the calibration all index by it. Changing that
means changing all of them.

**Hand the model a callable that re-streams `Site`s.** Simple, and it re-parses
the 625 MB gzip stream once per fold — the cost [0002](0002-feature-cache.md)
exists to eliminate, reintroduced at the worst possible point.

**Pad to a rectangular array.** 4.3 GB, 97% of it padding, and the padding then
has to be masked at every layer anyway.

**Put `ReadBlocks` in `crossval.py` to keep `data.py` minimal.** Tempting, and
wrong in one specific way: a MIL model that was ever shipped would need
`predict.py` to build read blocks, and `predict.py` cannot import `crossval`
(AGENTS.md section 4). It needs numpy alone, so the frozen path can hold it.

**Give `ReadBlocks` the site index so it can check itself.** Considered. It
would make the structure self-validating and make it a pandas object on the
frozen path, and it would still need the positional lookup at the join. The
check in `_reads_for` gets the same guarantee without the weight.

**Make reads always loaded.** 397 MB and a slower build for every run in the
project, to serve the one model class that wants them.

## Consequences

- **~397 MB of RAM and ~397 MB of cache per dataset that carries reads**, on the
  full training set. `build_reads` peaks at roughly twice that during the final
  concatenation, because a single streaming pass cannot know the total in
  advance and counting first means parsing the stream twice.
- **The attention-MIL model is expensive, and the number is bad.** Measured on
  this machine (CPU, 8 threads) with `configs/mil.yaml`: about **3.4 s per epoch
  per fold on 4,000 sites**, of which roughly 1.5 s is the training step and
  2 s is the per-epoch curve scoring. Extrapolated to the full training set that
  is **~1 hour for repetition 0** and **~10 hours for a `standard` run's ten
  repetitions**. That is not a laptop job and it is barely a Ronin job at the
  default profile. Nobody should discover this after starting the instance,
  which is why it is here and in GAPS.md rather than in a commit message.
- `Dataset` grew a field, `stack_rows` returns three values instead of two, and
  `BaseModel.predict_proba` grew an optional parameter. All additive; every
  existing model and every existing config behaves identically.
- A model that declares `CONSUMES_READS` **cannot currently be shipped through
  `scripts/predict.py`**. `predict.py` calls `extractor.transform(iter_sites(...))`
  and then `model.predict_proba(features)` with no reads, and teaching it to
  build a `ReadBlocks` means either a second pass over the input or restructuring
  the one it makes. That is a real gap, it is recorded in GAPS.md, and it means
  the MIL model is a research arm rather than a candidate for `models/final/`
  until someone closes it.
- The read cache is invalidated by `CACHE_VERSION` like everything else, and
  `make clean-cache` removes it (the glob is `*.npz`). At 397 MB it dominates
  the cache directory, so "why is `.cache/` suddenly 1.5 GB" now has an answer.

## How to check it still holds

Nothing about a site-level run may change. The regression check from
[0021](0021-models-report-how-the-fit-progressed.md) — per-fold PR AUC
0.4548 / 0.5081 / 0.4685 / 0.4704 / 0.4855 on `configs/quantiles.yaml` — covers
that, because `with_reads` defaults to False and `stack_rows` on one source
returns exactly what indexing returned before.

```bash
python scripts/train.py --config configs/mil.yaml --smoke --quick
```

runs the whole read-level path on 5,000 sites: reads extracted and cached, joined
to the feature rows, split by fold, fed to the network, a per-epoch training
curve recorded, and a model saved and reloaded.

Tests:

- `tests/test_evaluation.py::test_read_blocks_hold_exactly_the_reads_the_sites_had`
  — the ragged structure round-trips, and `take` preserves order and content.
- `::test_subsampling_blocks_matches_subsampling_sites` — the two draws agree
  read for read, which is what `kept_reads` exists to guarantee.
- `::test_reads_follow_the_label_join_and_are_checked_against_the_site_table` —
  the join reorders and drops, the reads follow, and a mismatch raises.
- `::test_a_read_level_model_refuses_a_dataset_with_no_reads` — the flag fails
  loudly instead of scoring on site features and looking merely bad.
- `tests/test_smoke.py::test_predict_path_has_no_heavy_imports` and
  `::test_registry_finds_the_builtin_implementations` — `attention_mil` is
  discoverable without torch installed, because the import is inside the methods.
