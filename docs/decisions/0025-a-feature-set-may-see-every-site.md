# 0025. A feature set may see every site, through one optional second pass

- **Date:** 2026-09-24
- **Status:** Accepted
- **Affects:** `src/m6a/features/base.py` (`FeatureExtractor.finalise`), `src/m6a/feature_cache.py` (`extract`), `src/m6a/features/neighbourhood.py`

## Context

Every feature set in this repo is built from `site_features(site) -> dict`, one
site at a time. That is the right default: it is trivially parallel, it cannot
leak, and it is exactly what `scripts/predict.py` needs.

It also makes one family of features impossible to write, and that family now
looks like the most promising thing left.

**m6A clusters along transcripts, and it is not a small effect.** Measured on
the training set on 2026-09-24, over 116,505 ordered pairs of adjacent candidate
sites on the same transcript:

| | P(neighbour positive) | lift vs the 4.49% base rate |
|---|---:|---:|
| this site negative | 0.0337 | 0.75x |
| this site **positive** | **0.2937** | **6.54x** |

It decomposes into two separate effects, both real:

- **Transcript-level, 3.60x.** A random *other* site on the same transcript is
  positive 16.16% of the time when this one is. Positives-per-transcript carries
  **4.68x the binomial variance**; 57.1% of transcripts with at least ten
  candidates have zero positives and 4.9% are at least a quarter positive.
- **Local adjacency, 1.82x on top of that.** The immediate neighbour beats a
  random same-transcript site, 0.2937 against 0.1616, and the effect decays with
  distance as a regional effect should - 10.9x at 11-25 nt, 1.66x beyond 250 nt.

It is **not** a motif artefact. Conditioning on the neighbour's motif, it
survives in all eighteen: within `GGACT` neighbours, 19.1% if this site is
negative against 77.5% if positive; within `AGACT`, 6.2% against 51.3%.

GAPS.md recorded the transcript-GNN premise as untested and priced it at +0.004.
The premise is not merely true, it is one of the strongest associations in the
dataset - comparable to the motif itself. What is still untested is whether a
model can **exploit** it, which is what this record makes possible to find out.

## Decision

`FeatureExtractor` gains one optional method:

```python
def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Add columns computed from every site at once. Default: return frame."""
    return frame
```

It is called once, after every row has been built, by the two places that build
a feature table: `FeatureExtractor.transform` (the path `predict.py` uses) and
`feature_cache.extract` (the path training and evaluation use). The default
implementation returns the frame unchanged, so **every existing feature set is
unaffected and nothing needs editing**.

`frame` is indexed by `(transcript_id, transcript_position)` and `sites` carries
`kmer`, `motif` and `n_reads` on the same index.

### 1. It sees features and identity. It never sees labels

`finalise` is handed the feature table and the site table. Neither contains
`label`, `gene_id` or the fold assignment, and it is called during extraction,
which happens **before** the label join and before the split
(`crossval.build_datasets`). So a neighbour-derived column is a function of
measurements and coordinates only.

This is the guard that matters, because the 6.54x above is a **label-label**
correlation and the obvious way to use it is the one that cannot work: at
prediction time every site of a held-out gene is held out together, so no
neighbour label is available, ever. A feature set that reached for one would
score brilliantly out of fold and be unusable in `predict.py`. It is not
prevented by discipline; it is prevented by `finalise` not being given the
labels.

What is left is legitimate and is the actual hypothesis: if a neighbouring site
is genuinely modified, **its reads carry that signature**, so its features are a
noisy observation of its latent state, and aggregating them is evidence about
the region.

### 2. Aggregates over neighbours must exclude the site itself

Every aggregate is leave-one-out. A mean over a window that includes the centre
site partly re-reads that site's own features under a new name, which inflates
nothing measurable in cross-validation and is simply a worse feature.

### 3. Edges never cross a fold boundary, by construction

All edges here are within a transcript, therefore within a gene, and the split
groups on `gene_id`. So a neighbour is always in the same fold as its site and
no neighbour feature can carry information across the split. This is a property
of the data, not of the implementation, and it is why cross-site features are
safe here when they would not be on a graph that spanned genes.

### 4. The cache key already covers it

`finalise` is deterministic given the file, the feature set and the depth, all
of which are in the key ([0002](0002-feature-cache.md)). A `--limit N` run
computes neighbourhoods over the first N sites only, which is correct and is
keyed by `limit`. No `CACHE_VERSION` bump is needed for adding the hook, because
no existing feature set's output changes.

## Why this and not the alternatives

**Let the extractor override `transform`.** It is already a public method and
already does the whole job. Rejected because `feature_cache.extract` does *not*
call `transform` - it uses its own blocked accumulator to keep the memory peak
flat over eleven depths (0002). An extractor that overrode `transform` would
behave one way in `predict.py` and another in training, silently. `finalise` is
called by both, and the existing test that asserts the two paths agree covers it.

**A separate `CrossSiteExtractor` base class.** Cleaner typing, and it splits the
registry into two kinds of thing that configs would have to distinguish between.
The capability-flag pattern already in the repo (`CONSUMES_READS`,
`REPORTS_TRAINING_CURVE`) is for *models*, where the harness has to change what
it passes. Here nothing upstream changes: the hook is a no-op unless overridden,
so a flag would be a second way of saying the same thing.

**Compute neighbour features in `crossval`, after the label join.** That is where
`gene_id` and the folds live, so it is where a leak becomes possible, and it
would put feature engineering in the module that does the splitting. It would
also miss `predict.py` entirely.

**Precompute a neighbour table once and join it on.** Fine for structural
features (distance, density), which do not depend on the measurements. Useless
for neighbour *feature* aggregates, which differ per feature set and per depth -
and those are where the signal is.

**Do it as a model instead of a feature set.** A model receives `X` after the
fold split and never sees site coordinates, so it could not build the
neighbourhood. Same reason 0022 gives for why `train_depths` is not a model
parameter.

## Consequences

- **`predict.py`'s behaviour now depends on which sites are in the input file.**
  This is the real cost and it has no clean fix. A neighbour aggregate over a
  file containing every DRACH candidate on a transcript is not the same quantity
  as one over a file containing a third of them, so a feature set using
  `finalise` is sensitive to the *candidate density of the evaluation file* in a
  way `quantiles_v1` is not. Our training file has 22.8 candidate sites per
  transcript at a median spacing of 36 nt; if the graded evaluation file is
  built the same way this is benign, and if it is a subset it is not. **Anything
  shipped in `models/final/` on a `finalise` feature set has to state this.**
- `finalise` runs once per (feature set, depth), on the whole table. At 121,838
  rows and eleven depths that is eleven passes of a groupby-and-rolling over
  5,333 transcripts - seconds, against the ~90 s the parse costs.
- Memory: the hook receives the whole frame, so the blocked accumulator's flat
  peak (0002) no longer holds for extractors that use it. ~50 MB per feature
  table, so this is affordable, but a `finalise` that builds an N x N anything
  is not.
- `FeatureExtractor` grows a method, which is shared infrastructure on the
  frozen import path (AGENTS.md section 4). It adds **no imports** - pandas and
  numpy are already there - and the default body is `return frame`.
- The hook makes a whole family writable, including ones nobody has proposed.
  That is the point, and it is also the risk: a feature set that quietly
  aggregates over all sites is harder to reason about than one that reads a
  single `Site`. The rule is in section 1 and it is enforced by what `finalise`
  is handed, not by asking people to be careful.

## How to check it still holds

Nothing about an existing feature set may change:

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --profile quick --depth-sweep
```

must still print per-fold PR AUC 0.4548 / 0.5081 / 0.4685 / 0.4704 / 0.4855 and
pooled 0.4759, because `QuantileFeatures` does not override `finalise`.

Tests:

- `tests/test_evaluation.py::test_cache_returns_exactly_what_recomputing_would_have`
  is the load-bearing one and it already exists - it compares the cache path
  against `FeatureExtractor.transform`, which is exactly the two call sites this
  record adds the hook to. If `finalise` is wired into one and not the other, it
  fails.
- `tests/test_neighbourhood_features.py::test_finalise_never_receives_a_label_or_a_gene`
  - the guard from section 1, asserted on the columns actually passed.
- `::test_neighbour_aggregates_exclude_the_site_itself` - section 2.
- `::test_a_feature_set_that_does_not_override_finalise_is_unchanged` - the
  no-op default, byte-for-byte.
