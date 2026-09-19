# 0015. A comparison arm can be pulled from W&B, but only against a fingerprint

- **Date:** 2026-09-19
- **Status:** Accepted
- **Affects:** `src/m6a/tracking.py` (`dataset_fingerprint`, `fetch_observations`, `require_same_dataset`), `scripts/evaluate.py` (`--compare-run`), `scripts/train.py`, the run config

## Context

Every comparison so far **refits** the other arm: `--compare-features` and
`--compare-with` both fit five (now fifty) more models locally. That is correct
and it is also the reason comparing against last week's work means re-running
last week's work, on a machine where last week's instance no longer exists.

[0009](0009-distributions-not-per-site-scores.md) anticipated this — *"cross-run
paired comparison now pairs on metric vectors pulled from W&B"* — and
[0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md) keyed every
observation on `(repetition, fold)` precisely so it could be pulled and paired.
The keys were there; nothing read them back.

There is one thing standing in the way, and it is not the plumbing.
`crossval.assert_same_folds` protects a local comparison by refusing to pair two
arms that did not see the same sites in the same folds. Pulling an arm out of
W&B there is **nothing to check** — just a vector of numbers with no memory of
where it came from. Compare against a run from before a data refresh, or against
a smoke run, and you get a perfectly plausible p-value for a comparison nobody
made. That is the exact failure AGENTS.md section 3 exists to prevent, and it
would be silent.

## Decision

### 1. Every run records what it was computed on

`tracking.dataset_fingerprint` goes into the W&B run config on every run:

```
data_digest      content hash of the signal JSON
labels_digest    content hash of the labels file
data_limit       0 for the whole file, else the site limit
split_seed  split_n_folds  split_group_by
```

`data_limit` is `0` and not `None` for "all of it", because W&B drops `None`
from a run config — so an unlimited run would come back indistinguishable from
one that never recorded the field, and those two are treated very differently
below.

### 2. `--compare-run <id-or-name>` pairs without refitting

It pulls `rep/{r}/fold/{f}/pr_auc` out of the run's summary, falling back to
`fold/{f}/pr_auc` for runs predating repeated CV, and pairs them against the
local arm through the same `paired_comparison` as everything else.

A run **name** that matches more than one run is an error listing the candidates,
not a pick of the newest. Reaching for the wrong arm is the failure this record
is about.

### 3. It refuses rather than trusts

`require_same_dataset` compares all six fingerprint fields and exits on any
mismatch, naming which. A run that records none of them — anything logged before
this shipped — cannot be compared this way at all, and is told to use
`--compare-with`, which refits and checks the folds directly.

### 4. The bootstrap is not available this way, permanently

A pulled comparison gets the corrected paired test and the per-stratum tests
(once those vectors are logged). It cannot get the **paired bootstrap on the
difference**, which resamples sites and therefore needs both arms' per-site
scores in memory at once. 0009 deliberately does not store those.

This is a real capability boundary, not an oversight, and it is stated in the
`--compare-run` help text and printed in the run's output.

## Why this and not the alternatives

**Trust the run you name.** The whole point of `assert_same_folds` is that two
runs on different data produce two plausible numbers, and nothing downstream
notices. Extending a comparison across machines without extending the check
would have removed the one guard that makes a paired test meaningful.

**Compare on `oof/pr_auc` alone.** Simpler, and it is the unpaired comparison
[0005](0005-paired-comparison-and-its-limits.md) exists to stop people making.

**Store the per-site scores after all, so the bootstrap works cross-run.**
That reverses 0009 and costs 9 MB per run per person to recover one interval
that must not be used as significance anyway (0006, "a trap this record did not
anticipate").

**Match on the config name rather than a content hash.** A name says what
someone called it, not what it was. The data can change underneath a name, and
that is precisely the case worth catching.

## Consequences

- Any run logged before this record cannot be a `--compare-run` target. That
  includes everything currently in the project. Re-running them is the fix, and
  the error says so.
- The fingerprint costs one content hash of the input — about a second on the
  180 MB file, memoised per process, and already computed by the feature cache.
- Cross-run comparison rests entirely on both runs having used seed 4262. The
  fingerprint checks that they claim to have; it cannot check they did.
- A pulled arm contributes no figures of its own to the local run. It does not
  need to: its own run has them, and the `curve/*` series
  ([0014](0014-curves-are-series-not-only-images.md)) overlay in the same panel
  without anything being copied.

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --compare-run <run-id>
```

- Against a run with no fingerprint, it exits naming the missing fields.
- Against a run on different data or a different split, it exits naming which
  field differs.
- Against a **name** shared by two runs, it exits listing their ids.
- Against a valid run it reports `n observations, pooled PR AUC ...` and pairs
  them, having fitted nothing but the local arm.

All four checked on 2026-09-19; the first and third were found by the check
rather than by reading the code.
