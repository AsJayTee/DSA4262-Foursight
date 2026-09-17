# 0009. Drop the out-of-fold table. Metrics are distributions, not per-site scores

- **Date:** 2026-09-16
- **Status:** Accepted — built 2026-09-17. Amends [0001](0001-retain-per-fold-metrics.md).
- **Affects:** `scripts/train.py`, `scripts/evaluate.py`, `src/m6a/crossval.py`, `src/m6a/figures.py`, `meta.json`

## Context

[0001](0001-retain-per-fold-metrics.md) decided two things:

1. **Keep per-fold metrics** instead of collapsing them to one pooled number.
2. **Write the full out-of-fold score table** (`oof.csv`, one row per site, ~9 MB)
   next to every model, so evaluation could be re-run without refitting.

The first stands and is the whole point of the harness. **The second no longer
earns its cost**, for two reasons that were not true when 0001 was written:

- [0007](0007-evaluating-without-a-baseline.md) makes `train.py` run the full
  evaluation inline. The questions the table existed to answer cheaply are now
  already answered and logged at the moment the run happens. Re-analysis was the
  justification; there is much less re-analysis to do.
- W&B is now the only durable store (instances are terminated with nothing pulled
  off them). It should hold figures and metric tables, not a 9 MB per-site CSV
  per experiment. That is the wrong granularity to push at it, repeatedly, from
  four people.

Separately, the right way to *show* five or fifty fold results is not a table of
numbers to squint at. It is a distribution.

## Decision

### 1. No out-of-fold table is stored

`oof.csv` is not written to disk and not uploaded. `evaluate.py --oof` and
`--compare-oof` are removed. The in-memory out-of-fold vector still exists during
a run — every metric, stratum, calibration figure and plot is computed from it
before it is discarded.

`crossval` keeps producing the table in memory; it simply stops being persisted.

### 2. Per-fold and per-repetition metrics are the unit that survives

What goes to W&B is the *metric vector*, not the scores behind it:

- with 5 folds: 5 PR AUCs, 5 ROC AUCs, per stratum too
- with repeated CV ([0006](0006-strengthening-the-comparison-test.md) #2,
  10 repetitions): 50 of each

Fifty floats instead of 9 MB, and it is everything a paired test needs.

### 3. Results are displayed as distributions

The primary figure for any run is the **distribution of the metric across folds
and repetitions** — a strip or violin with the individual points visible,
because with 50 points you should see them.

For a comparison, **overlay the two distributions on one axis.** That is the
plot that answers "is this better" at a glance, and it makes the fold-difficulty
problem visible rather than something you have to be told about.

### 4. The corrected t-test is the tie-breaker, and is promoted

When two overlaid distributions are too close to call by eye, the
Nadeau & Bengio corrected paired t-test ([0006](0006-strengthening-the-comparison-test.md) #3)
is the hard number that decides it. It is logged as a scalar
(`compare/{name}/p_value`, `compare/{name}/mean_difference`,
`compare/{name}/wins`) so it is sortable in the W&B run table.

This raises its priority. It is no longer a statistical nicety appended to the
comparison — it *is* the comparison, whenever the picture is ambiguous.

### 5. The bootstrap, if built, runs in-process

[0006](0006-strengthening-the-comparison-test.md) #1 resamples *sites*, so it
needs per-site scores. It can still be computed during the run, while the
out-of-fold vector is in memory, with only the resulting interval logged. It
drops to the lowest priority: repeated CV already supplies a distribution, which
is what the interval was wanted for.

## Why this and not the alternatives

**Keep `oof.csv` locally but do not upload it.** Pointless on a disposable
instance — the file dies with the machine, so it is storage with no reader.

**Upload it but compress / downsample it.** Rejected: complexity in exchange for
a capability nothing currently uses.

**Keep it only for the shipped model.** Considered. Rejected because the one
model you least need to re-analyse is the one already fully evaluated and
written up.

## Consequences — the real cost

**You can no longer ask a question of a finished run that you did not think to
ask during it.** Before, a stored table answered new questions in a second.
Now, a new question means re-running the experiment — and on a fresh instance
the feature cache is gone too, so that is a full download plus extraction plus
fit, not a one-minute refit.

This is an acceptable trade *only because* the standard profile is thorough
(0007 #3). It stops being acceptable the moment someone makes the default
profile leaner. **If you are tempted to trim `standard`, read this paragraph
first:** the two decisions are load-bearing for each other.

Also lost:

- **Site-level disagreement between two models** — "which sites does A get right
  that B does not" — is no longer computable after the fact. It was never
  supported, and it is the natural first question when considering an ensemble.
  Recorded in GAPS.md.
- **Cross-run paired comparison** now pairs on metric vectors pulled from W&B
  rather than on stored score tables. Adequate — a paired test never needed more
  than the per-fold numbers — but it means two runs can only be compared if both
  logged the same fold and repetition structure.

`meta.json` loses the `oof` key from 0001. Everything `predict.py` reads is
untouched, as ever.

## How to check it still holds

`crossval` must have no `save_oof`; `tests/test_evaluation.py::test_the_oof_table_reproduces_the_metrics_it_was_built_from`
asserts that alongside the guarantee that survived — the in-memory table still
reproduces the per-fold numbers reported with it.

`tests/test_evaluation.py::test_training_evaluates_inline_and_leaves_no_per_site_table`
is the other half: `meta.json` has no `oof` key and no `.csv` is written beside a
model.

No evaluation artifact over ~1 MB per run. On
[run y2lk7ilf](https://wandb.ai/dsa4262-team/dsa4262-project/runs/y2lk7ilf) the
report is 10 KB and the figures 0.37 MB; the 4.17 MB model artifact is the one
large upload and is deliberate (0007). The old `oof.csv` was ~9 MB per run.

The replacement for reading a score table after the fact is the metric vector in
the run table: `fold/{0..4}/pr_auc` pairs two runs from different sessions
without either of them still existing on disk.
