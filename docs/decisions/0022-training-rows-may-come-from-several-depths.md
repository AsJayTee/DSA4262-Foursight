# 0022. Training rows may be drawn at several read depths, and scoring never moves

- **Date:** 2026-09-22
- **Status:** Accepted
- **Affects:** `src/m6a/config.py` (`train_depths`), `src/m6a/crossval.py` (`stack_rows`, `cross_validate(train_on=)`), `src/m6a/report.py` (`depth_sweep`), `src/m6a/figures.py`, `scripts/train.py`, `scripts/evaluate.py`, `meta.json`, the run config

## Context

GAPS.md records, under Modelling, that a throwaway read-level probe beats
`quantiles_v1` badly at low read depth — 0.2692 against 0.1527 at one read, a
74% improvement in the regime that is a quarter of SG-NEx — and loses at full
depth. It records, in the very next entry, that **the comparison is confounded
and the control has not been run**:

> A read-level model is *implicitly trained at depth 1*, because each read is
> its own training row; it has no train/test mismatch at any depth. The
> site-level models were trained only at full depth.

So the table conflates two things: *read-level architecture*, and *trained at
the depth it is tested at*. Building an attention-MIL network to settle it would
answer a question nobody has established is the right one.

The control is far cheaper. Take the site-level model exactly as it is and train
it on rows drawn at the depths it will be tested at. Whatever that recovers is
what "trained at the depth it is tested at" was worth with the architecture held
fixed; whatever is left over is what the architecture might be worth.

[0003](0003-read-subsampling-in-data.md) anticipated this and put
`subsample_reads` in `data.py` rather than in the evaluation harness for exactly
this reason:

> The depth sweep uses it to build low-depth *test* sets; depth-augmented
> training — the obvious next experiment, and named in GAPS.md as the missing
> controlled comparison — needs the same function to build low-depth *training*
> rows.

The function was there. Nothing could reach it from the training side:
`cross_validate` fits on `X.loc[~holdout]` of exactly one `Dataset`.

## Decision

### 1. `train_depths`, a config key

```yaml
train_depths: [1, 3, 5, 10, null]   # null is full depth
```

The complete list of depths the **training** rows are drawn from. The default is
`[None]`, which is what every config did before this existed, so nothing already
recorded changes. `[1]` trains at depth 1 only — the direct analogue of a
read-level model's implicit training depth, and a control in its own right.

It is a config key and not a model parameter because the same model and the same
features trained at two different depths *is* the comparison. A model parameter
would make it invisible to `--compare-features`, which holds the model and its
parameters fixed.

### 2. `cross_validate(train_on=[...])`, and scoring never changes

`train_on` is a list of `Dataset`s to draw training rows from; empty means the
dataset being scored, which is the old behaviour to the row. `stack_rows`
concatenates each one's `~holdout` rows and tiles the labels.

**The held-out fold is always the full-depth one.** `oof/pr_auc`,
`fold/{0..4}/pr_auc`, the strata, the calibration and the thresholds therefore
mean exactly what they mean in every other run, and a depth-augmented run sorts
against the rest of the run table without a footnote. The depth sweep is where
the difference shows up, which is the right place for it.

Labels are simply repeated across depths. They come from m6ACE-Seq and not from
the nanopore reads, so dropping reads changes how much evidence a row carries
and not what is true about the site — [0003](0003-read-subsampling-in-data.md)
again.

### 3. The fold mask is positional, so alignment is checked rather than assumed

`build_datasets` already guarantees that every depth comes back in the same row
order with the same fold assignment, and raises if it does not.
`cross_validate` re-checks it on anything handed to `train_on`, because a
`Dataset` built some other way would train on the wrong sites and produce a
perfectly plausible number. Same reasoning as
[0015](0015-comparing-against-a-run-that-no-longer-exists.md): the failure is
silent, so the check is not optional.

This also means repeated cross-validation needs no special case. Re-splitting
([0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md)) changes
`dataset.folds` and nothing else; the extra depths are the same rows in the same
order, so the new mask applies to them unchanged.

### 4. The depth sweep is told what the model was trained at

The sweep's output said, unconditionally:

> Models are fitted on FULL-depth training folds and scored on read-subsampled
> copies of their own held-out fold.

That stopped being true the moment this shipped. The section and
`fig/depth_sweep`'s caption now both take the training depths and say what they
were, and a depth-augmented sweep says in as many words that **it is not the
same sweep** — the train/test mismatch it exists to measure is partly closed by
construction, which is the whole experiment.

A caption that is wrong about what was held fixed is worse than no caption,
because nothing about the numbers looks different.

`train_depths` also goes into `meta.json`, the W&B run config (as a string —
W&B drops `None`, the trap `data_limit` hit in
[0015](0015-comparing-against-a-run-that-no-longer-exists.md)) and the report's
`depth_sweep` block, so a run can always say how it was trained.

### 5. The shipped model is refitted the same way

`scripts/train.py`'s final refit on every row draws from the same depths the
fold models did. A shipped model trained differently from the models these
numbers describe is a model the numbers do not describe.

## Why this and not the alternatives

**A new feature set that does the subsampling.** Wrong layer: a
`FeatureExtractor` sees one `Site` and returns one row, so it cannot produce
five rows per site, and it would also change what `--compare-features` means.

**A wrapper model that augments internally.** A model receives `X` after the
fold split, so it never sees the reads and could not subsample them. It would
have to re-extract features inside `fit`, which is the 90-second parse
[0002](0002-feature-cache.md) exists to avoid, once per fold.

**A CLI flag instead of a config key.** Then the experiment is not reproducible
from the config, `--compare-with` cannot vary it, and the W&B run config does
not record it. Every other thing that defines an experiment here is in the YAML.

**Augment the held-out fold too.** Tempting for symmetry and it destroys
comparability: the headline number would no longer be measured on the same rows
as every other run, so nothing in the run table could be read against it. The
depth sweep already scores at reduced depth, on purpose, in a way that is
labelled.

**Weight the stacked copies, or subsample the stack back to the original size.**
Both are defensible and both add a knob to a control experiment whose job is to
be simple. Five unweighted copies is the plainest thing that answers the
question; if it turns out to matter, that is a finding worth its own record.

## Consequences

- **Training cost scales with the number of depths.** Five depths is five times
  the training rows per fold. On the full training set with
  `configs/quantiles_depth_augmented.yaml` that is 487,620 rows per fold against
  97,524. Feature extraction is unaffected — those depths are already extracted
  and cached for the depth sweep ([0002](0002-feature-cache.md)), so on a warm
  cache this costs fits and nothing else. On a **cold** cache it costs one
  streaming pass, which `build_datasets` was already making for the sweep.
- **The stacked rows are not new data, and the harness cannot tell.** The same
  97,524 training sites appear five times. Ordinary extra rows would shrink the
  variance of an estimate; these are five views of one site, correlated by
  construction. Nothing downstream knows that, so a model that reports a tighter
  training curve here is not better-evidenced — it has the same evidence
  arranged differently. This is the same class of caveat
  [0006](0006-strengthening-the-comparison-test.md) records about repeated CV:
  one dataset sliced more ways is still one dataset.
- The subsample draw is keyed, so the stacked rows are the same on every machine
  and do not move when the depth list changes
  ([0003](0003-read-subsampling-in-data.md)). They **do** move if
  `SUBSAMPLE_SEED` changes, which now affects training and not just the sweep.
  That is worth knowing before varying it: it used to be a safe knob.
- `train_depths` joins the config schema and the W&B run config. A config
  without it behaves exactly as before.
- A depth-augmented run's `depth/*` keys are **not** comparable with a
  full-depth run's as a like-for-like measurement of collapse — they are the
  answer to a different question, and comparing them is the point. The run
  config says which is which; the section and the figure say it in words.

## How to check it still holds

```bash
python scripts/train.py --config configs/lightgbm.yaml --smoke
```

must be unchanged in every respect — the default `train_depths` is `[None]` and
`stack_rows` on one Dataset returns exactly what indexing it would have. The
regression check in
[0021](0021-models-report-how-the-fit-progressed.md) covers the full-data
version of the same guarantee.

```bash
python scripts/evaluate.py --config configs/quantiles_depth_augmented.yaml \
       --compare-run 1cn5n37z
```

fits only the local arm and pairs it against the full-depth `quantiles_v1` run
without refitting that one ([0015](0015-comparing-against-a-run-that-no-longer-exists.md)).
The measured result is in the **What it showed** section below and in GAPS.md.

Tests:

- `tests/test_evaluation.py::test_stacked_training_rows_are_the_same_sites_at_several_depths`
  — the stack is `n_depths x` the rows, the labels are tiled to match, and no
  held-out site appears in it.
- `::test_one_training_set_is_indistinguishable_from_no_stacking` — the default
  path is byte-for-byte the old one.
- `::test_training_rows_in_a_different_order_are_refused` — the positional-mask
  guard, which is the silent failure this could otherwise produce.

## What it showed

Run on the full training set as
[psdwqobf](https://wandb.ai/dsa4262-team/dsa4262-project/runs/psdwqobf),
paired against [1cn5n37z](https://wandb.ai/dsa4262-team/dsa4262-project/runs/1cn5n37z)
with no refit. **It answers the question this record was built for, and the
answer is that the architecture was not doing the work.**

| reads per site | 1 | 3 | 5 | 10 | 20 | full |
|---|---:|---:|---:|---:|---:|---:|
| `quantiles_v1`, trained at full depth | 0.1527 | 0.2458 | 0.2991 | 0.3716 | 0.4277 | 0.4759 |
| **same model, trained at 1/3/5/10/full** | **0.2593** | **0.3434** | **0.3735** | **0.4166** | **0.4495** | **0.4844** |
| read-level probe (`mil_lite.py`) | 0.2692 | 0.3303 | — | 0.3453 | — | 0.3666 |

At one read the site-level model goes from 0.1527 to **0.2593** — a 70% gain,
and within 0.01 of the read-level probe that motivated all of this. At depth 3
it **passes** the probe (0.3434 against 0.3303). At depth 10 and at full depth
it is far ahead (0.4166 vs 0.3453; 0.4844 vs 0.3666).

So the probe's low-depth advantage was **training depth, not read-level
architecture**. GAPS.md said the two were conflated and that nobody knew which
was doing the work; it was the training depth, and the cheap control recovered
essentially all of it.

**And it costs nothing at full depth.** Paired over 50 observations against
1cn5n37z: mean difference **+0.0056**, **39/50** wins, corrected p = **0.2605**.
Not established as better — but not worse either, which is the result that
matters. The low-depth gain is not bought with high-depth performance.

Three caveats, all of which should travel with these numbers:

- **The depth-sweep rows are single values per depth with no per-fold vector
  underneath**, so there is no significance test on the +0.1066 at depth 1.
  [0018](0018-per-stratum-vectors-are-logged-for-later.md) records exactly this
  limitation. The effect is many times anything fold noise produces here, but it
  is an effect size and not a p-value.
- **The read-level row came from a different script with its own subsample
  draw** (`analysis/evaluation/scratch/mil_lite.py`), so treat the crossover as
  approximate. It is close enough to compare and it was not drawn together.
- **Both arms still inherit the depth >= 20 floor.** Subsampling simulates
  covariate shift and not label shift
  ([0003](0003-read-subsampling-in-data.md)), so none of this says what happens
  on genuinely shallow SG-NEx sites.

Per-stratum, the gain concentrates where it should: the 20-31 read band is
+0.0187 at 44/50 wins (corrected p = 0.0464, descriptive only — see
[0008](0008-comparisons-report-both-arms-and-strata.md) on multiplicity), and
every deeper band is flat.

**A side effect that looked like strong regularisation — and the claim was
wrong.** `fit/train_valid_gap` ([0021](0021-models-report-how-the-fit-progressed.md))
is **0.0561** here against **0.5188** for the plain run, and this record
originally read that as "five views of each site stop the booster memorising any
one of them".

**That comparison is not valid, and the panel shows why.** For this run the
training rows are drawn at depths 1/3/5/10/full and the held-out fold is full
depth *only*, so the model is scored on **easier rows than it trained on**. The
giveaway is that its training logloss sits **above** its held-out logloss —
0.2880 against 0.2479 — which essentially never happens when the two come from
one distribution:

| run | train logloss | valid logloss | train AP | valid AP | gap |
|---|---:|---:|---:|---:|---:|
| `lightgbm_quantiles` | 0.0538 | 0.1387 | 0.9963 | 0.4775 | 0.5188 |
| `lightgbm_pooled` | 0.1300 | 0.1849 | 0.8550 | 0.4653 | 0.3897 |
| `mlp_quantiles` | 0.2739 | 0.2995 | 0.5861 | 0.4803 | 0.1057 |
| **`quantiles_depth_augmented`** | **0.2880** | **0.2479** | 0.5429 | 0.4868 | 0.0561 |

So the gap shrank for two reasons mixed together: the model may genuinely
memorise less, and the training set certainly got harder — depth-1 rows are ones
this model only reaches 0.259 AP on. Nothing here separates them.
`fit/train_valid_gap` is an overfitting measure **only when `train_depths` is
`full`**, where the first three rows above are comparable with each other.

Whether depth augmentation regularises is therefore **unknown and untested**.
The clean measurement is the depth-augmented model's average precision on
full-depth *training* rows, which nothing currently logs. Recorded in GAPS.md.

This is the same mistake section 4 of this record exists to prevent, made one
level up: the depth sweep's caption was fixed so it could not claim the wrong
thing about what was held fixed, and then a scalar was compared across two runs
that were not holding the same thing fixed. It was caught by a teammate looking
at the training-curve panel and asking why one run's train line was above its
valid line.

**A bug this run caught.** `scripts/evaluate.py` built its W&B run config by
hand and did not include `train_depths`, so the first version of this run was
indistinguishable in the run table from a full-depth one — on exactly the
dimension that separates them. Fixed, and backfilled on `psdwqobf` and
`1cn5n37z` so the two runs this record cites are correctly labelled.
`scripts/train.py` was never affected: it passes `config.as_dict()`, which
carries the key.
