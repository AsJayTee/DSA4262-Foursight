# 0021. A model may report how its fit progressed, and it opts in

- **Date:** 2026-09-22
- **Status:** Accepted — **sections 1, 2 and 3 amended by [0024](0024-the-training-curve-is-for-gradient-descent-models.md)**, which
  scopes the curve to gradient-descent models and renames the flag to
  `REPORTS_TRAINING_CURVE`. LightGBM no longer reports one. The measurements
  below were taken while it did and are kept; see 0024 for how to retake them.
- **Affects:** `src/m6a/models/base.py` (`TRAINS_ITERATIVELY`, `history`, `fit(..., validation=)`), `src/m6a/models/lightgbm.py`, `src/m6a/crossval.py` (`record_history`), `src/m6a/report.py` (`training_curve`), `src/m6a/figures.py`, `src/m6a/tracking.py`, the `fit/*` and `curve/train/*` keys

## Context

No model in this repo recorded anything about its own fit. `LightGBMModel.fit`
called `lgb.train()` with no `valid_sets` and no callbacks, so the booster came
back and the 400 or 600 rounds that produced it left no trace.

That makes one question unanswerable: **is `n_estimators: 600` right?** Every
metric the harness computes is computed on a *finished* model, so a budget that
is twice what the model needs and a budget that is half of it produce the same
shape of report. GAPS.md records, under Modelling, that no hyperparameter value
in `configs/` has ever been tuned — and a full search was designed, costed at
~10 hours per architecture, and deliberately dropped. A training curve is not a
search. It is the cheapest thing that can say whether a hand-chosen number is
absurd, and it costs one fit rather than six thousand.

It is also the prerequisite for anything with real epochs. A torch model whose
loss is never plotted is a model nobody can debug.

## Decision

### 1. An opt-in capability flag, not a signature every model absorbs

```python
class BaseModel:
    TRAINS_ITERATIVELY = False          # renamed REPORTS_TRAINING_CURVE by 0024
    def fit(self, X, y, groups=None, validation=None) -> None: ...
```

`cross_validate` passes the fold's held-out rows as `validation` **only** when
the flag is set. A model that leaves the flag alone is never handed one and is
free to declare `fit(self, X, y, groups=None)` with no `validation` parameter at
all — `LogisticModel` does exactly that.

The single fit call site inside cross-validation is `src/m6a/crossval.py`, and
it is the only place that reads the flag.

Models fill `self.history`: one dict per step, `{"iteration": i, "train_*":
..., "valid_*": ...}`. `CVResult` grows a `histories` list, one entry per fold.

### 2. The field is called `iteration`, never `epoch`

There are no epochs in this repo. LightGBM has boosting rounds and a linear
model has solver iterations; a neural net would have epochs. One name that all
three can fill honestly is better than a name that two of them have to lie
about, and the alternative — each model inventing its own x — makes the curves
un-overlayable, which is the whole point of logging them
([0016](0016-everything-overlayable-lives-under-curve.md)).

Iterations are 1-based, matching what `n_estimators` counts, so the x axis reads
against the config value with no off-by-one.

### 3. Both objectives are logged, not just the one being optimised

`curve/train/{train,valid}_logloss` **and** `curve/train/{train,valid}_pr_auc`.

LightGBM minimises binary logloss; this project ranks on average precision. The
round where the two stop agreeing is the thing worth seeing, and one curve
cannot show it. Measured on `configs/quantiles.yaml`, full training set, seed
4262:

| picked by | iteration | value |
|---|---:|---:|
| held-out PR AUC | 431 | 0.4821 |
| held-out logloss | 600 | 0.1387 |
| where the fit stopped | 600 | 0.4775 |

Held-out logloss is **still falling at round 600** while held-out PR AUC peaked
at 431 and has been drifting down since. A logloss curve alone would say "keep
going"; the metric the project is actually judged on says the last 170 rounds
cost 0.0046 of mean per-fold PR AUC.

**Correction, added 2026-09-22 after running the full battery.** The row above
is accurate and the framing around it was too simple. Held-out logloss here is
**not monotone** — it dips by round 2, climbs to a peak near round 50, then
declines:

| iteration | 1 | 2 | 10 | 50 | 200 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lightgbm_pooled` | 0.1657 | **0.1633** | 0.1934 | 0.2902 | 0.2363 | 0.1849 | – |
| `lightgbm_quantiles` | 0.1647 | 0.1622 | 0.1859 | 0.2646 | 0.2060 | – | **0.1387** |

That hump is the `is_unbalance` miscalibration becoming visible during the fit:
early predictions sit near the base rate, boosting sharpens them onto a
rebalanced scale that is systematically too high, and logloss degrades while
PR AUC improves. It is the same defect `calib/count_ratio` measures on the
finished model, caught in the act.

**The consequence is a defect in this record's own scalar.**
`fit/logloss_best_iteration` is a plain argmin, so it reports **2** for
`lightgbm_pooled` — correct arithmetic and a nonsense stopping point (PR AUC at
round 2 is 0.338 against 0.465 at round 400). It reports 600 for
`lightgbm_quantiles` only because that config runs long enough for the late
decline to dip back under the early minimum. The two numbers are not comparable
across runs and the key invites exactly the misreading section 5 warns about.

It has been left as logged rather than changed mid-battery — redefining a key
while runs are in flight is the schema hazard
[0007](0007-evaluating-without-a-baseline.md) section 4 exists to prevent, and
a half-consistent run table is worse than a documented sharp edge. The fix, when
someone takes it, is either to report the argmin *after* the hump or to log the
hump's height alongside it, and it needs a record of its own. Recorded in
GAPS.md; the panel guide
([docs/wandb-panels.md](../wandb-panels.md)) says it in the place people will
actually read it.

LightGBM's `average_precision` is the same quantity
`sklearn.metrics.average_precision_score` computes, which is what
`m6a.evaluation.metrics` reports as `pr_auc` — checked rather than assumed, they
agree to 1e-15 on the same booster and the same rows. (`pr_auc` is average
precision, not a trapezoidal area; the key name stays and the prose says so.)

### 4. Repetition 0 only, meaned across its five folds

One curve per run, not five and not fifty. Consistent with `fold/*` and the
depth sweep already being repetition-0 quantities
([0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md),
[0013](0013-every-run-is-a-distribution.md)).

`curve/train/valid_pr_auc_sd` carries the spread over folds so the mean is not
read as more precise than it is — it is wider than most of the movement along
the curve, which is the honest impression to leave.

This is also why `record_history` is off for repetitions 1..9: scoring the
held-out fold every round is the whole cost of this record, and paying it once
instead of ten times makes it affordable at the `standard` profile.

### 5. It is a diagnosis, and the report says so in as many words

The peak is read off the same held-out folds the run's headline PR AUC is
computed on. Copying that iteration count into a config and then quoting this
run's score is selection on the test set — the flat-tuning trap GAPS.md records
under Modelling, arrived at one parameter at a time instead of forty.

The section prints that caveat every time, and says what the curve *is* good
for: shape. Still climbing at the last round means the budget is too small; a
flat held-out curve under a still-rising training curve means the rest of the
budget is buying memorisation. Changing the config and comparing the two configs
paired, like any other pair of runs, is the legitimate route.

### 6. Keys

| key | what |
|---|---|
| `curve/train/iteration` | x: the boosting round |
| `curve/train/{train,valid}_logloss` | the objective being minimised |
| `curve/train/{train,valid}_pr_auc` | the objective being reported |
| `curve/train/valid_pr_auc_sd` | spread over the five folds |
| `fit/n_iterations`, `fit/n_folds` | what the curve covers |
| `fit/best_iteration`, `fit/best_valid_pr_auc` | argmax of held-out PR AUC |
| `fit/logloss_best_iteration`, `fit/best_valid_logloss` | argmin of held-out logloss |
| `fit/final_valid_pr_auc` | the value the run actually used |
| `fit/train_valid_gap` | final train PR AUC minus final held-out PR AUC |

A model with no curve emits **none** of them. `fit/best_iteration = 0` would
read as "the fit peaked immediately", which is a claim; a blank reads as "this
model does not train in steps", which is the truth.

## Why this and not the alternatives

**Change `fit`'s signature for every model and let each ignore what it cannot
use.** The obvious move, and the trap. A model that accepts an evaluation set
and silently discards it reports no curve and no error, and the missing panel
reads as "this model has no interesting curve" rather than "nobody wired it up".
The flag makes the capability a statement the model has to make. Setting it
without accepting the argument raises a `TypeError` naming the model, which is
the loud version of the same mistake.

**Early stopping instead of a curve.** Tempting — LightGBM has it built in — and
it is the same selection-on-the-test-set problem with the evidence hidden. Early
stopping on the held-out fold *changes the model* using the data the run is
scored on. A curve leaves the model alone and puts the decision in front of a
person.

**Log five fold curves rather than one mean.** Five lines per run, so three runs
is fifteen lines on a panel built for cross-run overlay. The sd band carries what
matters about the spread at a fraction of the clutter.

**A new top-level namespace for the curve instead of `curve/train/`.**
[0016](0016-everything-overlayable-lives-under-curve.md) reserved `curve/` for
exactly this: anything meant to be drawn next to another run's. A second
namespace would mean two rules for the same kind of object.

**Twin y-axes on one panel for logloss and PR AUC.** Rejected in the figure: the
two live three orders of magnitude apart, and a twin axis makes the crossing
point of two lines look meaningful when it is an artefact of where the axes were
put. Stacked panels sharing an x show the one real thing — the two best rounds
are different rounds.

**Gate the whole thing on a profile.** Rejected for the reason
[0007](0007-evaluating-without-a-baseline.md) section 3 gives: optional
evaluation is evaluation that does not happen, and a run without a curve cannot
be compared against one that has it. Restricting it to repetition 0 makes the
cost small enough that gating buys nothing.

## Consequences

- **Every fold fit of an iterative model is slower**, because LightGBM scores
  the training fold and the held-out fold at every round. Measured on the full
  training set with `configs/quantiles.yaml` (600 rounds, 5 folds, warm cache),
  two runs of each:

  | | five folds |
  |---|---:|
  | without the curve | 40.2s, 45.7s |
  | with the curve | 66.4s, 76.0s |

  **1.65x on the repetition that records it.** Only repetition 0 does, so at the
  `standard` profile the whole cross-validation goes from about 7.2 minutes to
  about 7.6 minutes — **roughly 6%**, which is what makes this affordable as a
  default rather than a flag. Scoring the *training* fold is most of that cost
  and it is paid deliberately: a held-out curve on its own cannot say whether a
  flat tail is the model running out of signal or running out of capacity.
- **Run history deepens from 201 steps to `n_estimators`** (600 for
  `quantiles.yaml`). The ragged-series machinery
  ([0016](0016-everything-overlayable-lives-under-curve.md) section 2) handles
  it with no change: `log_curve_series` walks the longest series and emits only
  the keys present at each step, so the 201-point ROC curve simply stops
  contributing after step 200. That was the one thing that could have made this
  expensive and it was already solved.
- The local report JSON grows by ~5 series x `n_estimators` points. On
  `quantiles.yaml` that is about 40 KB on top of the 263 KB
  [0019](0019-a-threshold-sweep-because-a-ranking-cannot-count.md) recorded.
  Still inside the ~1 MB rule.
- The final model in `scripts/train.py` is refitted on every row with nothing
  held out, so it is fitted with no `validation` and records no history. That is
  correct — there is nothing honest to plot — and it means the shipped model
  costs exactly what it used to.
- `curve/train/*` and `fit/*` join the metric schema
  ([0007](0007-evaluating-without-a-baseline.md) section 4). Adding keys is free;
  these names are an interface now.
- Runs logged before today have neither, and show as blanks.

## How to check it still holds

The regression check first, because it is the one that matters: passing a
`valid_set` must cost time and **nothing else**.

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --profile quick
```

on the full training set must still print per-fold PR AUC
0.4548 / 0.5081 / 0.4685 / 0.4704 / 0.4855 and pooled 0.4759. Checked: it does,
to four decimals and beyond.
`tests/test_evaluation.py::test_recording_the_training_curve_does_not_change_the_model`
is the enforced version — it fits the same folds with and without the history
and asserts the out-of-fold scores are **bit-identical**, not merely close.

Then:

- `tests/test_evaluation.py::test_a_model_that_does_not_train_iteratively_is_never_handed_a_validation_set`
  — the guard on the flag. Its `_PlainModel.fit` has no `validation` parameter,
  so the test passing *is* the assertion that nothing tried to pass one.
- `::test_the_training_curve_is_the_mean_over_the_canonical_splits_folds` — one
  curve per run, meaned, with the fold spread beside it, and the two objectives
  peaking at different iterations.
- `::test_a_model_with_no_curve_logs_no_fit_keys` — absent, not zero.
Verified end to end on run `8nmucix2`, later re-verified on `3fyr70t8`
(`configs/quantiles.yaml`, full training set, `standard`). **Both have since
been deleted** — [0024](0024-the-training-curve-is-for-gradient-descent-models.md)
stopped LightGBM reporting a curve, so no live run reproduces the checks below
and they have to be re-taken the way 0024 describes. What was observed:

- `run.scan_history(keys=["curve/train/iteration", "curve/train/valid_pr_auc"])`
  returns **600** rows, step 0 to 599, iteration 1 to 600. **Use
  `scan_history`, or `history(samples=...)`** — `run.history()` downsamples to
  500 points by default and a 600-point curve comes back short, which looks
  like a logging bug and is not one.
- `curve/roc/*` is still 201 rows. The deeper history costs the ROC curve
  nothing: `log_curve_series` emits only the keys present at each step
  ([0016](0016-everything-overlayable-lives-under-curve.md) section 2), so the
  ROC series simply stops contributing after step 200.
- No key beginning `curve/` appears in `run.summary`; `fit/best_iteration` = 431
  and `fit/logloss_best_iteration` = 600 do.
- `fold/{0..4}/pr_auc` = 0.4548, 0.5081, 0.4685, 0.4704, 0.4855 and
  `oof/pr_auc` = 0.4759, unchanged.
- `fig/training_curve` is logged beside the other eleven figures.

One thing the run showed that is worth recording elsewhere rather than here:
`fit/train_valid_gap` is **0.5188** — training-fold average precision 0.9963
against 0.4775 held out. The booster very nearly memorises its training fold.
That is a modelling observation, so it goes in GAPS.md.
