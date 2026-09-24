# 0024. The training curve is for models that learn by gradient descent

- **Date:** 2026-09-22
- **Status:** Accepted. Amends [0021](0021-models-report-how-the-fit-progressed.md) sections 1, 2 and 3.
- **Affects:** `src/m6a/models/base.py` (`REPORTS_TRAINING_CURVE`, was `TRAINS_ITERATIVELY`), `src/m6a/models/lightgbm.py`, `src/m6a/models/mlp.py`, `src/m6a/models/mil.py`, `src/m6a/crossval.py`, `docs/wandb-panels.md`

## Context

[0021](0021-models-report-how-the-fit-progressed.md) gave every model that
trains in steps a training curve, and argued that the field should be called
`iteration` rather than `epoch` so that "a booster and a neural net can both
fill it honestly". Both halves of that are true. The conclusion drawn from them
was not.

The panel built from it, once the full battery had run, held **eight lines**:
four models x (train, valid). Three of those models are LightGBM and one is a
neural net, and their x axes are not the same quantity:

| | one iteration is | runs to |
|---|---|---:|
| `lightgbm_pooled` | one boosting round — one tree added | 400 |
| `lightgbm_quantiles` | one boosting round | 600 |
| `quantiles_depth_augmented` | one boosting round | 600 |
| `mlp_quantiles` | one **epoch** — a full pass over ~97,000 rows | 40 |

A boosting round and an epoch are different units of work by orders of
magnitude. Drawn on a shared axis they invite "the network converged 15x
faster", which is not a fact about anything. The MLP's line is also visibly
jagged where the boosters are smooth, because minibatch SGD is noisy and
boosting is deterministic — a second difference that reads as a property of the
models when it is a property of the training procedure.

The name was honest and the **panel** was not. Honesty about a field name does
not make two quantities comparable.

## Decision

**The training curve is for models that learn by gradient descent.** LightGBM
does not report one.

`TRAINS_ITERATIVELY` is renamed **`REPORTS_TRAINING_CURVE`**, because the old
name would now be a lie on `LightGBMModel`: it *does* train iteratively and is
choosing not to record it. A flag that states a falsehood is a flag someone
will "fix". The new name states a policy, which is what it is.

`LightGBMModel.fit` goes back to `fit(self, X, y, groups=None)` — no
`validation` parameter at all, so handing it one raises rather than being
quietly ignored, which is the same loudness rule 0021 section 1 established.

`curve/train/*` and `fit/*` are unchanged as a schema. They are simply filled by
fewer models, which the keys already tolerated: `baseline_logistic` never
emitted them, and 0021 section 6 already specifies absent-not-zero.

## Why this and not the alternatives

**Keep both families and split into one panel per family.** The obvious fix, and
it relies on every reader remembering to filter before drawing a conclusion. The
whole reason this project logs everything under one schema is so that comparison
is correct *by construction* ([0007](0007-evaluating-without-a-baseline.md)
section 3). A panel that is only correct when filtered is the opposite of that.

**Keep both and rely on the caption.** Tried, for about an hour. The caveat was
written into `docs/wandb-panels.md` and the panel still drew eight lines the
moment anyone added the obvious two Y keys.

**Normalise the x axis — rounds as a fraction of the budget, say.** Makes the
curves overlay and makes them mean less: "50% of the way through 600 rounds"
and "50% of the way through 40 epochs" are not comparable either, and the raw
iteration count is the number you would act on.

**Add a `--fit-curve` flag so a booster's curve is available on request.**
Genuinely useful and deliberately not built here — it is a new CLI surface for a
diagnostic nobody runs routinely, and the one-line local edit below does the
same job. Worth revisiting if anyone reaches for it twice.

## Consequences

- **A diagnostic is lost for the three booster configs**, and those are the
  candidates for the graded deliverable. 0021 existed to answer "is
  `n_estimators` right?", and for LightGBM it no longer can. **The measurements
  taken while it could are recorded in GAPS.md**, with their provenance, because
  they are no longer regenerable by any committed command:

  | config | `n_estimators` | held-out PR AUC peaks at | overfit gap |
  |---|---:|---:|---:|
  | `lightgbm_pooled` | 400 | 390 | 0.3897 |
  | `lightgbm_quantiles` | 600 | **431** | 0.5188 |
  | `quantiles_depth_augmented` | 600 | 548 | 0.0561 (not comparable — see [0022](0022-training-rows-may-come-from-several-depths.md)) |

  To take the measurement again: set `REPORTS_TRAINING_CURVE = True` on
  `LightGBMModel`, run, read it, and **do not commit the edit**.

- **The logloss hump is a booster observation and is now frozen too.** 0021's
  correction records that held-out logloss dips by round 2, peaks near round 50
  and then declines — the `is_unbalance` miscalibration visible during the fit.
  That stands as a measurement and stops being reproducible from the default
  code. The torch models show the same shape; `mlp_quantiles` ends at train
  logloss 0.2739 against valid 0.2995, so the effect is not LightGBM-specific.

- **LightGBM fits are faster again.** Passing a `valid_set` made it score the
  held-out fold every round, which cost 1.65x on the one repetition that
  recorded it, or about 6% of a `standard` run. That comes back.

- **The three booster runs that carried `fit/*` and `curve/train/*` have been
  re-run and deleted.** A run table where some boosters have those keys and
  later ones do not is exactly the inconsistency
  [0007](0007-evaluating-without-a-baseline.md) section 4 warns about, and the
  code change alone does not fix it — a run's history is already written, so the
  old runs would have kept drawing on the panel this record exists to clean up.
  Their replacements are `5vdhkur4` (`lightgbm_pooled`), `1cn5n37z`
  (`lightgbm_quantiles`) and `psdwqobf` (`quantiles_depth_augmented`); each was
  checked against its predecessor on 16 metric keys before the old one was
  deleted, and each carries zero curve points.

  **Verifying that takes care.** `run.scan_history(keys=["curve/train/iteration"])`
  returns a row for every history step with the key null-filled where it is
  absent, so a clean run comes back with 208 *rows* and 0 *values*. Count
  non-null values, not rows, or a clean run looks like a dirty one.

  Only `mlp_quantiles` (`1wzvr5hi`) draws on the training-curve panel now, which
  is the intended end state.

- `fit/logloss_best_iteration` was already recorded in GAPS.md as a badly-shaped
  scalar (a plain argmin over a non-monotone curve). Restricting the curve to
  gradient-descent models does not fix that; it reduces how many runs it can
  mislead anyone on.

## How to check it still holds

`tests/test_evaluation.py::test_the_booster_reports_no_training_curve` — asserts
`REPORTS_TRAINING_CURVE is False`, that `fit` has no `validation` parameter at
all, that cross-validation produces empty histories, and that no `fit/*` key
reaches the flat metrics.

`::test_recording_the_training_curve_does_not_change_the_model` is the guarantee
0021 shipped, moved to where it now lives: the MLP's out-of-fold scores must be
**bit-identical** with and without per-epoch curve scoring. It is skipped where
torch is not installed, because torch is not a dev dependency.

`::test_a_model_that_does_not_train_iteratively_is_never_handed_a_validation_set`
is unchanged and still covers the flag's contract.

On a run: `mlp_quantiles` and `attention_mil_quantiles` carry `curve/train/*`;
no `lightgbm_*` run made after today does.
