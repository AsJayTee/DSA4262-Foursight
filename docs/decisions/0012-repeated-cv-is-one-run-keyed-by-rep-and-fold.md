# 0012. Repeated cross-validation is one W&B run, and its observations are keyed (repetition, fold)

- **Date:** 2026-09-17
- **Status:** Accepted
- **Affects:** `src/m6a/crossval.py`, `src/m6a/compare.py`, `src/m6a/tracking.py`, `scripts/evaluate.py`, `scripts/train.py`, the `rep/*` metric keys

## Context

[0006](0006-strengthening-the-comparison-test.md) #2 decided *that* we run
seed-chained repeated cross-validation, and why: five folds is five paired
observations, ten repetitions of five folds is fifty, and the canonical split
survives as repetition 0 so nothing already recorded is invalidated.

It did not decide two things that have to be settled before any of it is built,
and both of them are about what the result looks like afterwards rather than how
it is computed:

1. **One W&B run, or ten?** [0007](0007-evaluating-without-a-baseline.md) turned
   the run table into the thing you compare against when you have no baseline.
   That only works if one experiment is one row.
2. **How are the fifty numbers keyed** so that two experiments run on different
   days, on different instances, can still be paired? A paired test needs the
   two vectors aligned observation by observation. Aligning them wrongly does not
   raise — it produces a plausible p-value for a comparison nobody made.

## Decision

### 1. One W&B run per experiment

Ten repetitions go into one run, not ten. `--repeats 10` changes what a run
*contains*, never how many rows it occupies.

Ten rows per experiment would make the run table unfilterable exactly when it
finally has enough runs to be worth filtering, and W&B's run grouping is a
display setting — it does not make `oof/pr_auc` sortable across grouped runs.

### 2. The observation key is `(repetition, fold)`, and repetition 0 is the canonical split

Every metric vector is a list of observations, each carrying both ids. Two runs
pair if and only if their `(repetition, fold)` id lists match; `paired_comparison`
refuses otherwise rather than zipping two vectors of equal length and hoping.

Repetition 0 uses seed **4262 exactly** — not a derived child of it. That is what
AGENTS.md section 3 protects and what every number recorded so far was computed
on. Repetitions 1 onward come from `np.random.SeedSequence(4262).spawn(n - 1)`.

### 3. The flat metric keys already in use do not change meaning

| key | what it is |
|---|---|
| `fold/{0..4}/pr_auc` | **repetition 0 only** — the canonical split, unchanged |
| `rep/{r}/fold/{f}/pr_auc` | every observation, including repetition 0's |
| `rep/pr_auc_mean`, `rep/pr_auc_sd` | over all `n_repeats x n_folds` of them |
| `rep/n_repeats`, `rep/n_observations` | how many there were |

A run made before this shipped, and a run made after it with `--repeats 1`, emit
an identical key set. That is the point: the `fold/*` keys are a schema
([0007](0007-evaluating-without-a-baseline.md) section 4), and redefining
`fold/0/pr_auc` to mean "the mean over repetitions" would silently change the
meaning of every historical run rather than orphaning it — which is worse, because
nothing would look broken.

### 4. Features are extracted once; only the fold assignment is recomputed

Fold assignment depends on the labels and the seed, not on the features. So a
repetition costs five model fits and nothing else — no re-extraction, no cache
miss. This is what makes ten repetitions affordable at all, and it is why
`--repeats` is cheap on a warm cache and brutal on a cold one.

### 5. `--repeats` is opt-in and off by default

Default 1: the canonical split, exactly as today. Ten repetitions multiply fit
time by ten, which is the wrong default for a quick check
([0006](0006-strengthening-the-comparison-test.md) says so explicitly).

## Why this and not the alternatives

**One W&B run per repetition, grouped.** The obvious W&B-native answer, and it
loses on the thing 0007 was written for: ten rows per experiment, and
`oof/pr_auc` no longer sorts meaningfully across the project. Grouping is a view,
not an aggregation.

**Key observations by a flat index 0..49.** Simpler, and wrong in the way that
does not announce itself: observation 17 of one run is only observation 17 of
another if both used the same repetition count *and* the same iteration order.
Change `--repeats` between two runs and the pairing silently misaligns. The
(repetition, fold) key cannot misalign, because it names what it is.

**Consecutive integer seeds (4262, 4263, ...).** Rejected in 0006 and worth
restating: they can produce correlated streams, and correlated splits would make
the extra repetitions look like evidence while adding less than they appear to.

**Derive repetition 0 from the spawn chain like every other repetition.**
Cleaner-looking, and it breaks AGENTS.md section 3. `SeedSequence(4262).spawn(n)[0]`
is not seed 4262 — it is 3903649664 — so every stored per-fold number in
GAPS.md and every `fold/*` key in W&B would stop being reproducible by the code
that claims to reproduce them.

**Recompute the features per repetition.** Correct and pointless: the split does
not enter feature extraction. It would turn a ten-minute job into an hour.

## Consequences

- **`spawn(n)[i]` is stable in `n`**, checked rather than assumed: the first five
  children of `spawn(10)` are the first five children of `spawn(5)`, which are
  the children of `spawn(3)` for the first three. So a `--repeats 5` run and a
  `--repeats 10` run **can** be paired on the five repetitions they share, and
  `paired_comparison` allows exactly that intersection rather than refusing. This
  is the same property [0003](0003-read-subsampling-in-data.md) demanded of the
  subsample draw, for the same reason: a result must not move because someone
  extended the list.
- Repeated CV narrows split-specific noise. It does not manufacture data. Fifty
  observations still come from 121,838 sites, the training sets still overlap
  heavily, and the corrected t-test
  ([0006](0006-strengthening-the-comparison-test.md) #3) is what stops fifty
  correlated numbers reading as fifty independent ones. The two ship together
  and should be read together.
- The fifty numbers are logged as `rep/*` scalars and as one table. That is ~50
  floats, which is what [0009](0009-distributions-not-per-site-scores.md) says
  survives a run.
- `--repeats` is not free on a cold cache: the first repetition pays the
  90-second extraction like any run, and nine more repetitions then pay five
  fits each.

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --repeats 10 \
       --compare-features pooled_v1
```

All three checked on W&B run `y2lk7ilf` (full training set):

- `rep/n_observations` = 50, `rep/n_repeats` = 10, and `fold/{0..4}/pr_auc` still
  0.4548, 0.5081, 0.4685, 0.4704, 0.4855 — unchanged from the canonical split, so
  every number in GAPS.md survived. Repetition 0's observations are identical to
  the canonical per-fold vector, not merely close.
- `rep/pr_auc_sd` = 0.0229 over 50 observations against `fold/pr_auc_sd` = 0.0203
  over 5 — **larger**, as it must be: more splits sample more of the
  split-to-split variation rather than averaging it away. A repeated run that
  reported a *narrower* spread would mean the repetitions were correlated, which
  is exactly what the spawned seed chain exists to prevent.
- The corrected p-value exceeds the uncorrected one: 9.5e-5 against 1.1e-20
  ([0006](0006-strengthening-the-comparison-test.md)).

The payoff, for anyone deciding whether `--repeats` is worth 25 minutes:
`quantiles_v1` vs `pooled_v1` is **not** resolvable on five folds (corrected
p = 0.1305, interval spanning zero) and is comfortably resolved on fifty
(p = 0.000095, 50/50 wins). Same data, same models, same seed.

`tests/test_evaluation.py::test_subsampling_does_not_depend_on_iteration_order_or_the_depth_list`
is the equivalent guard on the other keyed draw in this repo.
