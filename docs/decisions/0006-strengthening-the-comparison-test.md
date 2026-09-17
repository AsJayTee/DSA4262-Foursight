# 0006. Three additions to make "is this model actually better?" defensible

- **Date:** 2026-09-16
- **Status:** Accepted — all three built 2026-09-17, in the order
  [0009](0009-distributions-not-per-site-scores.md) set: repeated CV (#2) first,
  then the corrected t-test (#3), then the bootstrap (#1). How repeated CV is
  keyed and logged is [0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md),
  which this record left open.
- **Affects:** `src/m6a/compare.py`, `src/m6a/crossval.py`, `src/m6a/report.py`, `scripts/evaluate.py`, `--repeats`, `--bootstrap`

## Context

[0005](0005-paired-comparison-and-its-limits.md) established the paired
comparison and recorded its weakness: five folds, four degrees of freedom, and
training sets that overlap by 73.4%, which makes the p-value anti-conservative.

The harness is about to be used to compare *model architectures*, not just
feature sets. A test that can only detect large consistent effects, and whose
p-value reads stronger than it is, is not a good foundation for that.

Separately, there is no error bar on the headline number at all. `0.4759` is a
point estimate and nothing says how much it depends on which 121,838 sites
happened to be in the dataset.

There are two distinct sources of uncertainty and the current test conflates
them:

| source | question | addressed by |
|---|---|---|
| which sites are in our data | would this hold on a different sample of sites? | bootstrap over sites |
| which split we happened to use | would this hold under a different fold assignment? | repeated CV |

## Decision

Three additions, independently adoptable, in rough order of value per unit of
work.

### 1. Paired bootstrap over sites

Resample the 121,838 sites with replacement ~2,000 times. On each resample,
recompute both models' PR AUC and the difference. The spread gives a confidence
interval on both the headline number and the difference.

Must run **in-process, while the out-of-fold vector is still in memory**, with
only the resulting interval logged: per-site scores are no longer stored
([0009](0009-distributions-not-per-site-scores.md)). Lowest priority of the
three, because repeated CV already supplies a distribution and that is what the
interval was wanted for.

It gives us the thing we currently lack entirely: an error bar on 0.4759.

### 2. Seed-chained repeated cross-validation

`np.random.SeedSequence(4262).spawn(n)` yields independent child seeds derived
deterministically from 4262. Repetition 0 must use seed **4262 exactly**, so the
canonical split is preserved as rep 0 and every stored result stays valid and
comparable. Reps 1..n-1 are purely additive evidence.

This gets 50 paired observations instead of 5 without breaking AGENTS.md
section 3 — the rule is that the canonical split never changes, not that no
other split may ever be computed.

**The feature cache makes this affordable.** Features do not depend on the
split, so all repetitions read the same cached extraction and only the model
fits are paid for: ~10 min per model for 10 reps, unattended.

Consecutive integer seeds (4262, 4263, ...) are **not** an acceptable substitute
— they can produce correlated streams. Use `SeedSequence.spawn`.

### 3. Nadeau & Bengio corrected resampled t-test

The same paired t-test with the variance estimate inflated by a correction
accounting for train/test overlap. Roughly a one-line change to
`paired_comparison`, and it makes the fold-level p-value honest instead of
optimistic.

Report it **alongside** the uncorrected value during transition, not instead of
it, so the existing recorded numbers stay traceable.

## Why this and not the alternatives

**Dietterich's 5x2cv paired t-test.** The textbook fix: with 2 folds the two
training sets are completely disjoint, killing the overlap problem at its root.
Rejected as the primary route because each estimate trains on half the data, so
every number becomes worse and incomparable with everything already recorded.
Worth revisiting if the corrected t-test proves insufficient.

**Just report the win count and drop the p-value.** Tempting, and it is what
0005 tells people to read. But 5/5 tops out at p = 0.0625 two-sided, so on its
own it can never support a claim in the report.

**Break the seed and re-split freely.** Rejected: invalidates four people's
stored results, and does not fix the dependence anyway — every resample still
comes from the same 121,838 sites. More repetitions reduce split-specific luck;
they do not manufacture new data.

## Consequences

- None of these makes the estimates independent. One dataset sliced fifty ways
  is still one dataset. They narrow split-specific noise and make the stated
  uncertainty honest; they do not turn it into fifty experiments.
- Repeated CV multiplies fit time by the repetition count. Acceptable because
  extraction is cached, but it does mean `--repeats 10` should never be the
  default for a quick check.
- Once shipped, 0005's "do not quote p = 0.0465 unqualified" caveat should be
  revisited and this record's status updated to Accepted.

## How to check it still holds

The direction check, which is the one that catches the correction being applied
the wrong way round: **the corrected p-value must always exceed the uncorrected
one.** On `quantiles_v1` vs `pooled_v1` over the canonical five folds the
uncorrected paired test gives 0.0465 and the corrected test gives 0.1305. If you
ever see the corrected value come out smaller, the variance is being deflated
rather than inflated.

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --repeats 10 \
       --compare-features pooled_v1 --bootstrap 2000
```

Measured on that command (W&B run `y2lk7ilf`, full training set):

| | observations | mean difference | corrected p | wins |
|---|---:|---:|---:|---:|
| canonical 5 folds | 5 | +0.0148 | 0.1305 | 5/5 |
| 10 repetitions | 50 | +0.0164 | 0.000095 | 50/50 |

- `compare/features/p_value_corrected` > `compare/features/p_value`, always. At
  50 observations the naive test gives 1.1e-20 against the corrected 9.5e-5 —
  sixteen orders of magnitude, which is how badly the naive test misreads fifty
  correlated numbers.
- `rep/n_observations` = 50, and `fold/{0..4}/pr_auc` unchanged from the
  canonical split ([0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md)).
- `boot/difference/ci_low` and `ci_high` bracket the mean difference: measured
  +0.0145, 95% CI [+0.0074, +0.0217], positive in 100% of 2,000 resamples. The
  headline number itself comes out at 0.4761, 95% CI [0.4609, 0.4908].

**A trap this record did not anticipate.** The bootstrap interval on the
difference excludes zero decisively while the corrected 5-fold t-test does not.
That is not a contradiction and the bootstrap does not win: it **holds the split
fixed**, resampling sites against one set of fold models, so it is blind to
split-to-split variation - which is the entire thing the corrected test is
correcting for. Quoting the bootstrap in place of the corrected test would be
exactly the anti-conservatism this record exists to remove, arrived at by a
different route. The two are reported side by side and the printed output now
says so.

`corrected_variance` reduces to the naive `var(d)/n` only as `n_folds` goes to
infinity; at k = 5 it can never fall below `var(d)/4`, however many repetitions
are run. That floor is the point — repeating a split does not make the training
sets stop overlapping.
