# 0006. Three additions to make "is this model actually better?" defensible

- **Date:** 2026-09-16
- **Status:** **Proposed — agreed, not built.** Nothing below exists in the code yet.
  Priority reordered by [0009](0009-distributions-not-per-site-scores.md): the
  corrected t-test (#3) is promoted to the tie-breaker for overlaid
  distributions, repeated CV (#2) is what makes those distributions real, and
  the bootstrap (#1) drops to last and must run in-process.
- **Affects:** will affect `src/m6a/compare.py`, `src/m6a/crossval.py`, `scripts/evaluate.py`

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

Not built. When it is: the bootstrap CI on the `quantiles_v1` vs `pooled_v1`
difference should exclude zero, consistent with the current 5/5 win count, and
the corrected t-test's p-value should be **larger** than the uncorrected 0.0465
— if it is smaller, the correction is applied the wrong way round.
