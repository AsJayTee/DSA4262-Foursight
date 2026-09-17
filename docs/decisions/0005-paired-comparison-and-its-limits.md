# 0005. Compare two runs fold-by-fold, paired — and be honest about what that test can carry

- **Date:** 2026-09-16
- **Status:** Accepted
- **Affects:** `src/m6a/compare.py`, `evaluate.py --compare-features`

## Context

You change something and pooled PR AUC moves from 0.4614 to 0.4759. Is that real
or luck?

Two numbers cannot answer it. The five folds are the only replicates available —
but folds differ enormously in difficulty. The same model scores 0.4548 on fold
0 and 0.5081 on fold 1, a spread more than three times the effect being looked
for. Folds also differ in size (23,394 to 25,923 sites) and positive rate
(4.10% to 5.15%), and PR AUC is bounded below by the positive rate.

The team compared the gap to the fold-to-fold standard deviation, got 0.0148
against 0.0203, and concluded "noise". That is an unpaired test on a paired
design. Both models are scored on the *same* folds, so fold difficulty is common
to both and cancels in the difference.

On identical numbers: unpaired p = 0.31, paired p = 0.0465, all 5 folds positive.

## Decision

`paired_comparison` takes two runs' per-fold metrics and reports the per-fold
**differences**: mean difference, sd, paired t-test, 95% CI, and win count. The
unpaired framing is printed alongside so the gap between the two is visible
rather than asserted.

`crossval.assert_same_folds` refuses to pair two runs that were not scored on
the same sites with the same fold assignment, rather than trusting that they
were.

`--compare-features` holds the model and every hyperparameter fixed and varies
only the feature set, so a difference is attributable. `--compare-with` compares
two whole configs and says in its output that they may differ in more than one
thing.

## Why this and not the alternatives

**Compare pooled numbers against the fold sd.** The status quo. Wrong, and
demonstrably costly — see Context.

**Compare pooled numbers with no uncertainty estimate at all.** What the
pipeline invited by reporting only `metrics_oof`.

**A distribution-free test (sign / Wilcoxon) as the headline.** Honest, and it
is effectively what the win count reports — but with 5 folds the best possible
two-sided p is 0.0625, so it can never clear 0.05 and cannot be the only number.
Reported alongside instead.

## Consequences — read this part

**The p-value this produces is optimistic, and we ship it knowingly.**

The t-test assumes the five differences are five independent observations. They
are not. Each fold's model trains on the other four, so any two training sets
overlap heavily — measured on this data, **73.4% of fold 0's training rows are
also in fold 1's**. The five models are near-copies, so they agree with each
other more than five independent experiments would, the observed scatter
(sd 0.0116) understates the true uncertainty, and the p-value comes out smaller
than it should. This is the known anti-conservatism of the naive paired CV
t-test (Dietterich 1998; Nadeau & Bengio 2003).

Note the asymmetry: the **test** sets are properly independent — every site is
scored once, by a model that never saw its gene. It is the **training** sets
that overlap. Cross-validation protects against testing on what you trained on;
it does not hand you five independent experiments.

So:

- **Read the win count alongside the p-value.** 5/5 does not depend on the
  scatter estimate and is the more trustworthy signal.
- **Do not quote `p = 0.0465` unqualified in the report.** The effect is real;
  the p-value is softer than it reads.
- Five folds is four degrees of freedom either way. Only large, consistent
  effects are detectable.

[0006](0006-strengthening-the-comparison-test.md) is the agreed fix and is not
built yet.

Also: comparison is PR AUC only. It cannot say *which sites* two models disagree
about, which is the question behind "should we ensemble these".

## How to check it still holds

`tests/test_evaluation.py::test_pairing_finds_a_consistent_gain_that_the_unpaired_test_misses`
constructs the exact failure this decision is about — a small consistent gain
swamped by fold variation — and asserts paired finds it (p < 0.001) while
unpaired does not (p > 0.5).

`::test_pairing_does_not_bless_an_inconsistent_difference` guards the other
direction.
