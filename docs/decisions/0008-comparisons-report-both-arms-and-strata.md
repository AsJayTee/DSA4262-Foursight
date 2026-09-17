# 0008. A comparison reports both arms in full, and can be made inside a stratum

- **Date:** 2026-09-16
- **Status:** Accepted — built 2026-09-17
- **Affects:** `scripts/evaluate.py`, `src/m6a/compare.py`, `src/m6a/report.py`, `src/m6a/figures.py`, the `arm/*` metric keys

## Context

Two limitations, both found by actually running the harness on two models
(`--config configs/lightgbm.yaml --compare-with configs/baseline.yaml`) rather
than by reading the code.

**The comparison arm is a second-class citizen.** `--compare-with` fits the
baseline, prints its per-fold PR AUC and the paired test, and discards
everything else. No calibration, no strata, no stored out-of-fold table. To find
out that `baseline_logistic` overcounts positives by 6.10x, the run had to be
repeated standalone as `--config configs/baseline.yaml`. Fitting the same five
models twice to get numbers that were available the first time is the same waste
[0001](0001-retain-per-fold-metrics.md) was written to stop.

**A difference cannot be tested inside a stratum.** The by-depth tables show
that at 304+ reads, `quantiles_v1` holds 9.8x lift where `pooled_v1` gets 8.9x
and logistic gets 6.5x. Whether that gap is real is not a question the harness
can answer — `paired_comparison` runs on overall PR AUC and nothing else.

That is the wrong limitation to have right now. The next round of work is
architectures aimed at **low read depth**, where the question is never "is this
better on average" but "is this better *where we are currently weak*". A model
that trades 0.01 of pooled PR AUC for a large gain at depth 3 is exactly what
Task 2 needs and exactly what today's comparison would reject.

## Decision

### 1. Both arms get a full report

A comparison evaluates both arms at the same profile and logs both in full -
same sections, same figures, same W&B keys
([0007](0007-evaluating-without-a-baseline.md)) - rather than reducing the
baseline to a column of per-fold numbers. Per-site tables are not stored for
either arm ([0009](0009-distributions-not-per-site-scores.md)).

### 2. Paired comparison within strata

`--compare-* --by depth` pairs the two runs **fold by fold within each stratum**
and reports a difference per band:

```
depth band   baseline  candidate  mean diff  wins  p
20-31          0.4414     0.4553    +0.0139   4/5  0.087
...
304+           0.3854     0.4236    +0.0382   5/5  0.021
```

Strata thin enough to be unstable are reported as counts with no test, on the
same rule `metrics_by` already uses (`--min-positive`, default 10).

### 3. Multiplicity is stated, not corrected

Testing ten depth bands means ten p-values, and at least one will look
significant by chance. The report says so in the output rather than silently
applying a Bonferroni correction, because these tests are strongly correlated
(the same models, the same folds, overlapping evidence) and a correction
calibrated for independent tests would be wrong in the other direction.

Treat per-stratum p-values as *descriptive* — where the difference concentrates
— and the overall paired test as the confirmatory one. This sits on top of the
dependence problem already recorded in
[0005](0005-paired-comparison-and-its-limits.md), so per-stratum p-values are
the weakest numbers the harness produces and should be labelled as such.

## Why this and not the alternatives

**Re-run the baseline standalone when you want its calibration.** The current
workaround. Doubles the fitting and relies on remembering.

**Compare on a single low-depth number instead of every band.** Simpler, but
picking the band after seeing the results is exactly how you manufacture a
significant finding. Report every band.

**Bonferroni across bands.** Rejected as the default — see 3. It can be computed
by anyone who wants it from the reported p-values.

**Wait for [0006](0006-strengthening-the-comparison-test.md) to land first.**
Considered and rejected on ordering: 0006 makes existing numbers more honest,
0008 makes a question answerable that currently is not. The next phase of work
needs the second more, and they do not conflict — 0006's corrected test applies
per stratum too once both exist.

## Consequences

- A comparison costs a full evaluation of both arms rather than a partial one of
  the baseline. With the feature cache the marginal cost is the model fits.
- Per-stratum tests will be noisy and will sometimes disagree with the overall
  test. That is information, not a bug, but it needs the caveat printed next to
  it or someone will quote a band p-value as a headline.
- The 304+ anomaly recorded in GAPS.md becomes testable: is the high-depth drop
  significantly smaller for `quantiles_v1` than `pooled_v1`, or is it the same
  drop in all three models?

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/lightgbm.yaml --compare-features pooled_v1
```

compares a config against its own feature set, and must report a mean difference
of **exactly** zero in every stratum, with no p-value (a zero-variance difference
has no t-test, and saying so is not the same as failing one). Checked: it does.

Two runs whose folds differ inside a stratum fail loudly —
`stratified_paired_comparison` passes each stratum through `paired_comparison`,
which refuses mismatched `(repetition, fold)` ids rather than pairing on the
overlap. A stratum that only one arm could score is named in the `Not tested`
line rather than dropped silently: a band missing from the table reads as "no
difference" to anyone skimming, and that is not what it means.

The baseline arm's own numbers land under `arm/<name>/` in W&B and `arms.<name>`
in the report JSON — so the 6.10x overcount that started this record is now
recorded by the comparison run that produced it, with no standalone re-run.

**Shipped detail this record did not specify:** the arm keys are prefixed
`arm/<name>/` rather than sharing the run's own key names, because one W&B run
cannot hold two values under `oof/pr_auc`. The run's headline keys stay
unprefixed and keep meaning the primary arm.
