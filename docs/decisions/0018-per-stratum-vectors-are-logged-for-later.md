# 0018. Per-stratum metric vectors are logged on every run, so a later run can test inside a stratum

- **Date:** 2026-09-21
- **Status:** Accepted
- **Affects:** `src/m6a/report.py` (`strata_observations`, `STRATA_TABLE`), `src/m6a/tracking.py` (`fetch_strata`), `scripts/evaluate.py`, `scripts/train.py`, the `strata_observations` W&B table

## Context

[0015](0015-comparing-against-a-run-that-no-longer-exists.md) built
`--compare-run`, which pairs against a finished W&B run without refitting it. It
could only do the **overall** test, because the only vectors a run published
were `rep/{r}/fold/{f}/pr_auc` — one number per observation over all sites.

[0008](0008-comparisons-report-both-arms-and-strata.md) made "is this better
*where we are weak*" the question the next round of work asks. Answering it
needs a metric per (repetition, fold) **within each stratum**, and those were
computed locally during a comparison and then thrown away.

The concrete thing this blocks: at full depth `lightgbm_quantiles` beats
`lightgbm_pooled`, and at depth 3 the ranking **reverses** (0.2458 against
0.2557). That reversal is currently a picture, not a result — there is no paired
test behind it, because the depth-sweep numbers are single pooled values per
depth with no per-fold vector underneath.

## Decision

### 1. Every standard run logs its per-stratum vectors

`strata_observations` computes, for each of the read-depth bands and DRACH
motifs, one metric per (repetition, fold), and logs the lot as a single W&B
table. Roughly 50 observations x 22 strata ~ 1,100 rows.

It runs on **every** run at `standard` or above, not only on comparisons. A run
that did not log them can never be a stratified comparison target, and the
moment to gather evidence is while the instance exists
([0007](0007-evaluating-without-a-baseline.md)).

### 2. One table, one flat name

The table is named `strata_observations`, with columns
`kind, stratum, repetition, fold, n, n_positive, pr_auc`.

**The name deliberately contains no slash.** A W&B table key containing `/` is
sanitised into an artifact name with a random suffix —
`run-abc123-stratifiedfeatures_depth-DptPFg` — which cannot be reconstructed
later, and the entire point is retrieving it by name from a different machine.
One table for both kinds rather than one per kind, for the same reason: fewer
names to get right.

### 3. A missing table degrades, it does not fail

`fetch_strata` returns `{}` for a run that has no table — one from before this
record, or one evaluated at `--profile quick`. `--compare-run` says so and
falls back to the overall comparison. An overall test is still worth having, and
refusing to do one because the finer test is unavailable would be a worse trade.

### 4. It comes with a figure

`fig/band_distribution` draws the PR AUC distribution per depth band, all
repetitions and folds. The granularity view: 50 points per band rather than one
number per band.

## Why this and not the alternatives

**Log the vectors as flat scalars.** 1,100 keys per run. The run table becomes
unusable, which is the thing [0007](0007-evaluating-without-a-baseline.md)
section 4 exists to protect.

**Compute them only when a comparison is requested.** What happened before, and
why the reversal is still untested: by the time you want the comparison, the run
you want to compare against is gone along with its instance.

**Store the per-site scores and recompute strata on demand.** Reverses
[0009](0009-distributions-not-per-site-scores.md) — 9 MB per run per person — to
recover something a 1,100-row table already provides.

**Name the table `stratified/observations` for tidiness.** The slash is exactly
what makes it unretrievable. Tidiness loses to being able to find it.

## Consequences

- ~1,100 extra rows per run. Well inside the "nothing over ~1 MB" rule 0009 set,
  and three orders of magnitude below the score table it replaced.
- Computing them costs 50 x `metrics_by` over ~24k sites each. Seconds, against
  the minutes the fits take.
- **Thin strata are dropped per fold rather than scored.** A band that clears
  `min_positive` in seven folds out of ten contributes seven observations, not
  ten, and `paired_comparison` pairs on the ids both arms share. So a stratum's
  observation count varies, and a low count is itself a signal that the stratum
  is too thin to argue from.
- Per-stratum p-values remain the weakest numbers the harness produces. This
  record makes them *available across runs*; it does not make them stronger. The
  multiplicity caveat from 0008 applies unchanged.

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --compare-run <run-id>
```

against a run logged after this record must print the overall paired test **and**
a `Paired within read depth` table, having refitted only the local arm.

Against a run logged before it, the same command must print the overall test and
a line saying the per-stratum vectors are not there — not an error.

Verified 2026-09-21 on runs `3bw2tvmm` (logged the table) and the round trip
through `tracking.fetch_strata`, which returned 6 depth strata and 15 motif
strata with 10 observations each.
