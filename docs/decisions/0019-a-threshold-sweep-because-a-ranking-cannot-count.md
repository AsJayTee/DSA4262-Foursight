# 0019. A threshold sweep, because a ranking cannot count

- **Date:** 2026-09-21
- **Status:** Accepted
- **Affects:** `src/m6a/evaluation.py` (`threshold_table`, `operating_points`, `count_swing`), `src/m6a/report.py` (`thresholds`), `src/m6a/figures.py`, `src/m6a/tracking.py`, the `threshold/*` and `curve/threshold/*` keys

## Context

Every metric in this harness is threshold-free. PR AUC and ROC AUC integrate
over all thresholds; `metrics_by` does it inside a stratum; the depth sweep does
it at each depth. That is the right property for **comparing** models, and it is
exactly the wrong one for **using** one.

Task 2 has to count modified sites per cell line. A count requires calling each
site modified or not, and a call requires a number to compare the score against.
Nothing in the repo produced that number, and GAPS.md recorded the gap in as
many words: *"No resampling, alternative objective, or threshold work has been
tried"*, and *"There is no way to score an ensemble, a rule combining two
models, or a threshold choice."*

The nearest thing to an answer was 0.5, which nobody had checked. On a model
that overcounts positives by 1.80x when its scores are summed
([0007](0007-evaluating-without-a-baseline.md) and the calibration section),
0.5 is not a neutral default — it is an unexamined one.

## Decision

### 1. `threshold_table` computes the sweep, in `evaluation.py`, with no new imports

Precision, recall, F1, the number of sites called positive and the rate, on the
**same 201-point unit grid** the ROC and PR curves already use
([0014](0014-curves-are-series-not-only-images.md)). Scores are probabilities,
so a threshold means the same thing in every run and two runs' curves overlay
exactly.

It is pure numpy on one sorted copy of the scores — a `searchsorted` per
threshold — so the whole grid costs one sort. That matters because
`evaluation.py` is on `predict.py`'s import path and **may not grow a single
import** (AGENTS.md section 4, [0004](0004-comparison-stats-outside-evaluation.md)).
Reaching for `sklearn.metrics.precision_recall_curve` would have been the
obvious move and would have added a name to the one module in the repo that
cannot afford one.

A site is called positive when `score >= threshold`. Where nothing is called,
precision and F1 are **NaN**, not 1.0: a model that predicts nothing has no
precision, and sklearn's convention of calling that perfect would put a 1.0 at
the top of the run table.

### 2. Three named operating points, none of them a recommendation

| point | what it is |
|---|---|
| `f1_max` | balances the two errors. A default, not a finding — nothing here says precision and recall cost the same |
| `count_matched` | calls as many sites as really are positive. The point a **count** wants, and it says nothing about whether the right sites were picked |
| `half` | 0.5, reported so the cost of the unexamined default is on the page |

Measured on the full training set, `configs/quantiles.yaml`, seed 4262:

| point | threshold | precision | recall | F1 | sites called |
|---|---:|---:|---:|---:|---:|
| `f1_max` | 0.58 | 0.4849 | 0.5571 | 0.5185 | 6,290 |
| `count_matched` | 0.65 | 0.5133 | 0.5149 | 0.5141 | 5,492 |
| `half` | 0.50 | 0.4513 | 0.5976 | 0.5143 | 7,250 |

5,475 sites really are positive. **0.5 overcounts by 1.32x** — and note that is
a *different* number from the 1.80x overcount obtained by summing the scores.
Both are the same miscalibration seen from two angles, and they are not
interchangeable: one is what you get by adding up probabilities, the other by
thresholding them. Quoting either as "the" overcount is wrong.

### 3. The count curve is the headline, and it is logged as a series

`curve/threshold/*`, under the namespace
[0016](0016-everything-overlayable-lives-under-curve.md) reserved for anything
meant to be drawn next to another run:

| key | what |
|---|---|
| `curve/threshold/threshold` | x: the decision threshold |
| `curve/threshold/precision`, `/recall`, `/f1` | the trade-off |
| `curve/threshold/predicted_positives` | **the one that matters** |
| `curve/threshold/predicted_rate` | the count as a fraction, which is what transfers to a dataset of another size |
| `curve/threshold/log10_predicted_positives` | companion, because a W&B line panel has no log scale (0016 section 3) |

`threshold/count_swing` is the scalar summary: **2.1x** between threshold 0.3
(10,610 sites) and 0.7 (4,943). Both are thresholds someone could pick with a
straight face, so that ratio is the span of an unforced choice — not a
statistical interval, and not something a confidence interval would ever show.
Any Task 2 claim of the form "cell line X has N modified sites" has to carry it.

### 4. Ragged where the curve stops meaning anything

`predicted_positives` falls to zero at the top of the range, where precision and
F1 are undefined and `log10` of the count is `-inf`. Those three series are
**truncated** to the region where something is still being called, rather than
padded. `log_curve_series` already walks the longest series and emits only the
keys that have a value at each step (0016 section 2), so this costs nothing —
and a NaN sent to W&B would draw as a real point at zero, which is a lie about
what was measured. On the full training set the curves are 199 points against
the grid's 201.

### 5. Computed at every profile; only the figures are gated

Same rule as the curves (0016): the table costs one sort once the out-of-fold
vector exists, so `quick` gets it too. `fig/threshold_sweep` and
`fig/threshold_count` are gated on `profile.plots`.

## Why this and not the alternatives

**Pick one threshold and hardcode it.** The thing this record exists to prevent.
Whichever number were chosen would end up quoted as though it were measured, and
the sensitivity — the 2.1x — would never be seen.

**Just use 0.5.** It calls 7,250 sites where 5,475 are real. It is in the table
as one of three so that its cost is visible, which is not the same as adopting
it.

**Calibrate the scores first, then threshold.** The better long-term fix, and it
is a *modelling* change — isotonic or Platt scaling, which nobody has tried
(GAPS.md). This record deliberately does not do it: a threshold sweep is
evaluation infrastructure and works on whatever scores it is given, whereas
calibration changes the model's output and needs its own comparison. Calibrating
would move these operating points; it would not remove the need to report them.

**Use the thresholds `precision_recall_curve` returns.** They are the distinct
score values — up to 121,838 of them, all different per model, so two runs could
never share an axis. That is the same mistake 0014 rejected for the ROC curve,
and it would also mean a new import in `evaluation.py`.

**Sweep on a quantile grid of the scores instead of the unit interval.** Denser
where the scores actually live, and fatal for the thing the `curve/` namespace
is for: every model has a different score distribution, so no two runs would
share x values. The unit grid resolves this data well — `f1_max` lands at 0.58
and `count_matched` at 0.65, nowhere near the grid's edge.

**Report only the count curve, not precision/recall/F1.** The count is the
headline, but a count with no idea of its precision is a number with no error
of any kind attached. All four, count read first.

## Consequences

- `evaluation.py` grew, which is shared infrastructure on the frozen-import
  path. It gained **no imports**, which is the constraint that shaped the
  implementation.
- `threshold/*` and `curve/threshold/*` join the metric schema
  ([0007](0007-evaluating-without-a-baseline.md) section 4). Adding keys is
  free; these names are now an interface.
- Runs logged before today have no `threshold/*` keys and show as blanks.
- The local report JSON grows by ~11 KB (six series of ~200 points). Worth
  stating because
  [0010](0010-the-report-is-a-module-not-a-script.md) describes it as "~10 KB"
  and that is now well out of date: a full-data `standard` report is **263 KB**,
  almost all of it `strata_observations`
  ([0018](0018-per-stratum-vectors-are-logged-for-later.md), 121 KB) and the
  `curve/` series ([0016](0016-everything-overlayable-lives-under-curve.md),
  37 KB). Still comfortably inside the "nothing over ~1 MB per run" rule
  [0009](0009-distributions-not-per-site-scores.md) set, but the headroom is a
  third of what it looks like from 0010.
- **The operating points are computed on out-of-fold training data, at depth
  >= 20.** Every caveat about the depth floor applies to them unchanged: the
  threshold that matches the count here is not necessarily the one that matches
  it on SG-NEx at median depth 3, and nothing in this record measures that. It
  is a worse problem for a threshold than for a rank metric, because a threshold
  is an absolute cut on a score distribution that depth is known to shift.
- A third figure-series colour was needed and **the palette did not have one**.
  Checked with the validator rather than by eye: BLUE/ORANGE separate by
  dE 24.7 under protanopia, but BLUE/ORANGE/RED fails — RED against ORANGE is
  dE 10.8 at *normal* vision, below the 15 floor. The comment in `figures.py`
  claiming the first three slots held up together was wrong and has been
  corrected. F1 is drawn in the context grey instead, which is also the right
  encoding: it is a summary of the other two, not a third peer.

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --profile quick
```

on the full training set prints the three operating points above and a count
swing of 2.1x, and the report JSON carries `thresholds.operating_points` and
`thresholds.count_swing`.

`tests/test_evaluation.py::test_the_threshold_table_agrees_with_counting_by_hand`
is the guard that matters — it checks precision, recall and the predicted count
against a direct `(score >= t)` count at every threshold, which is the thing a
clever vectorised implementation gets subtly wrong.

`::test_thresholds_never_log_a_non_finite_point` covers section 4: nothing that
reaches `report.series` may be NaN or infinite, because W&B would draw it.
