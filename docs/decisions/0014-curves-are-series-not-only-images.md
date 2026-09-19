# 0014. ROC and PR curves are logged as series on a shared grid, not only as images

- **Date:** 2026-09-19
- **Status:** Accepted
- **Affects:** `src/m6a/report.py` (`curve_series`, `CURVE_GRID`), `src/m6a/tracking.py` (`log_curve_series`), the `curve/*` metric keys

## Context

[0007](0007-evaluating-without-a-baseline.md) section 5 said to log the ROC and
PR curves, and they were — as matplotlib PNGs.

A PNG cannot be overlaid with another PNG, filtered by the run selector, or put
on a shared axis. So the one question the curves exist to answer — *how do these
three models compare* — was the one question the logged form could not support.
You could open three run pages and flick between three pictures.

This was found by a teammate trying to build a panel, not by reading the code.
The images were never wrong; they were the wrong *object* for cross-run
comparison.

## Decision

Curves are logged **twice**, because the two forms do different jobs.

### 1. As a step series, on a grid every run shares

`curve_series` interpolates each curve onto `np.linspace(0, 1, 201)` and logs
one point per step:

| key | x | y |
|---|---|---|
| `curve/roc/fpr`, `curve/roc/tpr` | false positive rate | true positive rate |
| `curve/pr/recall`, `curve/pr/precision` | recall | precision |

A W&B line panel with y = `curve/roc/tpr` and x = `curve/roc/fpr` then draws one
line per run, and the run selector filters them.

**The shared grid is the load-bearing part.** A raw curve's x-values are its own
thresholds, so two models never share an axis and cannot be overlaid — or later
differenced. Interpolating first makes the overlay exact. It also cuts the data
by about 600x: a raw curve over 121,838 sites has up to that many thresholds,
and 201 points is visually indistinguishable at any zoom.

`define_metric(y, step_metric=x)` declares the pairing, so the panel picks the
right x-axis on its own instead of defaulting to `_step` and making every reader
fix it by hand. `summary="none"` keeps the last point of each curve out of the
run summary, where `curve/roc/fpr = 1.0` would sit as a meaningless column.

### 2. As images, unchanged

The PNGs keep the random-classifier line drawn at the 4.49% base rate and the
caption explaining it. That is what goes in the report, and neither annotation
survives into a line panel.

### 3. Only the primary arm becomes a series

A run owns one set of `curve/*` keys. A comparison arm's curve belongs to that
arm's own run, where W&B will draw it on the same panel anyway — which is the
whole point of making them overlayable.

## Why this and not the alternatives

**`wandb.plot.roc_curve`.** W&B's built-in. It produces a per-run Vega chart, so
it has exactly the problem this record is fixing.

**Log the curve as a `wandb.Table` and use a custom chart.** Works for one run;
custom charts do not overlay across runs in the run selector.

**Log the raw curve with no interpolation.** Up to 121,838 points per run, and
every run on different x-values, so nothing lines up. The size is the lesser
problem.

**Drop the PNGs now that there is a series.** Rejected: the base-rate line is
what makes a PR curve readable at a 4.49% positive rate, and a line panel cannot
carry it or the caption.

## Consequences

- A run's history now has 201 steps. Anything else logged with `run.log()` has
  to come after, or it lands at step 0 underneath the curve — `report.publish`
  sends the series first for that reason.
- `curve/*` joins the metric schema (0007 section 4), so those four names are
  now an interface like the rest.
- Every run carries the same 201-point grid, so two curves **can** be subtracted
  point-by-point later. Nothing does that yet.
- Curves are computed at every profile, including `quick`, because they cost
  milliseconds once the out-of-fold vector exists. Only the images are gated on
  `profile.plots`.

## How to check it still holds

```bash
python scripts/train.py --config configs/lightgbm.yaml --smoke
```

then, on the resulting run:

- `run.history(keys=["curve/roc/fpr", "curve/roc/tpr"])` returns **201 rows**.
- `curve/roc/fpr` is identical across every run in the project — that is what
  makes the overlay valid, and a run whose grid differs cannot be compared.
- No key beginning `curve/` appears in `run.summary`.

Verified on run `1hzgey5v`: 201 rows, grid shared, summary clean.
