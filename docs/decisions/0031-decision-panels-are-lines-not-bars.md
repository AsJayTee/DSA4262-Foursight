# 0031. The Decisions view draws cross-source results as lines, not bars

- **Date:** 2026-10-01
- **Status:** Accepted
- **Affects:** `src/m6a/report.py` (`cross_source_series`, `curve/ext/data1/*`), `src/m6a/tracking.py` (`log_curve_series` skips NaN), `scripts/build_dashboard.py`, `analysis/newdata/backfill_xsrc_curves.py` (new). Amends [0030](0030-the-wandb-dashboard-is-code.md).

## Context

The first Decisions view ([0030](0030-the-wandb-dashboard-is-code.md)) drew
the cross-source numbers as W&B bar charts of summary values. With four runs it
was already unreadable: W&B draws one labelled bar per run per metric, and a
panel of five arms became forty labels with the values too small to compare.
The one scatter plot - gain under dataset0's labels against gain under data1's -
asks "is this on the diagonal?" and W&B's scatter panel cannot draw a diagonal.

The original History panels never had this problem. They are `curve/*` step
series ([0016](0016-everything-overlayable-lives-under-curve.md)): a few points
per run, drawn as one line per run in a line panel.

## Decision

1. **Cross-source results are logged as curves too.**
   `report.cross_source_series` turns the block into five short series per run,
   logged beside the flat `xsrc/*` keys (which stay the schema for the run table):
   - `curve/xsrc/diag/*` - the shipped arm's two gains as one point, plus
     reference lines (y = x, and the -0.005 veto on each axis) at fixed
     coordinates (`REFERENCE_RANGE`), so every run draws the same lines.
   - `curve/xsrc/transfer/*` - x = 0 dataset0 labels, 1 data1 labels, 2 data1
     genes absent from dataset0. Flat = the gain transfers.
   - `curve/xsrc/crossed/*` - x = cells A..D.
   - `curve/xsrc/arm/*` - x = position in `ARM_ORDER`, ordered by how much
     data1 the arm trains on.
   - `curve/xsrc/path/*` - raw PR AUC, dataset0 against data1, one vertex per arm.
   Standard runs also log `curve/ext/data1/*` (x = new genes, new transcripts,
   new sites).
2. **No bar or scalar panels.** Every Decisions panel is a line panel over
   those curves. The diagonal panel draws the point with `line_marks="points"`
   and the references dashed/dotted.
3. **`log_curve_series` treats NaN as a hole.** A reference-line row carries
   no data point and vice versa; previously NaN would have been logged.
4. **Runs logged before this are backfilled** from their summary by
   `analysis/newdata/backfill_xsrc_curves.py`. `run_xsrc_batch.py` runs it,
   then rebuilds the dashboard, at the end of every batch.

## Why this and not the alternatives

**Fix the bar charts' settings.** The problem is the chart type: one bar per
run per metric does not scale past a handful of runs, whatever the styling.

**W&B custom (Vega) charts.** They can draw anything, error bars included, but
need a chart preset saved by hand in the W&B UI - the hand-built state 0030
removed - and a wandb.Table per run.

**A fake "reference" run that only logs the diagonal.** One line instead of N
overlapping ones, but it would sit in the run table, sort, and be filtered like
a model.

## Consequences

- **The x axes are integers.** A W&B line panel cannot name its ticks, so the
  axis title spells out what each integer means. Read the axis title.
- **Line styles are keyed by run id** (a W&B constraint), so the build styles
  the runs that exist. A run added since the last build draws its point as a
  plain (single-point) line and its references solid until
  `python scripts/build_dashboard.py` is run again.
- Reference lines are logged once per run, so they overlap and the legend lists
  them per run.
- **Backfilling a resumed run:** `wandb.init(resume="must").step` reads 0, and a
  row logged at a step already used is silently dropped. The backfill starts at
  the API's `lastHistoryStep + 1`; the first backfill lost each curve's first
  point to exactly this and was repaired.

## How to check it still holds

```bash
pytest tests/test_crosssource.py   # curves align; reference lines identical across runs
python scripts/build_dashboard.py --dry-run
```
