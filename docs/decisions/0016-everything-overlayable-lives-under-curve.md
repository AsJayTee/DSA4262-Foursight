# 0016. Anything meant to be compared across runs is a series under `curve/`

- **Date:** 2026-09-21
- **Status:** Accepted
- **Affects:** `src/m6a/report.py` (`Report.series`), `src/m6a/tracking.py` (`log_curve_series`), `src/m6a/evaluation.py` (`calibrated`), the `curve/*` keys, `DEFAULT_DEPTHS`

## Context

[0014](0014-curves-are-series-not-only-images.md) made the ROC and PR curves
overlayable by logging them as step series on a shared grid, and that worked.
It left three other per-run pictures as images only, all of which answer
questions that are *about the comparison between models*:

- the **depth sweep** — the collapse curve, the largest known risk in the project
- the **true-depth bands** — where the unexplained high-depth drop lives
- the **reliability diagram** — how a model's probabilities are wrong, not just
  how much

Each was one PNG per run, so comparing three models meant opening three run
pages and flicking between three pictures. The same limitation 0014 fixed,
in three more places.

## Decision

### 1. One namespace, one helper

`Report.series(x_key, xs, {y_key: ys})` registers an overlayable curve.
Everything registered lands under `curve/`:

| keys | x | y |
|---|---|---|
| `curve/roc/*` | false positive rate | true positive rate |
| `curve/pr/*` | recall | precision |
| `curve/depth/*` | reads kept per site | pr_auc, roc_auc, retained |
| `curve/band/*` | the band's **lower edge** in reads | pr_auc_lift, pr_auc |
| `curve/reliability/*` | mean predicted probability | observed rate |

`curve/` means "this is a line, and it is meant to be drawn next to another
run's". Scalars stay where they were; nothing moved.

**Band x-values are the band's lower edge** (20, 32, 47, 84, 304, 600) rather
than an index, because those are real read counts — the line is readable on a
log axis and means the same thing as the sweep's x.

**`full` is not in the depth series.** It has no number on a reads axis, it is
already `oof/pr_auc`, and inventing an x for "all of them" would need explaining
on a figure meant to be read at a glance. The PNG keeps it as a categorical tick.

### 2. Series may be ragged

201 points for a ROC curve, 10 for a depth sweep, 11 for bands. `log_curve_series`
walks the longest and emits only the keys that have a value at each step. W&B
tolerates a missing key at a step; it does not tolerate steps going backwards,
which is what logging each series over its own range would do.

### 3. Transform in the panel where you can, and log a companion where you cannot

A transformed *metric* cannot be untransformed, makes the key mean something
different from the day it changed, and orphans every run before it — the schema
hazard [0007](0007-evaluating-without-a-baseline.md) section 4 exists to prevent.
So the default is: log the raw quantity, let the axis do the work.

**Correction (2026-09-21): W&B cannot always do the work.** This section
originally asserted "W&B has a log-scale toggle per axis". That is false for
**line panels**, which expose only range min/max — checked against W&B's own
line-plot reference after a teammate went looking for the control and could not
find it. Only the scatter panel can transform an axis.

So a *companion* key is logged wherever the panel cannot transform. The rule is
that the raw key stays, stays the default, and remains the number to quote:

| companion | raw it accompanies | why the panel cannot do it |
|---|---|---|
| `calib/calibrated` = 1 - ece | `calib/ece` | a scatter axis cannot transform |
| `curve/band/log10_reads` | `curve/band/reads` | a line panel has no log scale |
| `curve/depth/log10_reads` | `curve/depth/reads` | same |

The band case is not cosmetic. On a linear 20..991 axis the first three bands
share **6% of the width**, so half the measurements are unreadable slivers.
Point the panel's x at the `log10_` key; switch back to the raw one to read real
read counts.

A companion is not a licence to transform freely. It is allowed only when the
panel demonstrably cannot, and it never replaces the raw key.

### 4. The depth sweep stops at 25

`DEFAULT_DEPTHS` becomes `1, 2, 3, 4, 5, 7, 10, 15, 20, 25` — dense through the
regime SG-NEx occupies (median 3, p25 1) and **capped on purpose**.

`subsample_reads` returns a site unchanged once it has fewer reads than asked
for. Measured on this training set:

| depth | sites returned untouched |
|---:|---:|
| 20 | 1.8% |
| 30 | 22.7% |
| 50 | 53.6% |
| 84 | 75.2% |

Past about 25 the row is mostly "full depth" wearing a label, and by 50 it is
meaningless. A sweep that quietly converges on `oof/pr_auc` while looking like a
measurement is worse than not having the point.

## Why this and not the alternatives

**Leave them as images.** What 0014 already rejected, for the same reason: the
question these figures exist to answer is a comparison, and an image cannot be
overlaid or filtered.

**One `wandb.Table` per curve and a custom chart.** Works for one run; custom
charts do not overlay across runs in the run selector.

**Band x = an index 0..10.** Then the axis means nothing, a log scale is
nonsense, and the sweep and band panels cannot be read against each other.

**Log `logit(pr_auc)` so good models spread out.** The transform-in-the-data
trap — see section 3. Not needed either: PR AUC here is ~0.4–0.5, nowhere near a
ceiling, and the scatter panel *can* transform its axes, so the companion
exception does not apply.

**Extend the sweep to 50 or 84 for a fuller picture.** See the table above. It
would add points that are mostly not measuring what they claim to.

## Consequences

- Runs now log 8 series. History is 201 steps deep (the ROC curve dominates);
  everything else rides along inside that range.
- `curve/*` joins the metric schema. Those names are an interface now.
- The denser depth list costs one extra subsample per site per new depth during
  the single streaming extraction pass, and ~50 MB of feature cache per depth.
  Fit time is unaffected — the sweep scores with existing fold models.
- **Adding depths does not move existing depth numbers**, which is exactly the
  property [0003](0003-read-subsampling-in-data.md) was designed to provide.
  Verified on the full training set: after going from 5 depths to 10, depths
  1/3/5/10/20 still read 0.1527 / 0.2458 / 0.2991 / 0.3716 / 0.4277, unchanged
  to four decimals.

## How to check it still holds

```bash
python scripts/train.py --config configs/lightgbm.yaml --smoke
```

The report's `curves.pairs` must list all eight (x, y) pairs, and on the run:

- `run.history(keys=["curve/depth/reads", "curve/depth/pr_auc"])` returns one
  row per numeric depth, with no `full`.
- `curve/band/reads` values are band lower edges, ascending.
- No key beginning `curve/` appears in `run.summary`.
- `calib/calibrated` equals `1 - calib/ece`.
