# The W&B workspace: which panels to build, and how to read them

<https://wandb.ai/dsa4262-team/dsa4262-project>

Every run logs the same keys, so the project **is** the comparison — you are
never evaluating against nothing. This page is how to set the workspace up once
so that a new run drops straight onto every panel, and how to read each one
without misinterpreting it.

Two things to know before you start.

**Panels are a property of the workspace, not of the runs.** Deleting runs does
not delete your panels, and a panel you build today draws every run anyone adds
tomorrow.

**Scalars go in the run table; curves go in line panels.** Anything meant to be
drawn next to another run's version of itself is logged under `curve/`
([0016](decisions/0016-everything-overlayable-lives-under-curve.md)). Everything
else is a column.

---

## Adding a line panel

In the workspace: **Add panel → Line plot**, then in the editor:

1. **X axis** — set it explicitly to the `curve/<family>/<x>` key named below.
   The code declares the pairing with `define_metric(step_metric=...)`, but a
   panel with more than one Y key falls back to `Step` and draws nonsense.
2. **Y** — add the `curve/<family>/*` keys listed.
3. **Group by: off.** One line per run is the entire point.

**There is no log-scale toggle on a W&B line panel.** Only the scatter panel can
transform an axis. That is why some families log a `log10_` companion key
alongside the raw one — point the panel's X at the companion, and switch back to
the raw key when you want to read real numbers off the axis
([0016 §3](decisions/0016-everything-overlayable-lives-under-curve.md)). The raw
key is always the one to quote.

---

## The panels

### 1. Fit progress — average precision

| | |
|---|---|
| X | `curve/train/iteration` |
| Y | `curve/train/valid_pr_auc`, `curve/train/train_pr_auc` |

**This panel is for deep-learning models only** — `mlp_quantiles`,
`attention_mil_quantiles`, and anything else that learns by gradient descent.
LightGBM and logistic regression draw nothing here, deliberately.

That is a change from how it first shipped
([0024](decisions/0024-the-training-curve-is-for-gradient-descent-models.md)
amends [0021](decisions/0021-models-report-how-the-fit-progressed.md)).
LightGBM *does* train iteratively — one boosting round per tree — and it used to
fill these keys, which made the panel draw eight lines across two model families
whose x axes are not the same quantity: a boosting round adds one tree, an epoch
is a full pass over ~97,000 rows. 600 of one beside 40 of the other invites "the
network converged 15× faster", which is not a fact about anything. Restricting
the panel to one kind of training makes every line on it comparable.

**Expect two lines per run** — train and valid. If that is too many, filter the
run selector rather than dropping a series: the gap between a run's two lines is
the point of the panel.

### The trap that remains

**`fit/train_valid_gap` is only an overfitting measure when the training and
held-out rows come from the same distribution.** For a run with
`train_depths: full` it is. For a depth-augmented run it is not: training rows
are drawn at depths 1/3/5/10/full and the held-out fold is full depth only, so
the model is being scored on *easier* rows than it trained on. The giveaway is a
train line sitting **above** its valid line, which otherwise essentially never
happens. Check `train_depths` before comparing this number across runs — see the
correction in
[0022](decisions/0022-training-rows-may-come-from-several-depths.md).

**How to read it.**

- **The gap between the two lines is overfitting.** `lightgbm_quantiles` reaches
  0.996 on its training fold against 0.478 held out — it very nearly memorises
  what it trains on. `quantiles_depth_augmented` sits at 0.543 against 0.487.
  Five views of each site turns out to be strong regularisation, which nobody
  set out to add.
- **Where the held-out line peaks is where the budget stops paying.** On
  `configs/quantiles.yaml` that is round 431, and the config runs to 600.
- **Still climbing at the right-hand edge** means `n_estimators` (or `epochs`)
  is too small.

**What not to do with it.** The peak is read off the same held-out folds the
run's headline PR AUC comes from. Copying that number into a config and then
quoting this run's score is selection on the test set — the flat-tuning trap
recorded under Modelling in [GAPS.md](../GAPS.md), arrived at one parameter at a
time. Change the config, run it, and compare the two runs paired like any other
pair.

### 2. Fit progress — logloss

| | |
|---|---|
| X | `curve/train/iteration` |
| Y | `curve/train/valid_logloss`, `curve/train/train_logloss` |

**A separate panel on purpose.** Logloss falls and average precision rises, on
scales three orders of magnitude apart; sharing an axis makes the crossing point
of two lines look meaningful when it is an artefact of the axis limits.

**The curve is hump-shaped, and this is the panel's whole point.** Held-out
logloss does *not* fall monotonically: it dips early, climbs to a peak, and only
then declines. **That hump is the miscalibration becoming visible.** Early on
the model predicts close to the base rate — logloss is low and the ranking is
useless. As training sharpens probabilities that the class rebalancing
(`is_unbalance`, `pos_weight`) has calibrated to a *rebalanced* world, they get
systematically too high, so logloss gets worse while PR AUC climbs. Later, the
model is accurate enough that logloss recovers.

It was measured most clearly on the boosters, before
[0024](decisions/0024-the-training-curve-is-for-gradient-descent-models.md)
scoped this panel to gradient-descent models:

| round | 1 | 2 | 10 | 50 | 200 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lightgbm_pooled` | 0.1657 | **0.1633** | 0.1934 | 0.2902 | 0.2363 | 0.1849 | – |
| `lightgbm_quantiles` | 0.1647 | 0.1622 | 0.1859 | 0.2646 | 0.2060 | – | **0.1387** |

The effect is not booster-specific — `mlp_quantiles` also ends with train
logloss 0.2739 against valid 0.2995 — so expect the same shape from the models
that still draw here.

**So do not read `fit/logloss_best_iteration` as "where the fit should stop".**
It is a plain argmin over a non-monotone curve, so it lands on the early dip
whenever a run is too short for the late decline to get back under it: on the
boosters it reported **2** for pooled (PR AUC 0.338 there against 0.465 at the
end) and 600 for quantiles, and those two numbers are not comparable. Read the
*shape* on this panel; take the budget from panel 1. Recorded as a known defect
in [GAPS.md](../GAPS.md).

This is the clearest evidence in the harness that the scores are not
probabilities, and it is visible nowhere else — `calib/*` measures the finished
model, and this shows it happening.

### 3. Depth collapse — the most important panel in the project

| | |
|---|---|
| X | `curve/depth/log10_reads` |
| Y | `curve/depth/pr_auc` |

Every training site has at least 20 reads; SG-NEx has a median of **3**
([docs/data.md#read-depth](data.md#read-depth) — depth is a count of distinct
RNA molecules, not repeated readings of one). This panel is what happens when
you take that evidence away.

**How to read it.** On the log axis, SG-NEx's median depth 3 is at **x ≈ 0.48**
and its 25th percentile of 1 read is at **x = 0**. Read the y value there, not
at the right-hand edge. Switch X to `curve/depth/reads` to see real read counts.

A motif-only classifier — an 18-way one-hot with no nanopore signal at all —
scores **0.1537**. A curve that drops to that level has stopped reading the pore
and is just recognising the sequence.

**Read the run's `train_depths` before comparing two lines here.** A run trained
at full depth and a run trained at `1,3,5,10,full` are answering different
questions on this axis, and the numbers are not like-for-like
([0022](decisions/0022-training-rows-may-come-from-several-depths.md)). The run
config records which.

### 4. Threshold → how many sites get called

| | |
|---|---|
| X | `curve/threshold/threshold` |
| Y | `curve/threshold/log10_predicted_positives` |

PR AUC and ROC AUC integrate over every threshold, which is right for comparing
models and useless for using one. Task 2 has to *count* modified sites, and a
count needs a cut ([0019](decisions/0019-a-threshold-sweep-because-a-ranking-cannot-count.md)).

**How to read it.** Any Task 2 claim of the form "cell line X has N modified
sites" is one point on this line. On the current model the count moves **2.15×**
between threshold 0.3 (10,610 sites) and 0.7 (4,943) — both defensible choices.
`threshold/count_swing` is that ratio as a sortable column. Quote it beside any
count, or you are reporting an arbitrary cut as a measurement.

0.5 is not neutral: it calls 7,250 sites where 5,475 are real.

### 5. Precision–recall

| | |
|---|---|
| X | `curve/pr/recall` |
| Y | `curve/pr/precision` |

**The panel cannot draw the base-rate line and the PNG can.** At a 4.49%
positive rate, a random classifier sits at precision 0.0449 — that is the floor,
and without it on the axis a PR curve is easy to over-read. Use this panel for
overlaying runs and the `fig/pr_curve` image on a run page when you want the
annotated version.

### 6. ROC

| | |
|---|---|
| X | `curve/roc/fpr` |
| Y | `curve/roc/tpr` |

Useful for shape, not for ranking models. At 4.49% positives ROC AUC flatters
everything — every model here is above 0.90 on it and they are not close on PR
AUC.

### 7. Reliability — are the scores probabilities?

| | |
|---|---|
| X | `curve/reliability/predicted` |
| Y | `curve/reliability/observed` |

**How to read it.** A perfectly calibrated model lies on y = x. Below the
diagonal means the model predicts higher than reality — it overcounts. Every
model in this repo sits below it, because every one of them handles the class
imbalance by reweighting (`is_unbalance`, `class_weight="balanced"`) and so
emits scores calibrated to a rebalanced world.

Rank metrics cannot see this at all. `calib/count_ratio` is the size of it as a
column: 1.80× for `lightgbm_quantiles`, 2.77× for `lightgbm_pooled`, 6.10× for
`baseline_logistic`.

### 8. Lift by *true* read depth

| | |
|---|---|
| X | `curve/band/log10_reads` |
| Y | `curve/band/pr_auc_lift` |

**Not the same question as panel 3, and the two are easy to confuse.** Panel 3
removes reads from the same sites. This one removes nothing — the sites simply
came with different coverage. X is each band's lower edge in reads.

Lift rather than PR AUC because PR AUC is bounded below by each band's own
positive rate.

**How to read it.** The 600+ band falls off a cliff — 8.84× against 11.07× for
304–599, which is the *best* band in the table
([0017](decisions/0017-split-the-top-depth-band.md)). More evidence should make
a site easier, not harder. That anomaly is unexplained, it is present in every
model, and it is 2,837 sites nobody has looked at.

### 9. Scatter — accuracy against calibration

**Add panel → Scatter plot.** X `oof/pr_auc`, Y `calib/calibrated`.

The scatter panel is the only W&B panel that can transform an axis, and this is
the natural "up and to the right is better" view: a good model that is also
honest about magnitude. `calib/calibrated` is `1 - ece`, logged as a companion
because a scatter axis cannot do the transform
([0016 §3](decisions/0016-everything-overlayable-lives-under-curve.md)).

---

## The run table

The scalar half. Add these columns (gear icon → Columns), roughly in this order:

| column | what it answers |
|---|---|
| `train_depths` | **set this first.** Which depths the training rows came from. `full` or e.g. `1,3,5,10,full`. Two runs differing here are not like-for-like on any depth number |
| `features`, `model` | what was varied |
| `oof/pr_auc` | the headline |
| `oof/pr_auc_lift` | how many times better than random. 1.0× means it learned nothing |
| `rep/pr_auc_mean`, `rep/pr_auc_sd` | the distribution over 50 observations, not 5 |
| `rep/n_observations` | **50 or 5.** A 5-point run is thinner, not wrong, and the two are not comparable |
| `depth/1/pr_auc`, `depth/3/pr_auc` | where Task 2 lives |
| `calib/count_ratio` | how badly summing the scores overcounts |
| `threshold/count_swing` | how arbitrary a site count is |
| `fit/best_iteration`, `fit/n_iterations` | is the budget right? |
| `fit/train_valid_gap` | how much it memorises |

Sort on `oof/pr_auc`; filter on `train_depths` before reading any `depth/*`
column.

---

## What W&B cannot draw, and what to use instead

Two figures the report needs that no panel can produce
([0020](decisions/0020-comparing-many-runs-is-a-script-not-a-panel.md)):

- **N metric distributions on one axis.** The per-run distribution is logged as
  an *image*, so a media panel shows three separate pictures at three different
  scales — the one comparison it exists to support.
- **A scatter with a y = x reference line and labelled points.** The scatter
  panel draws neither, and for "how far below full-depth performance does each
  model land at SG-NEx's depth" the diagonal *is* the claim.

```bash
python scripts/compare_runs.py <run-a> <run-b> <run-c> --pair
```

Fits nothing, downloads no data, needs no instance. It reads finished runs out
of W&B and writes `report/figures/runs_distribution.png` and `runs_scatter.png`,
and `--pair` adds the corrected paired test of each run against the first one
named.

**Pass run IDs, not names.** Names stop being unique the moment a config is
re-run, and `resolve_run` then refuses rather than guessing which you meant.
