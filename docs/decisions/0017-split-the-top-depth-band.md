# 0017. Split the top depth band at 600 reads

- **Date:** 2026-09-21
- **Status:** Accepted
- **Affects:** `src/m6a/evaluation.py` (`DEPTH_BAND_EDGES`, `DEPTH_BAND_LABELS`), every `band/*` metric key, `curve/band/*`

## Context

`DEPTH_BAND_EDGES` ended at 304 — this training set's p95 depth — leaving one
open-ended `304+` bucket of 6,095 sites and 263 positives.

Every model drops in it. PR AUC lift by band, from GAPS.md:

| band | 20-31 | 32-46 | 47-83 | 84-303 | **304+** |
|---|---:|---:|---:|---:|---:|
| `lightgbm_quantiles` | 10.1x | 10.7x | 11.1x | 10.8x | **9.8x** |
| `lightgbm_pooled` | 9.8x | 10.5x | 10.6x | 10.6x | **8.9x** |
| `baseline_logistic` | 8.8x | 9.5x | 9.6x | 9.5x | **6.5x** |

More evidence should make a site easier, not harder. The drop is consistent
across three models, so it is a property of the data or the labels rather than a
model quirk — and nobody has looked. The hypotheses on record are that 304+ sites
are the most highly expressed transcripts and may be a different biological
regime, that m6ACE-Seq's antibody background may differ there, or that the
summary statistics saturate.

**One open-ended bucket cannot distinguish between them**, because it cannot say
whether the drop starts at 304 and worsens, or appears only at some higher
threshold. Those imply different explanations.

## Decision

Add an edge at **600**: `304-599` and `600+`.

Measured first, rather than picked and hoped for:

| band | sites | positives |
|---|---:|---:|
| 304-599 | 3,258 | 134 |
| 600+ | 2,837 | 129 |

Both clear `min_positive` (10) comfortably and split the old bucket almost
evenly. True depth on this data tops out at **991** (p99 = 869), so there is
nothing above 600+ worth another edge — a third split would produce a band with
too few positives to score.

## What it immediately showed

Re-running `configs/quantiles.yaml` on the full training set:

| band | lift |
|---|---:|
| 84-303 | 10.82x |
| **304-599** | **11.07x** |
| **600+** | **8.84x** |

**304-599 is not depressed at all** — at 11.07x it is the best band in the table.
The entire drop lives above 600 reads.

So this is a **cliff above 600, not a slide from 304**, which is a materially
different fact from the one GAPS.md recorded and narrows the search to 2,837
sites. "Summary statistics saturate" is now a weaker hypothesis, since saturation
would predict a gradual decline; a threshold effect fits a distinct population
better.

## Why this and not the alternatives

**Leave it as `304+`.** The status quo, and it is why the question stayed open
for as long as it did.

**Split at the p99 (869).** Would give a top band of roughly 60 positives —
scoreable, barely, and far too noisy to base an argument on. 600 splits the
bucket near-evenly and keeps both halves above 125 positives.

**Compute band edges per dataset** rather than fixing them. Rejected for the
reason the edges were fixed in the first place: a band has to mean the same thing
in two reports, or the numbers cannot be read side by side.

**Add several bands at the top for resolution.** Each additional edge halves the
positives in the bands it creates. At 263 positives to start with, two is the
most the data supports.

## Consequences

- **Every `band/*` key changes meaning.** `band/304+/pr_auc_lift` no longer
  exists; `band/304-599/*` and `band/600+/*` replace it. Any run logged before
  today is not comparable on those keys.
- This was done **today specifically because W&B had just been cleared** — the
  project had zero runs, so the change orphaned nothing. It would have been a
  materially more expensive decision a week later, and more expensive still once
  the report starts quoting band numbers.
- PR AUC on ~130 positives is noisy. Both new bands should be read with that in
  mind, and the per-stratum caveat that already applies to every band test
  applies doubly here.
- `tests/test_evaluation.py::test_depth_bands_are_ordered_and_cover_the_sgnex_range`
  pins the labels and caught this change when it was made, which is what it is
  for.

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/quantiles.yaml
```

`band/304-599/pr_auc_lift` and `band/600+/pr_auc_lift` both exist, both are
scored (neither says "only N positive"), and `band/304+` does not appear.

The open question this was built to ask is now: **why does performance collapse
above 600 reads but not between 304 and 599?** That belongs in GAPS.md, not here.
