# 0029. Models are selected on their gain across both labellings, not on dataset0 alone

- **Date:** 2026-10-01
- **Status:** Proposed - agreed, not built. Built in the order of "Plan" below;
  change this line as each step ships.
- **Affects:** (when built) `src/m6a/external.py`, `src/m6a/report.py`, `src/m6a/tracking.py`, `scripts/evaluate.py`, a new cross-source mode, AGENTS.md section 6, docs/wandb-panels.md

## Context

Every model choice so far was made on gene-grouped cross-validation within
dataset0. [0028](0028-every-run-is-also-judged-on-data1-and-data2.md) added a
held-out score on data1, and on data1's genes unseen in dataset0 the project's
largest gain did not transfer:

| | CV on dataset0 | data1, new genes |
|---|---:|---:|
| `everything` minus `quantiles` | **+0.065** (50/50) | **-0.006** [-0.026, +0.015] |
| `quantiles` minus `pooled` (positive control) | +0.016 | +0.016 [+0.004, +0.028] |

Gene novelty cannot explain this: CV already scores unseen genes. Moving to
data1 changes three other things at once - the **labelling** (5.75% of shared
sites disagree; 65% of dataset0's positives are positive in data1), the
**sequencing run** (mean current r = 0.998 across runs, dwell only 0.80) and
the **gene population** (data1's new genes may be a different kind of gene).

An external literature review (ChatGPT deep research, 2026-09-30) framed this
as cross-study dataset shift and recommended the protocol below. Its load-bearing citations were checked: Bernau et al. 2014
(cross-study validation, doi:10.1093/bioinformatics/btu279) and Zhang et al.
2018 (doi:10.1093/biostatistics/kxy044) exist and say what it claimed. Two
supporting citations could not be found and are not relied on.

## Decision

### 1. Diagnose before redesigning: the shared-site crossed test

The 67,320 sites in both files each have two measurement sets (one per run) and
two labels (one per labelling). Each is scored by a CV fold model that never saw
its gene, four ways:

| | dataset0 labels | data1 labels |
|---|---|---|
| dataset0 measurements | A - today's CV | B - labelling changed |
| data1 measurements | C - run changed | D - both changed |

**Every cell reports a gain over `quantiles`, never a raw score** - the
question is where the +0.065 goes. Intervals: one gene-cluster bootstrap,
recomputing all four cells and every contrast from the same resampled genes.
Run for `everything` and for `everything` with one feature family removed at a
time, in this order of suspicion: same-transcript (cross-site) features; read
count used as a feature (`n_reads`, `log_n_reads`, `tx_n_reads_loo_mean`);
within-read correlations (coupling); flanking bases. The `quantiles`-over-
`pooled` gain is the positive control: it must survive every cell.

### 2. The data2 audit

Each feature's and each model's mean at each modification fraction. Published
tools rise with the fraction; ours fall at 95-100%. Features measuring spread
between reads (a *mix* of molecules) are the suspects.

### 3. The cross-source matrix

One five-fold split over the **union of genes** in both files; a gene, and both
run-copies of a shared site, always in the same fold. Per fold, train four ways
and score held-out genes of each file against that file's labels:

| trained on | scored on dataset0 genes | scored on data1 genes |
|---|---|---|
| dataset0 | within | across |
| data1 | across | within |
| both, pooled | | |
| the dataset0 and data1 models, scores averaged | | |

Pooled training is tested two ways for the shared sites: each run with its own
file's labels, and each run with **both** labels at a quarter weight each (a
disagreement becomes a 0.5 target). The second stops the model learning "run 1
measurements, so use run 1's labelling". A shared site counts once in total,
not twice for being sequenced twice.

### 4. The selection rule

Relative to `quantiles`, per labelling s in {dataset0, data1}:

- **choose on the mean gain**, (gain_dataset0 + gain_data1) / 2;
- **veto any model whose worse gain is below -0.005.**

The threshold is fixed here, before the next results, so it cannot be chosen
to suit them. Raw PR AUC, ROC AUC and the full matrix are always reported
beside the two numbers; a prevalence-adjusted PR AUC (both files at one
positive rate) is reported as a diagnostic only, because raw PR AUC rises with
the positive rate (4.49% against 7.26%).

### 5. Naming

Once data1 informs model choice it is **development data**, not an external
test. The leaderboard is the external test.

## Why this and not the alternatives

**Keep selecting on dataset0 CV and only report data1.** That is 0028 as
built, and it already let a +0.065 gain that is worth nothing elsewhere look
like the best model.

**Worst case of the raw PR AUCs.** With two imperfect labellings, the noisier
one would dominate every choice. A veto on the *gain* asks the question that
matters - does this change make us worse under either labelling.

**Pool both files and cross-validate once.** It mixes the two labellings
into one number and hides exactly the disagreement this record exists to see.

**Model the label noise** (Dawid-Skene, noise-corrected losses). Two labelling
sources give little leverage, and assay errors are likely sequence-dependent
rather than random. Revisit only if step 3 shows pooling helps.

## Consequences

- A cross-source evaluation fits up to four models per fold instead of one.
  The 10-repetition CV stays for comparisons within dataset0, so recorded
  results keep their meaning; new compute goes to the matrix.
- The shipped model may end up trained on both files (decided by step 3, not
  assumed).
- Any model choice now needs data1 and its gene table locally.

## Plan

1. Shared-site crossed test with feature-family removals (step 1) and the
   data2 audit (step 2) - evaluation only, research code first.
2. Cross-source matrix (step 3) - research code first.
3. Only then build steps 3-4 into `evaluate.py` and the report, with the W&B
   dashboard redone around them (docs/wandb-panels.md).

## How to check it still holds

When built: the `quantiles`-over-`pooled` positive control must show a gain on
both labellings, and `everything`-over-`quantiles` must reproduce
-0.006 [-0.026, +0.015] on data1's new genes.
