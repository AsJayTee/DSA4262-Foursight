# 0028. Every standard run is also judged on data1 and data2 - reported, not gated

- **Date:** 2026-09-29
- **Status:** Accepted
- **Affects:** `src/m6a/external.py` (new), `src/m6a/report.py` (`Profile.external`, `external`, `external_comparison`, `arm(held_out=)`), `src/m6a/tracking.py` (`ext/*`, `ext_compare/*`, `curve/data2/*`), `scripts/train.py` and `scripts/evaluate.py` (`--external / --no-external`), AGENTS.md section 6

## Context

Every number in this repo came from cross-validation on dataset0. The course
then released data1, a second labelled run of largely the same sites, and data2,
an in-vitro mixing series ([docs/data.md](../data.md#later-releases-data1-and-data2)).

The first time anyone scored the models on data1, most of the cross-validated
gain did not carry over (GAPS.md, "New data"). On the 15,137 data1 sites on
transcripts absent from dataset0:

| | CV on dataset0 | data1, new transcripts |
|---|---:|---:|
| `quantiles` | 0.4759 | 0.3689 |
| `everything` | 0.5408 | 0.3647 |
| `everything` minus cross-site | - | 0.3756 |

A +0.065 improvement in cross-validation was +0.000 here, and the largest
single component (cross-site features, +0.027 in CV) looked slightly harmful.
The harness had no way to see this: it only ever asked "is it better on our
data". Future changes should be seen against a second, independent labelling
as a matter of routine.

## Decision

**What is computed.** At `standard` and `full` (and at `quick` with
`--external`), the model a run would ship - fitted on every dataset0 training
row, depth copies included - scores:

- **data1**, in three slices, strictest first: **`new_genes`** (genes with no
  transcript in the training data; the headline), `new_transcripts`
  (transcripts absent from dataset0) and `new_sites` (sites absent from
  dataset0). Shared sites are never used: the model trained on them, under
  dataset0's differing labels. PR AUC, ROC AUC, lift, and a 95% interval from
  **2,000 resamples of whole genes** (the transcript, where no gene is known).

  **Why genes, amended the same day.** The first version's headline was
  `new_transcripts`. data1 has no gene ids, so a new transcript of a *training*
  gene - which shares its sequence, the exact leak AGENTS.md section 3 forbids -
  counted as unseen. Mapped via dataset0's own labels and then Ensembl
  (`analysis/newdata/map_genes.py`; 4,423 of 4,451 transcripts resolved, and
  Ensembl agreed with dataset0 on all 493 transcripts both could map), **2,718
  of the 15,137 `new_transcripts` sites (18%) were on training genes.**
  `new_genes` is 12,061 sites, 920 positive, 623 genes. The 28 transcripts
  nothing could map (358 sites) are kept out of it, since they may belong to a
  training gene. The mapping is `data1/transcript_genes.csv`, beside the data.
- **data2**: mean score at each modification fraction (0-100%), the Spearman
  correlation of site score with fraction, and ROC AUC of 100% against 0%.

**Comparisons.** `--compare-features` and `--compare-with` fit the baseline arm's
shipped model too, report its data1/data2 block in full, and add a **paired**
data1 difference: both arms on the identical transcript resamples, reporting
the difference, its interval, and the share of resamples the candidate wins.
`--compare-run` cannot pair (W&B holds no per-site scores) and says so.

**Report only.** Nothing is gated. A change that wins in cross-validation and
loses on data1 is shown as exactly that, and whoever reads it decides.

**W&B.** `ext/data1/<slice>/{pr_auc, roc_auc, pr_auc_lift, ci_low, ci_high, n,
n_positive}`, `ext/data2/{spearman, roc_auc_100_vs_0, mean_score/<N>pct}`,
`ext_compare/<key>/<slice>/{mean_difference, ci_low, ci_high, win_rate}`, and
`curve/data2/fraction` against `curve/data2/mean_score` as an overlayable line.

**Missing data fails at the start**, naming the two download commands, rather
than after an hour of fitting. `--no-external` opts out, and the run then lacks
the `ext/*` keys.

**It defaults off when `--json` names a custom training file**, and for smoke
and `--limit` runs. The data1 slices are defined as "not in dataset0", which
means nothing against another file. `--external` still forces it. (Found when
the train-then-predict tests, which train on a small generated file, all
stopped at the missing-data check.)

## Why this and not the alternatives

**A hard rule: "must not get worse on data1" or "must improve on data1".**
Offered and declined. data1 is one test set whose labels disagree with
dataset0's on 5.75% of shared sites; a strict rule would reject real
improvements on noise from a different labelling, and a lenient one would be a
rule nobody trusts. The numbers are logged on every run, so a rule can be
added later from recorded evidence.

**Score data1 with the five fold models and average.** Each fold model saw
only four fifths of the genes, so a data1 transcript absent from dataset0 is not
the same held-out quantity for each. The shipped model is the thing being
judged, and it is fitted on everything.

**Pool data1 into cross-validation.** Its labels conflict with dataset0's on
shared sites, and it has no gene ids for the grouped split. Using it as a
separate test answers the question that matters - does it transfer - without
deciding which labelling is right.

**Resample sites instead of transcripts.** Positives cluster by transcript
(docs/decisions/0025), so site-level resampling would understate the interval.

## Consequences

- **Every standard run costs one more fit on all rows** (two for a comparison),
  plus feature extraction on data1 the first time per feature set - a few
  minutes, then cached. The bootstrap takes seconds.
- Teammates must download data1 and data2 before a standard run
  (`download_data.py --set data1`, `--set data2`), or pass `--no-external`.
- **Runs before this record have no `ext/*` keys** and cannot be compared on
  them without being re-run.
- data1's labels are a different labelling, not ground truth. A low data1 score
  is agreement with it, not accuracy; on the shared sites dataset0's own labels
  score 0.326.

## How to check it still holds

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --profile quick --external \
       --compare-features pooled_v1 --no-wandb
```

must still print the recorded per-fold PR AUC (0.4548 / 0.5081 / 0.4685 /
0.4704 / 0.4855) and `quantiles`' data1 `new_transcripts` PR AUC 0.3689, the
number `analysis/newdata/score_new_data.py` measured independently.

Tests: `tests/test_external.py` - slices exclude what they claim, resamples are
whole transcripts and deterministic, identical arms differ by exactly zero,
arms scored on different sites are refused, the W&B keys.
