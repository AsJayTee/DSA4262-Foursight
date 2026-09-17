# 0001. Keep the per-fold results, and write the out-of-fold table to disk

- **Date:** 2026-09-16
- **Status:** Accepted — **partly amended by [0009](0009-distributions-not-per-site-scores.md)**,
  which drops the stored `oof.csv`. Keeping per-fold metrics, the first half of
  this decision and the point of the harness, stands unchanged.
- **Affects:** `scripts/train.py`, `src/m6a/crossval.py`, `meta.json`, a new `oof.csv` beside every model

## Context

`scripts/train.py` ran 5-fold gene-grouped cross-validation, then collapsed
every fold's predictions into one pooled PR AUC in `metrics_oof` and discarded
the rest.

One number cannot answer the questions that matter. It cannot say whether model
A beats model B or whether that gap is fold noise; it cannot say where the model
fails; and it cannot be checked by anyone who did not run it.

The cost was not hypothetical. Comparing two pooled numbers against the
fold-to-fold standard deviation (0.0203) led the team to dismiss a real
improvement of +0.0148 that wins on 5 folds out of 5.

## Decision

`cross_validate` keeps per-fold metrics, and `train.py` writes two new things
into the model directory:

- `meta.json` gains `metrics_per_fold` (one dict per fold) and `oof`, **added
  after the existing keys and changing none of them**.
- `oof.csv` — one row per site:
  `transcript_id, transcript_position, gene_id, motif, n_reads, fold, label, score`

Every site's score in that table came from a model that saw no transcript of its
gene. Per-fold metrics, metrics by depth or motif, calibration, and the paired
comparison between two runs are all derivable from it **without fitting
anything again**: `evaluate.py --oof models/<name>/oof.csv` runs in about a
second where a retrain takes 90.

## Why this and not the alternatives

**Recompute in the evaluation step instead of storing.** Rejected: recomputing
per-fold numbers means refitting five models to recover numbers that were
already computed and thrown away. It also lets training and evaluation drift
into disagreeing about the split.

**Store only the per-fold metric summaries, not the scores.** Rejected: the
summaries answer one question. The score table answers every question anyone has
asked so far, including ones not anticipated — stratify by anything present in
the row, calibrate, compare two runs site by site.

**Store the fitted fold models too, so the depth sweep needs no refit.** Not
done. Five boosters per run is real disk, and a stored model can go stale
against the code in a way a score table cannot. Refitting for a depth sweep
costs about a minute, which is cheap enough.

## Consequences

- `meta.json` grew. It is append-only by construction, and `predict.py` reads
  only `name`, `features`, `columns` and `model`, all untouched. A model trained
  before this change still loads.
- `oof.csv` is ~9 MB on the full training set. `models/final/oof.csv` is
  gitignored so it cannot be committed by accident when someone copies a model
  into `models/final/`. That is an addition to a file AGENTS.md section 5 says
  not to edit, and it is deliberate: it blocks data, it does not admit any.
- Scores are written to 9 significant figures. Far finer than any rank-based
  metric can notice, and a third smaller than full float repr.

## How to check it still holds

`tests/test_evaluation.py::test_the_oof_table_reproduces_the_metrics_it_was_built_from`
and `::test_evaluate_from_a_stored_oof_table_needs_no_fitting`.

The stored table must reproduce the pooled and per-fold numbers it came from to
1e-6.
