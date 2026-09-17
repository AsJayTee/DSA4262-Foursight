# Known gaps

What is missing, weak, or untested in this repo as of 2026-09-15.

**This file deliberately does not propose solutions.** It records what is
wrong, and the evidence for it, so that whoever picks up a piece of work can
reach their own conclusions about how to address it. If you find yourself
disagreeing with a framing here, that is fine — check it yourself.

Keep it current: when you close a gap, delete the entry. When you find a new
one, add it with whatever evidence you have.

Context for anything here: [docs/project-requirements.md](docs/project-requirements.md)
is what the project is graded on. [docs/data.md](docs/data.md) describes the
data — including [what read depth is](docs/data.md#read-depth), which several
entries below depend on and which is easy to misread.

---

## Evaluation

The harness is `scripts/evaluate.py`; what it reports is described in
[AGENTS.md section 6](AGENTS.md#6-metrics). Every number in this section can be
regenerated, and the JSON reports behind the current ones are in
`analysis/evaluation/reports/`. Quote numbers a command can reproduce.

- **Folds are unbalanced by positive rate**: 4.10%, 5.15%, 4.42%, 4.54%, 4.21%
  across folds 0-4, with fold sizes from 23,394 to 25,923 sites. PR AUC depends
  on the base rate, so a given model scores highest on fold 1 regardless of its
  merit. This is measured and reported per fold now, but it is not *fixed* —
  the split is frozen, so it cannot be. It is a reason to compare models paired
  rather than a reason to change anything.
- **Performance outside the training depth range is now measured, and it is
  bad.** *Depth* means how many separate RNA molecules gave us a measurement at
  one exact site — not repeated readings of one molecule, but distinct copies of
  that RNA from the cell. If this is the first time you are meeting the term,
  read [docs/data.md](docs/data.md#read-depth) before the numbers below, because
  almost everything here turns on it. Every site in the training set has at
  least 20 reads. The shipped model, scored on read-subsampled copies of its own
  held-out data (`--depth-sweep`, subsample seed 4262):

  | reads per site | 1 | 3 | 5 | 10 | 20 | full |
  |---|---:|---:|---:|---:|---:|---:|
  | `quantiles_v1` | 0.1527 | 0.2458 | 0.2991 | 0.3716 | 0.4277 | 0.4759 |
  | `pooled_v1` | 0.1448 | 0.2498 | 0.2996 | 0.3620 | 0.4094 | 0.4614 |

  A motif-only classifier (18-way one-hot, no signal at all) scores 0.1537. At
  one read the model is therefore doing nothing but reading the sequence
  pattern. SG-NEx A549 has median depth 3 and p25 depth 1 — see the Task 2
  section. **This is still the largest untested risk in the repo**: it is now
  measured, but nothing has been done about it, and every Task 2 number depends
  on it.
- **The scores are not calibrated, and the size of the error is measured.** The
  LightGBM model sets `is_unbalance=True`. Mean predicted probability is 0.0809
  against an actual positive rate of 0.0449 — a **1.80x overcount** (9,851
  expected positives vs 5,475 actual). By decile, the top bin predicts 0.608 and
  observes 0.327. ROC AUC and PR AUC are rank-based and unaffected, but any use
  of the scores as probabilities — counting modified sites per cell line in
  Task 2 — is currently wrong by close to a factor of two. The comment in
  `src/m6a/models/lightgbm.py` claiming `is_unbalance` keeps the outputs usable
  as probabilities is contradicted by this, and is still there. Nothing has been
  tried to fix it: no isotonic or Platt scaling, no threshold work, no
  alternative to `is_unbalance`.
- **The paired t-test over folds is anti-conservative, so its p-value reads
  stronger than it is.** Each fold's model trains on the other four, so any two
  training sets overlap heavily - measured here, 73.4% of fold 0's training rows
  are also in fold 1's. The five differences are therefore correlated, the
  observed scatter understates the true uncertainty, and p comes out too small
  (Dietterich 1998; Nadeau & Bengio 2003). The test sets are fine; it is the
  training sets that overlap. Read the win count (5/5) alongside any p-value,
  and do not quote `p = 0.0465` unqualified. The agreed fix is recorded in
  [docs/decisions/0006](docs/decisions/0006-strengthening-the-comparison-test.md)
  and is **not built**.
- **There is no error bar on the headline number itself.** 0.4759 is a point
  estimate. Nothing says how much it depends on which 121,838 sites happened to
  be in the dataset. The per-site scores are no longer stored
  ([0009](docs/decisions/0009-distributions-not-per-site-scores.md)), so a
  paired bootstrap over sites has to run in-process during the run; see
  [0006](docs/decisions/0006-strengthening-the-comparison-test.md) #1.
- **Comparisons between runs stop at PR AUC.** `--compare-features` pairs two
  runs fold by fold on one metric. It cannot say *which sites* two models
  disagree about, which is the question behind "should we ensemble these".
- **Every model is badly miscalibrated, and the size of the error tracks the
  rebalancing trick each one uses.** Measured with `evaluate.py --config ...`:

  | model | PR AUC | predicted positives | actual | overcount |
  |---|---:|---:|---:|---:|
  | `baseline_logistic` (`class_weight="balanced"`) | 0.4121 | 33,374 | 5,475 | **6.10x** |
  | `lightgbm_pooled` (`is_unbalance=True`) | 0.4634 | 15,192 | 5,475 | **2.77x** |
  | `lightgbm_quantiles` (`is_unbalance=True`) | 0.4759 | 9,851 | 5,475 | **1.80x** |

  So this is not one model's bug: every model in the repo handles the 4.49%
  positive rate by reweighting, and every one of them therefore emits scores
  calibrated to a rebalanced world rather than the real one. Nothing downstream
  of a score is currently safe to read as a probability.
- **Performance drops at the highest read depths, in every model, and nobody
  knows why.** PR AUC by depth band rises as expected up to 84-303 reads and
  then falls sharply in the 304+ band (6,095 sites, 263 positives):

  | depth band | 20-31 | 32-46 | 47-83 | 84-303 | **304+** |
  |---|---:|---:|---:|---:|---:|
  | `lightgbm_quantiles` lift | 10.1x | 10.7x | 11.1x | 10.8x | **9.8x** |
  | `lightgbm_pooled` lift | 9.8x | 10.5x | 10.6x | 10.6x | **8.9x** |
  | `baseline_logistic` lift | 8.8x | 9.5x | 9.6x | 9.5x | **6.5x** |

  Read the lift column, not PR AUC: the positive rate is near-identical across
  bands (4.32%-4.73%), so this is not a base-rate artefact. More evidence should
  make a site easier, not harder. The drop is consistent across three different
  models, so it is a property of the data or the labels rather than a model
  quirk, and it is worst for the weakest model. Untested hypotheses: 304+ sites
  are the most highly expressed transcripts and may be a different biological
  regime; m6ACE-Seq is an antibody method with known high background, so label
  quality may differ there; or the summary statistics may simply saturate. Nobody
  has looked.
- **Only single models are evaluated.** There is no way to score an ensemble, a
  rule combining two models, or a threshold choice.
- **A paired comparison reports only overall PR AUC, and only for the primary
  run.** `--compare-with` prints the comparison arm's per-fold numbers and
  nothing else - no calibration, no strata, no stored out-of-fold table - so
  answering "is the new model better *at low depth*" currently means running
  both configs separately and eyeballing two tables. That is the question the
  next round of work is about, so this is the most load-bearing thing the
  harness cannot do.
- **Cross-cell-line shift is smaller than this file previously assumed; depth
  shift is much larger.** Comparing the training data (Hct116) against SG-NEx
  A549 with depth held constant (both restricted to >=20 reads), the median of
  every one of the nine read features differs by less than 0.2 of the Hct116
  IQR, most by less than 0.1, and motif composition is near-identical (GGACT
  5.26% vs 5.89%, AAACA 8.16% vs 7.74%). The dominant covariate shift between
  our training data and any Task 2 dataset is depth, not cell line. (Caveat:
  7,129 A549 sites clear depth >=20 out of 62,744 sampled from 20 windows
  through the file, so this is directional, not a precise estimate. This one is
  not part of the harness — see the Infrastructure section.)
- **Changing the fold assignment invalidates every stored result.** The seed
  (4262) is frozen in `AGENTS.md` for that reason. The longer results accumulate
  in W&B, the more expensive any change to the split becomes: every `fold/*` key
  in every historical run stops being comparable, and nothing would warn anyone.

## The training set's provenance

- **The training set is SG-NEx Hct116 `replicate3_run1`, filtered to depth
  >= 20.** All 121,838 training sites appear in that SG-NEx sample — a 100%
  overlap on `(transcript_id, transcript_position)`, with an identical depth
  distribution (median 47, p25 32). SG-NEx has 1,391,230 sites for that sample
  and 39,378 transcripts against our 5,333.
- Two consequences. **The minimum depth of 20 is an artefact of the course's
  filtering, not a property of nanopore data** — m6Anet samples 20 reads per
  site, which is the likely reason for the threshold. And **Hct116 is not an
  independent cell line for us**: any Task 2 comparison that finds Hct116
  cleanest is at least partly circular.

## Modelling

- **Two feature sets exist**, both summary statistics over reads:
  `pooled_v1` (mean/std) and `quantiles_v1` (quantiles, IQR, tail spread).
  `quantiles_v1` is the better of the two: +0.0148 mean per-fold PR AUC, 5/5
  folds, paired p = 0.0465, 95% CI [+0.0004, +0.0292]. (The pooled gap is
  +0.0145; the mean of the per-fold differences is +0.0148. The paired test uses
  the latter.) Regenerate with
  `evaluate.py --config configs/quantiles.yaml --compare-features pooled_v1`.
- **The depth problem is not addressable by choosing between the existing
  feature sets.** Both collapse at the same rate: retention of full-depth PR
  AUC is 0.314 (pooled) vs 0.324 (quantiles) at one read, 0.524 vs 0.503 at
  three, 0.787 vs 0.784 at ten. The obvious hypothesis — that `quantiles_v1`
  should fail faster because 27 of its 101 columns go to exactly zero at depth
  1, against 9 of 38 for `pooled_v1` — was tested and is wrong. A
  gradient-boosted tree cannot split on a constant column, so it ignores it and
  falls back on what remains. At one read both feature sets reduce to the same
  nine raw measurements. Whatever addresses low depth, it is not feature
  selection within this family — but see the read-level entry below, where
  the same nine numbers produce a much better result under different training.
- **No model in the repo treats the reads as a set**, though the briefing
  frames this explicitly as a Multiple Instance Learning problem — the site
  carries the label, the reads do not, and only a fraction of reads at a
  positive site are modified. A throwaway torch-free probe suggests there is
  a lot here, and that it matters most exactly where the current models are
  weakest. A read-level gradient-boosted tree (every read inheriting its site's
  label, trained fold-out, read probabilities then combined by a fixed
  mean/max/90th-percentile rule with no learned second stage):

  | reads per site | 1 | 3 | 10 | full |
  |---|---:|---:|---:|---:|
  | read-level, best pooling | **0.2692** | **0.3303** | 0.3453 | 0.3666 |
  | `quantiles_v1` | 0.1527 | 0.2458 | **0.3716** | **0.4759** |

  (The read-level row came from `analysis/evaluation/scratch/mil_lite.py`, which
  is not part of the harness and draws its own subsample; the `quantiles_v1` row
  is from `--depth-sweep`. The two are close enough to compare but were not
  drawn together, so treat the crossover point as approximate.)

  The two approaches cross over somewhere around depth 5-10. At one read the
  read-level model scores 0.2692 against 0.1543 — a 74% improvement in the
  regime that makes up a quarter of SG-NEx. At full depth it loses badly.
- **The low-depth read-level result is confounded and the controlled
  experiment has not been run.** A read-level model is *implicitly trained at
  depth 1*, because each read is its own training row; it has no train/test
  mismatch at any depth. The site-level models were trained only at full depth.
  So the table above conflates "read-level architecture" with "trained at the
  depth it is tested at", and it is not known which is doing the work. The
  controlled comparison — a depth-augmented site-level model against the
  read-level one — is the obvious missing experiment, and it is far cheaper
  than building an attention-MIL network.
- **Read-level features are computed marginally, one feature at a time.**
  `mean_0_q95` is the 95th percentile of the `mean_0` column across reads, so
  the representation cannot express "these particular reads are jointly
  unusual across several measurements at once". Whether that joint structure
  carries signal is untested; it is a separate question from depth, and it is
  the one place a learned read encoder could beat a summary statistic at high
  depth, where the probe above currently loses.
- **Sequence information is barely used.** Only the central 5-mer is encoded,
  as an 18-way one-hot. The flanking bases of the 7-mer are discarded. The
  motif contributes +0.0123 PR AUC over signal-only features (5/5 folds, paired
  p = 0.0433), and 0.1537 on its own. Both from
  `evaluate.py --config configs/quantiles.yaml --ablate`.
- **No hyperparameter search has been run.** Every value in `configs/` was
  chosen by hand and none has been tuned.
- **Class imbalance is handled only by `is_unbalance=True`.** No resampling,
  alternative objective, or threshold work has been tried. See the calibration
  entry under Evaluation for what this currently costs.
- **No ensembling.**
- **No external training data has been sought**, although the handout
  explicitly permits searching for additional labels or training sets.
- The current best out-of-fold PR AUC is **0.4759** (quantile features,
  LightGBM). Two reference points now exist for reading that number: a
  motif-only classifier scores **0.1537** (3.42x random) with no signal data at
  all, and m6Anet's published site-level performance is **ROC AUC 0.83, PR AUC
  0.35** on HEK293T. The latter is a different cell line with different labels
  and is not a like-for-like comparison, but it does indicate 0.4759 is in a
  plausible range rather than suspiciously high.
- **`LightGBMModel.fit` fails on a DataFrame with non-string column names.** It
  passes `list(X.columns)` straight to `feature_name`, and LightGBM raises
  `AttributeError: 'int' object has no attribute 'encode'` from inside its own
  internals, naming neither the column nor the caller. Anyone building a frame
  from a numpy array hits this.

## Understanding the data

The briefing weights "do you fully understand the dataset, and its
limitations?" explicitly. Some of this is now measured; most is not.

- **No exploratory analysis exists.** Nobody has looked at how signal
  distributions differ between labelled and unlabelled sites.
- **Read depth does not predict the label, but it does drive model
  performance.** Positive and negative sites have near-identical depth
  (mean 91.2 vs 90.5, median 48 vs 47), and a depth-only classifier scores
  PR AUC 0.0457 against a base rate of 0.0449 — i.e. nothing. So the model is
  not learning coverage. That is a different question from how well it works
  *at* low depth, which is covered under Evaluation.
- **Per-motif base rates are now measured and vary by three orders of
  magnitude**: GGACT 22.58%, GAACT 11.25%, GGACA 8.94%, down to AAACC 0.17%,
  TAACC 0.07%, TAACA 0.02%. This is why a motif-only classifier reaches
  0.1537. Per-motif model performance is now measured too
  (`evaluate.py --by motif`): PR AUC lift ranges from 2.85x on GGACT to 19.5x on
  AGACA, and the two rarest motifs (TAACA, TAACC — 1 and 2 positives) cannot be
  scored at all. Why lift is *lowest* on the motif with the highest base rate is
  unexamined.
- **Positive sites cluster in 1,507 of 3,852 genes.** The structure of that
  clustering — whether positives concentrate in particular transcript regions,
  genes, or expression levels — is unexamined. Transcript position does not
  separate the classes on its own (mean 1594 for positives vs 1617 for
  negatives).
- **The labels' provenance is still undocumented, and it bears on the
  low-depth work.** m6ACE-Seq is an antibody-immunoprecipitation method: it
  achieves single-nucleotide resolution but depends on antibody quality, is
  known for high background, and does not quantify the modification level at a
  site. So a `0` conflates "unmodified" with "below the assay's detection
  limit". m6ACE-Seq is itself sequencing-based, so its sensitivity drops on
  lowly-expressed transcripts — which are exactly the transcripts with low
  nanopore depth. Any measurement of low-depth performance is therefore partly
  confounded with label quality, and read-subsampling experiments (which draw
  from high-expression sites with good m6ACE-Seq coverage) cannot see this.

## Task 2 - SG-NEx

No prediction has been run and no analysis exists. The data has now been
catalogued, and one blocker is much larger than expected.

- **The data is public, needs no credentials, and is already in our input
  format.** `s3://sg-nex-data/data/processed_data/m6Anet/` holds 23 direct
  RNA-Seq samples across 7 cell lines: H9 (6), HEYA8 (5), K562 (3), Hct116 (3),
  MCF7 (2), HepG2 (2), A549 (2). Each sample directory holds `data.json`,
  `data.info`, `data.index` and `data.readcount`. The `data.json` is the same
  nested structure as `dataset0.json.gz` with the same nine features, so
  `scripts/predict.py` runs on it unmodified — no nanopolish and no m6Anet
  install is needed for Task 2. Total volume is 33.4 GB uncompressed
  (mean 1.45 GB per sample, max 2.70 GB).
- **Read depth in SG-NEx is far below anything the model has seen.** For
  A549 `replicate6_run1`: 1,500,579 candidate sites, 20.2M reads, median depth
  **3**, p25 depth **1**. Only 12.5% of sites have the >= 20 reads that every
  training site has; 69.4% have >= 2 and 39.3% have >= 5. Read against the
  depth table under Evaluation, this means the majority of any naive Task 2
  prediction run would be at or near the motif-only floor. Cross-cell-line
  differences computed from those scores would track sequencing depth rather
  than biology.
- **Hct116 is not an independent cell line for us.** See "The training set's
  provenance".
- **Note H9 is not a cancer cell line** (it is an embryonic stem cell line),
  while the briefing scopes the task to 5 cancer cell lines. Which of the 7
  available lines to report on is undecided.
- No path exists for running prediction across many samples.
- No cross-cell-line comparison of any kind.
- No interactive visualisation. This is part of the 50% report component.

## Benchmarking against m6Anet

- **m6Anet has never been installed or run.** Two known obstacles are recorded
  in [analysis/m6anet/README.md](analysis/m6anet/README.md); neither has been
  resolved.
- No automated comparison against the simple baseline and random classifier,
  though the report requires both. A motif-only classifier (PR AUC 0.1537) is
  a third reference point worth including — it is stronger than random and
  requires no signal data.
- m6Anet's **published** figures are ROC AUC 0.83 / PR AUC 0.35 on HEK293T
  (Hendra et al. 2022). If m6Anet cannot be installed, that is the fallback
  comparison — on different data, which has to be stated plainly.

## Infrastructure

- **`manifest.json` does not exist in the R2 bucket.** Downloads currently fall
  back to comparing file sizes against the remote object, which will not catch
  a corrupted file whose length happens to match.
- **There is no upload path to R2.** Adding data to the bucket is manual and
  undocumented, which matters when the evaluation data is released.
- **`setup_remote.sh` and `bootstrap.sh` have never been run against a real
  VM.** They are written and reviewed but unproven. This is now the *only*
  unverified link in the chain: W&B logging is confirmed working end to end from
  a laptop (a full run with figures, tables and artifacts landed at
  [run y2lk7ilf](https://wandb.ai/dsa4262-team/dsa4262-project/runs/y2lk7ilf)),
  so what remains untested is specifically whether a fresh Ubuntu instance
  reaches the same state - apt packages, the venv, `pip install -e '.[train]'`,
  `wandb login`, and the R2 download. Nobody should discover on a paid instance
  that `bootstrap.sh` fails at step two.
- **`doctor` reports "R2 configured" based on the presence of keys, not their
  validity.** Placeholder values pass. The W&B half of this is fixed - `doctor`
  now calls the W&B API and FAILs on a key that does not work - but nothing
  checks that the R2 credentials can actually reach the bucket, so a broken
  download surfaces as a traceback in `download_data.py` rather than in the
  command whose job is to tell you what is wrong.
- **The SG-NEx figures in the Task 2 section came from an ad-hoc S3 crawl that
  is not in this repo.** The sample counts, the 33.4 GB total and the A549 depth
  distribution cannot be re-derived or challenged without redoing that crawl by
  hand. The evaluation measurements are no longer in this position - see
  `scripts/evaluate.py` - but the SG-NEx cataloguing still is.
- **`analysis/evaluation/scratch/` holds four rough scripts with hardcoded
  paths**, three of which are the only source of a number quoted above (the
  dataset profile, the read-level MIL probe, and the Hct116-vs-A549 comparison,
  which additionally needs an input file nothing in this repo produces). See the
  README there.
- **No CI.** Tests run only when someone remembers to run them.
- **The SSH access story is undecided** — whether instances come from Ronin or
  are self-funded, and how keys reach teammates.
- **Nobody has rehearsed the evaluator experience**: a clean machine, a fresh
  clone, the README followed literally and nothing else. That is what the 5%
  documentation grade measures.

## Deliverables not started

- The report.
- The interactive visualisation.
- **The AI-use log.** The report requires a table of AI tool usage plus at
  least two documented instances where an AI tool's output was wrong,
  misleading, or rested on an unverified assumption — including how it was
  detected and what was done. Nothing is being recorded yet. These are far
  harder to reconstruct after the fact than to note as they happen.
- Hackathon documents (shared Google doc per session, posted to Zulip).
