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
- **The repo's headline comparison does not survive a correct significance
  test, and the claim needs restating wherever it appears.** The paired t-test
  over five folds is anti-conservative: each fold's model trains on the other
  four, so 73.4% of fold 0's training rows are also in fold 1's, the five
  differences are correlated, and the observed scatter understates the true
  uncertainty (Dietterich 1998; Nadeau & Bengio 2003). The Nadeau & Bengio
  corrected test is now built and reported alongside the naive one. On
  `quantiles_v1` vs `pooled_v1`:

  | | mean difference | 95% CI | p |
  |---|---:|---|---:|
  | unpaired (wrong) | +0.0148 | - | 0.3095 |
  | paired, naive (optimistic) | +0.0148 | [+0.0004, +0.0292] | **0.0465** |
  | paired, corrected (quote this) | +0.0148 | [-0.0068, +0.0364] | **0.1305** |

  So the improvement that has been quoted at `p = 0.0465` is `p = 0.1305` once
  the overlap is accounted for, and the corrected interval includes zero. **Five
  folds cannot establish it**, and anywhere the report quotes 0.0465 as
  significant is wrong.

  Fifty can. `--repeats 10` runs the whole cross-validation over ten
  independently seeded splits (repetition 0 is the canonical seed-4262 one, so
  every number above is unchanged):

  | | observations | mean difference | corrected 95% CI | corrected p | wins |
  |---|---:|---:|---|---:|---:|
  | 5 folds | 5 | +0.0148 | [-0.0068, +0.0364] | 0.1305 | 5/5 |
  | 10 x 5 folds | 50 | +0.0164 | [+0.0086, +0.0241] | **0.000095** | **50/50** |

  **The result holds.** `quantiles_v1` beats `pooled_v1` on every one of fifty
  splits, and it survives the correction comfortably. The five-fold test was not
  wrong about the direction, it was underpowered. Regenerate with
  `evaluate.py --config configs/quantiles.yaml --compare-features pooled_v1 --repeats 10`
  (about 25 minutes on a warm cache; the 5-fold version is 100 seconds).
- **The headline number now has two error bars, and they answer different
  questions.** 0.4759 used to be a bare point estimate. Measured
  (`--bootstrap 2000`, resampling the 121,838 sites; and `--repeats 10`,
  resplitting them):

  | | what varies | `quantiles_v1` | `pooled_v1` |
  |---|---|---|---|
  | bootstrap over sites | which sites are in the dataset | 0.4761, 95% CI [0.4609, 0.4908] | 0.4616, [0.4460, 0.4765] |
  | repeated CV | which split was used | mean 0.4785, sd 0.0229, range [0.4263, 0.5264] | - |

  **Do not use the bootstrap to dodge the corrected t-test.** On the difference
  between the two feature sets the bootstrap gives +0.0145, 95% CI
  [+0.0074, +0.0217], positive in 100% of 2,000 resamples - which looks far more
  decisive than the corrected 5-fold test's p = 0.1305, and is not a
  contradiction. The bootstrap **holds the split fixed**: it resamples sites
  against one set of fold models, so it cannot see split-to-split variation at
  all. The corrected t-test is the one that carries the train/test overlap. Quote
  the repeated-CV result (50/50, corrected p = 0.000095) as the answer to "is
  this better"; quote the bootstrap as the answer to "how precise is 0.4759".

  What neither can tell you: every resample and every split inherits the
  depth >= 20 floor, so none of them says anything about how the model behaves
  on SG-NEx.
- **No comparison can say *which sites* two models disagree about**, which is
  the question behind "should we ensemble these". It was never supported, and
  it is now structurally impossible after the fact: per-site scores are not
  stored at all
  ([0009](docs/decisions/0009-distributions-not-per-site-scores.md)), so
  answering it means adding it to the run that computes the scores, while they
  are still in memory. A comparison does now test *inside* strata - by read depth
  band and by motif - so "is this better at low depth" is answerable even though
  "which sites" is not.
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
  `quantiles_v1` is the better of the two, and this is now **properly
  established** - though not by the number this file used to quote. Over ten
  repetitions of the 5-fold split (50 paired observations): mean difference
  **+0.0164**, winning **50 out of 50**, corrected paired p = **0.000095**, 95%
  CI [+0.0086, +0.0241]. On the canonical five folds alone it is +0.0148, 5/5,
  and corrected p = 0.1305 - directionally right but underpowered, so
  `p = 0.0465` should not be quoted anywhere. (The pooled gap is +0.0145; the
  paired test uses the mean of the per-fold differences.) Regenerate with
  `evaluate.py --config configs/quantiles.yaml --compare-features pooled_v1 --repeats 10`.
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
  motif contributes +0.0123 PR AUC over signal-only features (5/5 folds, naive
  paired p = 0.0433 - which carries the same correction problem as every other
  five-fold p-value here, see Evaluation), and 0.1537 on its own. Both from
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
