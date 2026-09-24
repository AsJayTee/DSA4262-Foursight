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
  as probabilities is contradicted by this, and is still there. **Nothing has
  been tried to fix it**: no isotonic or Platt scaling, no alternative to
  `is_unbalance`. (Threshold work now exists — see the next entry — but that
  measures the consequence rather than fixing the cause.)
- **A threshold sweep exists now, and 0.5 is not a neutral choice.** Every other
  metric here integrates over all thresholds, which is right for comparing
  models and useless for using one — and Task 2 has to *count* modified sites,
  which needs an operating point. Measured
  (`evaluate.py --config configs/quantiles.yaml`, full training set):

  | operating point | threshold | precision | recall | F1 | sites called |
  |---|---:|---:|---:|---:|---:|
  | best F1 | 0.58 | 0.4849 | 0.5571 | 0.5185 | 6,290 |
  | count-matched | 0.65 | 0.5133 | 0.5149 | 0.5141 | 5,492 |
  | 0.5, the default | 0.50 | 0.4513 | 0.5976 | 0.5143 | 7,250 |

  5,475 sites really are positive, so **0.5 overcounts by 1.32x** — a different
  number from the 1.80x above, because that one comes from *summing* the scores
  and this one from *thresholding* them. Both are the same miscalibration and
  they are not interchangeable; quoting either as "the" overcount is wrong.

  **The number to carry into Task 2 is the swing: a site count moves 2.1x
  between threshold 0.3 (10,610 sites) and 0.7 (4,943).** Both are choices
  someone could make with a straight face, so any claim of the form "cell line X
  has N modified sites" is partly a claim about a threshold nobody wrote down.
  That is not a statistical interval and no confidence interval would show it.

  What this does **not** do (**LOW PRIORITY** - deliberately parked
  2026-09-22; the project optimises PR AUC and no Task 2 count is being quoted
  yet, so this is not on the critical path): the operating points are fitted out
  of fold on depth >= 20 data. A threshold is an absolute cut on a score distribution, and
  depth is known to shift that distribution, so the threshold that counts
  correctly here is not necessarily the one that counts correctly on SG-NEx at
  median depth 3. Nobody has measured that, and it is a worse problem for a
  threshold than for a rank metric.
  See [0019](docs/decisions/0019-a-threshold-sweep-because-a-ranking-cannot-count.md).
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
- **The canonical five folds can manufacture an effect that is not there, and
  now there is a measured example of it.** Everything above and
  [0013](docs/decisions/0013-every-run-is-a-distribution.md) argues the five-fold
  case one way: it is *underpowered*, so it misses real effects. `quantiles_joint_v1`
  (2026-09-24) is the same coin's other face, and it is the more dangerous one,
  because a false positive is a result somebody quotes.

  | `quantiles_joint_v1` vs `quantiles_v1` | mean difference | wins | naive paired p | corrected p |
  |---|---:|---:|---:|---:|
  | repetition 0 - the canonical split | **+0.0097** | **5/5** | **0.0278** | 0.0874 |
  | all 10 repetitions | +0.0013 | 29/50 | 0.1819 | **0.7140** |

  On the canonical split this feature set sweeps every fold, at an effect size
  (+0.0097) close to the genuinely real +0.0109 that `quantiles_flank_v1`
  produced on the same day, and at a naive paired p that **clears 0.05**. Run it
  over fifty observations and it is +0.0013 at 29/50 - a coin flip.

  Two guards had to hold, and both did:

  - The **Nadeau & Bengio correction** refused it on five folds: 0.0278 naive
    against 0.0874 corrected. Anyone quoting the uncorrected paired test would
    have published a null as a finding.
  - The **repetitions** then showed the effect was not merely unproven but
    approximately zero. The correction alone would have left it as "promising,
    needs more evidence", which is not what it is.

  This is the strongest argument in the repo for why `standard` runs ten
  repetitions by default rather than offering it as a flag, and it cost one
  22-minute run to obtain. **Do not quote a 5-fold result, corrected or not.**
- **`oof/pr_auc` is a repetition-0 number, so the column everyone sorts the run
  table on can rank a null above a real result.** This is documented behaviour
  ([0012 section 3](docs/decisions/0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md):
  `fold/*` and the pooled out-of-fold metrics are the canonical split only) and
  it is still a trap, because
  [docs/wandb-panels.md](docs/wandb-panels.md) calls `oof/pr_auc` "the headline"
  and lists it above `rep/pr_auc_mean`.

  Measured on the two runs from 2026-09-24:

  | run | `oof/pr_auc` (5 folds) | `rep/pr_auc_mean` (50) | paired vs `quantiles_v1` |
  |---|---:|---:|---|
  | `quantiles_v1` | 0.4759 | 0.4785 | - |
  | `quantiles_joint_v1` | **0.4851** | 0.4798 | +0.0013, corrected p = 0.71 |
  | `quantiles_flank_v1` | 0.4897 | 0.4894 | +0.0109, corrected p = 0.0006 |

  Sorted on `oof/pr_auc`, the null arm (0.4851) sits above `mlp_quantiles`
  (0.4783) and within 0.0007 of `quantiles_depth_augmented` (0.4844), and a
  reader would reasonably conclude it is the project's second-best feature set.
  `rep/pr_auc_mean` tracks the paired result exactly - 0.4798 against 0.4785 is
  the +0.0013 the test reports.

  **Sort on `rep/pr_auc_mean` when comparing, and read `oof/pr_auc` as what it
  is: one split's number, kept unchanged so historical runs stay comparable.**
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
- **Performance collapses above 600 reads per site, and it is a cliff rather
  than a slide.** This entry used to say the drop began at 304. Splitting the
  top band at 600 ([0017](docs/decisions/0017-split-the-top-depth-band.md))
  shows that is wrong:

  | depth band | 20-31 | 32-46 | 47-83 | 84-303 | 304-599 | **600+** |
  |---|---:|---:|---:|---:|---:|---:|
  | `lightgbm_quantiles` lift | 10.1x | 10.7x | 11.1x | 10.8x | **11.07x** | **8.84x** |

  304-599 is the **best** band in the table. Everything above 600 falls off a
  cliff: 2,837 sites, 129 positives, 8.84x against a neighbour at 11.07x.

  Read the lift column, not PR AUC: the positive rate is near-identical across
  bands (4.32%-4.73%), so this is not a base-rate artefact. More evidence should
  make a site easier, not harder. The old 304+ number was recorded for all three
  models and the drop was present in each, so this is a property of the data or
  the labels rather than a model quirk - the split has only been re-measured on
  `lightgbm_quantiles` so far, and confirming it on the other two is one command.

  A threshold effect at 600 fits a **distinct population** better than it fits
  gradual saturation of the summary statistics, which would predict a decline
  starting well before 600. Untested hypotheses now narrowed to 2,837 sites:
  they are the most highly expressed transcripts and may be a different
  biological regime; or m6ACE-Seq's antibody background differs there. Nobody
  has looked at what those sites *are* - which transcripts, which genes, whether
  they are ribosomal or mitochondrial. That is now a tractable question rather
  than a vague one.
- **Only single models are evaluated** - and this needs **no harness change**.
  A rule combining two models ("model A above depth 10, model B below it", the
  obvious thing to try given the depth-sweep crossover under Modelling) is an
  *experiment*: a new model class in `src/m6a/models/` plus a config, evaluated
  by the existing pipeline exactly like any other arm. Recorded here so nobody
  rebuilds the comparison machinery to support it. **LOW PRIORITY** until
  somebody actually wants to run it.
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
- **Paralogous genes could leak across the split, and nobody has checked.**
  The split groups on `gene_id`, which stops transcripts of the *same* gene
  straddling a fold. It does nothing about two *different* genes that share
  near-identical sequence - paralogs, gene families, recent duplications. A
  paralog pair split across folds is a genuine leak: the model sees one copy in
  training and is scored on the other. Nobody has measured sequence similarity
  across fold boundaries, so the size of this is unknown. It is the only
  unguarded leakage vector identified so far, and it is the first thing to rule
  out whenever a number looks too good - see the m6Anet entry under Modelling.
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
- **A plain neural net ties LightGBM on the headline and is much worse where it
  matters.** `configs/mlp.yaml` is a 128/64 feed-forward net on the *same*
  `quantiles_v1` columns ([run 1wzvr5hi](https://wandb.ai/dsa4262-team/dsa4262-project/runs/1wzvr5hi)).
  Paired over 50 observations against `lightgbm_quantiles` it is **-0.0021,
  21/50 wins, corrected p = 0.7029** — indistinguishable. Then:

  | | `oof/pr_auc` | `depth/3` | `depth/1` | `calib/count_ratio` |
  |---|---:|---:|---:|---:|
  | `lightgbm_quantiles` | 0.4759 | 0.2458 | 0.1527 | 1.80x |
  | `mlp_quantiles` | 0.4783 | **0.2019** | **0.1045** | **4.31x** |

  **At one read the net scores 0.1045, below the 0.1537 a motif-only classifier
  gets with no signal data at all** — it has not merely degraded, it has become
  worse than reading the sequence. It also overcounts by 4.31x against 1.80x.
  Two models that are a coin-flip apart on the headline are nowhere near each
  other on the two numbers Task 2 actually depends on. Nobody has looked at why
  the net is so much more brittle to depth shift; depth-augmented training
  (`train_depths` on the mlp config) is the obvious first thing to try and has
  not been run.
- **The reads can now reach a model, and one does — but it has never been run
  on the full dataset and the cost is the problem.** The pipeline used to be
  structurally site-level end to end: `FeatureExtractor.site_features` returns
  one row per site, `Dataset` carried only `X` (site x feature), and
  `Site.reads` was consumed by extraction and discarded. Ragged per-read data
  now travels beside the feature table as a `ReadBlocks`
  ([0023](docs/decisions/0023-reads-travel-beside-the-feature-table.md)), and
  `configs/mil.yaml` is a gated-attention MIL model (Ilse et al. 2018) that
  consumes it.

  It works end to end on a 5,000-site smoke run. **It has not been run on the
  full training set, and on this machine it should not be**: measured at about
  3.4 s per epoch per fold on 4,000 sites, which extrapolates to **~1 hour for
  one repetition and ~10 hours for a `standard` run**, CPU-only. So the MIL arm
  needs a GPU instance or a deliberately reduced profile, and AGENTS.md section 6
  says a `--quick` number is not a number to quote. **Budget for it before
  starting the instance, not after.**

  When it is run, compare it against `configs/quantiles_depth_augmented.yaml`
  and not against `configs/quantiles.yaml` — otherwise it repeats the exact
  confound the entry two below resolved, and credits the architecture for what
  training depth does.
- **A read-level model cannot be shipped through `scripts/predict.py`.**
  `predict.py` calls `extractor.transform(iter_sites(...))` and then
  `model.predict_proba(features)` with no reads, so a model declaring
  `CONSUMES_READS` would raise there. Teaching it to build a `ReadBlocks` means
  either a second pass over the input file or restructuring the single pass it
  makes. Until that is done the MIL model is a research arm and not a candidate
  for `models/final/` — which matters, because the graded deliverable is what
  `predict.py` loads.
- **The read-level probe that started all of this, kept for reference.** A
  throwaway torch-free read-level gradient-boosted tree (every read inheriting
  its site's label, trained fold-out, read probabilities combined by a fixed
  mean/max/90th-percentile rule with no learned second stage):

  | reads per site | 1 | 3 | 10 | full |
  |---|---:|---:|---:|---:|
  | read-level, best pooling | 0.2692 | 0.3303 | 0.3453 | 0.3666 |
  | `quantiles_v1`, trained at full depth | 0.1527 | 0.2458 | **0.3716** | **0.4759** |

  (The read-level row came from `analysis/evaluation/scratch/mil_lite.py`, which
  is not part of the harness and draws its own subsample; the `quantiles_v1` row
  is from `--depth-sweep`. The two are close enough to compare but were not
  drawn together, so treat the crossover point as approximate.)

  **Read this table together with the resolved entry below, not on its own.**
  The low-depth gap it shows is almost entirely training depth rather than
  read-level architecture — a depth-augmented site-level model closes it.
- **RESOLVED: the read-level probe's low-depth win was training depth, not
  architecture.** This entry used to say the comparison was confounded — a
  read-level model is *implicitly trained at depth 1*, because each read is its
  own training row, while every site-level model here was trained only at full
  depth. The control has now been run. Train the **same** site-level model on
  rows drawn at several depths
  ([0022](docs/decisions/0022-training-rows-may-come-from-several-depths.md),
  `configs/quantiles_depth_augmented.yaml`) and almost all of the gap closes:

  | reads per site | 1 | 3 | 5 | 10 | 20 | full |
  |---|---:|---:|---:|---:|---:|---:|
  | `quantiles_v1`, trained at full depth | 0.1527 | 0.2458 | 0.2991 | 0.3716 | 0.4277 | 0.4759 |
  | **same, trained at 1/3/5/10/full** | **0.2593** | **0.3434** | **0.3735** | **0.4166** | **0.4495** | **0.4844** |
  | read-level probe | 0.2692 | 0.3303 | - | 0.3453 | - | 0.3666 |

  At one read the depth-augmented site-level model lands within 0.01 of the
  read-level probe; at depth 3 it passes it; at depth 10 and full depth it is
  far ahead. **And it costs nothing at full depth** — paired over 50
  observations against the plain run, +0.0056, 39/50 wins, corrected
  p = 0.2605: not established as better, not worse.

  Regenerate:
  ```bash
  python scripts/evaluate.py --config configs/quantiles_depth_augmented.yaml \
         --compare-run 1cn5n37z
  ```
  ([run psdwqobf](https://wandb.ai/dsa4262-team/dsa4262-project/runs/psdwqobf)
  against [1cn5n37z](https://wandb.ai/dsa4262-team/dsa4262-project/runs/1cn5n37z);
  about 37 minutes on a warm cache.)

  **What this does not establish.** The depth-sweep rows are single values per
  depth with no per-fold vector underneath
  ([0018](docs/decisions/0018-per-stratum-vectors-are-logged-for-later.md)), so
  +0.1066 at depth 1 is an effect size and not a p-value. The read-level row
  came from `mil_lite.py` with its own subsample draw, so the crossover is
  approximate. And both arms still inherit the depth >= 20 floor, so none of
  this says what happens on genuinely shallow sites.

  **The open question is now much narrower**: is there anything left for a
  read-level architecture *after* controlling for training depth? Two cheap
  things nobody has done — `train_depths: [1]` (trained at depth 1 only, the
  probe's implicit setting exactly), and running `configs/mil.yaml` against
  `configs/quantiles_depth_augmented.yaml` rather than against the plain one.
- **TESTED, AND IT DOES NOT PAY: joint within-read structure, hand-crafted, is
  worth +0.0013 and is not distinguishable from noise.** This entry used to say
  the representation is marginal - `mean_0_q95` is the 95th percentile of one
  column and `sd_0_q95` of another, so nothing can express that *the same
  molecule* was extreme on both - and that whether that structure carries signal
  was untested. It has now been tested.

  `quantiles_joint_v1`
  ([run jqoq58tn](https://wandb.ai/dsa4262-team/dsa4262-project/runs/jqoq58tn))
  adds 17 columns to `quantiles_v1`: quantiles of each read's Mahalanobis
  distance from its own site's read centroid (correlation matrix shrunk 0.1
  toward the identity, so it degrades to the independent reading rather than to
  a singular matrix), and the fraction of reads jointly extreme on two, three
  and four of the nine measurements, plus one-sided and centre-position
  variants.

  Paired over 50 observations: **+0.0013, 29/50 wins, corrected p = 0.7140**,
  95% CI [-0.0059, +0.0085]. Below the detection threshold and centred on zero.
  **No stratum rescues it** - by read-depth band the differences are +0.0047,
  -0.0039, +0.0036, -0.0005, -0.0059, -0.0092, none of them near significance,
  so it is not the case that joint structure helps where reads are plentiful and
  is diluted elsewhere.

  ```bash
  python scripts/evaluate.py --config configs/quantiles_joint.yaml \
         --compare-features quantiles_v1
  ```

  **What this licenses, and what it does not.** This is the cheap de-risking
  probe for B1/B2, and it came back negative, so the honest reading is that a
  learned per-read encoder is less likely to pay than the literature review
  argued. It is **not proof** that joint structure is absent: a hand-crafted
  Mahalanobis distance is one parameterisation of "jointly unusual", and a
  learned encoder could find a coupling this does not express. What it removes
  is the cheap positive evidence that would have justified ~30 GPU-hours. B1
  should not be booked on the strength of the joint-structure argument alone.

  One implementation caveat worth carrying if anyone revisits it: the distance
  is measured against **the site's own** centroid and covariance, so at a site
  where a minority of reads are genuinely modified, those reads inflate the
  covariance in exactly the direction that would have made them look unusual.
  The signal is attenuated by construction. A global (cross-site) whitening
  would not have that problem and does not fit `site_features(site)`, which sees
  one site at a time.
- **TESTED, NOT RESOLVED: the 3 x 3 measurement grid inside each read.** The
  nine numbers per read are three measurements at three **ordered, adjacent**
  pore positions - a 5-mer sits in the pore and the RNA ratchets one base at a
  time, so -1, 0 and +1 are consecutive states of the same molecule. Every
  feature set flattens that away and treats the nine columns as interchangeable;
  `attention_mil`'s encoder (`Linear(9 -> 64)`) does too.

  `quantiles_grid_v1`
  ([run 7ryz89r8](https://wandb.ai/dsa4262-team/dsa4262-project/runs/7ryz89r8))
  adds 21 columns: per-read centre-surround and gradient contrasts along the
  position axis - the only two length-3 convolutions that exist - quantiled
  across reads.

  Paired over 50 observations: **+0.0032, 29/50 wins, corrected p = 0.4070**,
  95% CI [-0.0045, +0.0109]. Below what the harness resolves, and the 29/50 win
  count says the direction is not consistent either. It is also **worse at one
  read** (0.1409 against 0.1527), which contradicts the prediction that a
  within-read feature would help where cross-read features cannot.

  So the CNN-over-the-position-axis premise is **weakly supported at best**. The
  information exists but the tree is evidently already extracting it from the
  marginal columns.

- **A method correction, recorded because it cost a run and will cost another
  one otherwise: a univariate screen predicts MARGINAL signal, not INCREMENTAL
  value.** Before building `quantiles_grid_v1` its columns were screened
  univariately on 40,000 sites and looked excellent - the quantile of per-read
  differences scored |AUC - 0.5| = **0.1198**, second only to `mean_m1_q50` at
  0.1320, while the *difference of quantiles* (which the model already has)
  scored 0.0008. That contrast is real and it is still the reason the feature is
  not derivable from what exists.

  It did not survive contact with the other 101 columns: **+0.0032, corrected
  p = 0.4070.** A column can be strongly predictive on its own and add nothing
  once a gradient-boosted tree has had a hundred correlated columns to work
  with.

  The screen is still worth the minute it costs - it is a cheap way to *reject*
  a feature with no marginal signal at all - but it must not be read as a
  prior on the paired result. Scoreboard so far: `flank` screened strong and won
  (+0.0109), `grid` screened strong and did not (+0.0032), `joint` was never
  screened and lost (+0.0013). One for two on the positive direction.
- **RESOLVED: the 7-mer's flanking bases were being thrown away, and they are
  worth +0.0109.** This entry used to say only that they were discarded and that
  nobody had measured the cost. They have now been measured, and recovering them
  is the largest established feature-set gain in the project.

  `motif_onehot` encodes `kmer[1:6]`, so the outer two bases of the 7-mer were
  parsed out of the file and dropped at extraction. `quantiles_flank_v1`
  ([run 0bw7mvz8](https://wandb.ai/dsa4262-team/dsa4262-project/runs/0bw7mvz8))
  adds them as 4 + 4 indicators and nothing else:

  | | `quantiles_v1` | `quantiles_flank_v1` |
  |---|---:|---:|
  | `oof/pr_auc` | 0.4759 | **0.4897** |
  | `oof/roc_auc` | 0.9169 | 0.9199 |
  | `rep/pr_auc_mean` over 50 | 0.4785 | 0.4894 |
  | `calib/count_ratio` | 1.80x | 1.76x |

  Paired over 50 observations: **+0.0109, 48/50 wins, corrected p = 0.0006**.
  Comfortably above the detection threshold, and at the top of the +0.003-0.009
  prior it was queued under. Regenerate:

  ```bash
  python scripts/evaluate.py --config configs/quantiles_flank.yaml \
         --compare-features quantiles_v1
  ```

  **Why 4 + 4 and not a 288-way 7-mer one-hot.** All 288 combinations occur
  (18 motifs x 4 x 4, every cell populated), but 5,475 positives spread over 288
  categories leaves the thin ones fitting noise. The effect is also largely
  *additive* across motifs, which two 4-way indicators capture and 288 fragment:
  measured before the run, the right flank moves the positive rate from 1.98%
  (A) to 6.88% (G) overall, and it survives conditioning on the motif - within
  `GGACT` the rate is 14.3% under a right-flank A against 27.8% under a G, and
  within `AGACT` 2.7% against 11.6%, the same direction in nearly every motif. A
  motif-only rate table scores 0.15474 mean logloss against 0.15010 for one
  keyed on the full 7-mer.

  **What is left of the original entry.** The motif still contributes +0.0123
  PR AUC over signal-only features (5/5 folds, naive paired p = 0.0433 - which
  carries the same correction problem as every other five-fold p-value here, see
  Evaluation), and 0.1537 on its own, from
  `evaluate.py --config configs/quantiles.yaml --ablate`. **That 0.1537 floor is
  now stale as a reference point**: it is an 18-way motif-only classifier, and
  the sequence-only model that would match the new feature set has flanks in it
  and has not been run. Anyone quoting "at one read the model is doing nothing
  but reading the sequence pattern" should re-run `--ablate` on
  `configs/quantiles_flank.yaml` first.

  Two things this opens rather than closes:

  - **It is orthogonal to depth augmentation and nobody has combined them.**
    Sequence context is depth-independent information, and the gain does show up
    at every depth (0.1633 against 0.1527 at one read, 0.2611 against 0.2458 at
    three - effect sizes only, since the depth sweep has no per-fold vector
    underneath it, see
    [0018](docs/decisions/0018-per-stratum-vectors-are-logged-for-later.md)).
    `quantiles_flank_v1` with `train_depths: [1, 3, 5, 10, null]` is one config
    file and would say whether the two gains stack.
  - **0.4897 is the highest pooled number in the project, and that is not the
    same as being the best model.** `quantiles_depth_augmented` sits at 0.4844
    and the two have never been paired. Comparing them is
    `--compare-run psdwqobf`, which refits nothing.
- **The experiment queue, ranked by what this harness can actually measure.**
  Derived from [docs/literature-review.md](docs/literature-review.md) and a
  follow-up exchange with it, then filtered through the detection threshold
  below. **The harness resolves effects between +0.006 and +0.016 at 50
  observations** - measured: +0.0164 came back at corrected p = 0.000095,
  +0.0056 at p = 0.2605, -0.0021 at p = 0.7029. An experiment whose honest prior
  is +0.004 cannot be established here however well it is run.

  Cheap first, because two of these are feature sets testable in ~15 minutes
  with full statistical power, and one of them de-risks a ~26-hour GPU job:

  | # | experiment | prior | cost | resolvable? | outcome |
  |---|---|---:|---|---|---|
  | A1 | recover the 7-mer flanking bases (26 sequence columns, not 18) | +0.003–0.009 | ~15 min | borderline | **WON: +0.0109, 48/50, corrected p = 0.0006** |
  | A2 | joint within-read features (per-read Mahalanobis from the site's own centroid) | unknown | ~15 min | — | **NULL: +0.0013, 29/50, corrected p = 0.7140** |
  | A3a | cross-site: candidate density and spacing (`nbr_struct`) | unknown | ~35 min | — | built, queued |
  | A3b | cross-site: neighbour *measurements*, +/-50 and +/-200 nt (`nbr_signal`) | unknown | ~45 min | — | built, queued |
  | A3c | cross-site: transcript-level leave-one-out (`transcript`) | unknown | ~35 min | — | built, queued |
  | A4 | the 3x3 within-read grid (`quantiles_grid_v1`) | unknown | ~35 min | — | **NOT RESOLVED: +0.0032, 29/50, p = 0.4070** |
  | A5 | `flank` x depth augmentation | stacked? | ~50 min | — | built, queued |
  | B1 | the existing attention-MIL + 7-mer conditioning + mean/std branch + log N | +0.009 | ~26 h GPU | borderline | **weakened by A2** |
  | B2 | contextual Deep Set (bag-context interaction layer) | +0.015 | GPU | yes | gated on B1 |
  | C1 | Platt calibration | 0 on PR AUC | ~1 h | n/a | not started |

  **A2 was run first with A1, and it did its job.** Both the literature review
  and the entry above identified marginal-vs-joint read features as the best
  remaining lead, and the neural answer costs ~30 GPU-hours. The hand-crafted
  version tested the same hypothesis for half an hour and came back at
  +0.0013 - so the cheap positive evidence that would have justified the GPU job
  is not there. That is the probe working as designed, not a wasted run.

  **The obvious next cheap experiment is A1 x depth augmentation**, which is one
  config file and no new code: `features: quantiles_flank_v1` with
  `train_depths: [1, 3, 5, 10, null]`. Sequence context and training depth are
  orthogonal, both gains are independently measured (+0.0109 established,
  +0.0056 not), and the `quantiles_flank_v1` cache is already warm at all eleven
  depths, so it costs fits only.

  **The ~15 min costs in this table are wrong, and the correction is worth
  keeping.** Measured on a warm cache, a laptop, 8 threads: a
  `--compare-features` run at the default `standard` profile is **~22 minutes**,
  because [0008](docs/decisions/0008-comparisons-report-both-arms-and-strata.md)
  evaluates *both* arms in full at 50 observations each - so it is 100 model
  fits, two depth sweeps and two sets of figures, not one. A new feature set
  also pays a **cold-cache extraction** first, over all eleven sweep depths in
  one streaming pass: 8.9 min for `quantiles_flank_v1` and 22.4 min for
  `quantiles_joint_v1`, whose per-site 9x9 solve is real arithmetic rather than
  parse time. Budget **30-45 minutes per cheap experiment**, not fifteen.

  **The ~15 min costs in this table are wrong, and the correction is worth
  keeping.** Measured on a warm cache, a laptop, 8 threads: a
  `--compare-features` run at the default `standard` profile is **~22 minutes**,
  because [0008](docs/decisions/0008-comparisons-report-both-arms-and-strata.md)
  evaluates *both* arms in full at 50 observations each - so it is 100 model
  fits, two depth sweeps and two sets of figures, not one. A new feature set
  also pays a **cold-cache extraction** first, over all eleven sweep depths in
  one streaming pass: 8.9 min for `quantiles_flank_v1` and 22.4 min for
  `quantiles_joint_v1`, whose per-site 9x9 solve is real arithmetic rather than
  parse time. Budget **30-45 minutes per cheap experiment**, not fifteen.

  **Not being built**: transcript GNN (+0.004), deeper residual read encoder
  (+0.002), Perceiver (~0). All below the detection threshold. A3 is the cheap
  test of whether the GNN premise - that m6A clusters along transcripts - shows
  up in our data at all.
- **No hyperparameter search has been run.** Every value in `configs/` was
  chosen by hand and none has been tuned. **One of them can now be sanity-checked
  without a search**: models record how the fit progressed
  ([0021](docs/decisions/0021-models-report-how-the-fit-progressed.md)), so
  `n_estimators` was briefly visible. Measured on `configs/quantiles.yaml`, full
  training set, on a run that **no longer exists** — see the provenance note
  below:

  | picked by | iteration | value |
  |---|---:|---:|
  | held-out PR AUC | 431 | 0.4821 |
  | held-out logloss | 600 (still falling) | 0.1387 |
  | where the fit stops | 600 | 0.4775 |

  So `n_estimators: 600` is **past** the held-out PR AUC peak, and the two
  objectives disagree about it — LightGBM minimises logloss, which is still
  improving at 600, while average precision peaked at 431 and has drifted down
  since. **This is a diagnosis and not a number to act on and then quote**: the
  peak is read off the same held-out folds the headline score comes from, so
  copying 431 into the config and reporting this run's PR AUC is the flat-tuning
  trap below, one parameter at a time. The honest route is to change the config
  and compare the two paired, like any other pair of runs. Nobody has.
- **The booster very nearly memorises its training fold, and nothing had
  measured it.** Same run: training-fold average precision **0.9963** against
  **0.4775** held out, a gap of 0.5188 (`fit/train_valid_gap`). That is not
  automatically a problem — a heavily-regularised model can be worse — but it
  says the capacity is nowhere near the constraint, so `num_leaves`,
  `min_child_samples` and `feature_fraction` are the parameters most likely to
  be badly chosen, and none of them has been varied either.

  `quantiles_depth_augmented` shows a gap of only 0.0561, and **that is not
  evidence of regularisation** — an earlier version of this entry said it was.
  Its training rows are drawn at depths 1/3/5/10/full while its held-out fold is
  full depth only, so it is scored on easier rows than it trained on; its train
  logloss sits *above* its valid logloss (0.2880 vs 0.2479), which is the
  giveaway. **`fit/train_valid_gap` only measures overfitting when
  `train_depths` is `full`.** Whether depth augmentation regularises is
  untested: the clean measurement is that model's average precision on
  full-depth *training* rows, which nothing logs. See
  [0022](docs/decisions/0022-training-rows-may-come-from-several-depths.md).

  Across the battery, `n_estimators` is **not** uniformly wrong — the training
  curve said it is wrong in one config out of three:

  | config | `n_estimators` | held-out PR AUC peaks at | overfit gap |
  |---|---:|---:|---:|
  | `lightgbm_pooled` | 400 | 390 | 0.3897 |
  | `lightgbm_quantiles` | 600 | **431** | 0.5188 |
  | `quantiles_depth_augmented` | 600 | 548 | 0.0561 (not an overfit measure — see above) |

  Only `quantiles.yaml` overshoots meaningfully. Changing it and comparing the
  two configs paired is the legitimate way to act on that; reading the peak and
  quoting this run's score is not. **Nobody has done it.**

  **These numbers are frozen, no committed command regenerates them, and the
  runs they came from have been deleted.** Measured 2026-09-22 on runs
  `yl6mkdo2`, `3fyr70t8` and `7trai6e4` — all since retired, because
  [0024](docs/decisions/0024-the-training-curve-is-for-gradient-descent-models.md)
  scoped the training curve to gradient-descent models — LightGBM no longer
  reports one, because a boosting round and an epoch on a shared axis make an
  unreadable panel. To retake the measurement: set
  `REPORTS_TRAINING_CURVE = True` on `LightGBMModel`, run, read it, and do not
  commit the edit. Whether that trade was right is worth revisiting if anyone
  wants to tune a booster; a `--fit-curve` flag was considered and not built.

  A nested-CV + Optuna protocol was designed and costed on 2026-09-22, then
  **dropped**, and the costing is worth keeping so it is not re-derived: a
  properly nested search (outer 5-fold gene-grouped, inner 3-fold, 40 TPE
  trials) is ~6,050 model fits per architecture at 10 repetitions, roughly
  **10 hours per architecture** on this data - about 30 for three. Cutting the
  outer loop to 3 repetitions brings it to ~1.7 hours each while widening the
  corrected standard error by only 8%, because the Nadeau & Bengio variance
  factor `1/n + 1/(k-1)` is dominated by the 0.25 floor at k=5 folds.

  Even at that price it was judged the wrong thing to spend an instance on
  versus read-level/MIL architectures, which the depth probe suggests are worth
  far more. **The trap to remember if anyone revives this:** tuning flat (on the
  same folds you report) does *not* wash out in a paired comparison, because the
  selection bias scales with search-space size - a 6-parameter GBT gains more
  from it than a 2-parameter logistic regression, so flat tuning hands one
  architecture an advantage and calls it a win.
- **Class imbalance is handled only by `is_unbalance=True`.** No resampling and
  no alternative objective has been tried. Threshold work now exists
  ([0019](docs/decisions/0019-a-threshold-sweep-because-a-ranking-cannot-count.md)),
  which measures what the rebalancing costs at the point of use without
  addressing the cause. See the calibration entry under Evaluation.
- **No ensembling.**
- **External labelled data: three routes, none of them started.** The handout
  explicitly permits additional labels or training sets, and the reason to want
  them is specific rather than general — every site we have is at depth >= 20,
  so nothing in this repo can measure performance at a realistic depth on
  *genuinely* low-depth sites rather than artificially thinned ones. The
  read-subsampling sweep cannot close that gap by construction
  ([0003](docs/decisions/0003-read-subsampling-in-data.md)), and neither can
  another run of our own cell line (see the dead idea under Task 2).

  In order of feasibility:

  **(a) GLORI labels + SG-NEx A549/K562 nanopore.** The only route where the
  nanopore side is already in our input format *and* at a realistic depth
  (median 2-3). Needs the GLORI site lists — GEO **GSE210563** — and conversion
  from genomic to transcript coordinates, which is the fiddly part and the most
  likely place to introduce a silent off-by-one. GLORI is
  chemical (glyoxal-and-nitrite) rather than antibody-based, so it is not
  subject to the m6ACE-Seq background problem recorded under "Understanding the
  data" — and that cuts both ways, because **cross-method label disagreement is
  then the main risk**: a site GLORI calls positive and m6ACE-Seq does not is
  not obviously either method's error. METTL3-knockout samples give
  high-confidence negatives, which is the strongest thing about this route.

  **(b) HEK293T + GLORI.** The best-characterised pairing and what the
  benchmarking papers use, so it is the route that would make our m6Anet
  comparison mean something. Blocked twice over: SG-NEx's HEK293T is fastq/fast5
  only, so it needs nanopolish eventalign and m6Anet dataprep first, and m6Anet
  is not installed (see Benchmarking).

  **(c) Aggregators — RMBase, m6A-Atlas, REPIC.** Easiest to obtain and the
  weakest labels: they pool sites called by different methods at different
  thresholds, so a positive means "somebody's assay called this once". Good for
  sanity-checking our own label set, risky to train on.

  Nobody has started any of them, and (a) is the one to start with.
- The current best out-of-fold PR AUC is **0.4759** (quantile features,
  LightGBM). One reference point is solid: a motif-only classifier scores
  **0.1537** (3.42x random) with no signal data at all.

  **The m6Anet comparison is not a second reference point, and this file used to
  claim it was.** m6Anet's published site-level performance is ROC AUC 0.83 /
  PR AUC 0.35 on HEK293T. This entry previously said that "indicates 0.4759 is
  in a plausible range rather than suspiciously high." That reasoning is
  backwards. Our **logistic regression baseline** scores 0.4122 - a linear model
  on 38 summary statistics, beating a published multiple-instance-learning
  network from Nature Methods. The correct response to that is suspicion, not
  reassurance.

  What would explain it, in order of likelihood:

  1. **PR AUC is not comparable across datasets at all** - and the base rate is
     now known, which retires most of the suspicion. It is bounded below by the
     positive rate. Ours is 4.49%, so 0.4759 is a **10.59x lift**. m6Anet's
     HEK293T figure reports 5,579 m6ACE-positive against 121,853 negative
     candidate positions, i.e. **4.38%** - so their 0.35 is roughly an **8.0x
     lift**, against our 10.59x. We are plausibly ahead rather than
     implausibly.

     Two caveats keep this from being settled. Their PR AUC is computed on
     positives defined as m6ACE **union miCLIP**, whereas 4.38% is the
     m6ACE-only count, so the prevalence behind the 0.35 is not established by
     that count. And their candidate universe is restricted to sites with >= 20
     reads, like ours, but is otherwise a different site set. Source:
     [docs/literature-review.md](docs/literature-review.md) - external, and the
     arithmetic is verified but the citation is not.
  2. **Different cell line, labels and site universe.** Theirs is HEK293T; ours
     is Hct116 with m6ACE-Seq labels and a site set the course pre-selected.
  3. **Possibly easier data.** Our own depth sweep says we score 0.2458 at depth
     3 and 0.1527 at depth 1 - both *below* 0.35. If their evaluation had
     realistic coverage the ordering flips entirely. **But** m6Anet samples 20
     reads per site, which is the likely reason the course filtered at >= 20 in
     the first place, so their evaluation may be depth-filtered too. Check
     before relying on this.
  4. **Leakage in our pipeline.** Least likely - the split is gene-grouped and
     features are computed per site from that site's own reads - but not ruled
     out; see the paralog entry below.
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
- **"SG-NEx has 7 cell lines" is wrong: it has 14, and 7 is the subset somebody
  else has already processed.** The raw direct RNA-Seq data under
  `data/sequencing_data_ont/fastq/` (and the matching `fast5/`) is **56 samples
  across 14 cell lines** — the 7 above plus **Hek293T (5 replicates)**,
  MCF7-EV (2), and IM95, Myeloma-N082, Myeloma-N104, Myeloma-N122 and NCC24
  with 1 each. Note the bucket spells it `Hek293T`, not `HEK293T`.

  The distinction is the whole story and it is easy to miss, because both halves
  are in the same bucket: the 7 are ready to predict on today, and the other 7
  are fastq/fast5 only. Using one of them means running nanopolish eventalign
  and m6Anet dataprep first, and m6Anet has never been installed here (see
  Benchmarking). **HEK293T is the painful one** — it is the best-characterised
  cell line for m6A, the one the benchmarking papers use, and it is on the wrong
  side of that line.

  Regenerate with `python analysis/sgnex/survey.py catalogue`.
- **Read depth in SG-NEx is far below anything the model has seen, and this is
  now reproducible rather than asserted.** Measured from each sample's
  `data.readcount` — one line per site, ~34 MB against the 2 GB `data.json`
  beside it, so a depth distribution costs well under a minute and no download
  (`python analysis/sgnex/survey.py depth <sample>`):

  | sample | sites | p25 | median | >= 2 | >= 5 | **< 20 reads** |
  |---|---:|---:|---:|---:|---:|---:|
  | A549 `replicate6_run1` | 1,500,579 | 1 | 3 | 69.4% | 39.3% | **87.5%** |
  | K562 `replicate4_run1` | 1,104,444 | 1 | 2 | 62.5% | 30.5% | **92.2%** |
  | Hct116 `replicate3_run1` | 1,391,230 | 1 | 3 | 67.2% | 36.8% | 88.8% |

  Every training site has >= 20 reads, so that last column is the fraction of
  each sample the model has never seen anything like. Read against the depth
  table under Evaluation, **the majority of any naive Task 2 prediction run
  would be at or near the motif-only floor**, and cross-cell-line differences
  computed from those scores would track sequencing depth rather than biology.

  The Hct116 row is a check on the method rather than a new fact: restricted to
  our 121,838 labelled sites it gives median 47 and p25 32, exactly the
  distribution recorded under "The training set's provenance" from the original
  ad-hoc crawl.
- **Hct116 is not an independent cell line for us.** See "The training set's
  provenance".
- **DEAD IDEA, recorded so nobody spends a day re-deriving it: the other two
  Hct116 samples cannot give us low-depth copies of our labelled sites.** The
  reasoning was good. Our depth sweep drops reads at random, which simulates
  *covariate* shift and structurally cannot see label shift
  ([0003](docs/decisions/0003-read-subsampling-in-data.md)); SG-NEx holds two
  more m6Anet-processed runs of our own cell line (`replicate3_run4`,
  `replicate4_run3`); if the same labelled sites were shallow in those, we would
  have real low-depth measurements of sites whose labels we already trust, and
  could check the sweep against them.

  They are not shallow. Measured on both:

  | sample | our sites found | min | p25 | median | below 20 reads |
  |---|---:|---:|---:|---:|---:|
  | Hct116 `replicate3_run4` | 121,838 (100%) | 20 | 28 | 42 | **0** |
  | Hct116 `replicate4_run3` | 121,838 (100%) | 20 | 51 | 76 | **0** |

  A 100% overlap and not one site below 20 reads in either. The reason is
  structural, which is why no other sample will fix it: **our sites have >= 20
  reads because their transcripts are abundant, and abundance is a property of
  the cell line, not of the sequencing run.** An abundant transcript is deep in
  every run of that cell line. The depth floor is not a filter that a different
  run happens to pass — it selects a population that stays deep.

  So real low-depth labelled data has to come from somewhere with *different*
  labels, not a different run of ours. See the external-data routes under
  Modelling. Regenerate with
  `python analysis/sgnex/survey.py depth SGNex_Hct116_directRNA_replicate3_run4 --labels data0/data.info.labelled`.
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
- **Nobody has read m6Anet's evaluation base rate out of the paper.** Without
  it, comparing our PR AUC to their 0.35 is meaningless, because PR AUC is
  bounded below by the positive rate. One number from one table would let us
  compare lift instead. This is the single cheapest open item in this file and
  it gates any claim the report makes about beating m6Anet.

## Infrastructure

- **A lower-profile run silently overwrites a recorded higher-profile report of
  the same name.** Found on 2026-09-24 by doing it.
  `evaluate.py --config configs/quantiles.yaml --profile quick` rewrote
  `analysis/evaluation/reports/lightgbm_quantiles.json`, replacing a `standard`
  report carrying 50 observations with a `quick` one carrying 5 - **9,686 lines
  deleted, 98 written**, including the whole `repeated_cv` block. It was
  recovered with `git checkout` only because that file happens to be tracked.

  [0010 section 2](docs/decisions/0010-the-report-is-a-module-not-a-script.md)
  anticipated exactly this shape of failure for *subset* runs and guards it: a
  run over a limited number of sites writes `<name>__limit5000.json` and can
  never clobber `<name>.json`. There is no equivalent guard on the **profile**,
  even though a `quick` report is a strictly poorer object than a `standard`
  one and the filename does not record which it is. The failure is silent: the
  file still parses, still carries the right config name, and still has a
  plausible `pooled.pr_auc` - it has simply lost the distribution underneath it.

  It bites hardest in the case the profiles exist to prevent, which is someone
  regenerating a number to check it and destroying the richer record in the act
  of verifying it.
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
- **The sample counts and depth distributions in the Task 2 section are
  reproducible now** — `analysis/sgnex/survey.py` does the crawl that used to be
  ad-hoc and untracked, and it was what found that SG-NEx has 14 cell lines
  rather than 7. Two numbers there still are not: the **33.4 GB** total volume
  and the per-sample sizes come from the original crawl and nothing regenerates
  them. They are cheap to re-derive (a `list_objects_v2` over the m6Anet prefix,
  summing `Size`) and nobody has.
- **`analysis/evaluation/scratch/` holds four rough scripts with hardcoded
  paths**, three of which are the only source of a number quoted above (the
  dataset profile, the read-level MIL probe, and the Hct116-vs-A549 comparison,
  which additionally needs an input file nothing in this repo produces). See the
  README there.
- **`fit/logloss_best_iteration` is a plain argmin over a non-monotone curve,
  and it reads as something it is not.** Held-out logloss here dips by round 2,
  humps to a peak near round 50, then declines — so the argmin lands on the
  early dip whenever the run is too short for the late decline to get back under
  it. It reports **2** for `lightgbm_pooled` (where PR AUC is 0.338 against
  0.465 at the end) and **600** for `lightgbm_quantiles`, and those two numbers
  are not comparable. Nobody should read either as "where the fit should stop".
  The key was left as logged rather than redefined mid-battery
  ([0007](docs/decisions/0007-evaluating-without-a-baseline.md) section 4); the
  fix is to report the argmin *after* the hump, or to log the hump's height
  beside it, and it needs a decision record. Documented in
  [docs/wandb-panels.md](docs/wandb-panels.md) meanwhile.
- **A W&B artifact download is committed to the repo.**
  `artifacts/run-3bw2tvmm-strata_observations-v0/strata_observations.table.json`
  is tracked. Nothing put it there deliberately: W&B's API downloads into
  `artifacts/` whenever something calls `.download()`, which
  `tracking.fetch_strata` does on **every** `--compare-run`, and the directory
  was not gitignored. It is now, so this stops growing — but the already-tracked
  file needs `git rm --cached` by someone who is sure nothing reads it from
  disk. It is a re-downloadable copy of what is already in W&B, so it is cache
  rather than record, and it is 121 KB per comparison anyone runs.
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
