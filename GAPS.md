# Known gaps

What is missing, weak, or untested in this repo as of 2026-09-15.

**This file deliberately does not propose solutions.** It records what is
wrong, and the evidence for it, so that whoever picks up a piece of work can
reach their own conclusions about how to address it. If you find yourself
disagreeing with a framing here, that is fine — check it yourself.

Keep it current: when you close a gap, delete the entry. When you find a new
one, add it with whatever evidence you have.

Context for anything here: [docs/project-requirements.md](docs/project-requirements.md)
is what the project is graded on. [docs/data.md](docs/data.md) describes the data.

---

## Evaluation

The current pipeline records one pooled out-of-fold number per run
(`metrics_oof` in each model's `meta.json`) and nothing else.

- **Per-fold results are computed and then discarded.** Nothing downstream can
  see how a model did on individual folds.
- **Fold-to-fold noise is larger than the difference between our two current
  feature sets.** Measured on the full training set: per-fold PR AUC has a
  standard deviation of 0.019 for pooled features, while the gap between
  pooled and quantile features is 0.0157 overall. Comparing two headline
  numbers from separate runs is therefore not reliable on its own.
- **Folds are unbalanced by positive rate**: 4.10%, 5.15%, 4.42%, 4.54%, 4.21%
  across folds 0–4, with fold sizes from 23,394 to 25,923 sites. PR AUC depends
  on the base rate, so a given model scores highest on fold 1 regardless of its
  merit.
- **No breakdown of performance by any stratum.** Read depth ranges from 20 to
  991 reads per site and there are 18 DRACH motifs; performance within those
  groups is unknown, so it is not known where the model fails.
- **No calibration assessment.** The LightGBM model sets `is_unbalance=True`,
  which affects the probability scale. ROC AUC and PR AUC are rank-based and
  unaffected, but any use of the scores as probabilities — counting modified
  sites in Task 2, for instance — rests on an unchecked assumption.
- **No mechanism for comparing runs.** Comparing two experiments currently
  means reading two terminal outputs or two W&B pages by hand.
- **No statistical treatment** of whether a difference between two models is
  meaningful.
- **Only one cell line has ever been validated against.** The training data is
  Hct116 only. The briefing states the hidden test set may be a different cell
  type or a different type of data. Cross-cell-line generalisation is entirely
  untested.
- **Changing the fold assignment invalidates every stored result.** The seed
  (4262) is frozen in `AGENTS.md` for that reason. The longer results
  accumulate, the more expensive any change to the split becomes.

## Modelling

- **Two feature sets exist**, both summary statistics over reads:
  `pooled_v1` (mean/std) and `quantiles_v1` (quantiles, IQR, tail spread).
- **No model treats the reads as a set.** The briefing frames this explicitly
  as a Multiple Instance Learning problem — the site carries the label, the
  reads do not, and only a fraction of reads at a positive site are modified.
  Nothing in the repo exploits that structure directly.
- **Sequence information is barely used.** Only the central 5-mer is encoded,
  as an 18-way one-hot. The flanking bases of the 7-mer are discarded.
- **No hyperparameter search has been run.** Every value in `configs/` was
  chosen by hand and none has been tuned.
- **Class imbalance is handled only by `is_unbalance=True`.** No resampling,
  alternative objective, or threshold work has been tried.
- **No ensembling.**
- **No external training data has been sought**, although the handout
  explicitly permits searching for additional labels or training sets.
- The current best out-of-fold PR AUC is **0.4759** (quantile features,
  LightGBM). Whether that is good is unknown — there is no reference point
  beyond the random classifier (0.045) and the logistic baseline (0.4121).

## Understanding the data

The briefing weights "do you fully understand the dataset, and its
limitations?" explicitly. Right now, no.

- **No exploratory analysis exists.** Nobody has looked at how signal
  distributions differ between labelled and unlabelled sites.
- **The relationship between read depth and label is unexamined.** If depth
  correlates with label, the models may be learning coverage rather than
  chemistry, and nothing currently would reveal that.
- **Positive sites cluster in 1,507 of 3,852 genes.** The structure of that
  clustering — whether positives concentrate in particular transcript regions,
  genes, or expression levels — is unexamined.
- **Per-motif base rates are unknown.** Whether some of the 18 DRACH motifs are
  modified far more often than others has not been checked.
- **The labels' provenance is undocumented.** m6ACE-Seq has its own sensitivity
  and specificity characteristics; what a `0` label actually means — truly
  unmodified, or below the assay's detection limit — is not written down
  anywhere in this repo, and it bears on every claim we make.

## Task 2 — SG-NEx

Nothing exists. Not started.

- No SG-NEx data has been downloaded, inspected, or catalogued.
- No record of which SG-NEx datasets are relevant, what the different data
  types are, or which samples are direct RNA-Seq.
- No path for running prediction across many samples.
- No cross-cell-line comparison of any kind.
- No interactive visualisation. This is part of the 50% report component.

## Benchmarking against m6Anet

- **m6Anet has never been installed or run.** Two known obstacles are recorded
  in [analysis/m6anet/README.md](analysis/m6anet/README.md); neither has been
  resolved.
- No automated comparison against the simple baseline and random classifier,
  though the report requires both.

## Infrastructure

- **`manifest.json` does not exist in the R2 bucket.** Downloads currently fall
  back to comparing file sizes against the remote object, which will not catch
  a corrupted file whose length happens to match.
- **There is no upload path to R2.** Adding data to the bucket is manual and
  undocumented, which matters when the evaluation data is released.
- **`setup_remote.sh` and `bootstrap.sh` have never been run against a real
  VM.** They are written and reviewed but unproven.
- **`doctor` reports "R2 configured" based on the presence of keys, not their
  validity.** Placeholder values pass the check.
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
