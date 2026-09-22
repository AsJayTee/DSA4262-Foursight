# Running experiments

**The default way to work in this repo is through a coding agent** — Claude
Code, Codex, or whatever you use. You describe the idea; the agent writes the
code, checks it runs, and hands you a command. You run that command on a Ronin
instance and look at the result in W&B.

You do not need to read the codebase to contribute to it. If you want to
understand, change, or extend something, **ask your agent** — it has
[AGENTS.md](../AGENTS.md) telling it how this repo is organised, and it can
read any file you are curious about and explain it in context.

Machine not set up yet? [setup.md](setup.md) first.

---

## The loop

### 1. Describe the idea to your agent

Be concrete about the *idea*, not the implementation. Good prompts look like:

> "Try using the maximum signal deviation across reads instead of the mean —
> the theory is that only some reads at a modified site are actually modified,
> so an extreme value might survive where an average washes out."

> "Add a model that weights each read by how unusual it is before aggregating."

> "Our current features throw away the flanking bases of the 7-mer. Add a
> feature set that encodes them."

Your agent knows the conventions: an experiment is **one new file plus one new
config**, and nothing existing gets edited. That is what keeps four people's
work from colliding.

### 2. Make the agent prove it runs

Before you spend any instance time, the agent should run:

```bash
make smoke CONFIG=configs/your_experiment.yaml
```

That is the full pipeline on 5,000 sites — about ten seconds. It catches the
errors that would otherwise surface after you have launched a VM and waited
five minutes.

**Ask for the smoke output if you are not shown it.** An experiment that has
not been smoke-tested is not ready to hand over.

### 3. Push, then run it for real

Commit and push from your laptop, then on the instance:

```bash
git pull
make train CONFIG=configs/your_experiment.yaml
```

About 100 seconds on the full dataset. That one command **trains and
evaluates** — per fold, by read depth, by motif, calibration, the depth sweep and
every figure — and pushes all of it to one W&B run, along with the model.

That is deliberate. Ronin instances are created, used and terminated with
nothing pulled off them, so anything left on the instance's disk is gone. A
training run that nobody evaluated before the instance died has lost the work,
and re-running it costs far more than evaluating it would have. Gather
everything while the machine exists.

Add `--quick` when you are iterating and do not need the depth sweep. Do not use
it for a run you intend to quote: a quick run has no depth numbers, and it
cannot be compared against a standard one on the dimension that matters most.

### 4. Look at the result

<https://wandb.ai/dsa4262-team/dsa4262-project>

Every run reports the same keys, so the run table *is* your baseline — you are
never comparing against nothing, you are comparing against every experiment
anyone has run. `oof/pr_auc` is the headline; `depth/1/pr_auc` is the one Task 2
lives or dies on; `calib/count_ratio` says whether a score may be read as a
probability. The figures are on the run page: start with the **PR AUC
distribution**, which shows all five folds as points rather than collapsing them
to one number.

**Read `oof/pr_auc` first.** At 4.49% positives, ROC AUC flatters everything —
every model we have tried sits above 0.90 on it. `oof/pr_auc_lift` tells you
how many times better than random the model is; 1.0× means it learned nothing.

Current reference points, out-of-fold on the gene-grouped split:

| | PR AUC | Lift |
|---|---:|---:|
| Random classifier | 0.045 | 1.0× |
| Motif only, no signal data at all | 0.1537 | 3.4× |
| Logistic regression, pooled features | 0.4121 | 9.2× |
| LightGBM, pooled features | 0.4634 | 10.3× |
| LightGBM, quantile features | 0.4759 | 10.6× |

The motif-only row is the one to keep in mind. An 18-way one-hot over the DRACH
motif, with no nanopore signal whatsoever, reaches 0.1537. Anything close to
that has not learned to read the pore.

### 5. Ask whether it is actually better

The pooled number above is one figure over five folds that differ in difficulty
— the same model scores 0.4548 on fold 0 and 0.5081 on fold 1. Comparing two
pooled numbers cannot separate an improvement from that variation. Comparing
the folds *pairwise* can, because both runs saw the same folds:

```bash
python scripts/evaluate.py --config configs/your_experiment.yaml \
       --compare-features quantiles_v1
```

```
      quantiles_v1  your_experiment  difference
fold
0           0.4548           0.4700      0.0152
...
unpaired  (wrong)      Welch p = 0.3095
paired    (optimistic) mean +0.0148  95% CI [+0.0004, +0.0292]  p = 0.0465  wins 5/5
corrected (quote this) 95% CI [-0.0068, +0.0364]  p = 0.1305
```

**Quote the corrected row.** The middle row assumes the five folds are five
independent observations. They are not — each fold's model trains on the other
four, so 73% of any two folds' training rows are shared, and the p-value comes
out smaller than the evidence supports. The correction inflates the variance to
account for that. It is always the largest of the three numbers.

Note what that does to the example: the improvement that looked significant at
p = 0.0465 is p = 0.1305 corrected, and the corrected interval includes zero.
The effect is probably real — it wins on all five folds — but five folds cannot
establish it.

Read the **win count** as well. 5/5 does not depend on the scatter estimate, so
it is the more trustworthy signal when you only have five observations.

**When the two are too close to call, get more observations:**

```bash
python scripts/evaluate.py --config configs/your_experiment.yaml \
       --compare-features quantiles_v1 --repeats 10
```

That runs the whole cross-validation ten times over independently seeded splits
— 50 paired observations instead of 5. Repetition 0 is always the canonical
seed-4262 split, so every number already recorded stays exactly as it was. It
costs ten times the model fits and no extra feature extraction: about 25 minutes
on the full set against 100 seconds.

It is worth it when the answer matters. On `quantiles_v1` vs `pooled_v1`, five
folds gives corrected p = 0.1305 with an interval spanning zero — no conclusion.
Fifty observations gives +0.0164, **50 wins out of 50**, corrected p = 0.000095.
Same data, same models; the five-fold test was simply underpowered.

Want an error bar on the headline number itself rather than on a comparison?

```bash
python scripts/evaluate.py --config configs/your_experiment.yaml --bootstrap 2000
```

That resamples the *sites* rather than the split, which is a different question:
how much does 0.4759 depend on which 121,838 sites we happen to have?

The same command answers two other questions worth asking of anything you build:

```bash
# where does it fail? by read depth, and by DRACH motif
python scripts/evaluate.py --config configs/your_experiment.yaml --by depth,motif

# what happens at SG-NEx's read depths, which is where Task 2 lives?
python scripts/evaluate.py --config configs/your_experiment.yaml --depth-sweep
```

**If your result is going to be a count of anything, read the Thresholds
section of the output.** PR AUC and ROC AUC rank sites; they never pick a
cut-off, and counting modified sites needs one. Every run now reports three
operating points and — the number that matters — how far the count swings
between them. On the current model a site count moves **2.1x** between
threshold 0.3 and 0.7, both of which are defensible choices. Quote the swing
beside any count, or you are reporting an arbitrary cut as a measurement.
See [docs/decisions/0019](decisions/0019-a-threshold-sweep-because-a-ranking-cannot-count.md).

### Several runs at once

Once a few experiments are in W&B, the question stops being "is A better than
B" and becomes "where does everything sit". W&B's own panels are awkward for
that — its distribution panels are one image per run, and its scatter cannot
draw a reference line or label a point:

```bash
python scripts/compare_runs.py lightgbm_quantiles lightgbm_pooled baseline_logistic --pair
```

That writes two figures to `report/figures/`: every run's metric distribution on
one axis, and one labelled point per run with the y = x diagonal — by default
full-depth PR AUC against PR AUC at depth 3, where the distance below the line
is the collapse. `--pair` adds the corrected paired test of each run against the
first one named. It fits nothing and needs no instance, so it works from a
laptop long after the machines are gone.

Every run writes a JSON report to `analysis/evaluation/reports/` **and** uploads
it to W&B, so a number you quote can be traced back to the run that produced it
from either place. Treat the local copy as a convenience: on a Ronin instance it
dies with the machine, and W&B is the copy that survives.

There is no stored per-site score table any more. Everything a comparison needs
is the per-fold metric vector, which is in the W&B run table under
`fold/0/pr_auc` … `fold/4/pr_auc` — so two runs from different weeks, on
different instances, can still be paired. The cost is that a question you did
not think to ask during a run means running it again; that is why the standard
profile is thorough. See
[docs/decisions/0009](decisions/0009-distributions-not-per-site-scores.md).

---

## Things to be careful about

**A small improvement may be noise — but two pooled numbers cannot tell you
which.** Fold-to-fold variation in PR AUC is about 0.020, larger than the gap
between our two current feature sets. That used to be read as "the two cannot be
distinguished", and it was wrong: on the *paired* per-fold differences
`quantiles_v1` beats `pooled_v1` by +0.0148, winning 5 folds out of 5. Use
`--compare-features`; do not eyeball two pooled numbers.

**But do not quote that comparison's p-value as 0.0465.** Corrected for the
overlap between training folds it is 0.1305, and the interval includes zero — not
established at five folds. It *is* established at fifty: `--repeats 10` gives
+0.0164, 50/50 wins, corrected p = 0.000095. Quote that one.

**A score of 0.6 does not mean 60%.** It means about 33% — the model overcounts
positives by 1.80×. That does not affect PR AUC or ROC AUC, which are rank-based,
but it does break any claim of the form "cell line X has N modified sites".

**Nothing has been tested below 20 reads per site.** Every training site has at
least 20; SG-NEx has a median of 3, and at one read this model scores what a
motif-only classifier scores. If your idea is aimed at Task 2, run
`--depth-sweep` before believing anything. What *depth* means precisely is in
[docs/data.md](data.md#read-depth) — a count of distinct RNA molecules, not
repeated readings of one — and that distinction carries most of the argument.

**Never change the split seed.** Every config uses `seed: 4262` grouped by
`gene_id`. If you change it, your numbers stop being comparable with everyone
else's — and nothing will warn you, because both numbers look plausible.

**Your agent can be confidently wrong.** It will sometimes produce code that
runs and is still doing the wrong thing — leaking labels, comparing against the
wrong baseline, silently dropping rows. If a result looks too good, it probably
is. Ask it to explain *why* the number improved, not just to confirm that it
did.

**Log it when that happens.** The report requires a table of AI tool usage plus
**at least two documented cases where an AI tool's output was wrong,
misleading, or rested on an unverified assumption** — how you noticed, and what
you did. These are much easier to write down the day they happen than to
reconstruct in late October. Keep a running note.

---

## Working without an agent

Everything is a plain command if you prefer:

```bash
make doctor                                   # is this machine healthy?
make smoke CONFIG=configs/lightgbm.yaml       # ~11s, 5,000 sites
make train CONFIG=configs/lightgbm.yaml       # ~100s, trains AND evaluates, to W&B
make evaluate CONFIG=configs/lightgbm.yaml      EVAL='--compare-features pooled_v1'      # compare against a baseline
make test                                     # test suite
make predict INPUT=<in.json.gz> OUTPUT=<out.csv>
make sample                                   # rebuild data/sample/
make clean-cache                              # drop the extracted-feature cache
```

A config is short enough to write by hand:

```yaml
name: exp_my_idea
features: quantiles_v1
model: lightgbm
model_params: {num_leaves: 127, learning_rate: 0.03}
split: {seed: 4262, n_folds: 5, group_by: gene_id}
notes: What you are testing and why.
```

To see what feature sets and models exist right now:

```bash
make doctor
```

It lists everything currently registered.

---

## What to work on

[GAPS.md](../GAPS.md) is the backlog — what is missing, weak, or untested,
with evidence but deliberately without proposed solutions, so you and your
agent can reach your own conclusions.

[project-requirements.md](project-requirements.md) is what the project is
actually graded on.

When you close a gap, delete its entry. When you find a new one, add it.
