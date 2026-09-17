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
paired   mean difference +0.0148  95% CI [+0.0004, +0.0292]  p = 0.0465  wins 5/5
```

Read the **win count** as well as the p-value. Five folds is five observations,
so a difference that is positive on all five is worth more than the p-value
alone suggests, and one that flips sign is worth much less.

The same command answers two other questions worth asking of anything you build:

```bash
# where does it fail? by read depth, and by DRACH motif
python scripts/evaluate.py --config configs/your_experiment.yaml --by depth,motif

# what happens at SG-NEx's read depths, which is where Task 2 lives?
python scripts/evaluate.py --config configs/your_experiment.yaml --depth-sweep
```

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
`quantiles_v1` beats `pooled_v1` by +0.0148, winning 5 folds out of 5,
p = 0.0465. Use `--compare-features`; do not eyeball two pooled numbers.

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
