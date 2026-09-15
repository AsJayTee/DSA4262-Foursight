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

About 90 seconds on the full dataset. Metrics and the trained model both go to
W&B automatically, so nothing is lost when the instance is terminated.

### 4. Look at the result

<https://wandb.ai/dsa4262-team/dsa4262-project>

**Read `oof/pr_auc` first.** At 4.49% positives, ROC AUC flatters everything —
every model we have tried sits above 0.90 on it. `oof/pr_auc_lift` tells you
how many times better than random the model is; 1.0× means it learned nothing.

Current reference points, out-of-fold on the gene-grouped split:

| | PR AUC | Lift |
|---|---:|---:|
| Random classifier | 0.045 | 1.0× |
| Logistic regression, pooled features | 0.4121 | 9.2× |
| LightGBM, pooled features | 0.4634 | 10.3× |
| LightGBM, quantile features | 0.4759 | 10.6× |

---

## Things to be careful about

**A small improvement may be noise.** Fold-to-fold variation in PR AUC is
around 0.019 — larger than the gap between our two current feature sets. Before
believing a 0.005 improvement, ask your agent how confident it actually is and
what would establish that. See [GAPS.md](../GAPS.md) for what our evaluation
currently does and does not tell us.

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
make train CONFIG=configs/lightgbm.yaml       # ~90s, full data, to W&B
make test                                     # test suite
make predict INPUT=<in.json.gz> OUTPUT=<out.csv>
make sample                                   # rebuild data/sample/
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
