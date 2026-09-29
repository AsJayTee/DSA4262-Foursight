# Catching up: where the m6A project stands

*Written 2026-09-28 for teammates who haven't opened the repo yet. No biology
assumed. It goes from the data to the current best model, including what
failed and why - the failures explain the design as much as the wins do.*

---

## 1. The problem, in data-science terms

### What we are predicting

Cells copy genes into **RNA** molecules - long strings over the alphabet
A, C, G, U (written T in our files). Some of the A's get a small chemical tag
called **m6A**. Whether a given A is tagged matters biologically (it's linked
to some cancers), and measuring it is expensive, so we want to predict it.

**One row of our problem = one candidate A on one RNA sequence (a "site").**
The label is 1 if that site is tagged (modified), 0 if not. It's binary
classification.

- m6A only happens at A's sitting inside specific 5-letter patterns, called
  **DRACH motifs**. There are 18 of them (e.g. `GGACT`). Every site in our data
  is already one of these candidates, so the question is never "could this be
  m6A?" but "is it?".
- A **transcript** is one RNA sequence; a **gene** can produce several similar
  transcripts. This matters for how we split the data (section 3).

### How the data is measured

A **nanopore** sequencer pulls a single RNA molecule through a tiny pore and
records the electrical current as the letters pass through. A tagged A
distorts the current slightly. So the input is **signal**, not the letters
themselves.

For each site we get the measurements from every molecule that passed through
the pore covering that site. Each molecule gives one **read** at that site: 9
numbers. The 9 numbers are 3 measurements (how long the molecule dwelt in the
pore, the spread of the signal, the mean current) at 3 neighbouring positions
(the base before, the site itself, the base after).

### The one concept everything depends on: read depth

**Depth = how many separate molecules were measured at a site.** Not 50
readings of one molecule, but 50 different copies of that RNA from the cell,
each measured once.

This is what makes the problem hard, and it's why the problem is framed as
**Multiple Instance Learning**:

- The **label belongs to the site**, not to any read. We never know which
  individual molecules carry the tag.
- At a modified site, **only some of the molecules are tagged** (say 30%). The
  rest look exactly like an unmodified site.
- So a site is a **bag of reads**, and the signal is "some fraction of this bag
  looks different", not "every read looks different". Averaging the reads
  washes that out.

Full explanation: [docs/data.md](data.md#read-depth).

---

## 2. The data we have

| | |
|---|---:|
| Sites (rows) | 121,838 |
| Reads (molecules x sites) | 11,027,106 |
| Transcripts | 5,333 |
| Genes | 3,852 |
| Positive sites | **5,475 (4.49%)** |
| Reads per site | 20 to 991, median 47 |

Two files: `dataset0.json.gz` (for every site, a 7-letter sequence and a
reads x 9 matrix) and `data.info.labelled` (gene, transcript, position, label).
Formats are in [docs/data.md](data.md).

**Three facts about this data that shape everything:**

1. **Only 4.5% of sites are positive.** Accuracy is meaningless (predict "no"
   everywhere and you get 95.5%). We use **PR AUC** (area under the
   precision-recall curve), which is judged against the 4.49% base rate: a
   useless model scores about 0.045.
2. **Every site has at least 20 reads - an artificial floor.** Real data (Task
   2) has a median of about 3 reads, and a quarter of sites have 1. So we have
   to worry about low depth even though we train on high depth.
3. **The labels come from a separate lab method (m6ACE-Seq) on one cell line
   (Hct116).** They're good, not perfect.

### What we're graded on

From [docs/project-requirements.md](project-requirements.md):

- **Task 1:** a model that predicts m6A from this kind of data, and a
  `predict.py` that other students run on an unseen file. Scored on **ROC AUC
  and PR AUC** on an unknown test set - 5% of the grade.
- **Task 2:** apply it to the public **SG-NEx** datasets (7 cell lines) and say
  something about m6A across them.
- **Benchmark against m6Anet** (the published tool).
- **The report is 50% of the grade**, and it explicitly rewards understanding
  the data's and the model's *limitations*. Much of what follows is exactly
  that material.

---

## 3. How we measure progress (why the numbers are trustworthy)

This part is the least exciting and the most important. Most "improvements" in
a project like this are noise, and the repo is built to tell the difference.

### Split by gene, never by row

Transcripts of the same gene share sequence. If one transcript of a gene is in
training and another in testing, the model has effectively seen the answer.
So **all 5 cross-validation folds are grouped by gene**, with a fixed seed
(4262) that nobody changes - otherwise two people's numbers aren't comparable.

### Every score is out-of-fold

A site's score always comes from a model that never saw any transcript of that
site's gene. We never quote training scores.

### Compare models paired, over 50 measurements

The same model scores anywhere from 0.455 to 0.508 depending on which fold it's
tested on - the folds just differ in difficulty. That spread is bigger than
most real improvements. So:

- Two models are compared **fold by fold** (paired), so fold difficulty cancels.
- Every run repeats the whole 5-fold cross-validation **10 times** with
  different splits: **50 paired measurements**, not 5.
- The p-value uses a **corrected t-test** (Nadeau & Bengio), because the 50
  measurements share training data and aren't independent. The uncorrected
  version is too optimistic.
- We quote the corrected p-value **and the win count** ("better on 48 of 50").

**What this buys us:** the harness reliably detects improvements of about
**+0.006 PR AUC or more**. Anything smaller can't be established, however good
the idea. That threshold decides which experiments are worth running.

### Everything is logged

Every run goes to a shared W&B project, with per-fold numbers, calibration,
threshold behaviour and a **depth sweep** (the model's score when held-out
sites are cut down to 1, 3, 10 reads - how it would fare on real low-depth
data).

---

## 4. The journey: from 0.46 to 0.54 (the established pipeline)

All models below are **LightGBM** (gradient-boosted trees) on hand-designed
features computed per site. Numbers are out-of-fold PR AUC.

### 4.1 Summarising the reads

A tree needs a fixed-length row, but each site has 20 to 991 reads. So the
first job is to summarise each site's bag of reads.

| feature set | what it computes per site | PR AUC |
|---|---|---:|
| motif only (no signal) | which of the 18 DRACH patterns | 0.1537 |
| `pooled_v1` | mean and std of each of the 9 measurements | 0.4614 |
| **`quantiles_v1`** | 5th/25th/50th/75th/95th percentiles of each, plus spreads | **0.4759** |

Quantiles win (+0.016, 50/50) because of the MIL point above: if 30% of
molecules are tagged, that shows up in the **tails** (the 95th percentile
moves) before it moves the mean.

### 4.2 The depth problem, and the fix

Scored at low depth, `quantiles_v1` collapses: 0.1527 at 1 read, 0.2458 at 3.
At one read it is barely better than just knowing the motif.

**Fix - depth-augmented training:** add copies of every training site with only
1, 3, 5 and 10 of its reads kept. The model learns what sparse evidence looks
like. Result: **0.2593 at 1 read** (+70%), at no cost at full depth.

*An important side story:* a read-level model had looked much better at low
depth, and it seemed to prove that "a network over reads" was needed. It
turned out the read-level model simply had been trained at low depth. The
cheap control (retrain the tree with depth augmentation) recovered almost all
of the gap. **Lesson: before crediting an architecture, check what else
changed.**

### 4.3 Using the sequence properly

The files give a 7-letter window around each site, but the features only used
the middle 5 letters (the motif). Adding the two outer letters: **+0.0109,
48/50**. Some neighbouring letters make modification 2-4x more likely.

### 4.4 How the three positions move together ("coupling")

A tagged molecule shifts the current at all three neighbouring positions *at
once*. Percentiles are computed one column at a time, so they can't see "these
two columns moved together on the same molecule". The **correlation between
positions, across reads** can: positives average 0.244, negatives 0.134.
Adding these correlations: **+0.0104, 47/50**.

Two earlier attempts at the same idea failed, and why is instructive: one
normalised the correlation away by accident; the other summarised it into
quantities the percentiles already fixed. Only computing E[XY] directly
worked.

### 4.5 The neighbours: the largest single gain

m6A clusters along transcripts. If a site is positive, its neighbour on the
same transcript is positive 29% of the time, against 4.5% overall.

We **can't use neighbours' labels** - at prediction time a whole gene is
unseen. But a modified neighbour's **reads** look modified. So we added
features computed from the *other sites on the same transcript*: each site's
relative position along the transcript, and averages of key measurements over
the other sites. **+0.027, 50/50** - the biggest single gain.

We kept only the "robust" subset: averages and positions, not counts or
maxima, because counts and maxima change when the input file has fewer sites
(section 6).

### 4.6 Stacking it all up

| config | contents | PR AUC | at 1 read | at 3 reads |
|---|---|---:|---:|---:|
| `quantiles.yaml` | the starting point | 0.4759 | 0.1527 | 0.2458 |
| `final_candidate.yaml` | + flanks + neighbours + depth augmentation | 0.5293 | 0.3096 | - |
| **`everything.yaml`** | + coupling | **0.5408** | **0.3121** | **0.3910** |

**+0.065 overall, and the one-read score doubled.** `everything.yaml` is the
current best model in the pipeline.

### 4.7 Calibration

The model's scores aren't probabilities: it predicts about 3x more modified
sites than there are. That doesn't affect ROC AUC or PR AUC (which only use the
ranking), but it would wreck Task 2, where we *count* modified sites. **Platt
scaling** (a logistic fit on the scores) takes the overcount from 3.83x to
1.02x without changing the ranking.

---

## 5. Against the published tool: m6Anet

We installed m6Anet and ran it two ways on our exact folds:

| | PR AUC |
|---|---:|
| m6Anet, retrained on our folds (the fair comparison) | 0.4879 |
| m6Anet, its own pretrained model | 0.5033 |
| our `final_candidate` | 0.5293 (+0.040 over retrained, 5/5, p = 0.005) |
| our `everything` | 0.5408 |

**How to say this honestly:** m6Anet is strong - it beats our starting point
(0.4759) and our sequence-and-depth work combined. We only pass it once we add
the neighbour features. **What beat it was describing the data better, not a
better learning algorithm.** (Its pretrained model likely saw this cell line in
training, which is why it scores higher than the retrained one.)

---

## 6. Risks we found (and haven't all fixed)

These matter for the "limitations" part of the report, and one of them
matters for the graded test.

### 6.1 The neighbour features depend on what's in the input file

Neighbour features are computed from whatever sites are in the file
`predict.py` receives. If that file has fewer sites per transcript than ours,
they change.

- **Realistic thinning** (a shallower sequencing run, where low-coverage sites
  drop out): costs only 0.001-0.003. Fine.
- **A file where most sites are alone on their transcript**: the best model
  drops from 0.5408 to **0.4478**. That's *below the same model without
  neighbour features* (0.5013), on every fold. The model reads "no neighbours"
  (zeros) as a rare, meaningful value.
- **Our own demo file `data/sample/` is exactly this case**: 778 of its 1,000
  sites are alone on their transcript. It's the first file other students will
  run.

**Not fixed yet.** The cheapest fix: let `predict.py` count sites per
transcript and switch to the no-neighbour model when the file is sparse.

We also tried "train on thinned copies of the file" as a fix. It cost 0.005-0.009
on normal files and didn't help on realistic ones, so it was dropped.

### 6.2 The shipped model is stale

`models/final/` still holds the original `quantiles_v1` model (0.4759), not
`everything`.

### 6.3 LightGBM needs a system library that fresh Ubuntu may not have

On a fresh Ubuntu machine, `import lightgbm` failed until we ran
`sudo apt-get install libgomp1`. Evaluators run `predict.py` on AWS Ubuntu.
If their image lacks it, prediction fails for everyone. Needs a line in the
README at minimum.

### 6.4 Every conclusion is in the >= 20 reads regime

We simulate low depth by dropping reads, which isn't the same as genuinely
shallow data. Task 2 lives at 1-3 reads.

---

## 7. The newest work: can a neural network read the reads better?

Everything above is a tree on features *we* designed. The question for this
round: can a network, given the raw reads, find something our features miss?

### 7.1 What we tested

Each candidate turns a site's bag of reads into a short vector. Then we ask two
questions: (a) is that vector useful on its own, and (b) does it add anything
on top of our 137 hand-designed features? We judged each vector with several
classifiers (linear, nearest-neighbour, small neural net, tree), because a tree
splits one axis at a time and can undersell information that lies at an angle.

| family | models | trained to |
|---|---|---|
| reconstruction | autoencoder with a 4-number bottleneck; LSTM and attention over the 3 pore positions; masked attention across reads; contrastive | rebuild or match reads - no labels |
| prediction | **deepset**; attention-MIL; bottleneck autoencoder with a prediction head; a wider deepset | predict the site's label |
| controls | untrained random network; a network over our own features | - |

**How `deepset` works**, because it's the one that matters:

```
each read (9 numbers + the 7-letter sequence)
   -> the same small network for every read -> 64 learned numbers per read
   -> across the site's reads: the MEAN and the SPREAD of each of the 64
   -> a second small network -> one score for the site
```

About 19,000 parameters, trained end to end on the site label only. Nobody
tells it which reads are modified. The "spread" part is what catches "some
molecules differ". It's the learned version of our percentile features: it
learns *which combinations of a read's 9 numbers* to summarise, instead of us
choosing them.

### 7.2 A trap we had to design around

If a network is trained on the labels and its score is then fed to a tree
trained on the *same* sites, the tree learns to over-trust a score the network
memorised. Every network score was therefore **cross-fitted**: the network is
trained on half the training genes and scores the other half, and vice versa.
A **stacking control** - the same procedure with a network over our *own*
features - came out null, confirming the procedure doesn't manufacture gains.

### 7.3 Results (50 paired measurements; run on the Ronin cloud machine)

| added to `everything` | PR AUC | gain | wins | corrected p |
|---|---:|---:|---:|---:|
| (nothing) | 0.5408 | - | - | - |
| stacking control | - | +0.005 (vs hand features) | 36/50 | 0.19 |
| **`deepset` score** | **0.5685** | **+0.028** | **50/50** | **< 0.0001** |
| `deepset` + `kmer_norm` | 0.5737 | +0.033 | 50/50 | < 0.0001 |
| all four prediction networks + `kmer_norm` | 0.5782 | +0.037 | 50/50 | < 0.0001 |

At 3 reads, `deepset` also helps: 0.3900 -> 0.4089 (+0.019, 50/50).

`kmer_norm` is a no-network idea from the same work: score each read against
the average read for its 7-letter sequence, then summarise per site. +0.013 on
its own, but only +0.005 on top of `deepset` (not established).

**What we learned:**

- **A plain network over raw reads adds +0.028 to our best model.** That's the
  largest single gain in the project, and it isn't a stacking artefact.
- **All four prediction networks learned the same thing.** Head to head they
  are statistically identical: attention didn't help, a wider network didn't
  help, an autoencoder with a prediction head didn't help. Pictures of their
  internal vectors (UMAP) show each one collapsing to a single "modification
  score".
- **Every reconstruction-only model found nothing**, even trained much longer.
  Rebuilding typical reads doesn't preserve the rare modified signal.
- **Most of the value is in combining each read's 9 numbers**, not in reading
  them in context of the sequence: removing the sequence from the per-read
  network kept most of the gain.

### 7.4 Round 3: were the networks underfitting? No - and bigger doesn't help

A diagnostic showed the networks fitting their own training data no better
than unseen data, which suggested underfitting. If so, deeper networks with
**residual connections**, or a **Set Transformer** (where reads attend to each
other, so it can see "a subgroup of molecules that agree"), might do better.

Tested on 5 folds, each paired against the original `deepset`, with a pass
rule fixed *before* seeing results (at least +0.005 and better on at least 4
of 5 folds):

| model | parameters | held-out PR AUC | vs deepset | folds better |
|---|---:|---:|---:|---:|
| `deepset` (original) | 18,817 | 0.4882 | - | - |
| `deepset`, better training recipe | 18,817 | 0.4965 | +0.008 | 3/5 |
| residual `deepset` | 443,201 | 0.4941 | +0.006 | 3/5 |
| residual attention-MIL | 443,394 | 0.4814 | -0.007 | 3/5 |
| Set Transformer | 136,065 | 0.4827 | -0.006 | 1/5 |

(These are a network's own score, trained on half the data - not comparable
with the 0.54-0.57 numbers above.)

**Nothing passes.** Two lessons:

- **The "underfitting" was mostly an artefact.** The early-stopping validation
  set was too small (185 positives), so training stopped on noise. With a
  larger one, even the original `deepset` trains 2x longer and fits its
  training data well (0.66).
- **24x more parameters, residual streams, or letting reads attend to each
  other gave nothing measurable.** Combined with round 2 (four architectures
  converging on the same score), the evidence now points to a ceiling set by
  the information in the reads and site-level labels, not by the network.
  `deepset` stays: it's the smallest and cheapest to ship.

---

## 8. What hasn't worked (so nobody re-tries it blind)

| idea | result | why |
|---|---|---|
| a plain neural net on our features | tied the tree (-0.002), much worse at low depth | on a table of numbers, trees are hard to beat |
| per-read "how unusual is this molecule" features | +0.001 | measured against the site's own reads, which normalises the signal away |
| position-contrast features along each read | +0.003, not established | redundant with the percentiles (0.86-0.97 correlated) |
| the other 24 within-read correlations | +0.001 (small run) | the useful ones were already in coupling |
| training on thinned copies of the file | -0.005 to -0.009 on normal files | protects against the wrong kind of thinning |
| reconstruction-based networks (5 designs) | ~0 | reconstruction ignores the rare signal |
| attention, width, autoencoder heads | = deepset | all learn the same score |
| residual streams, Set Transformer, 24x parameters | = deepset (none passes a pre-set bar) | the ceiling isn't the network |

---

## 9. Where we are now, and what's next

**Best established model:** `everything.yaml`, PR AUC **0.5408** (in the
pipeline). With the `deepset` score added: **0.5685**, established at 50/50 -
but not in the pipeline yet.

**Next steps, in order:**

1. **Fix the sparse-file failure** (section 6.1) - it affects the file other
   students will actually see.
2. **Ship `deepset` into the pipeline.** About a day's engineering:
   cross-fitting inside `train.py`, and the network's forward pass rewritten in
   numpy (the graded `predict.py` may not use torch).
3. **Update `models/final/`**, re-calibrate (Platt), and add the `libgomp1`
   note to the README.
4. **Task 2 (SG-NEx) hasn't started.** It's the other half of the report.

Architecture search is effectively closed: round 3 (section 7.4) found nothing
better than `deepset`. Remaining effort is better spent on items 1-4.

---

## 10. Where things live in the repo

| you want | look at |
|---|---|
| the rules for working here (read first) | [AGENTS.md](../AGENTS.md) |
| what's graded | [docs/project-requirements.md](project-requirements.md) |
| every open question, with evidence | [GAPS.md](../GAPS.md) |
| the data format and read depth | [docs/data.md](data.md) |
| why the pipeline is built the way it is | [docs/decisions/](decisions/) (26 short records) |
| what we assume about the test file | [docs/test-data-assumptions.md](test-data-assumptions.md) |
| the neural-network work | [analysis/representation/](../analysis/representation/) (PLAN.md, results/) |
| a model is | a config in `configs/` + feature code in `src/m6a/features/` |

Adding an experiment = one new feature or model file + one config. Don't edit
shared code as a side effect (AGENTS.md explains why - four people, one repo).

---

## 11. For the report's AI-use section

The report must document at least two cases where AI output was wrong or
misleading. This project has several, all caught by measurement:

- A claimed ~2x improvement from a read-level model turned out to be training
  depth, not architecture (4.2).
- A prescribed fix ("train on thinned copies") was based on the wrong model of
  how test files get thinner; a screen showed it would have hurt (6.1).
- A network that appeared to win was partly credited for using more labelled
  data than its baseline; the cross-fitted control removed the confound (7.2).
- An "underfitting" diagnosis was mostly an early-stopping artefact (7.4).
- A regression check overwrote a recorded report; a Python-side cache path
  silently duplicated 700 MB; a job was silently dropped by a shell loop -
  each caught by checking outputs rather than trusting the run.

---

## Glossary

| term | meaning |
|---|---|
| site | one candidate A on one transcript; one row |
| read | one molecule's measurements at a site: 9 numbers |
| depth | number of reads (molecules) at a site |
| m6A | the chemical tag we predict |
| DRACH motif | the 18 five-letter patterns where m6A can occur |
| transcript / gene | one RNA sequence / the gene it came from (split by gene) |
| MIL | Multiple Instance Learning: labels on bags, not on items |
| PR AUC | area under precision-recall; baseline = positive rate (4.49%) |
| paired test, 50/50 | better on all 50 fold-by-fold comparisons |
| corrected p | Nadeau & Bengio t-test, accounts for overlapping training sets |
| depth augmentation | training on copies of sites with reads removed |
| cross-fitting | a model's score for a site comes from a model that never saw that site |
| m6Anet | the published deep-learning tool we benchmark against |
| SG-NEx | public nanopore datasets for Task 2 |
