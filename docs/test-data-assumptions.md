# What we assume about the test data, and what breaks if we are wrong

The leaderboard scores us on a file nobody here has seen. Every modelling
choice in this repo quietly bets on what that file looks like. This page writes
the bets down, so that a choice can be judged by **how badly it fails if its
assumption is wrong**, not only by how well it scores on our own data.

The principle: our cross-validation measures performance on data built exactly
like ours. The test file may not be. A model that is 0.01 better on our data
and 0.04 worse on a plausible test file is the worse model. For each assumption,
we want either evidence that it holds, or a measurement of what it costs when it
does not, or a model that does not depend on it.

For scale: the leaderboard is **5%** of the grade, scored on ROC AUC and PR AUC
against a random classifier and a simple baseline
([project-requirements.md](project-requirements.md)).

---

## The assumptions, riskiest first

Status key: **measured** (we know the cost if wrong), **hedged** (the model
already protects against it), **unmeasured** (nobody knows).

### 1. Each transcript's sites are present together - MEASURED, and it fails badly

**What we assume.** The test file contains most candidate sites on each
transcript, as ours does (median 19 per transcript; only 1,172 of 121,838 sites,
about 1%, sit on transcripts with fewer than 5).

**What depends on it.** The cross-site features - relative position along the
transcript, and averages of the *other* sites' measurements on the same
transcript - are the largest gain in the project, about +0.027 PR AUC. They are
computed from whichever sites are in the input file
([0025](decisions/0025-a-feature-set-may-see-every-site.md)). A site that is
alone on its transcript gets zeros for all of them.

**What we know.**

| if the test file is thinned... | cost to the current best model |
|---|---:|
| by coverage (a shallower run: low-read sites drop out) | -0.001 to -0.003 |
| at random, keeping 75% / 50% / 25% of sites | -0.009 / -0.016 / -0.038 |
| at random, keeping 10% / 1% of sites | **-0.085 / -0.097** |
| every site alone on its transcript | **-0.093** |

On the sparse files the best model (0.4478) scores **below the same model with
the cross-site columns removed** (0.5013), on every fold. It treats "no
neighbours" as a meaningful, rare value rather than as missing information.

(One split, five folds; `analysis/evaluation/scratch/density_screen.py` and
`sparse_file.py`. Details in [GAPS.md](../GAPS.md), under the cross-site entry.)

Training on thinned copies was screened as a fix and **rejected**: it cost
0.005-0.009 on a normal file, on every fold, and did not help on coverage-thinned
files.

**Why the extreme case is not hypothetical.** Our own `data/sample/` - the file
other students will run `predict.py` on - is 1,000 sites drawn at random from
884 transcripts, and **778 of them are alone on their transcript.** That is the
worst case for these features, and it is the first file anyone outside the team
will see the model score. We made that file ourselves
(`scripts/make_sample.py`), so it says nothing about how the course builds its
test files - but it proves a randomly subsampled file is an easy thing to make.

**Options, none built:**
- Train on an extra copy of every site computed as if it were alone, so the
  model learns that zeros mean "no neighbours" rather than a real value. The
  milder version of this (random thinning to 25-75%) cost 0.005-0.009 on a
  normal file, so its price has to be measured, not assumed.
- Let `predict.py` look at the file it was given. It can count sites per
  transcript before predicting, so it can **detect** a sparse file rather than
  guess. It could then pick a model trained without cross-site features, or warn.
- Rebuild `data/sample/` from whole transcripts, so the demo file at least
  matches what the model expects.

### 2. Every site has at least 20 reads - HEDGED

**What we assume.** Like ours, every test site was measured on at least 20
separate RNA molecules. *Depth* means that count of molecules, not repeated
readings of one - see [data.md](data.md#read-depth).

**What depends on it.** Most per-site features are summaries across reads
(quantiles, correlations). The correlation ("coupling") features need at least
5 reads and are zero below that.

**What we know.** The floor comes from how the course prepared our file, which
matches m6Anet's 20-read sampling. If the test file keeps the same preparation,
this holds. If not:

| reads per site | 1 | 3 | full |
|---|---:|---:|---:|
| current best (`everything`) | 0.3121 | 0.3910 | 0.5408 |

**Hedge in place.** Depth augmentation: the model is also trained on copies of
each site with only 1, 3, 5 and 10 reads kept
([0022](decisions/0022-training-rows-may-come-from-several-depths.md)). That
roughly doubled the one-read score at no measured cost at full depth.

### 3. Sites were chosen the same way - UNMEASURED

**What we assume.** The test sites were selected by the same rule as ours.

**What we know.** Our 121,838 sites are about 78% of the roughly 156,000 sites
in the source sample (SG-NEx Hct116 `replicate3_run1`) that have at least 20
reads. So some rule beyond the read floor chose them - most likely which sites
m6ACE-Seq could label. We do not know that rule, so we do not know whether a
test file would follow it.

**What depends on it.** The positive rate (below), and assumption 1: a different
selection rule changes which neighbours are present.

### 4. About 4.5% of sites are modified - LOW RISK for the grade

**What we assume.** A positive rate near our 4.49%.

**What depends on it.** Calibration and thresholds - how many sites the model
calls modified. Platt scaling is fitted to our rate.

**Why it matters little here.** ROC AUC and PR AUC only use the *ranking* of
scores, so a different rate cannot change our ranking. It does change what a
"good" PR AUC looks like, because PR AUC starts from the positive rate. So a
leaderboard number is not comparable with our cross-validated one if the rates
differ. It matters a great deal for Task 2, where we count modified sites.

### 5. Same cell line and chemistry - MEASURED, small

**What we assume.** The test data looks like Hct116 data from the same kind of
sequencing run.

**What we know.** Comparing Hct116 with SG-NEx A549 at the same read depth, every
one of the nine per-read measurements differs by less than 0.2 of its spread,
and the mix of sequence motifs is near-identical. Depth is a much larger shift
than cell line. (Directional: about 7,000 A549 sites, not part of the harness.
See GAPS.md.)

### 6. The test genes are new to the model - AFFECTS HOW TO READ THE SCORE

**What we assume.** Our cross-validation holds out whole genes, so our numbers
describe genes the model has never seen.

**What we know.** If the test file is another Hct116 replicate, it probably
contains the *same* genes and sites we trained on. The leaderboard would then
score higher than our cross-validation predicts. That is fine for the grade and
misleading for the report: a leaderboard number from familiar genes is not
evidence that the model generalises.

### 7. File format and identifiers - LOW RISK

**What we assume.** m6Anet-processed JSON with the same nine measurements and a
7-letter sequence, and transcript IDs and positions from the same annotation.

**What we know.** The handout specifies the format. The neighbour code sorts
sites itself rather than trusting file order. A different annotation version
would shift positions and change which sites count as neighbours; nothing
checks for it.

---

## What to do with this

1. **Before shipping any model, read its row in this list.** A model built on
   cross-site features is making assumption 1. Say so beside its number.
2. **Measure the unmeasured ones that are cheap.** Assumption 1's extreme case
   is one scratch run. It is also the file other students will actually see.
3. **Prefer choices that fail gently.** Where two models are within noise on
   our data, ship the one that depends on fewer assumptions.
4. **Detect instead of guessing where we can.** `predict.py` sees the whole
   input file before it predicts. Sites per transcript, reads per site and motif
   mix are all measurable there. Nothing in this list has to be a blind bet at
   prediction time.

When an assumption is tested, update its status here and put the evidence in
[GAPS.md](../GAPS.md).
