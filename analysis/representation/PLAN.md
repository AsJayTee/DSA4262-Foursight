# Learned read representations: plan

**Question.** Every model in this repo sees a site through ~137 hand-designed
numbers (quantiles, correlations, sequence and transcript columns). Can a
network, given the raw reads, *retrieve* a representation that carries
information those numbers miss - or carries it in a form the hand features
cannot express?

Written before any result, so the plan cannot drift towards whatever wins.

---

## 1. What every encoder sees, and what it produces

**Input per site:** all of its reads (20-991 rows of 9 measurements: dwell,
signal sd and mean current at pore positions -1, 0, +1 - see
[docs/data.md](../../docs/data.md#read-depth)), its 7-mer, and its read count.
Dwell and sd are log-transformed (both are positive and heavy-tailed), then
every column is standardised on the training reads.

**Not input:** the hand features. The encoders must find their own
description; the hand features come back in at evaluation, where the question
is what the learned vector *adds* to them. One arm (`mlp_hand`) is the
exception, on purpose: it is a network over the hand features, which is the
literal "baseline NN on the points we already have".

**Output per site:** a vector z of roughly 16-32 numbers.

**Read subsampling while training.** Each step, each site shows the encoder a
random subset of its reads - its full count (capped at 48) half the time, a
uniform 1..48 the other half. Two reasons: it is the set-level analogue of the
depth augmentation that doubled low-depth performance
([0022](../../docs/decisions/0022-training-rows-may-come-from-several-depths.md)),
and it lets one encoder embed a site at any depth, which evaluation E5 needs.
At embedding time every read is used.

Pooling is by **mean and standard deviation**, never sum or max: those grow with
the number of reads, so an encoder trained on 48 reads would see a different
distribution at 991.

## 2. The encoders

| # | name | architecture | objective | uses labels | why it is here |
|---|---|---|---|---|---|
| 0 | `hand` | the 137 `quantiles_all_v1` columns | - | - | the reference everything is measured against |
| 0b | `hand_pca` | PCA of the above to 32 dims | - | - | the reference at the same width as z |
| 0c | `random` | encoder 2's architecture, untrained | - | no | **control**: random projections of reads, pooled, can be surprisingly good; a learned z must beat this |
| 1 | `mlp_hand` | MLP over the 137 hand columns, z = penultimate layer | predict | yes | the requested baseline NN on the points we already have |
| 2 | `deepset` | per-read MLP([read, 7-mer]) -> mean+sd pool + log n -> MLP -> z(32) -> logit | predict | yes | the requested baseline NN on the raw reads; the simplest permutation-invariant network |
| 3 | `read_ae` | per-read autoencoder, **bottleneck 4**, encoder and decoder both see the 7-mer | reconstruct each read | no | the bottleneck AE. Conditioning on sequence is essential: without it the bottleneck spends itself encoding which bases are in the pore |
| 4 | `read_ae_pred` | encoder 3 + a site head on the pooled codes | reconstruct + predict | yes | the requested "bottleneck AE with goal to predict" |
| 5 | `pos_lstm_ae` | per read, an LSTM over the **3 pore positions** (each step: 3 measurements + that window's 5-mer); final hidden -> bottleneck 4 -> LSTM decoder | reconstruct | no | the requested RNN, final state as the code. See the note below on what the sequence is |
| 6 | `pos_attn_ae` | encoder 5 with self-attention over the 3 positions instead of the LSTM | reconstruct | no | the requested "same with attention" |
| 7 | `set_masked` | transformer over the **reads of a site** (2 layers, d=32) plus a 7-mer token; hide 25% of the reads and predict their measurements from the rest | reconstruct (masked) | no | attention where it can actually matter: across reads. Forces a model of the site's read distribution |
| 8 | `attn_mil` | gated-attention MIL (Ilse et al. 2018) over reads + 7-mer | predict | yes | the supervised attention counterpart; the architecture `src/m6a/models/mil.py` implements |
| 9 | `contrastive` | encoder 2, trained so two **disjoint random subsets of one site's reads** map to the same z (InfoNCE) | agree across subsets | no | **recommended addition.** It learns exactly the property a site-level description should have - stable whichever molecules you happened to sample - which is the depth problem stated as an objective |

**On the RNN's sequence.** Reads are a *set*: there is no order among them,
and an LSTM run over them in file order would learn an artefact of that order.
Inside one read, though, the three pore positions *are* a sequence - the RNA
ratchets through the pore one base at a time. So the RNN and its attention twin
run over that axis, with sequence-aware inputs, and attention *across* reads is
covered by `set_masked` and `attn_mil`, which are permutation-invariant by
construction.

**For the per-read encoders (3-6), the site vector** is the mean, sd and 10th /
50th / 90th percentiles of the reads' codes, plus the mean, median, 90th and
95th percentile of reconstruction error, the fraction of reads above the
training reads' 95th-percentile error, and log n. Reconstruction error is the
anomaly score: if modified molecules are unusual, a site with more of them
should reconstruct worse.

## 3. The protocol that keeps it honest

**Split.** The canonical gene-grouped folds (seed 4262). Only repetition 0, so
5 paired observations per comparison, not the harness's 50. Everything here is
a **screen**: a representation that clears it graduates to the real harness as
a feature extractor.

**Train the encoder and the probe on different sites.** A supervised encoder's
vector on its own training sites is over-confident, and a classifier trained on
it learns to trust it too much - the stacking leak. So inside each training
fold, genes are split in half by a fixed hash:

- **half A** trains the encoder (with 10% of A's genes held back to pick the
  epoch, for supervised encoders);
- **half B** trains every probe;
- **the held-out fold** is scored.

Every representation, including `hand`, goes through this same protocol, so
they are comparable with each other. They are **not** comparable with the
harness's 0.5408: probes see half the training data and no depth augmentation.
`hand` under this protocol is the number to beat.

## 4. How the representations are judged

The concern is real. A gradient-boosted tree splits one axis at a time, so a
representation whose information lies along *oblique* directions can look
weak to a tree and strong to anything else. So a range of lenses, none of which
is the tree alone.

**E1. Probes of different shapes, on z alone** (PR AUC on the held-out fold):

| probe | what it can see |
|---|---|
| logistic regression | information along any single direction: linear separability |
| k-nearest neighbours (k = 100) | local neighbourhood structure, any shape, no axes |
| small MLP (1 hidden layer) | smooth non-linear, oblique boundaries |
| LightGBM | axis-aligned, piecewise |
| LightGBM on a **randomly rotated** z | the same, with any axis alignment destroyed |

**E2. Does the axis matter? The rotation test.** Compare the tree on z and on a
random rotation of z. If the tree does *better* after rotation, or the MLP and
logistic probes beat the tree on z but not on `hand`, the representation's
information is oblique and a tree is the wrong judge of it - the concern above,
made measurable. **Positive control:** rotating `hand` should *hurt* the tree,
because hand features are built to be meaningful one column at a time. If it
does not, the test is not sensitive.

**E3. Incremental value - the practical question.** LightGBM on hand + z
against hand alone, and logistic on hand + z against hand alone, paired over
the 5 folds with the corrected test (`m6a.compare.paired_comparison`). The
logistic version answers the same question without the tree's axis bias.

**E4. What is in the vector.**
- *Redundancy:* how much of z a ridge regression on the hand features explains
  (R^2), and linear CKA between z and hand. High = z re-derived what we already
  had.
- *Nuisance:* how well z predicts read depth and the motif. A representation
  that mostly encodes coverage or sequence is spending its capacity on things
  the model already knows.
- *Effective dimensionality:* the participation ratio of z's covariance - how
  many directions it actually uses.

**E5. Depth robustness.** Embed the held-out sites from 1, 3 and 10 reads (the
harness's keyed subsample, `m6a.data.subsample_blocks`) and score them with
the probes trained at full depth. Compare against hand features recomputed at
the same depths.

**E6. Pictures.** UMAP and t-SNE of the held-out fold-0 sites (all positives
plus a random sample of negatives), coloured by label, motif family and read
depth, for `hand` and each z. They are there to *see* whether positives form
a region or a scatter, and whether the dominant structure is label or
nuisance. Pictures are not evidence of separability on their own - a 2-D
projection can merge what is separable in 32 dimensions and separate what is
not - so every visual claim is checked against E1-E4.

## 5. Order and budget

CPU only (4 cores). Each encoder is time-boxed per fold. Order: controls and
`hand` first, then the requested models (1, 2, 3, 4, 5, 6), then the attention
set models (7, 8), then `contrastive`. If time runs short, the order says what
gets dropped.

## 6. What would count as a result

- **Headline:** any z where hand + z beats hand by at least ~+0.006 on 5/5 or
  4/5 folds under both the tree and logistic (E3). That graduates to the harness.
- **Useful even if null:** whether the learned vectors are redundant with the
  hand features (E4), and whether the tree was hiding anything (E2).
- **Not a result:** a z that beats `hand` alone under one probe only, or a
  pretty picture.
