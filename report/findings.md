# Findings for the report

A record of what the project found, with the numbers, so the report can be
written from it. Every result says **where it came from** (script and output
file), so any number can be regenerated or checked. Recorded 2026-10-03 to
2026-10-07; the analyses ran on the Ronin machine and on a laptop.

> **Read this first: the two labelled files are different cell lines.**
> dataset0 and data1 (the course's second labelled release) come from
> different cell lines - different reads AND different labels (course staff,
> 2026-10-05). Early project documents describe data1 as "a second sequencing
> run, labelled differently"; that interpretation was wrong. Throughout this
> file, "cell line 1" = dataset0 and "cell line 2" = data1. The final test
> data includes data that is not dataset0, may contain partial transcripts
> (positions without enough reads are not reported), and is scored on
> ranking only (ROC AUC, PR AUC).

Terms used throughout:
- **PR AUC** is the headline metric (positives are 4.5% of dataset0 sites,
  7.3% of data1), ROC AUC flatters every model.
- **Held-out genes:** every number is out of fold on a gene-grouped split; a
  test gene has no site, transcript or read in training, in either file.
- **Gain** = a model's PR AUC minus the baseline's on identical sites, with
  95% intervals from resampling whole genes.
- **Depth** = number of reads (molecules) at a site; see docs/data.md.

---

## Contents

1. [How models were evaluated](#1-how-models-were-evaluated)
2. [The shipped model and how it compares](#2-the-shipped-model-and-how-it-compares)
3. [What did and did not work (architecture search)](#3-what-did-and-did-not-work)
4. [Low read depth (Task 2 relevance)](#4-low-read-depth)
5. [Why ~100 nt of neighbours transfers between cell lines](#5-why-100-nt-transfers)
6. [m6A biology reproduced in our labels](#6-m6a-biology-reproduced)
7. [RNA folding: tested, not supported](#7-rna-folding-tested-not-supported)
8. [Why the graph network works](#8-why-the-graph-network-works)
9. [Leaderboard submission](#9-leaderboard-submission)
10. [Caveats to state in the report](#10-caveats)
11. [Suggested report figures](#11-suggested-figures)

---

## 1. How models were evaluated

**Problem found:** dataset0 cross-validation alone over-ranked models. The
hand-engineered `everything` feature set gained **+0.065** PR AUC over the
quantile-feature baseline under dataset0's labels, but only **+0.016** under
data1's, and **-0.003** on data1 genes absent from dataset0
(docs/decisions/0029).

**Cross-source evaluation (decision 0029):** one gene-grouped 5-fold split
over both files. Each model is trained either on dataset0 only ("dataset0
arm") or on both files ("both-files arm"), and scored as a gain over
quantiles + LightGBM on held-out genes of **both** files. Extra checks:
data1 genes absent from dataset0 ("new genes"), and a crossed test that
swaps the cell line's reads and labels separately on the 67,320 shared
sites.

**Selection rule (decision 0032):** rank on the **worse** of the two gains,
because the test set's cell line / labelling is unknown. Gaps under ~0.005
are ties, broken on the new-genes gain. Under the earlier mean-of-two rule
`everything` ranked first on a dataset0 gain that did not transfer.

**Networks were trained to convergence:** early stopping after 25 epochs
without a validation gain, with a plateau-halving learning rate (recipe v3).
A first attempt with a 45-minute cap left every network still improving; it
was stopped and rerun. Every reported network was checked by
`analysis/representation/convergence.py` (0 undertrained in the final runs).

Two labellings disagree (shared sites, n = 67,320): 2,136 positive in both,
1,155 positive only in dataset0, 2,717 positive only in data1 (94.25%
agreement). dataset0's own labels score PR AUC **0.326** against data1's.

---

## 2. The shipped model and how it compares

**Shipped (models/final, decision 0033):** an ensemble of four graph
networks, two designs x two seeds, trained on every labelled site of both
cell lines, scored in plain numpy (no torch needed), combined as the mean of
within-file ranks:

| network | what each site sees | output | parameters |
|---|---|---|---|
| `h2gcn_twohead_aux` (x2 seeds) | own reads + neighbours within 200 nt + transcript summary | one head per cell line, averaged | 46,082 |
| `h2gcn_aux` (x2 seeds) | own reads + neighbours within 50 nt | one score | 37,697 |

Both were trained with an auxiliary head scoring each site from its own reads
alone (discarded at prediction). Final fits all stopped on patience (best
epochs 33-48; validation AP 0.497-0.514).

**Architecture name for the report:** a DeepSets read encoder (Zaheer et al.
2017) feeding an H2GCN-style heterophily-aware graph network (Zhu et al.,
NeurIPS 2020) over each transcript's candidate sites: ego/neighbour
separation, a transcript-level context channel in place of 2-hop neighbours,
jumping-knowledge combination of layers (Xu et al. 2018); trained
inductively (Hamilton et al. 2017) on held-out genes.

**Cross-validated performance (held-out genes, both-files arm):**

| model | data1 PR AUC | data1 gain over LightGBM | dataset0 PR AUC |
|---|---|---|---|
| quantiles + LightGBM (previous shipped model) | 0.392 | - | 0.472 |
| `h2gcn_twohead`, mean of 3 seeds | 0.431 | +0.039 (seed sd 0.002) | 0.582 |
| `h2gcn_aux` | 0.436 | +0.044 | 0.553 |
| `h2gcn_twohead_aux` | 0.437 | +0.045 | 0.586 |
| **ensemble `h2gcn_twohead_aux` + `h2gcn_aux`** | **0.450** | **+0.058** [+0.048, +0.067] | **0.595** |

Ensembling two *different* designs gave +0.014 over the best single network;
three seeds of one design gave only +0.009 (data1 gain +0.048 vs +0.039).

**Against m6Anet** (pretrained HCT116_RNA002; it was likely trained on data
overlapping dataset0's cell line, so its numbers are its best case):

| | m6Anet PR AUC | m6Anet ROC AUC | our held-out PR AUC |
|---|---|---|---|
| dataset0 | 0.503 | 0.932 | 0.595 |
| data1 | 0.356 | 0.819 | 0.450 |

Every model as a gain over m6Anet (trained on dataset0, held-out genes):

| model | dataset0 | data1 | data1 new genes |
|---|---|---|---|
| `h2gcn` | +0.062 | +0.044 [+0.035, +0.055] | +0.058 [+0.032, +0.082] |
| `h2gcn_local` | +0.035 | +0.044 | +0.041 |
| deepset | +0.018 | +0.034 | +0.038 |
| quantiles + LightGBM | -0.027 | +0.016 | +0.022 |

m6Anet's own crossed test (gain over LightGBM): +0.042 on dataset0's reads
and labels, -0.013 on data1's reads and labels, -0.022 on data1's new genes -
the signature of a model trained on dataset0-like data.

**data2 (in vitro, never trained on):** one synthetic sequence, 189 positions,
sequenced at 0/25/50/70/75/95/100% modified molecules. Mean ensemble score
rises 0.14 (0%) -> 0.43 (25%) -> 0.57 (50%) -> 0.60 (70%), then plateaus
(0.57-0.60 to 100%). Earlier LightGBM models were non-monotone here (their
read-count features are confounded by the samples' different depths,
550-1,205 median reads).

Sources: analysis/representation/results/*_nn__xsrc.json, ensembles*.json,
analysis/m6anet/m6anet_pretrained__xsrc.json, W&B Decisions view.

---

## 3. What did and did not work

All trained on both files unless noted; gains are PR AUC over quantiles +
LightGBM on data1 held-out genes (the worse-gain measure).

**Worked:**
- **Graph over the transcript's sites, keeping a site's own evidence separate
  (H2GCN-style):** `h2gcn` +0.036; +0.010-0.016 over the same reads without a
  graph (deepset), interval excluding zero.
- **Training on both cell lines** rather than one: +0.044 vs +0.033 for
  `h2gcn_aux`.
- **Two output heads, one per cell line:** most stable model (seed sd 0.002)
  and best of the three finalists across 3 seeds (+0.039 vs +0.031/+0.033).
- **Auxiliary own-reads loss (deep supervision):** +0.044 vs +0.033 on the
  same base (`h2gcn_aux` vs `h2gcn_local`); new genes +0.029 vs +0.019.
- **Ensembling two designs:** +0.058 vs +0.045 best single.

**Did not work:**
- **Plain GCN** (site averaged with its neighbours): collapsed, **-0.10**
  data1, -0.14 dataset0 - over-smoothing; the site's own evidence drowned.
- **GAT** (attention over neighbours): +0.044 on dataset0, ~0 on data1 -
  attention learned the training cell line's clustering.
- **Attention over reads** (gated attention MIL, Set Transformer): no better
  than mean + spread pooling, even when trained to convergence.
- **Hand features inside the network** (`deepset_hand`): -0.021 vs deepset.
- **k-NN graph over a site's reads** (mutual, k = 4): no better than a control
  with the graph shuffled across reads (static vs shuffled -0.005 / -0.003 on
  data1), though k = 4 was chosen from in-vitro data where the graph was
  strongly homophilous (adjusted homophily 0.38 at 47 reads).
- **More capacity, three times:** deep residual encoders (138k parameters)
  -0.017 vs their base on data1 (+0.016 vs +0.033, and no gain over
  deepset); quantile pooling of read encodings -0.009; larger
  networks in earlier rounds tied. Winners added structure or training
  signal, never capacity.
- **Noise-adaptation (two "noisy annotators")**: tied two-head on score; its
  learned rates were not identifiable (true-positive rates stayed at their
  initial 0.90).
- **Transcript-wide context only** (`h2gcn_transcript`): +0.002 over deepset
  on data1 - its gain is cell-line-specific.

**Hand-engineered features (LightGBM era):** removing cross-site
(neighbour) features from `everything` cut its dataset0 gain from +0.065 to
+0.025 but raised its data1 gain (+0.016 -> +0.020): those features encoded
dataset0-specific clustering. Removing depth augmentation helped everywhere
(`everything_fulldepth` data1 +0.022, new genes +0.009).

---

## 4. Low read depth

Held-out data1 PR AUC when every site is thinned to the stated reads (all
models fitted at full depth):

| reads per site | quantiles + LightGBM | `h2gcn_local` | `h2gcn_aux` | `h2gcn_twohead` | `h2gcn_twohead_aux` |
|---|---|---|---|---|---|
| 1 | 0.136 | 0.237 | 0.243 | 0.252 | 0.248 |
| 3 (SG-NEx median) | 0.200 | 0.299 | 0.299 | 0.320 | 0.322 |
| 10 | 0.306 | 0.368 | 0.372 | 0.378 | 0.383 |

At 3 reads the networks lead by ~+0.10-0.12 (about 50% better) against
~+0.04 at full depth. Sources: results/depth_rescore*.json.

See section 8 for how much of this comes from neighbours.

---

## 5. Why ~100 nt transfers

**Radius test** (`h2gcn_aux` design, only the neighbour radius changed, one
seed; gain in PR AUC over LightGBM):

| radius | trained on dataset0 -> data1 (unseen cell line) | data1 new genes | both files -> data1 | both files -> dataset0 | crossed-test drop when labels change cell line |
|---|---|---|---|---|---|
| 50 nt | +0.033 | +0.029 | +0.044 | +0.081 | 0.017 |
| **100 nt** | **+0.043** | **+0.054** | **+0.047** | +0.096 | 0.029 |
| 200 nt | +0.034 | +0.048 | +0.038 | +0.103 | 0.029 |
| 400 nt | +0.032 | +0.044 | +0.033 | **+0.112** | **0.045** |

On the training cell line, wider is always better; on the other cell line,
~100 nt is best and 200-400 nt fall back to 50 nt's level; cell-line
specificity rises steadily with radius. The earlier impression that "200 nt
beats 50 nt" came from the transcript node and held on dataset0 only.

**A. Co-modification by distance (labels only).** For site pairs on the same
transcript, positive-positive pairs relative to two expectations; "local" =
beyond what positive-rich transcripts explain (95% CI from resampling
transcripts):

| distance | local, cell line 1 | local, cell line 2 | transcript-level, cell line 1 | transcript-level, cell line 2 |
|---|---|---|---|---|
| 1-25 nt | 2.20x [2.04, 2.38] | 2.13x [1.96, 2.29] | 3.4x | 2.1x |
| 25-50 nt | 2.10x | 2.00x | 3.3x | 2.0x |
| 50-100 nt | 1.88x | 1.73x | 3.3x | 2.0x |
| 100-200 nt | 1.46x | 1.34x | 3.3x | 2.0x |
| 200-400 nt | 1.11x | 1.02x | 3.2x | 1.9x |
| 400-800 nt | 0.80x | 0.74x | 3.0x | 1.8x |

Shared sites only (positions fixed, cell line changed): local 2.28x vs 2.06x
within 25 nt; transcript-level 3.06x vs 2.15x. Local co-modification is
shared between cell lines and fades over ~100-200 nt; transcript-level
enrichment is constant with distance and cell-line-specific. (Below 1x at
long range partly follows arithmetically from excess at short range.)

An earlier coarser version: positive-neighbour lift within 50 nt was 10.3x
(cell line 1) vs 5.2x (cell line 2), of which shuffling within transcripts
explains 3.9x vs 2.3x, leaving 2.6x vs 2.3x local.

**B. Neighbours per radius** (cell line 1; cell line 2 within 1-2 points):

| radius | no neighbours | 1 neighbour | 2+ neighbours | median |
|---|---|---|---|---|
| 25 nt | 42% | 38% | 21% | 1 |
| 50 nt | 17% | 29% | 54% | 2 |
| **100 nt** | 4% | 11% | **85%** | 3 |
| 200 nt | 1% | 2% | 97% | 7 |
| 400 nt | 0% | 0% | 99% | 13 |

**C. What the trained 400-nt model uses** (remove one band of neighbours at
prediction time; change in PR AUC):

| band removed | trained on dataset0: dataset0 | trained on dataset0: data1 | both files: dataset0 | both files: data1 |
|---|---|---|---|---|
| 0-50 nt | -0.004 | -0.004 | -0.009 | -0.009 |
| 50-100 nt | -0.005 | -0.002 | -0.004 | -0.004 |
| 100-200 nt | -0.007 | -0.004 | -0.008 | -0.003 |
| 200-400 nt | **-0.014** | **+0.001** | -0.010 | +0.001 |

Distant neighbours help only the training cell line; nearby ones help both.

**Synthesis:** m6A sits in position-dependent zones of ~100-200 nt shared
between cell types (section 6), so modified sites co-cluster locally over
that range in both cell lines (A); 100 nt is the smallest radius that gives
most sites several neighbours (B); beyond ~200 nt the neighbour average
becomes a transcript-level signal that is cell-type-specific (A, C).

Sources: analysis/newdata/distance_bands/, results/band_ablation.csv,
results/h2gcn_aux_r*_nn__xsrc.json.

---

## 6. m6A biology reproduced

Positive rate relative to each cell line's average (lift), positions in
transcript coordinates from Ensembl release 91. Validation: every labelled
site's 7-mer matches the reference sequence (100% of a 19,856-site sample)
and every annotated stop codon reads TAA/TAG/TGA (5,676 transcripts).

| context | cell line 1 | cell line 2 |
|---|---|---|
| within 25 nt of an exon-exon junction | 0.10x | 0.28x |
| 25-50 nt from a junction | 0.13x | 0.41x |
| 50-100 nt from a junction | 0.59x | 0.72x |
| 100-200 nt from a junction | 2.60x | 2.17x |
| 200-400 nt from a junction | 2.87x | 2.41x |
| > 400 nt from a junction | 1.19x | 1.25x |
| exon < 200 nt | 0.08x | 0.34x |
| exon 400-800 nt | 2.46x | 2.04x |
| exon > 1,600 nt | 1.44x | 1.50x |
| internal exon | 0.25x | 0.41x |
| last exon | 1.62x | 1.65x |
| 3' UTR | 1.41x | 1.51x |
| coding sequence | 0.70x | 0.71x |
| 100 nt before the stop codon | 1.63x | 1.54x |
| **first 100 nt after the stop codon** | **2.68x** | **2.40x** |
| 100-200 nt after | 2.63x | 2.35x |
| 500-1,000 nt before the stop codon | 0.40-0.68x | 0.43-0.58x |

Both cell lines show the same shapes (cell line 1 more extreme): the
exclusion zone near exon junctions matches the exon junction complex's known
suppression of m6A deposition; enrichment after the stop codon and in long
exons matches the canonical m6A distribution.

Source: analysis/newdata/annotation_context.py -> annotation_context/*.csv.

---

## 7. RNA folding: tested, not supported

Hypothesis: RNA folding brings sites that are far apart in sequence close in
space, so they co-modify. Test: fold every transcript, link two sites when
their +-5 nt windows are predicted to pair, and compare co-modification of
linked vs unlinked pairs **at the same distance on the same transcripts**
(transcript-composition controlled; repeated within the junction-permissive
zone). Two predictors with the same Vienna energy model: RNAplfold (local,
window 480, span 400) and LinearPartition (global, no span limit). 6,278
transcripts.

Linked / unlinked co-modification (1.0 = folding adds nothing; pairing
probability >= 0.1):

| distance | cell line 1, LinearPartition | cell line 1, RNAplfold | cell line 2, LinearPartition | cell line 2, RNAplfold |
|---|---|---|---|---|
| 25-50 nt | 0.94 [0.72, 1.16] | 1.00 [0.85, 1.17] | 1.15 [0.94, 1.36] | 1.18 [0.99, 1.42] |
| 50-100 nt | 1.17 [0.94, 1.46] | 0.98 [0.78, 1.18] | 1.35 [1.09, 1.67] | 1.33 [1.08, 1.65] |
| 100-200 nt | 1.02 [0.67, 1.43] | 0.97 [0.76, 1.22] | 1.03 [0.71, 1.41] | 1.07 [0.80, 1.34] |
| 200-400 nt | 1.09 [0.56, 1.59] | 0.97 [0.74, 1.18] | 1.19 [0.62, 1.75] | 1.21 [0.99, 1.53] |
| 400-800 nt | 1.27 [0.53, 2.01] | - | 1.96 [0.89, 3.32] | - |

- Cell line 1: no effect with either predictor at any distance.
- Cell line 2: a modest 50-100 nt enrichment (~1.35x) with both predictors -
  but both largely agree on short local hairpins, and it appears in one cell
  line only although folding is set by the (shared) sequence.
- Long range (100-800 nt), where folding would matter: nothing detectable.
- Predicted accessibility of the A: no robust association (RNAplfold most-
  unpaired quintile 0.92x / 0.88x; LinearPartition 0.95-1.09x everywhere).
- The radius test also argues against folding: a sequence-determined effect
  should transfer between cell lines, and beyond ~100 nt it does not.

Conclusion: a clean negative result - predicted RNA structure does not
explain m6A co-modification beyond position and distance. Caveat: both tools
predict test-tube structure of the *unmodified* sequence.

Sources: analysis/newdata/fold_comodification.py,
fold_comodification/ and fold_comodification_linearpartition/.

---

## 8. Why the graph network works

`h2gcn_aux` networks (neighbours within 50 nt), saved cross-validation
weights, nothing retrained. Numbers below: trained on both files, scored on
data1 (dataset0-trained networks show the same pattern).

**Gain over deepset by neighbour status** (labels used for analysis only):

| site's neighbours within 50 nt | sites | positive rate | gain over deepset |
|---|---|---|---|
| at least one modified | 9,308 | 26.4% | -0.007 [-0.018, +0.007] |
| only unmodified | 64,238 | 4.7% | +0.013 [+0.001, +0.027] |
| none | 16,906 | 6.3% | +0.003 [-0.011, +0.019] |
| all sites | 90,452 | 7.3% | +0.024 [+0.015, +0.032] |

Within each group the gain is ~0; overall it is +0.024. The graph's whole
advantage is telling the groups apart (a site next to a modified site is ~6x
more likely to be modified) - corroboration. Same on dataset0 (modified
neighbour -0.011, unmodified only +0.001, none -0.005; dataset0-trained).

Gain by own read count (both files -> data1): 20-30 reads +0.032
[+0.019, +0.046]; 30-50 +0.017; 50-100 +0.014; 100+ +0.019. By zone:
>= 100 nt from a junction +0.026 [+0.016, +0.036]; < 100 nt +0.010.

**Swapping what neighbours provide** (scored site unchanged; PR AUC on data1):

| neighbours given | PR AUC | change |
|---|---|---|
| real inputs (as trained) | 0.436 | - |
| nothing | 0.405 | -0.030 [-0.038, -0.023] |
| reads with their transcript's average removed | 0.388 | -0.048 |
| reads borrowed from same-5-mer sites elsewhere | 0.323 | -0.113 |
| a random site's reads and 7-mer | 0.310 | -0.126 |
| real reads but a random 7-mer | 0.273 | -0.163 |

Wrong neighbour evidence is far worse than none: the model trusts neighbours'
measurements. A wrong 7-mer is most damaging: it reads each neighbour's
current relative to what its sequence should produce - i.e. modification
evidence. Removing the transcript baseline removes the whole neighbour
benefit, which fits reading each neighbour's own deviation rather than
normalising against a local baseline.

**Depth asymmetry** (scored site thinned; PR AUC on data1):

| scored site's reads | neighbours with all reads | neighbours equally thinned | no neighbours |
|---|---|---|---|
| 1 | 0.367 | 0.243 | 0.225 |
| 3 | 0.381 | 0.299 | 0.269 |
| 10 | 0.408 | 0.372 | 0.338 |
| all | 0.436 | 0.436 | 0.405 |

A 1-read site with well-covered neighbours (0.367) approaches a full-coverage
site without neighbours (0.405); the neighbour benefit falls from +0.142 at
1 read to +0.030 at full depth. When neighbours are equally sparse (a whole
low-coverage sample, as in SG-NEx) the benefit is +0.030 at 3 reads.

**Mechanism paragraph for the report:**
The graph network's advantage comes from corroborating modification evidence
on nearby sites. For each neighbour within 50 nt it learns whether that
neighbour's current deviates from what its sequence should produce, and
because m6A sites co-occur locally (~2x within 50 nt, equally in both cell
lines), evidence of nearby modification raises a site's probability. Within
groups defined by whether a neighbour is truly modified, the graph adds
nothing over a read-only model; its entire gain is separating sites in
modified neighbourhoods from isolated ones. The effect is strongest where a
site's own evidence is weakest. Neither neighbours' sequence alone nor a
transcript-level baseline accounts for it.

**Why keeping the site's own evidence separate matters:** the plain GCN that
averaged a site into its neighbours collapsed (-0.10 on data1); `h2gcn` keeps
self, neighbour and transcript channels separate and the head sees every
layer. With no neighbours `h2gcn_aux` falls back to deepset level (0.405 vs
0.412), while `h2gcn_twohead_aux` - which relies on its transcript channel -
drops to 0.327 (ensemble alone: 0.391, LightGBM level).

Sources: results/gain_decomposition.csv, results/ablate_graph.csv,
results/isolated_rescore.csv.

---

## 9. Leaderboard submission

Intermediate leaderboard (7 Oct): `foursight_dataset0.csv` (121,838 sites),
`foursight_dataset1.csv` (90,810), `foursight_dataset2.csv` (1,323), from
the shipped ensemble; format, coverage and line endings verified. In-sample
scores (the model trained on these labels): dataset0 PR AUC 0.665 / ROC AUC
0.972; dataset1 0.513 / 0.900. Held-out estimates are lower (0.595 / 0.450).

---

## 10. Caveats

- **Cell lines vs labelling.** The two files differ in cell line; part of the
  cell-line-specific signal could still reflect how each was labelled
  (thresholds, expression affecting detection).
- **Single seeds** for most variants; seed spread for the finalists was
  0.002-0.005 PR AUC, so gaps under ~0.005-0.01 are ties.
- **Correlation, not mechanism,** for the biology (section 6): the zones fit
  the exon-junction-complex model but do not prove it.
- **Prediction-time ablations** (sections 5C and 8) feed networks inputs they
  never saw in training: they measure reliance, not what an ideal model would
  do.
- **Predicted structure** (section 7) is for the unmodified sequence in vitro.
- **m6Anet** comparison uses the pretrained model (likely its best case on
  dataset0); a retrained m6Anet on our folds was not run.
- **Missing neighbours.** Sites with no reported neighbour drop the ensemble
  to LightGBM level (0.391); test data may contain partial transcripts. A
  safeguard (use only `h2gcn_aux` for lone sites) is designed but not shipped.
- **Scores are ranks,** not calibrated probabilities (allowed: only ranking
  is scored).

---

## 11. Suggested figures

1. Cross-source evaluation schematic, and the dataset0-CV vs data1 gain
   scatter (CV over-ranking).
2. Model comparison: data1 PR AUC of LightGBM, deepset, h2gcn variants,
   ensemble, m6Anet.
3. PR AUC vs reads per site (section 4).
4. Radius test: gain vs radius for the unseen cell line and the training
   cell line (section 5).
5. Local vs transcript-level co-modification by distance, both cell lines
   (section 5A).
6. Positive-rate profiles around exon junctions and the stop codon, both
   cell lines (section 6).
7. Folding test: linked/unlinked ratios with intervals, both predictors
   (section 7).
8. Gain by neighbour status, and the depth-asymmetry curves (section 8).
9. data2 dose-response: mean score vs fraction modified.
