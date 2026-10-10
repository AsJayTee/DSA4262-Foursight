# Findings for the report

A record of what the project found, with the numbers, so the report can be
written from it. Every result says **where it came from** (script and output
file), so any number can be regenerated or checked. Recorded 2026-10-03 to
2026-10-08; the analyses ran on the Ronin machine and on a laptop.

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
12. [Final batch (8 Oct): designs, literature motivation, results](#12-final-batch-8-oct-designs-literature-motivation-results)
13. [Where the data comes from, and public data beyond the course](#13-where-the-data-comes-from-and-public-data-beyond-the-course)
14. [Models the report discusses](#14-models-the-report-discusses-agreed-10-oct)
15. [Candidate entries for the AI-use table](#15-candidate-entries-for-the-ai-use-table-recorded-10-oct-choose-later)
16. [Appendix: full tables](#appendix-full-tables) - every band, cell line,
    threshold and bin for the clustering-by-distance (A), neighbours-per-radius
    (B), 400-nt band-use (C), position-on-transcript (D) and RNA-folding (E)
    analyses, generated from the result files

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

**Against m6Anet** (pretrained HCT116_RNA002). **Checked 9 Oct:** m6Anet was
trained on exactly dataset0's setting:
- **Same cell line and labels:** SG-NEx HCT116 direct RNA (replicate 2
  run 1; ENA PRJEB44348) with HCT116 m6ACE-seq labels. Every m6ACE-seq
  site counts as modified; other positions with the same 5-mer are
  unmodified.
- **Same pipeline:** at least 20 reads per site, DRACH motifs only,
  nanopolish eventalign features. The course data is in m6Anet's own
  format.
- **The same source sample:** dataset0 itself comes from SG-NEx HCT116
  replicate 3 run 1 (docs/test-data-assumptions.md).

So on dataset0, pretrained m6Anet has seen the same cell line's labels,
almost certainly for many of the same sites. Its dataset0 number is
optimistic, and data1 is the fair comparison. m6Anet's own published
cross-cell-line test (HEK293T, m6ACE-seq + miCLIP labels): ROC AUC 0.83,
PR AUC 0.35 (Hendra et al., Nat Methods 2022, doi:10.1038/s41592-022-01666-1).
That is close to its data1 result here (0.819 / 0.356).

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

---

## 12. Final batch (8 Oct): designs, literature motivation, results

**Status: complete.** The batch ran on Ronin from 04:53 to 22:41 UTC on
8 Oct. All networks converged (convergence.py: 0 undertrained). Results are
in 12C. The headline: constrained corroboration transfers better between
cell lines; a graph transformer does not.

**The question.** Section 8 shows the graph network works by corroboration:
a site gains when nearby sites look modified. Section 5 shows that only about
100 nt of context carries over to an unseen cell line. Two possibilities
remain:

- an architecture that builds in "nearby sites corroborate, and the effect
  fades with distance" transfers better than a free neighbour mean;
- a more expressive network (a graph transformer) does better still.

This batch tests both and adds controls for confounds. Every design keeps
the own-reads auxiliary head (weight 0.5). Every design is compared to
`h2gcn_aux` trained on the same seeds. The ranking rule is unchanged (0032:
the worse of the two cross-source gains).

### 12A. Designs, what each tests, and its control

| Design | What it changes | What it tests | Control (what a null result means) |
|---|---|---|---|
| `fk_band` | Neighbour message = weighted sum of mean neighbours in 0-30, 30-75 and 75-150 nt bands. The weights are fixed, not learned: per band, max(log co-modification ratio, 0) from **cell line 1 training labels** of each fold. | Does imposing the measured distance profile transfer better than letting the network learn it? | `fk_band_shuffled`: the same bands with the weights reversed. If it matches `fk_band`, the profile itself carries nothing. |
| `fk_kernel` | Neighbours weighted by exp(-d/λ) within 150 nt, row-normalised. λ is **fitted per fold** to the decay of co-modification in cell line 1 training labels: 101.5-122.1 nt. | A smooth, data-set decay instead of a hard 50-nt cutoff. | `h2gcn_aux` (hard 50 nt) and `h2gcn_aux_r100` (hard 100 nt), all trained on the same seeds. |
| `res_gate` | Score = own-reads score + g × (neighbour correction) × has-neighbour, with g = sigmoid(a·log reads + b·\|own score\| + c). Edge dropout 0.3 and node dropout 0.15 in training. Same kernel as `fk_kernel`. | "Safe" corroboration: neighbours can only adjust the own-reads score, more when the site has few reads or an uncertain score. This targets the lone-site and partial-transcript risk (section 10). | `res_nogate` (g = 1: is the depth gate needed?); `res_nodrop` (gate, no dropout: does dropout help sites with few neighbours?). |
| `scalar_msg` | Neighbours pass only their own-reads **scores** (kernel-weighted mean, max within 75 nt, mean log reads), not learned embeddings. A small MLP turns these into a correction to the site's own score. | Is corroboration just "neighbours look modified"? If so, a one-number message should suffice and generalise at least as well. | `scalar_random`: the same model, but the scores come from unrelated sites. A gain that survives this is not corroboration. |
| `gps` | GraphGPS-style layer: the local 50-nt message passing plus global attention over every site on the transcript, with a learned distance bias. | Does a more expressive, modern architecture (transformer-style, long-range) help? Prediction from sections 5 and 8: it helps cell line 1 and not the unseen cell line. | `h2gcn_aux` on the same seeds. |

Confound analyses run alongside:

- **Gain by neighbour count** (0 / 1 / 2-3 / 4+ within 50 nt) and **by
  transcript size** (quartiles of sites per transcript). Having neighbours
  depends on expression (positions need enough reads to be reported). The
  question is whether section 8's neighbour effect is really a coverage
  effect. A ROC AUC gain column is added.
- **Band ablation with 0-10 and 10-20 nt split out.** Nanopore signal from
  one m6A spreads about ±10 nt (DeepRM, below). So a very close "neighbour"
  may partly re-measure the scored site's own modification rather than
  corroborate it. The earlier coarse-band table is kept as
  `band_ablation_v1.csv`.

Order: seed 0 of every design and control; then seeds 1 and 2 of the main
designs plus `h2gcn_aux` and `h2gcn_aux_r100`. Each design gets three seeds
against matched baselines, which matters because the seed spread is
0.002-0.005 PR AUC (section 10). Source: analysis/representation/run_day.py,
xsrc_nets.py (`GRAPH_VARIANTS`), graph.py (`forward_v2`,
`comod_band_weights`, `comod_decay_scale`).

**A finding from setting up the batch.** The decay scale fitted to the
labels differs by cell line:

- cell line 1 training labels: 101.5-122.1 nt across the 5 folds;
- cell line 2 labels: about 59 nt (computed for reference only; no model
  uses cell line 2 labels to set λ).

~~Co-modification fades about twice as fast in cell line 2.~~ **Retracted
after checking (9 Oct): the difference is not robust.**
`analysis/newdata/decay_scale_check.py`, with 1,000 transcript resamples,
gives:

| Labels | Sites | Positive % | λ (nt) | 95% interval |
|---|---|---|---|---|
| Cell line 1, all sites | 121,838 | 4.5 | 112.7 | 81.5-143.7 |
| Cell line 2, all sites | 90,810 | 7.3 | 58.8 | 38.2-122.8 |
| Shared sites, cell line 1 labels | 67,320 | 4.9 | 100.7 | 40.2-123.4 |
| Shared sites, cell line 2 labels | 67,320 | 7.2 | 102.6 | 42.6-126.0 |
| Cell line 2 thinned to 4.49% positives (50 draws) | 90,810 | 4.5 | 72.8 | 36.2-114.8 |

- The intervals overlap heavily.
- On the sites both files share, the two cell lines' labels give the
  same scale (101 vs 103 nt).
- The short 59 nt comes from cell line 2's *other* sites, not from its
  labels decaying faster on the same positions.

Report it as: the decay scale is about 100 nt in both cell lines, with
wide uncertainty. That is consistent with section 5A (local
co-modification is shared between cell lines) and with the ~100 nt
exon-junction zone (12B). Source: analysis/newdata/decay_scale/decay_scale.csv.

### 12B. Literature motivation (each claim checked against the paper, 8 Oct)

**Biology: m6A sites cluster, and the clustering fades with distance.**

- **m6Aiso**: *Single-molecule m6A detection empowered by endogenous
  labeling unveils complexities across RNA isoforms*, Molecular Cell 2025,
  doi:10.1016/j.molcel.2025.01.014 (Jinkai Wang's group, Sun Yat-sen
  University; preprint bioRxiv 10.1101/2024.01.30.577990).
  - Single-molecule linkage between m6A sites is "relatively weak but
    nonnegligible" below 200 bp. It decreases gradually with distance and
    is not recognisable beyond 1 kb.
  - 39.1% of m6A sites cluster with another within 50 bp on the same gene,
    and 44.5% on the same isoform.
  - Motivates: a distance-decaying neighbour weight (`fk_kernel`,
    `res_gate`) and our own measured profile (section 5A: 2.2× below 25 nt,
    falling to 1.0× by 200-400 nt).
  - *Earlier notes cited this as "Guo et al. 2025". The first author is not
    confirmed, so cite it by title and DOI until someone checks the
    published author list.*
- **DeepRM**: Kang, Hwang & Baek, *Comprehensive discovery of m6A sites in
  the human transcriptome at single-molecule resolution*, Nat Commun 2025,
  doi:10.1038/s41467-025-67417-w.
  - "4819 pairs of significantly co-occurring m6A sites across the HEK293T
    transcriptome".
  - In their example, the pair co-occurs on 40.0% of reads against 21.8%
    expected (positive co-occurrence on the same molecule).
  - Motivates: corroboration has a molecular basis. Modified sites tend to
    be modified together on the same molecule.

**Biology: why about 100 nt is shared between cell lines.** Explains
section 6. *These are not used as model inputs*: the data has no
annotation, and annotation was ruled out as an input.

- **Uzonyi et al.**, Mol Cell 2023, 83(2):237-251,
  doi:10.1016/j.molcel.2022.12.026. The exon junction complex excludes m6A
  within about 100 nt of splice junctions.
- **He et al.**, Science 2023, doi:10.1126/science.abj9090. Exon junction
  complexes suppress m6A. Their range covers average-length internal exons,
  not long internal or terminal exons. That matches the stop-codon and
  last-exon enrichment in section 6.
- **Luo et al.**, Nat Commun 2023, 14:4172,
  doi:10.1038/s41467-023-39897-1. The exon-intron boundary represses 12-34%
  of m6A at adjacent exons, over about 100 nt.
- Reading: the main rule shaping m6A placement is gene structure on a scale
  of about 100 nt, and it is shared across cell types. That fits a ~100-nt
  context transferring between cell lines while wider context does not.
  This is correlation, not mechanism (section 10).

**Signal: why very close neighbours are a confound.**

- **DeepRM** (above): "modification-dependent variations in electric
  current and base quality were observed over a wide range of up to ±10 nts
  from m6A". "Electric current signatures from two m6As within a 20-nt
  window will overlap and therefore interfere with each other."
  - Motivates: the 0-10 / 10-20 nt split in the band ablation.
- **Nanocompore**: Leger et al., Nat Commun 2021,
  doi:10.1038/s41467-021-27393-3. About 5 nucleotides sit in the R9 pore's
  reader head at once, so one modification shifts the current of several
  consecutive k-mers.
  - This is why each site's 9 features cover three overlapping 5-mers.

**Machine learning: where each design comes from.** These are standard
papers, identified from memory; claims about their methods are from the
papers' own titles and abstracts.

| Paper | Idea | Used in |
|---|---|---|
| H2GCN: Zhu et al., *Beyond Homophily in Graph Neural Networks*, NeurIPS 2020 | Keep the node's own embedding separate from its neighbours' | The base architecture (sections 2, 8) |
| GraphGPS: Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, NeurIPS 2022 | Local message passing + global attention in each layer | `gps` |
| SchNet: Schütt et al., NeurIPS 2017 | Continuous filters as a function of distance | Distance-weighted messages (`fk_kernel`) and the learned distance bias in `gps` |
| C&S: Huang et al., *Combining Label Propagation and Simple Models Out-performs GNNs*, ICLR 2021 | A base predictor, then propagate its errors and predictions over the graph | `res_gate` (correct the own-reads score), `scalar_msg` (propagate scores, not embeddings) |
| DropEdge: Rong et al., ICLR 2020; GRAND (DropNode): Feng et al., NeurIPS 2020 | Randomly remove edges and nodes in training | Dropout in `res_gate`; `res_nodrop` is the control |
| GPR-GNN: Chien et al., ICLR 2021; ACM-GCN: Luan et al., NeurIPS 2022 | Learned weighting of neighbour information, per node and channel | The depth gate in `res_gate` |
| PNA: Corso et al., NeurIPS 2020 | Several aggregators instead of one mean | Per-band means in `fk_band` |
| Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR 2021; V-REx: Krueger et al., ICML 2021; group DRO: Sagawa et al., ICLR 2020 | Domain-generalisation methods rarely beat plain training under a fair model selection rule; select on the worst domain | Why designs are ranked on the worse of two gains (0032), and why we change inductive bias rather than add DG penalties |

**What the batch could conclude** (written before the results; see 12C for
what happened):

- If `fk_band` or `fk_kernel` beats `h2gcn_aux` on the unseen cell line
  and its control does not, the measured distance profile is a transferable
  prior.
- If `scalar_msg` matches `h2gcn_aux` and `scalar_random` falls to
  deepset level, corroboration is fully captured by one score per neighbour.
- If `gps` gains on cell line 1 but not on the unseen cell line, more
  capacity buys cell-line-specific context, as predicted. Null results are
  reported as results.

### 12C. Results

**The designs in plain terms.** All share the same read encoder, which turns
a site's bag of reads into a vector and an "own-reads" score: how modified
the site looks from its own reads alone.

- `h2gcn_aux` (current design, the reference): each site averages the
  learned vectors of every neighbour within 50 nt, equally weighted, and
  combines that with its own vector.
- `h2gcn_aux_r100`: the same, with a 100 nt radius.
- `fk_band`: neighbours out to 150 nt, averaged separately in three distance
  bands (0-30, 30-75, 75-150 nt). The bands are mixed with fixed weights
  measured from the training labels.
- `fk_kernel`: neighbours out to 150 nt, weighted smoothly by distance,
  exp(-d/λ), with λ (~100-120 nt) fitted to the training labels.
- `res_gate` ("residual + gate"): the own-reads score is the starting
  answer. The neighbours (weighted as in `fk_kernel`) may only add a
  correction to it. A learned gate decides how much of the correction to
  apply, using the site's read count and how confident the own-reads score
  is. In training, 30% of neighbour links and 15% of neighbour sites are
  dropped at random, so the model cannot rely on any one neighbour.
- `scalar_msg` ("scalar messages"): neighbours pass only one number each,
  their own-reads score (plus read counts), not a learned vector. A small
  network combines the site's own score with three summaries:
  - the distance-weighted mean of neighbour scores;
  - the highest neighbour score within 75 nt;
  - the neighbours' read counts.

  This is corroboration in its barest form: "do the sites near me look
  modified?"
- `gps`: a graph transformer (GraphGPS-style). `h2gcn_aux` plus attention
  over every site on the transcript, with a learned distance bias. It has
  the most capacity and the longest reach.

**Main table.** PR AUC gain over quantiles + LightGBM, mean ± SD over seeds 0-2
(n = 1 for controls). The matched difference is from `h2gcn_aux` on the same
seeds; positive = better. Seed SD is 0.001-0.005 for most designs (up to
0.011 for `h2gcn_aux_r100`), so 3-seed differences above ~0.01 are real.

| Model | Unseen cell line (train 1 → score 2) | data1 new genes | Both → cell line 2 | Both → cell line 1 | Matched diff: unseen | Matched diff: cell line 1 |
|---|---|---|---|---|---|---|
| `h2gcn_aux` | +0.0332 ± 0.0029 | +0.0260 ± 0.0040 | +0.0421 ± 0.0025 | +0.0763 ± 0.0052 | - | - |
| `h2gcn_aux_r100` | +0.0372 ± 0.0048 | +0.0445 ± 0.0086 | +0.0425 ± 0.0069 | +0.0873 ± 0.0112 | +0.0040 | +0.0111 |
| `fk_band` | +0.0447 ± 0.0011 | +0.0530 ± 0.0047 | +0.0454 ± 0.0039 | +0.0908 ± 0.0041 | +0.0115 | +0.0145 |
| `fk_band_shuffled` | +0.0408 | +0.0550 | +0.0436 | +0.0918 | +0.0081 | +0.0107 |
| `fk_kernel` | +0.0430 ± 0.0035 | +0.0450 ± 0.0032 | +0.0519 ± 0.0051 | +0.0956 ± 0.0076 | +0.0097 | +0.0194 |
| **`res_gate`** | **+0.0505 ± 0.0011** | +0.0530 ± 0.0071 | +0.0534 ± 0.0050 | **+0.1017 ± 0.0074** | +0.0173 | +0.0255 |
| `res_nogate` | +0.0479 | +0.0453 | +0.0585 | +0.1076 | +0.0152 | +0.0265 |
| `res_nodrop` | +0.0393 | +0.0404 | +0.0497 | +0.1022 | +0.0066 | +0.0211 |
| **`scalar_msg`** | **+0.0550 ± 0.0020** | +0.0518 ± 0.0046 | **+0.0542 ± 0.0015** | +0.0752 ± 0.0003 | **+0.0218** | -0.0011 |
| `scalar_random` | +0.0147 | +0.0164 | +0.0278 | +0.0570 | -0.0180 | -0.0242 |
| `gps` | +0.0281 ± 0.0028 | +0.0207 ± 0.0046 | +0.0383 ± 0.0049 | +0.1001 ± 0.0028 | -0.0051 | +0.0238 |

Source: analysis/representation/results/day_summary.md (from
`summarize_day.py`), per-run `results/<model>_nn__xsrc[__seedN].json`.

**Findings.**

1. **Constraining how neighbours are used transfers better than a free
   neighbour mean.** Every constrained design beats `h2gcn_aux` on the
   unseen cell line: by +0.010 to +0.022, against a seed SD of about
   0.003. The worst column (the 0032 rule) improves from +0.026
   (`h2gcn_aux`) to +0.045 (`fk_kernel`), +0.045 (`fk_band`), +0.050
   (`res_gate`) and +0.052 (`scalar_msg`).
2. **`res_gate` is the best all-rounder.** It is top or near-top in every
   column: +0.051 on the unseen cell line, +0.102 on cell line 1.
3. **`scalar_msg` transfers best but adds nothing on the training cell
   line.** It gains +0.022 over `h2gcn_aux` on the unseen cell line and
   -0.001 on cell line 1. When neighbours can pass only "how modified do I
   look", the model learns what transfers and nothing that does not. It
   also has the smallest seed spread (SD 0.0003-0.0046).
4. **Corroboration, tested directly.** `scalar_random` feeds the same model
   scores from unrelated sites, and the gain collapses: unseen cell line
   +0.055 → +0.015 (-0.040). The neighbours' own-reads scores are the
   information. This complements section 8 (gain ~0 within
   neighbour-status groups) with an intervention.
5. **The measured distance profile is not what matters.** Reversing the
   band weights (`fk_band_shuffled`) keeps most of the gain (+0.041 vs
   +0.045 unseen; n = 1). The gain comes from reaching 150 nt with
   distance-separated averages, not from the specific weights.
   - A smooth kernel (`fk_kernel`) and fixed bands (`fk_band`) perform
     about the same.
   - Both beat a hard 100-nt radius (`h2gcn_aux_r100`, +0.004).
6. **Neighbour dropout is the active ingredient of `res_gate`; the gate
   adds little.** Unseen cell line, n = 1 for each control (*confirmed on
   3 seeds in 12F: +0.0403 without dropout, +0.0492 without the gate*):
   - without dropout (`res_nodrop`): +0.051 → +0.039;
   - without the depth gate (`res_nogate`): +0.048.

   Dropping neighbours in training stops the model leaning on any single
   one (DropEdge / DropNode, 12B). That also makes it more robust where
   neighbours are missing (section 10, partial transcripts).
7. **More capacity does not transfer (the negative result).** `gps` gains
   +0.024 over `h2gcn_aux` on cell line 1, the second-best gain there.
   On the unseen cell line it loses 0.005 (+0.028), and on data1 new genes
   it is the worst design (+0.021). This is the outcome predicted from
   sections 5 and 8: extra expressiveness and range buy cell-line-specific
   context. A stronger, modern architecture is not the route to better
   cross-cell-line prediction.

**Why the graph transformer failed: what the evidence says so far.**
`gps` is the fifth time a more flexible or longer-range use of context has
helped cell line 1 and not cell line 2:

| Model | What it adds | Cell line 1 | Unseen cell line | Source |
|---|---|---|---|---|
| GAT | Learned attention over neighbours | +0.044 | ~0 | section 3 |
| `h2gcn_transcript` | Transcript-wide mean | gain | +0.002 over deepset | section 3 |
| `everything` (LightGBM) | Hand-made neighbour features | +0.065 | +0.016 | section 3 |
| Radius 400 nt | Wider neighbourhood | +0.112 | +0.032 | section 5 |
| `gps` | Attention over the whole transcript | +0.100 | +0.028 | 12C |

A useful single number is the **cell-line gap**: for the model trained on
both cell lines, its gain on cell line 1 minus its gain on cell line 2. It
measures how much of what a model learns is specific to one cell line.

| Model | Cell-line gap |
|---|---|
| `scalar_msg` | 0.021 |
| `h2gcn_aux` | 0.034 |
| `fk_kernel` | 0.044 |
| `fk_band` | 0.045 |
| `h2gcn_aux_r100` | 0.045 |
| `res_gate` | 0.048 |
| `gps` | **0.062** (largest) |

The radius test shows the same rise with reach: 0.017 at 50 nt, 0.029 at
100-200 nt, 0.045 at 400 nt.

The working explanation:
- **Local co-modification is shared** between cell lines (about 2.2× below
  25 nt in both, section 5A).
- **Transcript-level enrichment is not shared**: 3.3× in cell line 1 vs
  2.0× in cell line 2. *Which* transcripts are heavily modified depends on
  the cell type.
- **Attention over the whole transcript lets `gps` learn cell line 1's
  transcript-level pattern**, which scores well on cell line 1 and misleads
  on cell line 2.
- This is a failure of what the model is free to learn, not of
  optimisation: all 30 `gps` networks converged, and its cell line 1 gain is
  the second-highest of any design.

**Not yet tested directly.** The explanation fits every row above, but no
experiment has isolated it. Three checks would need the `gps` weights on
Ronin, scoring only, with no retraining:
1. switch the global attention off at scoring time. Prediction: cell line 1
   falls, cell line 2 holds or rises;
2. measure how much attention goes beyond 150 nt;
3. give each site another transcript's distant context. Prediction: this
   hurts cell line 1 more than cell line 2.

Related ML literature, not checked against the papers:
- tuned message-passing networks match graph transformers on long-range
  benchmarks (Tönshoff et al., "Where Did the Gap Go?", 2023);
- out-of-distribution graph benchmarks find more expressive models do not
  generalise better (GOOD, Gui et al., NeurIPS 2022).

**Confound 1: neighbour count and transcript size**
(`gain_decomposition.py`). Gain is `h2gcn_aux` over `deepset` (the same
read encoder, no graph). The first table is in PR AUC with 95% gene-bootstrap
intervals, trained on cell line 1 only.

| Neighbours within 50 nt | Sites (cell line 1) | Gain, cell line 1 | Gain, cell line 2 |
|---|---|---|---|
| 0 | 20,742 | -0.005 (-0.026, 0.014) | +0.004 (-0.010, 0.018) |
| 1 | 35,764 | +0.010 (-0.014, 0.033) | +0.015 (-0.001, 0.029) |
| 2-3 | 52,909 | +0.032 (0.014, 0.049) | +0.019 (0.006, 0.031) |
| 4+ | 12,423 | +0.060 (0.026, 0.096) | +0.017 (-0.006, 0.042) |

Trained on both cell lines, the same four rows on cell line 1 are -0.005 /
+0.016 / +0.045 / +0.075; on cell line 2, +0.003 / +0.020 / +0.025 / +0.045.

Sites per transcript, in quartiles (trained on cell line 1):

| | Smallest | 2nd | 3rd | Largest |
|---|---|---|---|---|
| Cell line 1 | +0.007 | +0.026 | +0.036 | +0.011 |
| Cell line 2 | -0.003 | +0.022 | +0.025 | +0.023 |

- The gain rises with the number of neighbours. It is steeper on cell
  line 1, which fits the cell-line-specific part of wider context.
- The gain does **not** rise steadily with transcript size, a proxy for
  expression and coverage: the largest quartile gains little on cell
  line 1.
- So "more neighbours helps" is not simply "better-covered transcripts
  help". This supports corroboration over a coverage artefact. It does not
  fully separate the two, since neighbour count itself depends on coverage.
- ROC AUC gains follow the same pattern and are smaller.

Source: results/gain_decomposition.csv.

**Confound 2: very close neighbours** (`band_ablation.py`, `h2gcn_aux_r400`
seed 0, scored without retraining). Values are PR AUC on cell line 1 / cell
line 2, trained on cell line 1.

| Neighbours used | Cell line 1 | Cell line 2 |
|---|---|---|
| all (within 400 nt) | 0.5840 | 0.4042 |
| none | 0.3464 | 0.2792 |
| only 0-10 nt | 0.3927 | 0.3193 |
| only 10-20 nt | 0.4085 | 0.3240 |
| drop 0-10 nt | 0.5839 | 0.4032 |
| drop 10-20 nt | 0.5835 | 0.4039 |

Removing all neighbours within 0-10 or 10-20 nt costs at most 0.0010 PR
AUC (0.0005 on cell line 1). The nanopore signal overlap DeepRM describes
(±10 nt) is therefore not what the graph exploits; the gain comes from
genuinely separate sites. The pattern is the same when trained on both cell
lines.

Source: results/band_ablation.csv. The coarser earlier bands are in
results/band_ablation_v1.csv.

### 12D. Do the gains hold across genes? And can the models tell the cell lines apart? (9 Oct)

`analysis/representation/day_significance.py` runs locally from the cached
out-of-fold predictions and needs no torch.
- Each model's three seeds are rank-averaged.
- The difference from `h2gcn_aux` (also seed-averaged) is computed on
  identical sites.
- The 95% interval and the win rate come from 1,000 resamples of whole
  genes.

Source: results/day_significance.csv.

**Paired difference from `h2gcn_aux`, PR AUC [95% gene interval] (gene
resamples won):**

| Model | Unseen cell line | New genes | Both → cell line 2 | Both → cell line 1 |
|---|---|---|---|---|
| `res_gate` | **+0.017 [+0.011, +0.023]** 100% | +0.028 [+0.012, +0.042] 100% | +0.009 [+0.003, +0.016] 100% | +0.019 [+0.010, +0.028] 100% |
| `scalar_msg` | **+0.020 [+0.013, +0.028]** 100% | +0.024 [+0.007, +0.041] 100% | +0.010 [+0.002, +0.017] 99% | -0.007 [-0.017, +0.004] 10% |
| `res_gate` + `scalar_msg` (rank average) | **+0.024 [+0.018, +0.029]** 100% | +0.029 [+0.012, +0.045] 100% | +0.016 [+0.009, +0.022] 100% | +0.019 [+0.010, +0.027] 100% |
| `h2gcn_aux_r100` | +0.005 [-0.002, +0.011] 92% | +0.022 [+0.004, +0.039] 99% | -0.000 [-0.008, +0.006] 44% | +0.009 [-0.001, +0.018] 96% |
| `gps` | -0.003 [-0.011, +0.005] 26% | -0.003 [-0.018, +0.012] 34% | -0.002 [-0.009, +0.005] 24% | **+0.025 [+0.015, +0.035]** 100% |
| `scalar_random` (seed 0) | -0.018 [-0.026, -0.010] 0% | -0.013 [-0.032, +0.007] 13% | -0.016 [-0.024, -0.008] 0% | -0.024 [-0.036, -0.013] 0% |

- **`res_gate` beats `h2gcn_aux` everywhere.** Every interval excludes
  zero.
- **`scalar_msg` beats it only on cell line 2.** On cell line 1 it is a
  tie.
- **Their ensemble is the best on the unseen cell line:** +0.024
  [+0.018, +0.029].
- **`gps` is significantly better *only* on the training cell line,**
  which confirms finding 7.
- **`scalar_random` is significantly worse.** That is the corroboration
  test, now with intervals.

**Against the shipped ensemble (seed 0 only; superseded by the 3-seed comparison in 12F).** The shipped model is
`h2gcn_twohead_aux` + `h2gcn_aux`; only seed 0 of `h2gcn_twohead_aux` is
cached, so this is a stand-in. Both-cell-lines arm. Numbers are in
results/day_significance.csv (rows with vs = "shipped ensemble"):

- `res_gate` alone **loses** on cell line 1: -0.0135 [-0.022, -0.006].
  On cell line 2 it ties: -0.002.
- `res_gate` + `scalar_msg` gains on cell line 2 (+0.007, interval above
  zero) but loses on cell line 1 (-0.013). The worse-gain rule (0032)
  rejects it.
- **`h2gcn_twohead_aux` + `res_gate` + `scalar_msg`** gains on both:
  - cell line 2: +0.0074 [+0.003, +0.012], 100% of gene resamples;
  - cell line 1: +0.0044 [-0.002, +0.010], 92% of gene resamples.

  Its worse gain is +0.004, positive but inside the ~0.005 tie band. It is
  the only candidate so far that does not lose on either cell line.
- Adding all four models (shipped + both new ones) gives +0.008 / +0.003.

The shipped model's strength on cell line 1 comes from `h2gcn_twohead_aux`,
which has a separate head per cell line and the transcript channel. The new
designs add what transfers. Confirming this needs seeds 1-2 and an export
path for the new designs (not run).

**Discordant sites: can a model tell which cell line carries the
modification?** 67,320 sites appear in both files. Their labels:

| | Count |
|---|---|
| Modified in both | 2,136 |
| Modified in cell line 1 only | 1,155 |
| Modified in cell line 2 only | 2,717 |
| Unmodified in both | 61,312 |

Of the 6,008 shared sites modified in either cell line, **64% are modified
in only one** (labels agree on 36%, by Jaccard overlap).

For each discordant site, the test compares the model's score in cell line
2's reads with its score in cell line 1's reads, as percentile ranks within
each file. The models were trained on cell line 1 labels only.

| Model (seed 0) | Which-line AUC | Rank correlation between the two files, shared sites |
|---|---|---|
| quantiles + LightGBM | 0.52 | 0.72 |
| deepset (reads only) | 0.50 | 0.88 |
| `h2gcn_aux` | 0.48 | 0.91 |
| `res_gate` | 0.48 | 0.91 |
| `scalar_msg` | 0.48 | 0.91 |
| `gps` | 0.45 | 0.93 |

- **No model can tell which cell line carries the modification** at sites
  where the labels disagree. The which-line AUC is about 0.5 throughout.
- A site gets nearly the same rank in both cell lines' reads (correlation
  0.88-0.93 for the networks). That includes deepset, which sees only the
  site's own reads.
- The scores shift by the same small amount at discordant and concordant
  sites (-0.02 to -0.03 in rank).

So at the 3,872 sites where the two labellings disagree, the reads give
the networks no detectable cell-line difference. There are two readings,
and this test cannot separate them:
1. **Labels:** much of the disagreement is labelling (m6ACE-seq thresholds,
   coverage, antibody background) rather than a difference in the
   molecules.
2. **Sensitivity:** the molecules do differ, by modification fraction, but
   the signal is below what these models resolve.

Either way, there is a **ceiling on cross-cell-line accuracy that no model
can pass using the reads alone.** It also explains why context features
that encode one cell line's labelling do not transfer.

A direct test would be to compare the raw read measurements of each
discordant site between the two files. Not run.

**The label oracle (10 Oct): a hard bound on transfer.** What if a model
had learned cell line 1's labelling *perfectly*? On the 67,320 shared
sites, cell line 1's true labels were used as the score for cell line 2's
labels (`label_oracle.py`; results/label_oracle.csv):

| Score for cell line 2's labels (positive rate 7.2%) | PR AUC | ROC AUC |
|---|---|---|
| Cell line 1's true labels | **0.326** | 0.711 |
| Cell line 1's true labels, ties broken by `h2gcn_aux`'s score | 0.485 | 0.837 |
| `h2gcn_aux` trained on cell line 1 (seeds averaged) | **0.422** | 0.823 |

- **Perfect knowledge of cell line 1's labels scores only 0.33 on cell line
  2**, below our reads-based model (0.42).
- Even combined with our model's score, it reaches 0.49. Our model already
  gets 87% of that.
- The reverse direction: cell line 2's labels score 0.30 on cell line 1's
  labels (0.63 with tie-breaking), against 0.59 for our model.

So **no amount of learning cell line 1's labelling can close the
cross-cell-line gap.** The two labellings disagree too much. What our
model transfers is worth more than the labels themselves. This is the
strongest single statement that we are "doing what the data allows".

(The tie-broken row is not a strict upper bound: some other combination
could score higher. The "labels alone" row is exact.)

**The raw-signal check (10 Oct): the reads barely differ where the labels
do.** This is model-free (`discordant_signal.py`; results/discordant_signal.csv).

Method:
- Per site and per file: mean and spread of the 9 read measurements, centred
  on that file's mean for the site's 7-mer.
- A linear "modification direction" is fitted within one file, on sites
  *not* shared by both files.
- Each shared site is scored in both files' reads.

| Direction | Within-file AUC, cell line 1 | Within-file AUC, cell line 2 | Which-line AUC at the 3,872 discordant sites |
|---|---|---|---|
| Fitted on cell line 1 | 0.818 | 0.712 | **0.519** |
| Fitted on cell line 2 | 0.819 | 0.714 | **0.516** |
| None: centre-position current only | 0.301 (lower = modified) | 0.385 | **0.542** |

- **Within a file, the direction clearly separates modified sites**: AUC
  0.71-0.82.
- **Across files, at the sites where the labels disagree, it barely moves**:
  0.52-0.54, where 0.5 means no difference.
- So the molecules show at most a faint trace of the label difference. That
  agrees with every trained model (which-line AUC about 0.5, above).

The cross-cell-line label disagreement is therefore mostly not visible in
these reads. Two readings remain:
1. **Labels:** different labelling (m6ACE-seq thresholds, coverage,
   antibody background).
2. **Stoichiometry:** small differences in modification fraction below what
   averaged reads resolve.

Either way, the reads cannot recover it. This is the closest we can come to
confirming the ceiling from this data.

Caveat: the direction is linear, on site averages. A difference confined to
a few reads per site would need a read-level test.

Source: results/discordant_sites.csv.

### 12E. Inside the designs: scoring-only probes (9 Oct)

`analysis/representation/probe_day.py` uses the seed-0 fold models on every
held-out site, with no retraining.

**1. Why `gps` fails to transfer: confirmed directly.** It was rescored with
its whole-transcript attention cut to nearby sites. Its local channels were
untouched.

| `gps` trained on cell line 1, attention limited to | Cell line 1 PR AUC | Cell line 2 PR AUC |
|---|---|---|
| whole transcript (as trained) | 0.569 | 0.403 |
| 150 nt | 0.530 (-0.039) | 0.388 (-0.015) |
| 50 nt | 0.477 (-0.092) | 0.372 (-0.031) |
| reference: `h2gcn_aux` | 0.544 | 0.405 |

- **Cutting the long-range attention costs the training cell line 2.6-3×
  more than the unseen one.** The same holds when trained on both lines:
  cell line 1 drops by 0.048 / 0.100, cell line 2 by 0.024 / 0.053. (The
  network never saw cut attention in training, so both drops are inflated.
  The asymmetry is the evidence.)
- **Where the attention goes:** about half of `gps`'s attention lands beyond
  150 nt, and 31% beyond 400 nt. The site itself gets 9-21%, 0-50 nt 14-15%,
  and 50-150 nt 15-20%. That is the range where nearby sites no longer
  co-modify (section 5A), so `gps` is reading transcript-wide context, whose
  meaning differs by cell line (3.3× vs 2.0× transcript-level enrichment).
- Even with full attention, `gps` only matches `h2gcn_aux` on cell line 2
  (0.403 vs 0.405).

Source: results/probe_gps.csv, probe_gps_attention.csv.

**2. What `res_gate`'s gate learned.** The gate is
g = sigmoid(a·log reads + b·|own score| + c), across 5 folds × 3 seeds:

- **Trained on cell line 1 only:** a = -0.004 ± 0.017, b = -0.032 ± 0.023.
  The gate stayed at a constant ~0.50-0.52 and learned nothing. That fits
  `res_nogate` ≈ `res_gate`.
- **Trained on both cell lines:** a = -0.12 ± 0.02, b = -0.18 ± 0.03. It
  learned the intended direction: neighbours get more weight when a site has
  fewer reads or a less certain own score.
  - by read count, 20-30 → 100+ reads: g falls from 0.36 to 0.31;
  - by own-score certainty, lowest → highest fifth: g falls from 0.39 to
    0.26.
- The effect is modest because every training site has at least 20 reads.
  The gate has never seen the 1-3-read regime of SG-NEx.

Source: results/probe_gate_params.csv; per-site values in
.cache/representation/probe_gate.csv.gz (contains labels, so not in git).

**3. The corroboration curve (`scalar_msg`).** This is the correction, in
logits, added to a site's own score as one neighbour summary varies. The
other inputs stay at their real values, averaged over held-out sites that
have neighbours (partial dependence). The x-axis is the summary's quantile.

| Neighbour summary, quantile | 2% | 14% | 26% | 50% | 74% | 86% | 98% |
|---|---|---|---|---|---|---|---|
| Distance-weighted mean neighbour score, trained on cell line 1 | -2.71 | -1.90 | -1.55 | -1.00 | -0.37 | +0.18 | +1.83 |
| The same, trained on both | -0.73 | -0.46 | -0.34 | -0.14 | +0.11 | +0.38 | +1.26 |
| Best neighbour score within 75 nt, trained on cell line 1 | -0.97 | -1.14 | -1.17 | -1.14 | -0.93 | -0.44 | +0.57 |
| The same, trained on both | +0.12 | -0.07 | -0.15 | -0.21 | -0.16 | +0.11 | +0.91 |

- **Corroboration is monotone.** The more modified the neighbours look, the
  higher the site's score, over a range of 3-4.5 logits. Most of the work
  is *lowering* sites whose neighbours look unmodified, which is most sites.
- **One clearly modified close neighbour matters only at the top.** The
  best-neighbour curve is flat until about the 75th percentile, then rises.
- **Size of the effect:** +1.8 logits turns a site at probability 0.20 into
  0.61.
- **The model trained on both lines uses a gentler curve.** That fits
  corroboration being partly cell-line-specific in strength.
- This is a figure for the report and for the "why this score" view of the
  Task 2 platform.

Source: results/probe_scalar_curve.csv (25 points per curve). The inputs
are correlated, so a partial-dependence curve illustrates the effect; it is
not a controlled experiment.

### 12F. 9-10 Oct batch: controls on 3 seeds, scalar messages with dropout, and the premise test

All 135 new networks converged (convergence.py: 0 undertrained). Source:
results/day_summary.md (regenerated) and batch2 logs.

**The controls now have 3 seeds, and every single-seed claim in 12C
holds.** Values are gains over LightGBM, mean ± SD over 3 seeds, on the
unseen cell line:

| Pair | Design | Control | Difference | What it shows |
|---|---|---|---|---|
| `res_gate` vs `res_nodrop` | +0.0505 ± 0.0011 | +0.0403 ± 0.0017 | **+0.010** | Neighbour dropout is the active ingredient (finding 6) |
| `res_gate` vs `res_nogate` | +0.0505 ± 0.0011 | +0.0492 ± 0.0014 | +0.001 | The depth gate adds nothing. Without it, trained on both lines, the model even scores slightly higher on cell line 2 (+0.057 vs +0.053) |
| `fk_band` vs `fk_band_shuffled` | +0.0447 ± 0.0011 | +0.0432 ± 0.0021 | +0.0015 | The measured band weights don't matter (finding 5) |
| `scalar_msg` vs `scalar_random` | +0.0550 ± 0.0020 | +0.0171 ± 0.0034 | **-0.038** | Corroboration: neighbours' real scores are the signal (finding 4) |

**`scalar_drop` (`scalar_msg` + neighbour dropout) fixes `scalar_msg`'s
weak spot.**

| | Unseen cell line | New genes | Both → cell line 2 | Both → cell line 1 |
|---|---|---|---|---|
| `scalar_msg` | +0.0550 | +0.0518 | +0.0542 | +0.0752 |
| `scalar_drop` | +0.0537 ± 0.0016 | +0.0502 | **+0.0585** (best of any design) | +0.0875 (+0.012) |

- With dropout, the one-number-message model keeps its transfer and
  recovers 0.012 on cell line 1.
- Dropout helps both constrained designs.

**`h2gcn_twohead_aux` on 3 seeds.** Both → cell line 1 +0.1120 ± 0.0035
(the highest of any model), both → cell line 2 +0.0462 ± 0.0020.

**The premise test: does training on cell line 2 itself help on cell
line 2?** (`data1_arm_eval.py`, seed 0.) `h2gcn_aux` and `gps` were trained
on cell line 2 only and scored on cell line 2's held-out genes, the same
sites as every other row. Source: results/data1_arm.csv.

| PR AUC on cell line 2, trained on | `h2gcn_aux` | `gps` |
|---|---|---|
| cell line 1 only | 0.405 | 0.403 |
| cell line 2 only | 0.412 | 0.422 |
| both | 0.436 | 0.436 |

| Paired difference, 95% gene interval (win %) | Value |
|---|---|
| `gps` - `h2gcn_aux`, both trained on cell line 1 only | -0.001 [-0.010, +0.008] (39%) |
| **`gps` - `h2gcn_aux`, both trained on cell line 2 only** | **+0.010 [+0.000, +0.020] (98%)** |
| `gps` - `h2gcn_aux`, both trained on both | +0.000 [-0.008, +0.010] (51%) |
| `h2gcn_aux`: trained on cell line 2 - trained on cell line 1 | +0.007 [-0.002, +0.017] (93%) |
| **`gps`: trained on cell line 2 - trained on cell line 1** | **+0.019 [+0.007, +0.030] (100%)** |

1. **Capacity does help within a cell line.** Trained and tested on cell
   line 2, `gps` beats `h2gcn_aux` (+0.010, interval just above zero), just
   as it does on cell line 1 (+0.025, 12D).
   - It loses that advantage only when the training and test cell lines
     differ.
   - So `gps`'s cross-line failure is the **labelling differing between
     cell lines** (the premise breaking), not a lack of generalisation in
     general.
   - Report framing: extra capacity learns more of the *training cell
     line's* labelling, which is real within that line and does not carry
     over.
2. **The constrained model barely needs the target cell line's labels.**
   Its own labels give `h2gcn_aux` only +0.007 (not significant), against
   +0.019 for `gps`. What `h2gcn_aux` learns from cell line 1 is almost
   all transferable, and that is what constraining buys.
3. **Most of the drop from cell line 1 to cell line 2 is not a transfer
   failure.**
   - `h2gcn_aux` scores 0.544 on cell line 1 and 0.405 on cell line 2.
   - Training on cell line 2 itself recovers only 0.007 of that gap.
   - Cell line 2's labels are simply harder to predict from these reads
     (lift over its positive rate: 0.41 / 0.073 = 5.6× vs 0.544 / 0.045 =
     12× on cell line 1). This fits the discordant-site ceiling in 12D.
4. **Training on both lines is best on cell line 2 for both models**
   (0.436). More, varied training data beats the target line's labels
   alone.

Caveats:
- Cell line 2 alone has fewer training sites: about 73k, against 122k for
  cell line 1.
- Seed 0 only; the `gps` interval within cell line 2 just clears zero.

**Ensembles against the shipped one, every component averaged over 3
seeds.** Both-cell-lines arm, paired, 95% gene interval (win %). Source:
results/day_significance.csv (vs = "shipped ensemble"); the new-genes
column is from the same function.

| Candidate | Cell line 1 | Cell line 2 | data1 new genes |
|---|---|---|---|
| `res_gate` alone | -0.019 [-0.028, -0.010] 0% | -0.001 | - |
| `scalar_drop` alone | -0.038 0% | +0.001 | - |
| `res_gate` + `scalar_msg` | -0.019 0% | +0.006 98% | - |
| `h2gcn_twohead_aux` + `res_gate` | +0.004 [-0.002, +0.009] 92% | -0.000 | - |
| `h2gcn_twohead_aux` + `res_gate` + `scalar_msg` | -0.001 [-0.006, +0.005] 35% | +0.006 [+0.002, +0.009] 100% | +0.012 [+0.000, +0.022] 98% |
| **`h2gcn_twohead_aux` + `res_gate` + `scalar_drop`** | **+0.001 [-0.005, +0.006] 61%** | **+0.006 [+0.002, +0.009] 100%** | **+0.012 [+0.000, +0.023] 98%** |
| shipped + `res_gate` + `scalar_msg` | -0.003 8% | +0.006 100% | - |

- **The seed-0 result in 12D was optimistic.** With every component
  seed-averaged, the shipped ensemble is stronger on cell line 1. The
  candidate's +0.004 there became +0.001, a tie.
- **What holds up:** about +0.006 on cell line 2 and +0.012 on its new
  genes, at no cost on cell line 1.
- On the two other HCT116 sequencing runs (13), the `scalar_msg` version
  also beat the shipped one: +0.009 / +0.012 (seed 0).

**Implications.**

- **What to ship (updated 12F, 3 seeds).** The best candidate is
  `h2gcn_twohead_aux` + `res_gate` + `scalar_drop`. Against the shipped
  ensemble it:
  - ties on cell line 1 (+0.001);
  - gains +0.006 on cell line 2 (interval above zero);
  - gains +0.012 on data1's new genes.

  That is a small but consistent improvement on the cell line we didn't
  train on. Before it ships: a final fit, a numpy export of `res_gate` and
  `scalar_drop` (new inference code: shared infrastructure, needs a
  decision record) and a predict.py test. None of these have been run.
- **A labelling ceiling (12D).** At sites where the two cell lines' labels
  disagree, no model sees a difference in the reads. Cross-cell-line PR AUC
  is bounded by label agreement, not only by the model.
- For the report, a paper-ready story:
  - corroboration (sections 8 and 12C finding 4);
  - limited to ~100-150 nt to transfer (sections 5 and 12C finding 5);
  - best exploited by constraining the model (findings 1-3, 6), not by
    adding capacity (finding 7).

---

## 13. Where the data comes from, and public data beyond the course

**dataset0 is SG-NEx HCT116 direct RNA, replicate 3 run 1, exactly.**
`analysis/newdata/sgnex_source_match.py` compared every course site, with
its read count, against the `data.readcount` file of all 22
m6Anet-processed runs in the public SG-NEx bucket (`s3://sg-nex-data`, no
credentials). For HCT116 replicate 3 run 1:
- 100% of dataset0's sites are present;
- 100% have an identical read count.

Every other run has the sites (94-100% present) but only 0.1-3.5% identical
read counts. Labels: HCT116 m6ACE-seq (docs/project-requirements.md).

**data1 does not come from any processed SG-NEx run.**
- No run matches more than 1.4% of its read counts.
- That rules out A549, H9, HEYA8, HCT116, HepG2, K562 and MCF7 as processed
  in the bucket.
- The course says only that it is a different cell line. A natural guess
  is HEK293T: it is the other line with public m6ACE-seq labels, and
  m6Anet's own test line. This is **unconfirmed**; course staff could be
  asked.

**What m6Anet trained and tested on** (Hendra et al., Nat Methods 2022,
doi:10.1038/s41592-022-01666-1; full text checked 9 Oct):

| | m6Anet |
|---|---|
| Training reads | SG-NEx HCT116 direct RNA, replicate 2 run 1 (ENA PRJEB44348) |
| Training labels | HCT116 m6ACE-seq; every m6ACE-seq site counts as modified; other positions with the same 5-mer are unmodified |
| Site rule | DRACH 5-mers only (66 of them); 20 reads sampled per site; modified sites oversampled to balance classes |
| Split | 5-fold by gene (75/25); cross-cell-line tests on 500 genes shared by both lines |
| Test 1: HEK293T (ENA PRJEB40872, includes METTL3 knock-out) | m6ACE-seq + miCLIP labels; ROC AUC 0.83, PR AUC 0.35 (all 18 DRACH motifs) |
| Test 2: synthetic "curlcake" RNA (GEO GSE124309) | Read-level ROC AUC 0.90, PR AUC 0.91; site-level ROC AUC > 0.98 at ≥ 50% modified |
| Test 3: Arabidopsis (VIR mutant, ENA PRJEB32782) | Trained and tested across species; no numbers in the text |

Consequences for our report:
- Pretrained m6Anet's dataset0 score (0.503) is not a held-out number. Its
  data1 score (0.356 PR AUC, 0.819 ROC AUC) is the fair comparison, and it
  closely matches the paper's own HEK293T result (0.35 / 0.83).
- **Our model beats m6Anet on m6Anet's own home ground (dataset0) and on
  an unseen cell line (data1).**

**Public data for testing generalisation beyond the course data:**

1. **Other SG-NEx cell lines, already processed (easy):** A549, H9, HEYA8,
   HepG2, K562 and MCF7, in exactly our input format.
   - They have no single-base m6A labels.
   - We can still test whether predictions behave biologically in every
     line: the post-stop-codon peak, exclusion within ~100 nt of junctions,
     and agreement between replicates.
   - Overlaps with Task 2.
2. **Other HCT116 runs (easy, labelled):** replicate 3 run 4 and replicate
   4 run 3 are the same cell line, so dataset0's m6ACE-seq labels apply.
   - Scoring them with the fold models on held-out genes gives "new
     sequencing run, same biology". That is probably what an HCT116 test
     file looks like.
   - It also tests real depth variation rather than our subsampling.
3. **HEK293T (hard, best-labelled line):**
   - Labels: m6ACE-seq, miCLIP, miCLIP2, m6A-SAC-seq and GLORI (absolute
     single-base levels; Liu et al., Nat Biotech 2023). GLORI is
     reportedly among the most accurate assays.
   - Direct RNA reads: ENA PRJEB40872, wild type and METTL3 knock-out.
   - Not processed in SG-NEx: it needs basecalling, alignment, nanopolish
     eventalign and m6Anet dataprep from raw signal. Roughly a day of Ronin
     compute plus setup.
   - The METTL3 knock-out gives a label-free test: scores at the same sites
     should collapse when the enzyme that writes m6A is gone.
   - **Done (9 Oct), see "Other HCT116 runs" below.**
4. **Synthetic curlcake RNA** (GEO GSE124309): fully known modification
   status, like data2. Also needs processing from raw signal.

**Other HCT116 runs: a new sequencing run of the same biology (9 Oct).**
`analysis/representation/score_hct116_runs.py` scored two more SG-NEx HCT116
direct-RNA runs from the public bucket:
- replicate 3 run 4: the same RNA sample as dataset0, sequenced again;
- replicate 4 run 3: a separately grown batch of cells.

Setup:
- Every one of dataset0's 121,838 labelled sites is present in both runs,
  with dataset0's m6ACE-seq labels.
- Each site was scored by the seed-0 fold model that held out its gene,
  using only that run's reads.
- Comparison: the same models' out-of-fold scores on dataset0's own run, on
  identical sites.
- Median depth is 42 reads in replicate 3 run 4. **Every site has at least
  20 reads in both runs**, so this tests run-to-run variation, *not* low
  depth.

PR AUC (identical sites):

| Model (trained on) | Replicate 3 run 4 | Replicate 4 run 3 | dataset0's own run |
|---|---|---|---|
| `h2gcn_aux` (cell line 1) | 0.523 | 0.533 | 0.544 |
| `res_gate` (cell line 1) | 0.562 | 0.567 | 0.584 |
| `scalar_msg` (cell line 1) | 0.539 | 0.549 | 0.560 |
| `gps` (cell line 1) | 0.554 | 0.551 | 0.569 |
| `h2gcn_twohead_aux` (both) | 0.561 | 0.560 | 0.586 |
| shipped ensemble (both, seed 0 stand-in) | 0.575 | 0.578 | 0.595 |
| **`h2gcn_twohead_aux` + `res_gate` + `scalar_msg` (both)** | **0.584** | **0.590** | **0.599** |

- **A new run of the same cell line costs 0.01-0.025 PR AUC**, about the
  size of the seed spread and far smaller than the cross-cell-line drop.
  Run-to-run noise is a minor issue.
- **The two runs agree.** The separately grown cells (replicate 4) score no
  worse than the re-sequenced sample, so biological replicate variation is
  negligible here.
- **The ranking holds on new runs.** The three-model ensemble is best on
  both, ahead of the shipped one by +0.009 / +0.012. Within cell line 1,
  `res_gate` beats `gps`, which beats `h2gcn_aux`, as in cross-validation.
- **What this means for the leaderboard:** if the test file is another
  HCT116 run, expect roughly our cross-validated number minus 0.01-0.02. A
  new cell line costs far more.

Source: analysis/representation/results/hct116_runs.csv (by depth band too).

Accessions are as given in the m6Anet paper. The specific claims about
GLORI and other assays come from search summaries and were not checked
against the papers.

---

## 14. Models the report discusses (agreed 10 Oct)

Each model stands for one machine-learning mechanism and teaches one thing
about learning m6A from nanopore reads. Gains are PR AUC over quantiles +
LightGBM, given as unseen cell line / cell line 1, from results/day_summary.md
and section 3. Figure versions are generated by `report_models.py`, see
below.

**In the figures (18):**

| # | Model | Mechanism | What it teaches |
|---|---|---|---|
| 1 | Logistic regression | Linear model on per-site mean/sd of the reads | The required simple baseline (configs/baseline.yaml) |
| 2 | Quantiles + LightGBM | Hand-made per-site read-distribution features, boosted trees | A strong classical baseline. The reference for every gain |
| 3 | m6Anet (pretrained) | Published multiple-instance learner over reads | The state of the art, trained on dataset0's own cell line and labels: a home advantage on cell line 1, a fair test on cell line 2 |
| 4 | DeepSet | A learned encoder per read, mean + spread pooling | Learning from raw reads beats hand-made summaries |
| 5 | Set Transformer | Attention between a site's reads | No gain: a site's reads are interchangeable molecules, so simple pooling suffices |
| 6 | GCN | A site averaged with its neighbours | Fails (-0.10 / -0.14): the site's own evidence is drowned (over-smoothing). Keep self and neighbours separate |
| 7 | GAT | Learned attention over neighbours | +0.044 on cell line 1, ~0 on cell line 2: attention learns the training line's clustering |
| 8 | H2GCN + own-reads head (`h2gcn_aux`) | Separate self / neighbour channels; an auxiliary loss on the site's own reads | Corroboration works: the base graph model, +0.033 / +0.076 |
| 9 | H2GCN, 400 nt (`h2gcn_aux_r400`) | Wider neighbourhood | Wider context helps only the training line (+0.032 / +0.112); ~100 nt transfers |
| 10 | Graph transformer (`gps`) | Attention over the whole transcript | Better *within* each cell line, not *across* (12E-12F): capacity learns that line's labelling |
| 11 | Two-head H2GCN (`h2gcn_twohead_aux`) | One output head per cell line's labelling | Models the label difference explicitly; strongest on cell line 1 (+0.112) |
| 12 | Scalar messages (`scalar_msg`) | Neighbours pass one number (their own-reads score) | The barest corroboration transfers best (+0.055 / +0.075) |
| 13 | ... control: random neighbours (`scalar_random`) | The same, with scores from unrelated sites | The gain collapses (+0.017): proves corroboration |
| 14 | Residual + dropout (`res_gate`) | Own-reads score + gated neighbour correction, with neighbour dropout | Constrained and robust; the best all-rounder (+0.051 / +0.102) |
| 15 | ... control: no dropout (`res_nodrop`) | Without dropout | Dropout is the active ingredient (-0.010) |
| 16 | Scalar messages + dropout (`scalar_drop`) | 12 + 14's dropout | Dropout also fixes the scalar design's cell line 1 weakness (+0.054 / +0.088) |
| 17 | Intermediate leaderboard ensemble | Two-head + `h2gcn_aux` (decision 0033) | What was submitted on 7 Oct |
| 18 | Final ensemble | Two-head + `res_gate` + `scalar_drop` (decision 0034) | Ties 17 on cell line 1, +0.006 on cell line 2, +0.012 on new genes |

**In the text, one line each:**

- **k-NN graph over a site's reads** (link each read to its 4 most similar
  reads): no better than the same graph shuffled across reads (-0.005 /
  -0.003). That is despite k = 4 being chosen on in-vitro data where such
  graphs were strongly homophilous. Reads at a site do not form
  informative sub-groups at these depths.
- **Attention MIL** (gated attention over reads): no better than mean +
  spread pooling, even trained to convergence. Same lesson as the Set
  Transformer.
- **Hand-made features inside the network** (`deepset_hand`): -0.021
  against DeepSet. Learned read encodings already contain them, and adding
  them hurts.
- **Quantile pooling of read encodings** (`h2gcn_aux_q`): -0.009.
  Mean + spread captures the read distribution; more summary statistics
  add capacity, not information.
- **A distance prior** (`fk_band`, `fk_kernel`): weighting neighbours by
  distance out to 150 nt helps (+0.010 to +0.012 over `h2gcn_aux`). The
  shuffled-weights control shows the measured profile itself does not
  matter.
- **The depth gate** (`res_nogate`): +0.001. It adds nothing; dropout does
  the work.

**Figures** (`analysis/representation/report_models.py`, no torch; two
versions each):
- `report/figures/models_distribution_{cl1,both}.png`: per-fold PR AUC gain
  over LightGBM, same fold and seed.
- `report/figures/models_scatter_{cl1,both}.png`: PR AUC and ROC AUC,
  cell line 1 against cell line 2.

`cl1` = trained on cell line 1 only (cell line 2 is unseen). `both` = trained
on both cell lines, the shipped setting. The two-head model and the
ensembles appear only in `both`. Numbers: results/report_models_pooled.csv,
report_models_folds.csv.

**The shipped ensemble at low read depth (10 Oct).** Reads (and each
neighbour's reads) were thinned at random to 1, 3 or 10 per site; the models
were trained at full depth, on both cell lines (held-out genes)
(`depth_rescore.py`, `depth_ensembles.py`; results/depth_ensembles.csv,
report/figures/fig_depth.png).

| PR AUC | 1 read | 3 reads | 10 reads | all (>= 20) |
|---|---|---|---|---|
| Final ensemble, cell line 1 | 0.333 | 0.451 | 0.534 | 0.607 |
| Intermediate ensemble, cell line 1 | 0.319 | 0.443 | 0.532 | 0.606 |
| LightGBM, cell line 1 | 0.140 | 0.221 | 0.368 | 0.472 |
| Final ensemble, cell line 2 | 0.279 | 0.341 | 0.397 | 0.460 |
| Intermediate ensemble, cell line 2 | 0.272 | 0.337 | 0.396 | 0.454 |
| LightGBM, cell line 2 | 0.136 | 0.200 | 0.306 | 0.392 |

- The networks keep roughly twice LightGBM's PR AUC at 1-3 reads, the
  depths SG-NEx mostly has.
- The final ensemble is slightly ahead of the intermediate one at every
  depth.
- Caveat: the depths below 20 are simulated. Real low-depth sites come from
  lowly expressed genes.

**Do dataset0's labels map onto the SG-NEx samples? (10 Oct)**
`analysis/sgnex/label_mapping_check.py`; analysis/sgnex/label_mapping_check.csv.

- **Where the labels come from:** m6ACE-seq on HCT116, a separate lab
  experiment. A label describes the cell line's RNA at a position, not one
  sequencing run.
- **Why each SG-NEx sample has different reads:** each sample reads
  different molecules of the same kind of cells.
- **The coordinates are proven to match.** At every one of dataset0's
  121,838 labelled sites, in all 22 SG-NEx samples, the 7-mer is identical
  to dataset0's: 100.0% everywhere. That includes MCF7 replicate 3 after
  its version suffix is removed. A one-nucleotide offset would change the
  7-mer.
- The labelled sites are present in 100% of each HCT116 sample (82-99% in
  other lines).
- **Every labelled site has at least 20 reads in all three HCT116 samples**
  (median 42-76). The labelled set is well-covered sites.
  - Real low-depth accuracy **cannot** be measured with these labels; the
    simulated thinning (`fig_depth`) is the only low-depth evidence.
  - In other cell lines, many labelled positions have few reads (12-74%
    under 20), but HCT116's labels do not apply there.

**Related work: graph and attention models for m6A (checked 10 Oct).**

| Model | What it does | Status |
|---|---|---|
| m6Anet (Hendra et al., Nat Methods 2022) | Multiple-instance learning over a site's reads; every site scored independently | Verified |
| m6ATM (Brief Bioinform 2024, 25(6):bbae529) | Deep network over nanopore reads; its authors note RNA structure is not used | Verified |
| DeepRM (Kang et al., Nat Commun 2025) | Transformer over current signal and a 21-nt context; co-occurring m6A on single molecules | Verified (12B) |
| Xron (Genome Res 2024) | Signal-to-methylated-base encoder-decoder; context within a read only | Verified |
| structRFM (bioRxiv 2025) | Structure-guided RNA foundation model (sequence + secondary structure) | Verified (preprint) |
| NanoFM | Nanopore signal + structRFM embeddings + cross-attention | GitHub repo verified; it gives no paper, training data or results. Its claimed DOI (10.1016/j.ijbiomac.2026.152629) resolves but could not be read |
| m6A-IIN (Li et al., Commun Biol 8:1022, 2025; doi:10.1038/s42003-025-08265-8) | Sequence-only: a 41-nt fragment plus RNAfold secondary structure, invertible neural network; one fragment per prediction, no nanopore signal | **Verified** (corrected 10 Oct; first search missed it). The first AI summary's "graph wavelet" description was wrong |

**Prior-art search, round 2 (two deep-research reports, 10 Oct; key claims
re-checked by us).**

- **Our novelty claim must be narrow.** DNA methylation has clear
  precedent for using neighbouring sites:
  - **DeepMod** (Liu et al., Nat Commun 2019, doi:10.1038/s41467-019-10168-2;
    nanopore DNA 5mC). A second network takes a CpG's predicted
    methylation percentage plus its neighbouring sites' (both strands) and
    outputs a revised percentage, to exploit the "cluster effect" of CpG
    methylation. **Verified** from the repo docs (docs/Usage.md: "5mC in
    CpG motifs has cluster effect"; `hm_cluster_predict.py`) and a search
    summary of the paper. The improvement it reports (claimed +1-3% AP,
    +3-5% AUC) is **not** verified. This is essentially our scalar-messages
    idea in DNA, without the gating, residual design, dropout or
    cross-domain evaluation. **Cite it.**
  - **DeepCpG** (Genome Biol 2017), **CpG Transformer** (Bioinformatics
    2022) and GraphCpG (Bioinformatics 2023) impute single-cell CpG states
    from observed neighbouring states. ccsmeth and hifimeth (PacBio) use
    neighbouring CpGs' read-level calls. *(Reported by the research; not
    individually re-checked.)*
  - MeRIP-seq peak callers (HEPeak, MeTPeak, BaySeqPeak) use HMMs across
    neighbouring bins. That is coarse region smoothing, not sites.
    *(Reported; not re-checked.)*
- **RNA side:** no nanopore RNA-modification method was found that feeds
  measured evidence from other candidate sites into a site's prediction.
  - Both reports agree, covering about 15 tools: m6Anet, m6ATM, DeepRM,
    CHEUI, TandemMod, SingleMod, m6Aiso, MultiNano, RNANO, ORCA, Xron,
    MINES, Nanocompore and others.
  - Co-occurrence between sites appears only as post-hoc analysis (DeepRM,
    m6Aiso, CHEUI, ORCA, Nanocompore).
  - Nanocompore combines neighbouring k-mers, but that is one
    modification's own signal footprint, not other sites.
- **Defensible wording:** "To our knowledge, the first method to propagate
  read-derived evidence between candidate RNA modification sites along a
  transcript in nanopore direct RNA sequencing, and to characterise how such
  propagation behaves when the labels change between cell lines. Analogous
  neighbour-site models exist for DNA methylation (DeepMod; DeepCpG; CpG
  Transformer)."
- **Supporting results we checked:**
  - **Graph transformers out of distribution** (Niv & Rabin, arXiv:2506.20575,
    2025). GPS generalised better than message passing on 4 of 6 GOOD
    benchmarks, but "vGIN leads by roughly 2% on the size shift and 5% on
    the scaffold shift ... locality in message passing can be better suited
    for certain distribution shifts". So "capacity hurts out of
    distribution" is **not** a general law. Our result is evidence for
    *label-shift between cell lines* in this domain. Phrase it that way.
  - **Neighbour-reliant models and data splits** (single-cell methylation
    imputation benchmark, Brief Bioinform 2026, 27(4):bbag434). "CpG
    Transformer and MambaCpG showed strong sensitivity to splitting
    protocol, with severe drops in MCC under the chromosome-based split
    framework"; "random splitting inevitably places highly similar
    neighboring sites in both training and test sets". **Verified.** Our
    split is by gene, so a site and its neighbours (same transcript) are
    always on the same side of it. Worth a sentence in Methods.
  - **Low site-level agreement between assays:** NP-mFinder reports 28%
    site-level and 85% gene-level concordance with GLORI v2.0. Abstract
    verified via a mirror; the venue (Front Genet 2026) is not confirmed.
  - **m6Anet's own cross-cell-line claim** ("generalizes robustly to other
    cell lines without a loss in accuracy") was made with labels from a
    similar assay in both lines. Our setting has different labels per
    cell line, which is where we see the drop.
  - **Not found in either report:** a paper arguing that cross-cell-line
    accuracy is bounded by label disagreement. Our label oracle and
    discordant-site results appear to be new as stated, quantified results.
- **Errors in the first report (do not cite from it):**
  - EpiNano is given as "Science 2019"; it was Nat Commun 2019.
  - GraphGPS is attributed to "Dwivedi et al., ICLR 2022"; it is Rampášek et
    al., NeurIPS 2022.
  - SMART-m6A's year conflicts with its DOI.
  - It also says m6A-IIN is not nanopore-based; that part is correct.

  Every citation from either report must be checked before use.

- **None of the nanopore RNA methods above shares information between
  candidate sites on a transcript.** Every one found scores each site
  independently. The "graph" models build graphs over nucleotides within
  one molecule (structure), not over sites.
- Report wording: "to our knowledge, ..." (above).
- **Not pursued, with reasons:**
  - RNA structure: our folding test was null (section 7).
  - Expressive pair attention: our graph transformer learns the training
    line's labelling (12E-12F).
  - Foundation-model sequence embeddings: they need the reference
    transcript sequence (outside annotation, ruled out as a model input)
    and torch on the prediction path.
- **Future work this suggests:** DeepRM shows co-occurrence *on the same
  molecule*. The m6Anet data format discards read identities, so we
  corroborate per site rather than per molecule. Linking reads across
  sites would enable molecule-level corroboration.

---

## 15. Candidate entries for the AI-use table (recorded 10 Oct; choose later)

The brief asks for at least two instances where an AI tool's output was
wrong, misleading, or rested on an unverified assumption: how it was
detected, and what was done.

| # | What the AI said or did | How it was caught | What was done |
|---|---|---|---|
| 1 | Reported that co-modification "fades twice as fast in cell line 2" (decay scale 59 vs 113 nt), from point estimates | A robustness check: the transcript-bootstrap intervals overlap (38-123 vs 82-144 nt), and on shared sites the two labellings give 101 vs 103 nt | Retracted in findings 12A; reported as ~100 nt in both |
| 2 | A literature summary cited m6Aiso as "Guo et al. 2025, Mol Cell" | Checking each citation against the paper: the paper exists (doi:10.1016/j.molcel.2025.01.014), the first author could not be confirmed | Cited by title and DOI |
| 3 | Early project documents described data1 as "a second sequencing run, labelled differently" | Course staff, 5 Oct: different cell lines | Every analysis re-framed as cross-cell-line |
| 4 | AI-written parallel code for the RNA-folding test silently ran the default folding program (RNAplfold) when LinearPartition was requested: Python 3.14's process start method did not pass the setting to workers | The "LinearPartition" results were identical to RNAplfold's | Fixed with a worker initialiser; workers now report the predictor they used; rerun |
| 5 | Recommended the three-model ensemble on a seed-0 comparison (+0.004 on cell line 1) | Repeating with every component averaged over 3 seeds: +0.001, a tie | Claim restated as "ties on cell line 1, +0.006 on cell line 2" (12F, decision 0034) |
| 6 | Wrote a wrong number into findings (deep model -0.028; the result file says -0.017) | Re-reading the result file before quoting it | Corrected |
| 7 | (Agent action, not output) Launched Ronin jobs the user had not approved | The user noticed the extra jobs | Killed; an approval rule was adopted for every compute job |

---

## Appendix: full tables

Generated directly from the result files named under each table (no
hand-copying). Section numbers refer to the summaries above.


### A. Co-modification by distance band (section 5A)

Positive-positive site pairs on the same transcript, relative to the file's overall rate (total), split into the part explained by positive-rich transcripts (transcript-level) and the rest (local). Source: `analysis/newdata/distance_bands.py` -> `distance_bands/bands.csv`.


**dataset0 (cell line 1), all sites**

| band | pairs | total enrichment | transcript-level part | local part | local 95% low | local 95% high |
|---|---|---|---|---|---|---|
| 1-25 nt | 49209 | 7.447 | 3.383 | 2.201 | 2.041 | 2.376 |
| 25-50 nt | 56749 | 6.946 | 3.309 | 2.099 | 1.901 | 2.258 |
| 50-100 nt | 109632 | 6.143 | 3.271 | 1.878 | 1.757 | 1.987 |
| 100-200 nt | 205612 | 4.771 | 3.258 | 1.464 | 1.388 | 1.556 |
| 200-400 nt | 357940 | 3.539 | 3.178 | 1.114 | 1.051 | 1.178 |
| 400-800 nt | 528014 | 2.381 | 2.974 | 0.801 | 0.747 | 0.858 |
| 800-1600 nt | 535586 | 1.726 | 2.765 | 0.624 | 0.536 | 0.705 |

**data1 (cell line 2), all sites**

| band | pairs | total enrichment | transcript-level part | local part | local 95% low | local 95% high |
|---|---|---|---|---|---|---|
| 1-25 nt | 35186 | 4.534 | 2.127 | 2.132 | 1.964 | 2.293 |
| 25-50 nt | 40576 | 4.072 | 2.039 | 1.997 | 1.847 | 2.160 |
| 50-100 nt | 78864 | 3.536 | 2.044 | 1.730 | 1.627 | 1.825 |
| 100-200 nt | 145148 | 2.712 | 2.022 | 1.341 | 1.278 | 1.399 |
| 200-400 nt | 250421 | 1.956 | 1.923 | 1.017 | 0.970 | 1.061 |
| 400-800 nt | 358084 | 1.340 | 1.821 | 0.736 | 0.689 | 0.783 |
| 800-1600 nt | 335991 | 1.178 | 1.665 | 0.708 | 0.646 | 0.774 |

**shared sites, cell line 1 labels**

| band | pairs | total enrichment | transcript-level part | local part | local 95% low | local 95% high |
|---|---|---|---|---|---|---|
| 1-25 nt | 26511 | 6.961 | 3.058 | 2.276 | 2.044 | 2.492 |
| 25-50 nt | 30688 | 6.409 | 2.884 | 2.222 | 2.003 | 2.467 |
| 50-100 nt | 59260 | 5.621 | 2.949 | 1.906 | 1.757 | 2.029 |
| 100-200 nt | 108948 | 3.960 | 2.926 | 1.353 | 1.264 | 1.430 |
| 200-400 nt | 185689 | 2.810 | 2.852 | 0.985 | 0.915 | 1.047 |
| 400-800 nt | 259816 | 1.921 | 2.680 | 0.717 | 0.654 | 0.787 |
| 800-1600 nt | 232755 | 1.656 | 2.519 | 0.657 | 0.554 | 0.761 |

**shared sites, cell line 2 labels**

| band | pairs | total enrichment | transcript-level part | local part | local 95% low | local 95% high |
|---|---|---|---|---|---|---|
| 1-25 nt | 26511 | 4.428 | 2.153 | 2.056 | 1.866 | 2.211 |
| 25-50 nt | 30688 | 4.070 | 2.055 | 1.981 | 1.805 | 2.134 |
| 50-100 nt | 59260 | 3.478 | 2.053 | 1.694 | 1.588 | 1.793 |
| 100-200 nt | 108948 | 2.626 | 2.021 | 1.299 | 1.238 | 1.363 |
| 200-400 nt | 185689 | 1.871 | 1.917 | 0.976 | 0.920 | 1.025 |
| 400-800 nt | 259816 | 1.315 | 1.789 | 0.735 | 0.687 | 0.780 |
| 800-1600 nt | 232755 | 1.072 | 1.564 | 0.686 | 0.611 | 0.774 |

### B. Neighbours within each radius (section 5B)

Other candidate sites on the same transcript within the radius. Source: `distance_bands/neighbours.csv`.

| data | radius | no neighbours | 1 neighbour | 2+ neighbours | median | mean |
|---|---|---|---|---|---|---|
| dataset0 | 25 | 42% | 38% | 21% | 1.00 | 0.84 |
| dataset0 | 50 | 17% | 29% | 54% | 2.00 | 1.77 |
| dataset0 | 100 | 4% | 11% | 85% | 3.00 | 3.57 |
| dataset0 | 200 | 1% | 2% | 97% | 7.00 | 6.94 |
| dataset0 | 400 | 0% | 0% | 99% | 13.00 | 12.81 |
| data1 | 25 | 43% | 37% | 19% | 1.00 | 0.81 |
| data1 | 50 | 19% | 30% | 51% | 2.00 | 1.70 |
| data1 | 100 | 5% | 12% | 83% | 3.00 | 3.44 |
| data1 | 200 | 1% | 3% | 96% | 6.00 | 6.63 |
| data1 | 400 | 0% | 0% | 99% | 12.00 | 12.14 |

### C. What the trained 400-nt model uses (section 5C)

`h2gcn_aux_r400` (seed 0) rescored with neighbours restricted at prediction time: `all` as trained; `none`; `only a-b` keeps only neighbours a < d <= b nt away; `drop a-b` removes them. Out of fold, both training arms. Source: `analysis/representation/band_ablation.py` -> `results/band_ablation.csv`.

| trained on | neighbours | PR AUC on dataset0 | PR AUC on data1 |
|---|---|---|---|
| dataset0 | all | 0.5840 | 0.4042 |
| dataset0 | none | 0.3464 | 0.2792 |
| dataset0 | only 0-50 | 0.4965 | 0.3834 |
| dataset0 | only 50-100 | 0.4998 | 0.3703 |
| dataset0 | only 100-200 | 0.5392 | 0.3859 |
| dataset0 | only 200-400 | 0.5551 | 0.3875 |
| dataset0 | drop 0-50 | 0.5803 | 0.3998 |
| dataset0 | drop 50-100 | 0.5792 | 0.4023 |
| dataset0 | drop 100-200 | 0.5771 | 0.4007 |
| dataset0 | drop 200-400 | 0.5702 | 0.4048 |
| pooled_both | all | 0.5847 | 0.4244 |
| pooled_both | none | 0.4219 | 0.3439 |
| pooled_both | only 0-50 | 0.5087 | 0.4014 |
| pooled_both | only 50-100 | 0.5073 | 0.3909 |
| pooled_both | only 100-200 | 0.5386 | 0.4002 |
| pooled_both | only 200-400 | 0.5462 | 0.3974 |
| pooled_both | drop 0-50 | 0.5761 | 0.4159 |
| pooled_both | drop 50-100 | 0.5804 | 0.4208 |
| pooled_both | drop 100-200 | 0.5763 | 0.4213 |
| pooled_both | drop 200-400 | 0.5743 | 0.4252 |

### D. Position on the transcript (section 6)

Positive rate, lift relative to each cell line's average, and number of sites, per bin. Ensembl 91 transcript coordinates. Source: `analysis/newdata/annotation_context.py` -> `annotation_context/*.csv`.


**Distance from the modified A to the nearest exon-exon junction (nt)**

| bin | positive rate (cell line 2) | positive rate (cell line 1) | lift vs file rate (cell line 2) | lift vs file rate (cell line 1) | sites (cell line 2) | sites (cell line 1) |
|---|---|---|---|---|---|---|
| [0, 25) | 0.020 | 0.005 | 0.279 | 0.101 | 19569 | 24350 |
| [25, 50) | 0.030 | 0.006 | 0.411 | 0.129 | 16704 | 21045 |
| [50, 100) | 0.052 | 0.026 | 0.723 | 0.585 | 13779 | 17468 |
| [100, 200) | 0.157 | 0.117 | 2.166 | 2.595 | 8185 | 9948 |
| [200, 400) | 0.175 | 0.129 | 2.413 | 2.872 | 9116 | 11623 |
| [400, 1000000000) | 0.091 | 0.054 | 1.250 | 1.192 | 19110 | 33196 |

**Exon the site sits in**

| bin | positive rate (cell line 2) | positive rate (cell line 1) | lift vs file rate (cell line 2) | lift vs file rate (cell line 1) | sites (cell line 2) | sites (cell line 1) |
|---|---|---|---|---|---|---|
| first | 0.042 | 0.045 | 0.572 | 1.002 | 4237 | 4351 |
| internal | 0.030 | 0.011 | 0.414 | 0.254 | 42366 | 54120 |
| last | 0.120 | 0.073 | 1.654 | 1.624 | 39860 | 59159 |
| single | 0.191 | 0.223 | 2.626 | 4.961 | 1878 | 1552 |

**Length of that exon (nt)**

| bin | positive rate (cell line 2) | positive rate (cell line 1) | lift vs file rate (cell line 2) | lift vs file rate (cell line 1) | sites (cell line 2) | sites (cell line 1) |
|---|---|---|---|---|---|---|
| [0, 200) | 0.024 | 0.004 | 0.336 | 0.080 | 37374 | 46906 |
| [200, 400) | 0.063 | 0.034 | 0.869 | 0.765 | 11094 | 13125 |
| [400, 800) | 0.148 | 0.111 | 2.037 | 2.464 | 11142 | 12563 |
| [800, 1600) | 0.125 | 0.090 | 1.717 | 1.993 | 13254 | 18187 |
| [1600, 1000000000) | 0.109 | 0.065 | 1.495 | 1.438 | 15477 | 28401 |

**Transcript region**

| bin | positive rate (cell line 2) | positive rate (cell line 1) | lift vs file rate (cell line 2) | lift vs file rate (cell line 1) | sites (cell line 2) | sites (cell line 1) |
|---|---|---|---|---|---|---|
| 3'UTR | 0.110 | 0.064 | 1.511 | 1.414 | 31455 | 47989 |
| 5'UTR | 0.046 | 0.042 | 0.631 | 0.930 | 2489 | 2680 |
| CDS | 0.052 | 0.031 | 0.713 | 0.701 | 48848 | 62590 |
| noncoding | 0.090 | 0.058 | 1.236 | 1.285 | 5549 | 5923 |

**Distance to the stop codon (nt; negative = coding sequence)**

| bin | positive rate (cell line 2) | positive rate (cell line 1) | lift vs file rate (cell line 2) | lift vs file rate (cell line 1) | sites (cell line 2) | sites (cell line 1) |
|---|---|---|---|---|---|---|
| [-1000, -900) | 0.033 | 0.030 | 0.456 | 0.678 | 2056 | 2625 |
| [-900, -800) | 0.042 | 0.027 | 0.576 | 0.603 | 2414 | 3024 |
| [-800, -700) | 0.037 | 0.018 | 0.504 | 0.401 | 2735 | 3496 |
| [-700, -600) | 0.032 | 0.023 | 0.447 | 0.521 | 3208 | 4182 |
| [-600, -500) | 0.031 | 0.020 | 0.428 | 0.455 | 3793 | 4945 |
| [-500, -400) | 0.041 | 0.023 | 0.563 | 0.520 | 4501 | 5608 |
| [-400, -300) | 0.052 | 0.026 | 0.712 | 0.588 | 5029 | 6280 |
| [-300, -200) | 0.055 | 0.031 | 0.756 | 0.689 | 5703 | 6909 |
| [-200, -100) | 0.057 | 0.039 | 0.788 | 0.870 | 6047 | 7493 |
| [-100, 0) | 0.112 | 0.073 | 1.544 | 1.625 | 6164 | 7764 |
| [0, 100) | 0.175 | 0.121 | 2.404 | 2.682 | 6015 | 7419 |
| [100, 200) | 0.170 | 0.118 | 2.346 | 2.626 | 4409 | 5635 |
| [200, 300) | 0.124 | 0.088 | 1.704 | 1.960 | 3363 | 4531 |
| [300, 400) | 0.097 | 0.064 | 1.338 | 1.413 | 2686 | 3795 |
| [400, 500) | 0.103 | 0.057 | 1.412 | 1.261 | 2282 | 3212 |
| [500, 600) | 0.067 | 0.039 | 0.919 | 0.858 | 1888 | 2853 |
| [600, 700) | 0.062 | 0.043 | 0.848 | 0.962 | 1624 | 2499 |
| [700, 800) | 0.060 | 0.031 | 0.823 | 0.694 | 1356 | 2181 |
| [800, 900) | 0.065 | 0.031 | 0.898 | 0.694 | 1197 | 1925 |
| [900, 1000) | 0.057 | 0.028 | 0.788 | 0.622 | 926 | 1646 |

### E. RNA folding test, every threshold and stratum (section 7)

Co-modification of fold-linked vs unlinked site pairs at the same distance, each relative to the within-transcript expectation; `linked / unlinked` with a 95% interval from resampling transcripts. tau = minimum predicted pairing probability between the two sites' +-5 nt windows. Sources: `analysis/newdata/fold_comodification.py` -> `fold_comodification/` (RNAplfold, W 480, L 400) and `fold_comodification_linearpartition/` (LinearPartition -V, global).


**RNAplfold - dataset0 (cell line 1)**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 8042 | 48315 | 2.105 | 2.095 | 1.005 | 0.852 | 1.174 |
| 0.100 | 50-100 nt | 6823 | 102176 | 1.836 | 1.881 | 0.976 | 0.780 | 1.183 |
| 0.100 | 100-200 nt | 6617 | 197845 | 1.419 | 1.469 | 0.966 | 0.763 | 1.223 |
| 0.100 | 200-400 nt | 11316 | 344705 | 1.088 | 1.117 | 0.974 | 0.736 | 1.183 |
| 0.300 | 25-50 nt | 4049 | 52308 | 1.755 | 2.127 | 0.825 | 0.578 | 1.067 |
| 0.300 | 50-100 nt | 3267 | 105732 | 1.715 | 1.883 | 0.911 | 0.619 | 1.196 |
| 0.300 | 100-200 nt | 2684 | 201778 | 1.702 | 1.464 | 1.163 | 0.775 | 1.524 |
| 0.300 | 200-400 nt | 3725 | 352296 | 0.991 | 1.118 | 0.887 | 0.589 | 1.223 |

**RNAplfold - dataset0 (cell line 1), both sites >=100 nt from a junction**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 3209 | 18954 | 3.001 | 2.815 | 1.066 | 0.881 | 1.285 |
| 0.100 | 50-100 nt | 2612 | 38155 | 2.314 | 2.602 | 0.889 | 0.708 | 1.099 |
| 0.100 | 100-200 nt | 2170 | 68555 | 1.803 | 2.032 | 0.887 | 0.670 | 1.115 |
| 0.100 | 200-400 nt | 3423 | 106821 | 1.547 | 1.626 | 0.951 | 0.735 | 1.220 |
| 0.300 | 25-50 nt | 1630 | 20533 | 2.479 | 2.871 | 0.863 | 0.586 | 1.160 |
| 0.300 | 50-100 nt | 1280 | 39487 | 2.399 | 2.589 | 0.927 | 0.694 | 1.239 |
| 0.300 | 100-200 nt | 926 | 69799 | 2.212 | 2.022 | 1.094 | 0.669 | 1.597 |
| 0.300 | 200-400 nt | 1097 | 109147 | 1.507 | 1.625 | 0.928 | 0.581 | 1.316 |

**RNAplfold - data1 (cell line 2)**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 5814 | 34356 | 2.320 | 1.964 | 1.182 | 0.987 | 1.422 |
| 0.100 | 50-100 nt | 4877 | 73313 | 2.251 | 1.692 | 1.330 | 1.081 | 1.645 |
| 0.100 | 100-200 nt | 4679 | 139278 | 1.439 | 1.341 | 1.073 | 0.802 | 1.338 |
| 0.100 | 200-400 nt | 8099 | 240306 | 1.230 | 1.012 | 1.215 | 0.991 | 1.534 |
| 0.300 | 25-50 nt | 2953 | 37217 | 1.908 | 2.021 | 0.944 | 0.737 | 1.161 |
| 0.300 | 50-100 nt | 2289 | 75901 | 2.229 | 1.712 | 1.302 | 0.969 | 1.658 |
| 0.300 | 100-200 nt | 1873 | 142084 | 1.504 | 1.342 | 1.121 | 0.701 | 1.502 |
| 0.300 | 200-400 nt | 2709 | 245696 | 1.050 | 1.019 | 1.030 | 0.702 | 1.419 |

**RNAplfold - data1 (cell line 2), both sites >=100 nt from a junction**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 2033 | 11814 | 3.876 | 2.823 | 1.373 | 1.115 | 1.676 |
| 0.100 | 50-100 nt | 1619 | 23524 | 3.683 | 2.492 | 1.478 | 1.172 | 1.884 |
| 0.100 | 100-200 nt | 1271 | 39982 | 2.170 | 1.943 | 1.117 | 0.828 | 1.433 |
| 0.100 | 200-400 nt | 1972 | 59080 | 1.527 | 1.456 | 1.048 | 0.780 | 1.258 |
| 0.300 | 25-50 nt | 1082 | 12765 | 2.955 | 2.963 | 0.997 | 0.700 | 1.328 |
| 0.300 | 50-100 nt | 766 | 24377 | 3.359 | 2.540 | 1.322 | 0.991 | 1.772 |
| 0.300 | 100-200 nt | 484 | 40769 | 2.157 | 1.948 | 1.107 | 0.689 | 1.694 |
| 0.300 | 200-400 nt | 664 | 60388 | 0.923 | 1.464 | 0.631 | 0.316 | 1.032 |

**RNAplfold - positive rate by predicted unpaired probability of the A (quintiles)**


cell line 1

| unpaired bin | positive rate | sites | lift |
|---|---|---|---|
| (-0.0007030000000000001, 0.26] | 0.0451 | 24216 | 1.0015 |
| (0.26, 0.512] | 0.0462 | 24216 | 1.0262 |
| (0.512, 0.733] | 0.0466 | 24215 | 1.0355 |
| (0.733, 0.897] | 0.0458 | 24216 | 1.0189 |
| (0.897, 1.0] | 0.0413 | 24216 | 0.9179 |

cell line 2

| unpaired bin | positive rate | sites | lift |
|---|---|---|---|
| (-0.0007030000000000001, 0.267] | 0.0729 | 18005 | 1.0050 |
| (0.267, 0.527] | 0.0768 | 18004 | 1.0595 |
| (0.527, 0.747] | 0.0774 | 18005 | 1.0678 |
| (0.747, 0.905] | 0.0716 | 18003 | 0.9875 |
| (0.905, 1.0] | 0.0638 | 18004 | 0.8802 |

**LinearPartition - dataset0 (cell line 1)**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 5462 | 50895 | 1.988 | 2.110 | 0.942 | 0.723 | 1.155 |
| 0.100 | 50-100 nt | 3962 | 105037 | 2.178 | 1.864 | 1.169 | 0.941 | 1.464 |
| 0.100 | 100-200 nt | 2737 | 201725 | 1.490 | 1.467 | 1.015 | 0.672 | 1.428 |
| 0.100 | 200-400 nt | 2058 | 353963 | 1.222 | 1.116 | 1.095 | 0.561 | 1.593 |
| 0.100 | 400-800 nt | 1253 | 524465 | 1.015 | 0.800 | 1.269 | 0.530 | 2.007 |
| 0.300 | 25-50 nt | 3890 | 52467 | 1.956 | 2.108 | 0.928 | 0.691 | 1.164 |
| 0.300 | 50-100 nt | 2830 | 106169 | 2.055 | 1.872 | 1.098 | 0.797 | 1.468 |
| 0.300 | 100-200 nt | 1960 | 202502 | 1.565 | 1.467 | 1.067 | 0.550 | 1.528 |
| 0.300 | 200-400 nt | 1431 | 354590 | 1.522 | 1.115 | 1.365 | 0.718 | 2.177 |
| 0.300 | 400-800 nt | 876 | 524842 | 0.593 | 0.801 | 0.741 | 0.000 | 1.369 |

**LinearPartition - dataset0 (cell line 1), both sites >=100 nt from a junction**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 2214 | 19949 | 2.990 | 2.825 | 1.058 | 0.818 | 1.326 |
| 0.100 | 50-100 nt | 1536 | 39231 | 2.447 | 2.589 | 0.945 | 0.708 | 1.206 |
| 0.100 | 100-200 nt | 923 | 69802 | 1.613 | 2.030 | 0.795 | 0.454 | 1.163 |
| 0.100 | 200-400 nt | 579 | 109665 | 1.714 | 1.623 | 1.056 | 0.542 | 1.676 |
| 0.100 | 400-800 nt | 284 | 141802 | 1.004 | 1.159 | 0.867 | 0.000 | 1.829 |
| 0.300 | 25-50 nt | 1548 | 20615 | 2.863 | 2.840 | 1.008 | 0.738 | 1.353 |
| 0.300 | 50-100 nt | 1077 | 39690 | 2.533 | 2.584 | 0.980 | 0.662 | 1.342 |
| 0.300 | 100-200 nt | 653 | 70072 | 1.630 | 2.028 | 0.804 | 0.366 | 1.452 |
| 0.300 | 200-400 nt | 403 | 109841 | 2.226 | 1.621 | 1.373 | 0.552 | 2.187 |
| 0.300 | 400-800 nt | 199 | 141887 | 0.484 | 1.159 | 0.418 | 0.000 | 1.551 |

**LinearPartition - data1 (cell line 2)**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 3930 | 36240 | 2.288 | 1.982 | 1.154 | 0.941 | 1.363 |
| 0.100 | 50-100 nt | 2832 | 75358 | 2.293 | 1.704 | 1.346 | 1.089 | 1.673 |
| 0.100 | 100-200 nt | 2000 | 141957 | 1.384 | 1.344 | 1.030 | 0.705 | 1.409 |
| 0.100 | 200-400 nt | 1562 | 246843 | 1.214 | 1.018 | 1.192 | 0.623 | 1.752 |
| 0.100 | 400-800 nt | 895 | 354798 | 1.443 | 0.734 | 1.965 | 0.893 | 3.319 |
| 0.300 | 25-50 nt | 2842 | 37328 | 2.148 | 2.002 | 1.073 | 0.808 | 1.333 |
| 0.300 | 50-100 nt | 2046 | 76144 | 2.358 | 1.709 | 1.380 | 1.052 | 1.816 |
| 0.300 | 100-200 nt | 1385 | 142572 | 1.461 | 1.343 | 1.088 | 0.630 | 1.580 |
| 0.300 | 200-400 nt | 1114 | 247291 | 1.164 | 1.019 | 1.143 | 0.496 | 1.821 |
| 0.300 | 400-800 nt | 616 | 355077 | 1.493 | 0.735 | 2.033 | 0.597 | 3.897 |

**LinearPartition - data1 (cell line 2), both sites >=100 nt from a junction**

| tau | band | linked pairs | unlinked pairs | linked local | unlinked local | linked / unlinked | 95% low | 95% high |
|---|---|---|---|---|---|---|---|---|
| 0.100 | 25-50 nt | 1402 | 12445 | 3.987 | 2.854 | 1.397 | 1.096 | 1.704 |
| 0.100 | 50-100 nt | 929 | 24214 | 3.303 | 2.534 | 1.303 | 0.968 | 1.734 |
| 0.100 | 100-200 nt | 547 | 40706 | 1.694 | 1.955 | 0.866 | 0.531 | 1.259 |
| 0.100 | 200-400 nt | 349 | 60703 | 1.381 | 1.459 | 0.947 | 0.267 | 1.709 |
| 0.100 | 400-800 nt | 144 | 74185 | 1.275 | 1.034 | 1.234 | 0.000 | 2.620 |
| 0.300 | 25-50 nt | 1001 | 12846 | 3.647 | 2.911 | 1.253 | 0.856 | 1.607 |
| 0.300 | 50-100 nt | 683 | 24460 | 3.465 | 2.539 | 1.364 | 0.895 | 1.843 |
| 0.300 | 100-200 nt | 355 | 40898 | 1.488 | 1.956 | 0.761 | 0.385 | 1.327 |
| 0.300 | 200-400 nt | 243 | 60809 | 1.469 | 1.458 | 1.007 | 0.145 | 1.954 |
| 0.300 | 400-800 nt | 86 | 74243 | 0.801 | 1.034 | 0.774 | 0.000 | 2.859 |

**LinearPartition - positive rate by predicted unpaired probability of the A (quintiles)**


cell line 1

| unpaired bin | positive rate | sites | lift |
|---|---|---|---|
| (-0.0159, 0.083] | 0.0488 | 24219 | 1.0858 |
| (0.083, 0.42] | 0.0438 | 24213 | 0.9740 |
| (0.42, 0.852] | 0.0429 | 24215 | 0.9547 |
| (0.852, 0.986] | 0.0425 | 24216 | 0.9436 |
| (0.986, 1.0] | 0.0469 | 24216 | 1.0418 |

cell line 2

| unpaired bin | positive rate | sites | lift |
|---|---|---|---|
| (-0.0159, 0.0846] | 0.0709 | 18005 | 0.9774 |
| (0.0846, 0.44] | 0.0758 | 18004 | 1.0449 |
| (0.44, 0.87] | 0.0713 | 18004 | 0.9829 |
| (0.87, 0.988] | 0.0735 | 18005 | 1.0134 |
| (0.988, 1.0] | 0.0712 | 18003 | 0.9814 |
