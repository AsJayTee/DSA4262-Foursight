# SG-NEx and GLORI: findings and definitions

The source of truth for the Task 2 (SG-NEx) findings that rest on GLORI. Every
term the report uses for these results is defined here, and every number names
the file and command that produce it. Use of GLORI was agreed by the team and
the course coordinator on 26 Oct.

If the report and this file disagree, check this file against the result
files listed in [section 9](#9-provenance-and-regeneration) before changing
either.

Contents:

1. [What the analysis is for](#1-what-the-analysis-is-for)
2. [Definitions](#2-definitions)
3. [The GLORI data](#3-the-glori-data)
4. [How a site gets its label](#4-how-a-site-gets-its-label)
5. [Finding 1: the model's SG-NEx predictions, graded by GLORI](#5-finding-1-the-models-sg-nex-predictions-graded-by-glori)
6. [Finding 2: comparison with m6Anet](#6-finding-2-comparison-with-m6anet)
7. [Finding 3: which kinds of genes carry more m6A](#7-finding-3-which-kinds-of-genes-carry-more-m6a)
8. [Limitations](#8-limitations)
9. [Provenance and regeneration](#9-provenance-and-regeneration)

---

## 1. What the analysis is for

Task 2 asks us to predict m6A across the SG-NEx direct RNA sequencing samples
and compare cell lines. SG-NEx has no m6A labels, so without outside data the
predictions can only be shown to look plausible: they reproduce known
patterns, such as the rise after the stop codon and the dip near exon
junctions.

GLORI is an independent laboratory method that maps m6A at single-base
resolution. Public GLORI site lists exist for four of the seven SG-NEx cell
lines. Using them as labels turns the SG-NEx predictions into a held-out test,
in cell lines the model was not trained on and at SG-NEx's real read depths.

GLORI is used **only to grade**. It is never an input to the model. The
predictions were made before GLORI was obtained, and they are unchanged.

---

## 2. Definitions

**m6A.** A chemical mark (N6-methyladenosine) on individual A bases of RNA.
Not every A can carry it. Most m6A sits in a short sequence context called
DRACH, where D = A/G/T, R = A/G, H = A/C/T, and the A in the middle is the
modified base.

**Candidate site.** An A at the centre of a DRACH 5-mer in a transcript, as
listed by SG-NEx's m6Anet-processed files. Only candidate sites are ever
scored, by our model or by m6Anet. A GLORI site outside a DRACH context
cannot be a candidate, so it is never counted, as a hit or a miss
([section 4](#4-how-a-site-gets-its-label)). Every statement here about "sites"
means candidate sites.

**Transcript copy vs genomic site.** A gene usually has several transcripts
(isoforms), and one A in the genome can appear in several of them. SG-NEx
reports each (transcript, position) pair separately, and assigns each read
to one transcript, so each copy has its own reads and its own score. GLORI
reports genomic positions. **All analyses here use one row per genomic A**,
represented by the transcript copy with the most reads. Section 4 gives the
reason.

**Read and read depth.** A read is one RNA molecule passing through the
nanopore. A site's depth (`n_reads`) is the number of distinct molecules
measured at it. It is not repeated measurements of one molecule
([docs/data.md](../../docs/data.md#read-depth)). Depth mostly reflects how
abundant the transcript is.

**Pooled line.** Each cell line has 2-6 SG-NEx samples (separate batches and
runs). For every site, the reads from all of that line's samples are
combined, and the site is scored once on the combined reads. The depths
quoted here are pooled depths.

**Depth band.** Sites grouped by pooled depth. Findings 1 and 2 use four
bands: 1-2, 3-9, 10-19 and 20+ reads. Finding 3 uses the repo's eleven finer
bands (1, 2, 3-4, 5-9, 10-19, 20-31, 32-46, 47-83, 84-303, 304-599, 600+;
`m6a.evaluation.depth_bands`).

**Model score (`score_raw`).** The mean of the six networks' raw outputs in the
shipped ensemble (`models/final`). It is a ranking score on one fixed scale,
comparable between files, and **not a probability**. The leaderboard and
`scripts/predict.py` output a rank within the file instead; the analyses here
use `score_raw`, so that sites can be compared across depth bands.

**The shipped model** was trained on every labelled site of dataset0 (HCT116)
and data1 (HEK293T). It has not seen A549, H9 or HepG2. It has seen HCT116,
so HCT116 results exclude the training sites and are not a held-out cell line.

**GLORI site (positive).** A genomic A that the GLORI study reported as m6A in
that cell line, in the file listed in [section 3](#3-the-glori-data). We apply
no extra threshold of our own: a site is positive if and only if it is in the
authors' list.

**GLORI ratio (m6A level).** For a GLORI site, the fraction of the molecules
GLORI read there that carried m6A (0-1). It measures how much of the RNA at
that site is modified, which is called stoichiometry. Only positives have one.

**Negative.** A candidate site in a gene that GLORI covered (next definition)
that is not in GLORI's list. Negatives are inferred, not measured. GLORI
publishes its positive calls only, not the sites it tested and found
unmodified.

**GLORI-covered gene.** A gene in which GLORI called at least one site that is
also one of our candidates. Genes without any GLORI call are left out of
Findings 1 and 2: there, a missing call might mean "not modified" or might
mean "GLORI never measured this gene", and the two cannot be told apart.

**Positive rate.** The share of sites in a group that are GLORI positives. A
random ranking would score a PR AUC about equal to it.

**PR AUC.** Average precision: one number for how well a ranking places
positives above negatives across all cut-offs (`m6a.evaluation.metrics`). It is
the project's headline metric, because positives are rare.

**Lift.** PR AUC divided by the positive rate. 1x is no better than random.
Lift lets groups with different positive rates be compared.

**Top 1% precision.** Among the 1% of sites the model ranks highest within a
group, the share that are GLORI positives. Read it as: if a researcher took
the model's top 1% to the lab, how many would GLORI confirm.

**Motif-only baseline.** A ranking that uses only each site's DRACH 5-mer: every
site gets its 5-mer's modification rate in our two labelled training lines
(dataset0 and data1, averaged; `data/dashboard/trust.json`). It is the
"sequence alone" rule our model has to beat.

**Spearman correlation (ρ).** A rank correlation: +1 if two quantities order
the sites identically, 0 if unrelated.

---

## 3. The GLORI data

GLORI (glyoxal and nitrite-mediated deamination) chemically converts every
*unmodified* A to inosine, which sequencing reads as G. m6A is protected and
still reads as A. At each A, the share of reads that still show A is the
site's m6A level. The method is antibody-free, unlike the m6ACE-seq behind
our course labels.

| SG-NEx line | GEO series, sample | File used | GLORI sites | What the authors' list contains |
|---|---|---|---:|---|
| A549 (lung cancer) | GSE290408 | `GSE290408_A549_GLORI3_hg38_SNuc_KDremoved.bed.gz` | 47,259 | Sites in wild-type cells, **minus any site that persists after METTL3 knockdown** |
| H9 (embryonic stem cells) | GSE290408 | `GSE290408_H9_GLORI3_hg38_SNuc_KDremoved.bed.gz` | 34,906 | As for A549. The series calls the samples "hESC"; the file is named H9 |
| HepG2 (liver cancer) | GSE301419, GSM9083038 | `GSM9083038_HepG2_sgCTRL.totalm6A.FDR.csv.gz` | 7,356 | GLORI-tools default calls in **engineered** cells (below) |
| HCT116 (colon cancer) | GSE301419, GSM9083040 | `GSM9083040_HCT116_sgCTRL.totalm6A.FDR.csv.gz` | 6,680 | As for HepG2 |

What this means for the labels:

- **A549 and H9 positives are confirmed, writer-dependent m6A.** METTL3 is the
  enzyme that deposits m6A on mRNA. A site that survives METTL3 knockdown is
  removed as unlikely to be METTL3-made m6A. That is a strict definition, and
  it should make false positives rare. GEO processing note: "authentic m6A
  modification sites ... identified by filtering out sites that persist in
  METTL3-knockdown (KD) cells" (PubMed 41560347).
- **HepG2 and HCT116 come from engineered cells.** GSE301419 (the FOCAS study,
  PubMed 41478283) uses cells expressing dCas13b-FTO, an m6A *eraser* that a
  guide RNA steers to chosen sites. We used the control-guide (sgCTRL)
  samples. The control guide targets nothing, but some background
  demethylation by the eraser cannot be ruled out. These two GLORI runs also
  called far fewer sites (about 7k, against 35-47k), so more real sites are
  missing from their lists. HepG2 and HCT116 numbers should be read as the
  weaker evidence.
- **GSE210563, the original GLORI paper's series, has no SG-NEx line**
  (HEK293T, HeLa and mouse only). An earlier GAPS.md entry wrongly recommended
  it.
- GSE272469 also has A549 GLORI (wild type, two replicates). It was not used.
  It could check how far two GLORI studies of one cell line agree.

---

## 4. How a site gets its label

Steps, all in `analysis/sgnex/glori_check.py`:

1. **Score.** Our model's `score_raw` for every candidate site of the pooled
   line (`data/sgnex_scores/<line>_pooled.csv.gz`, from
   `analysis/sgnex/score_sgnex.py`).
2. **Locate the A.** The position convention is checked, not assumed: the
   site's 7-mer matches the Ensembl 91 cDNA at that position for 97.2% of
   sites, with `transcript_position` being the A itself.
3. **Map to the genome.** The A's transcript position goes through the
   transcript's exons (Ensembl 91 GTF, GRCh38) to a genomic coordinate and
   strand.
4. **Collapse isoforms.** For each genomic A, keep the transcript copy with
   the most reads. GLORI gives one call per genomic A; counting a site once
   per isoform would weight genes by how many isoforms they have.
5. **Match to GLORI.** Same chromosome, position and strand.
   - With no shift, 50-74% of GLORI sites land exactly on one of our
     candidates: A549 50.1%, H9 53.3%, HepG2 71.0%, HCT116 74.4%.
   - Shifted by one base either way, at most 0.23% do. A coordinate error
     would show up as a near-zero match, so the mapping is exact.
   - The GLORI sites that match nothing are mostly non-DRACH sites, or
     sites on transcripts SG-NEx did not cover. We did not break this down
     further.
6. **Restrict to GLORI-covered genes** (definition in section 2).
7. **HCT116 only:** drop the 121,838 sites in dataset0, the model's training
   set.

Sites graded after these steps: A549 518,544; H9 570,700; HepG2 162,764;
HCT116 99,576.

---

## 5. Finding 1: the model's SG-NEx predictions, graded by GLORI

### 5.1 Results

| line | depth band | sites | GLORI sites | positive rate | PR AUC model | PR AUC motif-only | lift model | lift motif-only | top 1% model | top 1% motif-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A549 | 1-2 reads | 140,503 | 4,570 | 3.3% | 0.135 | 0.073 | 4.2x | 2.3x | 19.2% | 8.6% |
| A549 | 3-9 reads | 171,655 | 7,352 | 4.3% | 0.212 | 0.097 | 5.0x | 2.3x | 25.9% | 12.6% |
| A549 | 10-19 reads | 83,513 | 4,201 | 5.0% | 0.264 | 0.113 | 5.2x | 2.2x | 29.0% | 12.8% |
| A549 | 20+ reads | 122,873 | 7,546 | 6.1% | 0.366 | 0.150 | 6.0x | 2.4x | 45.4% | 22.0% |
| H9 | 1-2 reads | 104,168 | 1,081 | 1.0% | 0.041 | 0.021 | 4.0x | 2.0x | 5.4% | 2.5% |
| H9 | 3-9 reads | 145,018 | 3,139 | 2.2% | 0.103 | 0.044 | 4.8x | 2.0x | 11.0% | 4.8% |
| H9 | 10-19 reads | 97,075 | 2,954 | 3.0% | 0.149 | 0.063 | 4.9x | 2.1x | 12.8% | 6.5% |
| H9 | 20+ reads | 224,439 | 11,437 | 5.1% | 0.300 | 0.107 | 5.9x | 2.1x | 39.5% | 18.4% |
| HepG2 | 1-2 reads | 21,904 | 93 | 0.4% | 0.022 | 0.012 | 5.2x | 2.8x | 2.7% | 1.4% |
| HepG2 | 3-9 reads | 23,740 | 277 | 1.2% | 0.086 | 0.035 | 7.4x | 3.0x | 14.3% | 5.9% |
| HepG2 | 10-19 reads | 18,377 | 352 | 1.9% | 0.138 | 0.057 | 7.2x | 3.0x | 15.2% | 10.9% |
| HepG2 | 20+ reads | 98,743 | 4,504 | 4.6% | 0.381 | 0.141 | 8.3x | 3.1x | 51.0% | 34.3% |
| HCT116\* | 1-2 reads | 19,490 | 42 | 0.2% | 0.015 | 0.008 | 6.9x | 3.5x | 3.6% | 2.1% |
| HCT116\* | 3-9 reads | 18,899 | 181 | 1.0% | 0.072 | 0.025 | 7.5x | 2.6x | 8.5% | 3.7% |
| HCT116\* | 10-19 reads | 14,207 | 227 | 1.6% | 0.097 | 0.043 | 6.1x | 2.7x | 12.7% | 10.6% |
| HCT116\* | 20+ reads | 46,980 | 1,328 | 2.8% | 0.194 | 0.085 | 6.9x | 3.0x | 21.5% | 14.3% |

\* Training sites removed. HCT116 is the model's training cell line, so it is
not a held-out line.

Rows that pool all depth bands are in `analysis/sgnex/glori/metrics.csv`. They
are not quoted as headline numbers: pooling mixes groups whose positive rate
and score both rise with depth, which inflates lift (H9 pooled 7.2x, against
4.0-5.9x within bands).

### 5.2 What it shows

- **The predictions carry real signal in cell lines the model never saw.**
  Within every depth band of every line, the model's lift is about twice the
  motif-only baseline's. At 20+ reads in A549, 45% of the model's top 1% are
  GLORI sites, against 22% for motif-only and 6.1% for a random pick.
- **Low-depth sites remain informative.** At 1-2 reads, which is about a
  third of SG-NEx, the lift is 4.0-5.2x in the held-out lines, against
  2.0-2.8x for motif-only. The earlier read-dropping simulation had suggested
  the model falls to the motif-only level at 1 read. This is the first
  measurement on real low-depth sites. Two possible reasons for the
  difference: real low-depth sites keep deeply read neighbours, which the
  site-graph model uses, and the 1-2 band includes 2-read sites.
- **Lift is lower than on our own labels:** 6-8x at 20+ reads here, against
  about 12x out-of-fold on dataset0. A change of cell line *and* of labelling
  method is expected to cost this much. It matches the data1 result
  (decision 0029).

### 5.3 How far the two laboratory methods agree (HCT116)

HCT116 is the only line with both our course labels (m6ACE-seq) and GLORI. On
the 48,620 training sites inside GLORI-covered genes, treating the course
labels as a predictor of GLORI:

- **precision 0.477:** of the course's positives, 47.7% are in GLORI's list;
- **recall 0.443:** of GLORI's sites, the course labels mark 44.3%;
- GLORI positive rate on these sites: 6.6%.

So the two assays agree on fewer than half of the sites in the same cell line.
A model trained on one assay's labels cannot be expected to match the other
assay closely. This puts a rough ceiling on Finding 1.

The model's precision on held-out lines, at the same recall of 44.3%:

| line, 20+ reads | precision at 44.3% recall |
|---|---:|
| A549 | 42.5% |
| HepG2 | 39.5% |
| H9 | 33.3% |
| HCT116 (non-training sites) | 22.0% |

The comparison is indicative, not like for like. The course labels' 0.477 is
measured on HCT116's training sites, which are all abundant (20+ reads),
against HCT116's shallow GLORI run. The model's figures come from other lines
and other sites.

### 5.4 The score tracks how much of a site is modified

Among GLORI positives, Spearman correlation between the model's score and the
GLORI ratio:

| line | all positives | positives with 20+ reads | 20+ reads, GGACT motif only (n) |
|---|---:|---:|---|
| A549 | 0.432 | 0.585 | 0.493 (1,186) |
| H9 | 0.393 | 0.546 | 0.346 (1,320) |
| HepG2 | 0.572 | 0.639 | 0.642 (1,073) |
| HCT116 | 0.624 | 0.677 | 0.674 (321) |

The model was trained on yes/no labels and was never shown a modification
level. The correlation holds within a single motif, so it is not only that
some motifs are both more often and more heavily modified.

---

## 6. Finding 2: comparison with m6Anet

### 6.1 Set-up

- **m6Anet** 2.1.0, the pretrained `HCT116_RNA002` model (Hendra et al.,
  Nature Methods 2022). It was trained on HCT116. Our model was trained on
  HCT116 and HEK293T. Neither has seen A549, H9 or HepG2, so all three are
  held out for both. HCT116 is left out of this comparison because m6Anet
  was trained on it.
- **The same sites:** exactly the graded sites of section 4.
- **The same reads:** each site's reads pooled over all of the line's samples,
  as for our model. m6Anet read at most 50 reads per site, sampled at random
  (seed 4262), to keep the run affordable. Its site probability is built from
  20 reads sampled with replacement, so the cap changes its scores very
  little. Our model used every read. Depth bands use the uncapped count for
  both models.
- **m6Anet's score** is its site-level `probability_modified`.
- **Two settings.** As published, m6Anet drops every site with fewer than 20
  reads. We also ran it with that minimum lowered to 1
  (`analysis/m6anet/infer_low_depth.py`). Each site's probability depends only
  on its own reads, so the 20+ rows below are what m6Anet as published
  produces.
- **Uncertainty:** the difference in PR AUC is recomputed over 200 resamples
  of genes. Genes are resampled, not sites, because sites in one gene are not
  independent. The interval is the middle 95% of those resamples.
- **Coverage:** m6Anet scored all but 34 (A549) and 60 (H9) of the graded
  sites, and every HepG2 site.
- Run on Ronin on 26 Oct (`analysis/sgnex/run_m6anet_glori.sh`), compared
  locally (`analysis/sgnex/m6anet_glori_compare.py`).

### 6.2 What m6Anet as published can score

| line | graded sites with 20+ reads (scored by m6Anet as published) | GLORI sites among them |
|---|---:|---:|
| A549 | 23.7% | 31.9% of A549's GLORI sites |
| H9 | 39.3% | 61.5% |
| HepG2 | 60.7% | 86.2% |

Our model scores every site.

### 6.3 Head to head, identical sites

| line | depth band | sites | positive rate | PR AUC ours | PR AUC m6Anet | difference (95% interval) | resamples ours wins | top 1% ours | top 1% m6Anet |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| A549 | 1-2 reads | 140,469 | 3.3% | 0.135 | 0.114 | +0.021 (+0.016 to +0.026) | 100% | 19.1% | 17.4% |
| A549 | 3-9 reads | 171,655 | 4.3% | 0.212 | 0.205 | +0.008 (+0.003 to +0.014) | 100% | 25.9% | 30.5% |
| A549 | 10-19 reads | 83,513 | 5.0% | 0.264 | 0.275 | -0.011 (-0.018 to -0.004) | 0% | 29.0% | 39.2% |
| A549 | **20+ reads** | 122,873 | 6.1% | 0.366 | 0.357 | +0.008 (-0.001 to +0.017) | 94% | 45.4% | 49.1% |
| A549 | all depths | 518,510 | 4.6% | 0.270 | 0.201 | +0.069 (+0.064 to +0.074) | 100% | 40.9% | 23.6% |
| H9 | 1-2 reads | 104,156 | 1.0% | 0.041 | 0.037 | +0.004 (+0.001 to +0.007) | 100% | 5.4% | 5.8% |
| H9 | 3-9 reads | 145,013 | 2.2% | 0.103 | 0.103 | +0.000 (-0.004 to +0.005) | 62% | 11.0% | 14.4% |
| H9 | 10-19 reads | 97,067 | 3.0% | 0.149 | 0.158 | -0.008 (-0.013 to -0.004) | 0% | 12.8% | 17.9% |
| H9 | **20+ reads** | 224,404 | 5.1% | 0.300 | 0.262 | +0.038 (+0.032 to +0.044) | 100% | 39.5% | 32.0% |
| H9 | all depths | 570,640 | 3.3% | 0.233 | 0.146 | +0.088 (+0.082 to +0.094) | 100% | 35.2% | 13.3% |
| HepG2 | 1-2 reads | 21,904 | 0.4% | 0.022 | 0.016 | +0.006 (-0.002 to +0.033) | 94% | 2.7% | 2.3% |
| HepG2 | 3-9 reads | 23,740 | 1.2% | 0.086 | 0.065 | +0.022 (+0.008 to +0.038) | 100% | 14.3% | 11.4% |
| HepG2 | 10-19 reads | 18,377 | 1.9% | 0.138 | 0.126 | +0.012 (-0.001 to +0.028) | 93% | 15.2% | 14.7% |
| HepG2 | **20+ reads** | 98,743 | 4.6% | 0.381 | 0.278 | +0.103 (+0.087 to +0.119) | 100% | 51.0% | 33.7% |
| HepG2 | all depths | 162,764 | 3.2% | 0.341 | 0.173 | +0.169 (+0.153 to +0.185) | 100% | 47.4% | 16.6% |

How to read "difference": positive means ours is higher. "Better" means the
95% interval excludes zero. A "tie" means it includes zero.

### 6.4 What it shows

- **Where m6Anet as published works (20+ reads), ours ties or is better:**
  a tie on A549 (+0.008, interval -0.001 to +0.017), better on H9 (+0.038)
  and on HepG2 (+0.103).
- **m6Anet as published says nothing about most SG-NEx sites:** it skips 39%
  (HepG2) to 76% (A549) of them.
- **Below 20 reads, with m6Anet's minimum lowered, the two are close.** Ours
  is ahead at 1-2 reads. m6Anet is slightly ahead at 10-19 reads in A549 and
  H9 (by about 0.01). HepG2 favours ours throughout.
- **Across all depths ranked together, ours is well ahead** (0.270 vs 0.201,
  0.233 vs 0.146, 0.341 vs 0.173). This is the setting of a researcher
  ranking one gene's sites, deep and shallow together. A likely reason:
  `score_raw` keeps one scale across depths, while m6Anet's probability
  shifts with depth. We have not tested that reason directly.
- **Two respects where m6Anet does better:**
  - *Its top 1% within a band.* m6Anet's top 1% are confirmed more often
    than ours in most A549 and H9 bands, including A549 at 20+ reads (49.1%
    vs 45.4%), even where its PR AUC is lower. Its highest-confidence calls
    are sharper. Ours is better over the whole ranking.
  - *Tracking the m6A level.* Spearman correlation with the GLORI ratio,
    positives with 20+ reads: m6Anet 0.623 / 0.606 / 0.708 against ours
    0.585 / 0.546 / 0.639 (A549 / H9 / HepG2). m6Anet estimates the
    fraction of modified reads by design; ours was trained on yes/no labels.
- **Why this comparison is preferred over the earlier HCT116 benchmark:**
  m6Anet's pretrained model was very likely trained on SG-NEx HCT116 with
  m6ACE-seq labels, i.e. on our dataset0's own sites. Here neither model has
  seen the cell lines, and the labels come from a third method.

---

## 7. Finding 3: which kinds of genes carry more m6A

### 7.1 What "more m6A" means here

"Class X carries more m6A" is shorthand for one precise statement:

> **A larger share of the candidate sites in class X's genes rank among the
> top 5% of sites with the same read depth in the same cell line, compared
> with other protein-coding genes.**

It is built in four steps (`analysis/sgnex/gene_classes.py`):

1. **Sites.** Every candidate site of the pooled line, collapsed to one row
   per genomic A, as in section 4. HCT116's training sites are removed. This
   uses all genes, not only GLORI-covered ones.
2. **"High" site: normalised for depth.** Within each cell line and each of
   the eleven fine depth bands, sites are ranked by `score_raw`. A site is
   "high" if it is in the top 5% of its own band. By construction, exactly 5%
   of the sites in every band are high (5.00% overall in A549), so a gene
   gains nothing from being deeply sequenced.
3. **Gene share: normalised for the number of sites.** A gene's m6A share is
   the fraction of its sites that are high. This is a **proportion, not a
   count**: a long gene with many sites does not score higher for having
   them. Genes with fewer than 3 sites are dropped, because their shares are
   too coarse (0, 1/2, 1). A549 keeps 14,824 genes, with a median of 47 sites
   per gene.
4. **Class effect.** The mean share over the class's genes, minus the mean
   share over all other protein-coding genes. For long non-coding RNAs the
   comparison group is protein-coding genes. Each gene counts once, however
   many sites it has. The effect is given in **percentage points**. The 95%
   interval comes from resampling the class's genes and the comparison genes
   (500 resamples).

Worked example (A549, transcription factors):

- The average transcription-factor gene has 8.08% of its sites in its depth
  band's top 5%. The average other protein-coding gene has 5.38%. The effect
  is **+2.7 percentage points**.
- In plain terms: a transcription factor's sites are about 1.5 times as likely
  as a typical gene's sites to be among the model's strongest calls for
  their depth.

What the measure **does not** mean:

- **Not more modified molecules.** It ranks sites by how likely they are to be
  modified. It does not measure the fraction of molecules carrying m6A at
  them (the GLORI ratio does that).
- **Not more m6A sites in absolute number.** Counts would favour long genes.
- **Not a count of sites above a probability cut-off.** The score is not a
  probability.
- **Not independent of sequence.** The motif is part of the score, so a class
  whose genes have more GGACT sites scores higher partly for that. That is
  still m6A biology, since motif drives deposition, but it has not been
  separated out.

### 7.2 The GLORI side of the same question

The same class comparison is repeated on GLORI's own calls, wherever GLORI
exists (A549, H9, HepG2, HCT116). It is depth-matched too, because GLORI also
detects more sites on abundant transcripts:

1. The graded sites of section 4 (GLORI-covered genes), with at least 10
   reads in SG-NEx.
2. Each site's **excess**: its GLORI label (1 or 0) minus the GLORI positive
   rate of its depth band. SG-NEx depth stands in for transcript abundance.
3. A gene's value: the mean excess of its sites (genes with at least 3
   sites).
4. The class effect: the class mean minus the comparison-group mean, in
   percentage points of GLORI call rate, with a gene-resampled interval.

This side uses no model output. It is an independent check.

### 7.3 When a class counts as a finding

All four must hold:

- the model's effect has the same sign in **all 7** SG-NEx lines;
- its 95% interval excludes zero in at least 4 of the 7;
- GLORI's effect has that sign in **all 4** GLORI lines;
- GLORI's interval excludes zero in at least 2 of the 4.

54 classes were tested: ribosomal proteins, mitochondrially encoded genes,
transcription factors (Lambert et al., Cell 2018), long non-coding RNAs, and
the 50 MSigDB hallmark gene sets (v2023.2). There is no correction for
testing many classes, so the results are descriptive. The rule's requirement
of agreement across lines and across two independent measures is what guards
against chance.

### 7.4 Results

| class | genes (A549) | model, range over 7 lines (pp) | GLORI, range over 4 lines (pp) | raw score, A549 (not depth-matched) | finding |
|---|---:|---|---|---:|---|
| TNF-alpha signalling via NF-kB (immediate-early genes) | 166 | +2.9 to +4.4 | +1.7 to +4.9 | +0.35 | yes |
| transcription factors | 1,082 | +2.0 to +3.2 | +0.7 to +2.3 | +0.23 | yes |
| early oestrogen response | 169 | +1.1 to +2.3 | +0.3 to +4.0 | +0.23 | yes |
| long non-coding RNAs (vs protein-coding) | 1,026 | +0.5 to +1.4 | +3.3 to +5.6 | +0.21 | yes |
| mitotic spindle | 194 | -1.6 to -1.1 | -1.7 to -1.3 | -0.23 | yes |
| mTORC1 signalling | 188 | -3.0 to -1.9 | -1.1 to -0.5 | -0.14 | yes |
| E2F targets | 196 | -3.1 to -2.4 | -2.8 to -1.1 | -0.27 | yes |
| MYC targets (v1) | 195 | -3.9 to -3.1 | -2.5 to -1.7 | -0.16 | yes |
| protein secretion | 92 | -3.8 to -3.5 | -1.9 to -0.8 | -0.55 | yes |
| ribosomal proteins | 79 | -3.7 to -1.9 | -0.5 to +1.5 | **+0.38** | no: GLORI disagrees |
| mitochondrially encoded | 12 | -5.7 to -5.4 | no GLORI calls | -0.48 | negative control |

pp = percentage points. All 54 classes are in
`analysis/sgnex/gene_classes/summary.md`.

### 7.5 What it shows

- **Regulatory genes carry more m6A; growth and housekeeping programmes carry
  less.** Transcription factors and immediate-early inflammatory genes (the
  NF-kB set) are enriched. Proliferation and biosynthesis programmes (MYC and
  E2F targets, mTORC1, mitotic spindle, protein secretion) are depleted. Each
  holds in all seven lines and in GLORI. This fits the published picture of
  m6A marking short-lived regulatory transcripts for faster decay. We have
  not tested the decay link with half-life data.
- **Long non-coding RNAs:** both measures say more, but GLORI's effect is
  much larger than the model's (+3.3 to +5.6 against +0.5 to +1.4).
- **Depth matching reverses the raw result for ribosomal proteins.** On raw
  scores, ribosomal-protein genes are the highest-scoring class in A549
  (+0.38). After depth matching they sit 1.9-3.7 points *below* other genes
  in every line. Their transcripts are among the most abundant, so their
  sites are deeply read, and raw scores rise with depth. GLORI places them
  near average (-0.5 to +1.5), so the direction itself is unresolved; the
  size of the raw-score artefact is the point. **Any gene-level claim from
  raw scores, ours or anyone's, needs depth matching.**
- **Mitochondrially encoded genes have essentially no high sites** (about
  -5.5 points against an expected 5%, i.e. close to 0%). These 12-13 genes are
  transcribed inside mitochondria, where the nuclear m6A writer complex does
  not act. It is the clearest negative control in the data. GLORI has no
  calls on them, so it cannot confirm.
- **The effects are nearly the same in every line.** This agrees with the
  earlier SG-NEx analyses (`analysis/sgnex/explore/summary.md`, A2 and E):
  the seven lines' m6A landscapes look alike.

---

## 8. Limitations

Each of these applies to the findings above and should travel with them into
the report.

1. **GLORI lists only what it detected.** Negatives are inferred. A real m6A
   site that GLORI missed counts against both models as a false positive, so
   every precision, PR AUC and lift here is a **lower bound**. This is worst
   for HepG2 and HCT116, whose GLORI runs called about 7k sites. It affects
   our model and m6Anet equally, so Finding 2 still holds.
2. **GLORI detects more on abundant transcripts.** Its positive rate rises
   with SG-NEx depth (A549 3.3% at 1-2 reads to 6.1% at 20+; HepG2 0.4% to
   4.6%). Part of that rise is GLORI missing sites on rare transcripts, not
   less m6A there. So compare within a depth band, not across bands.
3. **Different samples, same cell line.** GLORI and SG-NEx were run in
   different laboratories on different cultures of each line. HepG2 and
   HCT116 GLORI came from engineered cells carrying an m6A eraser
   (section 3).
4. **Only DRACH candidates are scored.** Non-DRACH m6A is outside both models
   and outside these results.
5. **Agreement between methods is limited.** GLORI and m6ACE-seq agree on
   fewer than half of HCT116's sites (section 5.3). The report should not
   present GLORI as the truth, only as an independent second measurement.
6. **Only four of seven lines have GLORI.** K562, MCF7 and HEYA8 are covered
   only by the model's side of Finding 3.
7. **Finding 3 is descriptive.** 54 classes, no multiplicity correction, and
   the motif is not separated from the gene effect.
8. **HCT116 is not held out.** The model was trained on other HCT116 sites.

---

## 9. Provenance and regeneration

| Result | File | Command |
|---|---|---|
| Finding 1 (all tables), mapping checks, label agreement | `analysis/sgnex/glori/{summary.md, metrics.csv, mapping.csv}` | `python analysis/sgnex/glori_check.py` (~2.5 min) |
| Finding 2 | `analysis/sgnex/m6anet_glori/{summary.md, metrics.csv}` | Ronin: `bash analysis/sgnex/run_m6anet_glori.sh` (~1.5 h, 64 vCPUs); local: `python analysis/sgnex/m6anet_glori_compare.py` (~2 min) |
| m6Anet input preparation | `data/m6anet_glori/<line>_input/` (not committed) | `python analysis/sgnex/m6anet_prep.py --line <line> ...` (called by the Ronin script) |
| Finding 3 | `analysis/sgnex/gene_classes/{summary.md, effects.csv}` | `python analysis/sgnex/gene_classes.py` (~3 min, after `glori_check.py`) |
| Worked example in 7.1 (8.08% vs 5.38%) | not stored; recomputed from the same steps on 27 Oct | the gene shares behind `effects.csv`, A549 |

Inputs:

- SG-NEx pooled scores: `data/sgnex_scores/`, from
  `analysis/sgnex/score_sgnex.py`.
- Ensembl 91 annotation: `data/annotation/`, from
  `analysis/newdata/annotation_context.py`.
- GLORI files, the transcription-factor list and the hallmark sets:
  `data/external/glori/`. Each script prints download commands if they are
  missing.

Nothing under `data/` is committed. The labelled per-site files in
`data/external/glori/` include HCT116 training flags derived from the course
labels.

Related summaries:

- `report/notes/findings.md`, section 16: the same results in the findings log
  (written before the m6Anet run, so it does not include Finding 2).
- `report/notes/sgnex_glori_notes.md`: the plain-language version prepared for the
  team meeting.
