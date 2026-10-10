# Task 2 briefing: where we are, and what to decide

For the group meeting. Status as of **10 October 2026**.

Detailed numbers:
- [../analysis/sgnex/explore/summary.md](../analysis/sgnex/explore/summary.md) (tables);
- [../analysis/sgnex/explore/overview.png](../analysis/sgnex/explore/overview.png) (figure);
- [findings.md](findings.md), section 13.

---

## 1. What Task 2 asks for (from the course brief)

1. **Predict** m6A in all SG-NEx direct-RNA datasets.
2. **Describe and compare** the results across cell lines. Open-ended:
   "summarise and visualise".
3. **An interactive platform** for other researchers to explore the
   results. We have decided to host it on **GitHub Pages in this
   repository, built with React**. A starter template is in
   [`dashboard/`](../dashboard/).

It is written up as **Results (2)** in the report, about 1.25 pages.

## 2. What is already done

**Deliverable 1, predictions: done.**
- **Scope:** all 22 SG-NEx samples that are already in the course's input
  format, covering **7 cell lines**: A549 (lung), H9 (embryonic stem
  cells), HEYA8 (ovarian), HCT116 (colon), HepG2 (liver), K562
  (leukaemia), MCF7 (breast).
- The remaining SG-NEx lines exist only as raw signal files. Converting
  them is very expensive; not done, and the report should say so.
- **Model:** the final shipped model.
- **Two versions:**
  - each sample on its own;
  - each cell line's samples **combined** (more reads per site, so more
    reliable).
- **For every candidate site** (1.5-2 million per line) we saved:
  - the **rank score**: what the prediction script outputs. It orders
    sites *within one sample*; it cannot compare cell lines, because
    every sample always has 5% of its sites in its own top 5%;
  - the **raw score**: the model's underlying output, on one fixed scale,
    so it **can** be compared between cell lines. It is not a calibrated
    probability, and sites with fewer reads score closer to the middle,
    so compare lines at similar read counts;
  - **how many reads** support the site;
  - the split into "the site's own evidence" and "its neighbours'
    correction", for the model explainer;
  - **the actual reads** of 300 example genes per line, also for the
    explainer.
- Where it lives: `data/sgnex_scores/` (2.4 GB, not in git). Produced by
  `analysis/sgnex/score_sgnex.py`.

**Two data issues found and fixed:**
- one MCF7 sample wrote transcript names in a different format (with a
  version suffix), so its sites did not combine;
- about half the samples contain *spike-ins* (synthetic control RNAs the
  lab adds): dropped.

**Preliminary analyses A-E: done** (section 4 below).

## 3. Facts we accept (no point re-measuring)

- **Only HCT116 has lab-measured labels**, and only at 121,838
  well-covered sites (all have 20+ reads in every HCT116 sample).
  - Accuracy at the low read counts typical of SG-NEx (1-3 reads) can only
    be *simulated*, by thinning reads (report figure `fig_depth`).
  - The labels map correctly onto all SG-NEx samples: at every labelled
    site, the 7-letter sequence is identical in all 22 samples.
- **The model detects whether a site is modified, not how much of it is.**
  On the in-vitro series, its scores separate 0% from modified RNA
  sharply but plateau above ~50% modified (`fig_data2_fractions`).
- **Differences between cell lines at single sites are mostly not
  visible in the reads.** We showed this on our two labelled cell lines
  (findings 12D), and the SG-NEx analyses below confirm it.

---

## 4. Preliminary results (A-E): what was tested, what it means

All comparisons use the comparable **raw score**, with each line's samples
combined.

### A1. How many reads do sites have?

- **Tested:** share of sites with 1-2, 3-9, 10-19 and 20+ reads. Each read
  is one RNA molecule.
- **Result:**
  - 34-42% of sites have only 1-2 reads; only 14-26% have 20+ (the level
    the model was trained at);
  - median reads per site: 3 (A549, K562, MCF7) or 4-5 (H9, HEYA8,
    HCT116, HepG2).
- **Meaning:** most sites have little evidence behind them. Every
  comparison must be made within the same read-count range, or the line
  with deeper sequencing simply looks different.

### A2. Do two samples of the same line agree more than two different lines?

- **Tested:** for every pair of samples, how similarly they rank the sites
  they share. Spearman correlation: 1 = identical order, 0 = unrelated.
  - "Same line": two samples of one cell line (34 pairs).
  - "Different lines": one sample from each of two lines (21 pairs).

| | 1-2 reads | 3-9 | 10-19 | 20+ reads |
|---|---|---|---|---|
| Same line | 0.815 | 0.853 | 0.889 | 0.928 |
| Different lines | 0.800 | 0.851 | 0.885 | 0.917 |

- **Meaning (the key result):**
  - Scores agree strongly *everywhere*. Most of a site's score comes from
    what all lines share: sequence motif, neighbour layout, m6A biology.
  - The cell-line-specific part, the gap between the two rows, is tiny
    (0.01-0.015).
  - **At single sites, real differences between cell lines cannot be told
    apart from sample-to-sample noise.**

### A3. Does the strongest known m6A motif score highest in every line?

- **Tested:** mean score for each of the 18 DRACH motifs (sites with 10+
  reads), per line.
- **Result:** GGACU is top in all 7 lines (GAACU second), as in our
  labelled data and the literature.
- **Meaning:** the scores behave consistently in lines without labels.
- **Caveat:** the model sees the sequence, so this partly reflects what it
  learned. A consistency check, not independent proof.

### B. Where m6A sits along genes

- **Tested:** each site matched to the gene annotation (Ensembl 91,
  analysis only; the model never sees it): gene region, distance to the
  stop codon, distance to the nearest exon junction. Mean score per
  distance bin (sites with 10+ reads).
- **Result, the same in all 7 lines:**
  - lowest scores within 50 nt of an exon junction, peaking 100-400 nt
    away;
  - a sharp peak just after the stop codon;
  - the protein-coding region scores lowest.
- **Meaning:** this is the published m6A pattern. Proteins left at exon
  junctions block m6A nearby, and m6A concentrates near the stop codon.
  **The model is never told where a site is in its gene, yet the pattern
  appears in every line**: good evidence the scores capture real m6A
  biology where we have no labels.
- **Caveat:** sequence composition differs between gene regions, so part
  of this may come through sequence.

### C. How much m6A does each line have?

- **C1, average score per line within each read range:**
  - narrow spread;
  - HCT116 highest (-1.15 at 20+ reads), MCF7 lowest (-1.52), the rest
    -1.37 to -1.40.
- **C2, can a cut-off from labelled data be trusted on another cell
  line?**
  - On cell line 1's labelled sites, find the score cut-off where 50% of
    the sites above it are truly modified (precision 50%).
  - Apply the **same** cut-off to cell line 2's labelled sites, where the
    answer is known.
  - **Result:** 53% precision there (20+ reads), calling a similar share of
    sites (6.5% vs 6.7%); 47% at 1-2 reads.
  - **Meaning:** a cut-off keeps roughly its meaning across cell lines, so
    SG-NEx site counts can be stated with a reliability. (Checked on two
    labelled lines only.)
- **C3, applied to SG-NEx (20+ reads):**
  - 6.7-9.2% of sites called modified per line at the 50%-precision
    cut-off;
  - **range 2-19%** across stricter or looser cut-offs: always quote the
    range.
  - Sanity check: at 50% precision about half the calls are true, so
    ~3.5-4.5% of well-covered sites are truly modified. That is close to
    the labelled rates (4.5% and 7.2%).
- **HCT116 is partly an artefact.** HCT116 is the model's training cell
  line; dataset0 *is* one of its SG-NEx samples.
  - Its 121,838 training sites score far higher (-0.76) than its other
    sites (-1.30).
  - Excluding training sites shrinks its lead by about a third.
  - **Compare lines on non-training sites only.**

### D. Which genes carry m6A?

- **Tested:** only sites with 10+ reads in **all 7 lines** (212,478 sites,
  5,687 genes), so coverage differences cancel. Gene score = its
  highest-scoring site.
- **Result:**
  - **188 genes are in the top 10% in every line** (e.g. POLDIP2, TSR3,
    PSMB1, PTBP1, HSPA5): a shared core of heavily modified genes;
  - **almost no line-specific genes** (top 10% in one line, bottom half in
    all others): one in HEYA8, none elsewhere.
- **Meaning:** at gene level too, the lines look alike, as expected after
  A2. The shared core is a concrete, reportable result.

### E. Which cell lines look alike?

- **Tested:** correlation between every pair of lines (same sites; same
  genes), then grouping lines by similarity.
- **Result:**
  - all pairs 0.89-0.93;
  - **K562 is the most different**: the only blood-derived line, which is
    biologically plausible;
  - H9, HEYA8 and HCT116 are closest.
- **Meaning: suggestive only.**
  - The grouping also follows read depth: the lower-depth lines (K562,
    MCF7, A549) sit together.
  - The differences are about the size of the sample noise (A2).
  - HCT116 carries the training artefact.

### Overall

- **Solid and reportable:**
  - known m6A biology appears in every line, from a model blind to gene
    position;
  - consistent motif preference;
  - cut-offs that carry across cell lines with stated reliability;
  - a shared core of heavily modified genes.
- **Weak:** differences between lines. They are small, within noise, or
  confounded by the training cell line and by read depth. That agrees
  with our labelled-data finding, and it is an honest limitation to
  report (the brief rewards this).

---

## 5. To decide at the meeting

### a) What else to investigate

| Option | What it would add | Cost |
|---|---|---|
| Redo C-E **without HCT116's training sites** | Removes the training artefact from the comparison | Small (laptop) |
| **Equalise read depth** (thin every site to the same number of reads, re-score) | Removes depth as a confound in E | ~1 h on Ronin (compute is limited) |
| What the 188 shared genes **do** (gene-set enrichment) | Biological interpretation of the shared core | Small-medium |
| **Isoforms**: the same position modified in one version of a gene but not another | Distinctive for long-read SG-NEx data | Larger |
| Compare to **public m6A maps** (e.g. HepG2) | Partial external check | Medium; resolution differs |

### b) Dashboard (GitHub Pages + React)

- **Pages/views:**
  - **Search:** a gene → its sites across the 7 lines, coloured by score,
    marked by read count.
  - **Overview:** the A-E charts, interactive (filter by read count, pick
    lines).
  - **How the model works:** a site's reads, its neighbours, and its score
    built from its own evidence plus its neighbours' correction. Sliders
    remove reads or neighbours. The data for this is saved.
  - **What this can't tell you:** a limitations panel (section 3 above).
- **Data size:** about 13 million site rows across 7 lines is too big to
  ship whole.
  - Options: per-gene files loaded on demand; pre-filtered to well-covered
    sites; or a compact columnar format (Parquet) queried in the browser
    (DuckDB-WASM).
  - Decide what the Search view needs.
- **What must never go on the public site:** the course's labels or raw
  read data (repo rule: the course data is unpublished). Model scores on
  public SG-NEx data are fine. The showcase reads for the explainer are
  SG-NEx's public data and fine in small amounts. **Check the course
  rules.**
- **Who builds what**, and by when (final deadline 28 Oct).

### c) Report, Results (2)

- One figure (candidates: the stop-codon profile across lines; the A2
  agreement chart; the C3 range).
- One short table (C2/C3).
- Text: the "shared biology, weak differences" story above, with its
  caveats.
