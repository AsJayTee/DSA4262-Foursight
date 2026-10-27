# 0040. GLORI is the independent check on SG-NEx predictions

- **Date:** 2026-10-27
- **Status:** Accepted (agreed by the team and the course coordinator, 26 Oct)
- **Affects:** `analysis/sgnex/glori_check.py`, `gene_classes.py`, `m6anet_prep.py`,
  `run_m6anet_glori.sh`, `m6anet_glori_compare.py`, `export_glori_stats.py` (all new);
  `dashboard/public/data/glori_stats.json` (new file the Model stats page reads);
  `report/sgnex_glori_findings.md` (the definitions behind every number)

## Context

SG-NEx has no m6A labels, so Task 2's predictions could only be shown to look
plausible (stop-codon and exon-junction patterns, top motif). Low-depth
performance had only ever been simulated by dropping reads
([0003](0003-read-subsampling-in-data.md)), and the m6Anet benchmark on HCT116
favoured m6Anet, whose released model was very likely trained on those sites.

## Decision

Published GLORI site lists (a chemical, antibody-free single-base m6A assay) grade
the SG-NEx predictions in the four lines that have them: A549 and H9 (GSE290408),
HepG2 and HCT116 (GSE301419). GLORI is used **only to grade, never as a model
input**. A site's label is its presence in the authors' list. Negatives are
candidate sites in genes where GLORI called at least one site. One row per genomic
A (the isoform copy with the most reads). HCT116's training sites are excluded,
and HCT116 is kept out of the m6Anet comparison and the dashboard because both
models were trained on it. m6Anet is compared on identical sites and reads.

## Why this and not the alternatives

- **GSE210563** (the GLORI paper's own series) has no SG-NEx line.
- **Aggregators** (m6A-Atlas, RMBase, REPIC) pool calls from mixed methods and
  thresholds; a positive means only "some assay called this once".
- **HEK293T + GLORI** is the best-characterised pairing, but SG-NEx's HEK293T is
  raw signal only and needs nanopolish and m6Anet dataprep first.
- **Treating unlisted sites as negatives everywhere** would count genes GLORI never
  measured as unmodified; restricting to GLORI-covered genes avoids that, at the
  cost of a smaller universe.

## Consequences

- Every GLORI precision, PR AUC and lift is a lower bound: GLORI lists only what it
  detected, and the HepG2 and HCT116 runs were shallow (~7k sites) and came from
  engineered cells carrying an m6A eraser with a control guide.
- GLORI and the course's m6ACE-seq labels agree on fewer than half of HCT116's
  sites, so GLORI is a second measurement, not ground truth.
- The m6Anet comparison needed m6Anet itself (Ronin, about 1.5 h on 64 vCPUs);
  its outputs live under `data/` and are not committed, so re-running the
  comparison needs that run again.

## How to check it still holds

`python analysis/sgnex/glori_check.py` must report position shift 0 with 50-74%
of GLORI sites on candidates, and at most 0.23% at a shift of +-1
(`analysis/sgnex/glori/mapping.csv`). Headline: A549 PR AUC at 20+ reads 0.366
against 0.150 for the motif rule (`analysis/sgnex/glori/metrics.csv`).
