# 0038 — Connect public SG-NEx preview subsets

Status: accepted on dashboard-josh, 2026-10-20

Connect seven first-1,000-row pooled prediction subsets using generated static
JSON. Preserve existing model outputs exactly. No raw reads, course labels,
showcase selection, or model changes are included. The standalone reference
Explorer is replaced by a readable component using the same visual styles.

Use original transcript positions and show annotation metadata, rather than
placing sites on an unverified genomic coordinate system. A matching FASTA and
sequence-position check are needed before exposing genome-mapped site tracks.
This temporarily omits the reference's illustrative exon drawing.

Load one cell line at a time. The manifest describes subset selection; source
CSV values are checked against all 7,000 generated records. Within-file ranks
refer to the full original prediction files. Other reference tabs retain their
separate provenance and synthetic labels.

Validation: production build; seven files with 1,000 unique sites each; exact
numeric score, sequence and read-count comparison against subset CSVs.
Browser interaction testing has not been performed in this restricted session.
