# 0036. Row browses precomputed transcript predictions

- Date: 2026-10-17
- Status: Implemented on dashboard-josh
- Scope: dashboard UI and separate data-preparation utilities only.

Row loads one transcript at a time from a small manifest or a local JSON export.
Candidate sites and coverage share a transcript-coordinate viewport with pan,
zoom and keyboard-accessible site selection. Coordinates remain exactly those
of the input; no unverified genomic mapping or full transcript length is inferred.

`export_browser_data.py` exports selected transcripts from public SG-NEx scoring
CSVs. `prepare_browser.py` optionally runs the existing saved ensemble through
the existing score_sgnex functions on a complete processed input, then exports.
Neither changes model definitions, weights or rank aggregation. Selecting sites
before scoring is avoided because it would change graph context and ranks.

Per-site SG-NEx inputs/results are absent from this checkout. The UI explicitly
offers a synthetic example, never silently substituting it for biological data.
No course reads or labels are exported. Raw scores and within-file ranks are
labelled as uncalibrated; no modification classification threshold is invented.

The static GitHub Pages deployment serves prepared outputs; inference is a local
preparation step, not an on-click browser computation. Gene-name search becomes
available when gene_name/gene_id annotations are added to manifest entries;
without annotation, search uses transcript IDs. Gene structure is not fabricated.
