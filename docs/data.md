# Data dictionary

All figures below were measured directly from the training set on 2026-09-15.

## Files

| File | Size | Contents |
|---|---:|---|
| `dataset0.json.gz` | 180 MB (625 MB raw) | Signal features, one site per line |
| `data.info.labelled` | 4.7 MB | m6A labels, one site per row |

Both live in the R2 bucket `dsa4262-group-project` under `course/`, and land in
`data/raw/` via `scripts/download_data.py`. Neither is ever committed.

## `dataset0.json.gz` — signal features

One JSON object per line:

```json
{"ENST00000000233": {"244": {"AAGACCA": [[0.00299, 2.06, 125.0,
                                          0.01770, 10.40, 122.0,
                                          0.00930, 10.90, 84.1], ...]}}}
```

| Level | Meaning |
|---|---|
| `ENST00000000233` | Transcript ID (Ensembl) |
| `244` | Position within that transcript |
| `AAGACCA` | The 7-mer spanning positions 243–245 |
| Inner lists | One per read aligned to this site — 9 features each |

The 7-mer covers three overlapping 5-mers, because a 5-mer sits in the pore at
any moment and shifts one base at a time: `AAGAC` (243), `AGACC` (244),
`GACCA` (245). **The central 5-mer is always one of the 18 DRACH motifs**
(D = A/G/T, R = A/G, H = A/C/T).

### The nine features per read

Three measurements × three positions, in this order:

| Index | Name | Position | Measurement | Observed range |
|---:|---|---|---|---|
| 0 | `dwell_m1` | −1 | Dwell time (s) | 0.0017 – 0.12 |
| 1 | `sd_m1` | −1 | Signal std. dev. | 0.094 – 206 |
| 2 | `mean_m1` | −1 | Mean current (pA) | 73.2 – 153 |
| 3 | `dwell_0` | 0 | Dwell time (s) | 0.0017 – 0.14 |
| 4 | `sd_0` | 0 | Signal std. dev. | 0.044 – 206 |
| 5 | `mean_0` | 0 | Mean current (pA) | 75.4 – 156 |
| 6 | `dwell_p1` | +1 | Dwell time (s) | 0.0017 – 0.10 |
| 7 | `sd_p1` | +1 | Signal std. dev. | 0.136 – 184 |
| 8 | `mean_p1` | +1 | Mean current (pA) | 61.0 – 143 |

## `data.info.labelled` — labels

```csv
gene_id,transcript_id,transcript_position,label
ENSG00000004059,ENST00000000233,244,0
```

`label` is 1 if the site carries an m6A modification (per m6ACE-Seq), else 0.

> **This is not m6Anet's `data.info`.** m6Anet's file of that name is a
> byte-offset index with an entirely different format. See
> [../analysis/m6anet/README.md](../analysis/m6anet/README.md).

## Shape

| | |
|---|---:|
| Sites | 121,838 |
| Reads | 11,027,106 |
| Transcripts | 5,333 |
| Genes | 3,852 |
| Positive sites | 5,475 (4.49%) |
| Genes with ≥1 positive | 1,507 |
| Distinct 7-mers | 288 |
| Distinct central 5-mers | 18 (all DRACH) |

Reads per site: min 20, p25 32, median 47, p75 84, p95 304, max 991, mean 90.5.

**This variance is the modelling problem.** Only a fraction of reads at a
modified site actually carry the modification, so mean-pooling washes out the
signal — this is a Multiple Instance Learning problem in disguise. See
`src/m6a/features/quantiles.py`.

## Joining the two files

Join on `(transcript_id, transcript_position)`.

The two training files happen to be line-for-line aligned — verified, all
121,838 rows in identical order. **Do not rely on this.** Evaluation data
arrives without labels and in its own order, and `m6a.data.align_to_features`
always joins on the key.

## Splitting

Split by `gene_id`, never randomly. A gene has 1.38 transcripts on average (max
9), which share sequence and positions; a random split puts near-duplicate rows
on both sides and inflates AUC. No transcript in this dataset belongs to more
than one gene, so grouping is unambiguous.

`m6a.data.assign_folds` implements this with a fixed seed (4262) and no
dependency on library internals, so the same seed gives the same folds
everywhere. Do not change the seed — see [../AGENTS.md](../AGENTS.md).
