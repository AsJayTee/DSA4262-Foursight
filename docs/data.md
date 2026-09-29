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

## Read depth

**Depth is the single most important variable in this dataset, so it is worth
being precise about what it means before any number below is read.**

Depth is *how many separate RNA molecules gave us a measurement for this exact
site.* Take one particular `A` on one transcript:

| | |
|---|---|
| **depth = 1** | Exactly 1 RNA molecule passed through the nanopore covering that position. We have a single observation of what that site looked like. |
| **depth = 3** | We measured that exact position on 3 separate RNA molecules. |
| **depth = 50** | We measured it on 50 separate RNA molecules. |

**The part that is easy to get wrong:** those 50 reads are *not* 50
measurements of the same physical molecule. They are 50 **different copies** of
that RNA from the cell, each one passing through the pore once and being
measured once. Depth is a count of molecules, not a count of repeated readings.

That distinction is what makes this a Multiple Instance Learning problem. The
m6A tag is attached to *individual molecules*, and only some copies of a
modified site carry it. So at a site labelled positive, some of those 50 reads
come from modified molecules and some do not — and we are never told which.

Two consequences follow, and both are load-bearing:

- **Depth sets how much evidence we have.** At depth 1, if the site is modified
  at 50% stoichiometry, there is roughly a coin-flip chance the one molecule we
  measured was not modified at all. At depth 50 we have almost certainly
  sampled several modified molecules. Low depth is genuinely less information,
  not merely noisier information.
- **Depth is a property of the sequencing run, not of the site.** It is driven
  by how abundant that transcript was in the cell and how much sequencing was
  done. Abundant transcripts get hundreds of reads; rare ones get one or two.
  The same site can have depth 200 in one dataset and depth 1 in another.

**Every site in this training set has at least 20 reads, and that floor is an
artefact of how the course prepared the data, not a property of nanopore
sequencing.** Real datasets have no such floor — SG-NEx samples have a median
depth of about 3. Any claim about model performance made on this data is a
claim about the depth ≥ 20 regime only. See [../GAPS.md](../GAPS.md).

To find out what a model does outside that regime, drop reads from the held-out
data and score it again:

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --depth-sweep
```

`m6a.data.subsample_reads` does the dropping, keyed on
(seed, depth, transcript, position) so the same site yields the same reads on
every machine and regardless of what else is being computed. It is here rather
than in the evaluation code because depth-augmented *training* will want the
same function. Labels stay valid under subsampling: they come from m6ACE-Seq,
not from the nanopore reads, so removing reads changes how much evidence the
model has and not what is true about the site.

**This variance is the modelling problem.** Only a fraction of reads at a
modified site actually carry the modification, so mean-pooling washes out the
signal. See `src/m6a/features/quantiles.py`.

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

## Later releases: `data1` and `data2`

Released on Canvas on 2026-09-29 as `Team_Project_export.zip`. In the R2
bucket under `data1/` and `data2/`; fetch with
`python scripts/download_data.py --set data1` (or `--set data2`), which puts
them in `data/raw/data1/` and `data/raw/data2/`
([0027](decisions/0027-later-data-releases-keep-their-folder.md)).

Each folder has a signal file in exactly `dataset0.json.gz`'s format (9
features per read, 7-mer, central 5-mer always DRACH) and a `data.info` in
**m6Anet's format**: `transcript_id, transcript_position, n_reads, label,
start, end`. `start`/`end` are byte offsets into the JSON. Unlike
`data.info.labelled`, there is **no `gene_id` column**.

### `data1`: a second labelled run of largely the same sites

| | `data1` | `dataset0` |
|---|---:|---:|
| sites | 90,810 | 121,838 |
| transcripts | 4,451 | 5,333 |
| reads | 7,907,952 | 11,027,106 |
| positive sites | **6,593 (7.26%)** | 5,475 (4.49%) |
| reads per site | min 20, median 40, p95 312, max 994 | min 20, median 47, p95 304, max 991 |
| sites per transcript | median 17 | median 19 |

- **Built the same way as `dataset0`**: whole transcripts (only 1.2% of sites
  sit on transcripts with fewer than 5 sites), the same >= 20-read floor.
- **74.1% of its sites are also in `dataset0`** (83.3% on a shared transcript),
  with different read counts: a different sequencing run of largely the same
  sites.
- **Its labels are not `dataset0`'s.** On the 67,320 shared sites they agree
  94.25% of the time, but only 2,136 of the 3,291 `dataset0` positives there
  (65%) are positive in `data1`, and `data1` calls 2,717 sites positive that
  `dataset0` calls negative. So it was labelled from a different experiment,
  cell line or threshold. Which one is not documented.
- **No gene ids.** Using it in the gene-grouped split means mapping transcripts
  to genes; 83% of its sites are on transcripts `dataset0` already maps.

### `data2`: an in-vitro mixing series, labelled by fraction modified

| transcript | label | median reads |
|---|---:|---:|
| `tx_id_0` | 1.00 | 1,124 |
| `tx_id_1` | 0.95 | 550 |
| `tx_id_2` | 0.75 | 589 |
| `tx_id_3` | 0.70 | 640 |
| `tx_id_4` | 0.50 | 850 |
| `tx_id_5` | 0.25 | 672 |
| `tx_id_6` | 0.00 | 1,205 |

- **All seven are the same synthetic sequence**: 189 sites each, one every 10
  bases from 0 to 1880, identical 7-mer at every position. The label is the
  **fraction of molecules modified**, one value per transcript, not a 0/1
  label per site.
- **The signal tracks the fraction.** Mean centre-position current, averaged
  over sites, moves away from the 0% sample in proportion: 3.55 pA at 100%,
  2.65 at 75%, 1.77 at 50%, 0.89 at 25%.
- **Use it to test and calibrate, not to train.** It is a direct test of
  whether a site score rises with modification stoichiometry. It is synthetic,
  one sequence, and every candidate A is modified on every modified molecule,
  which real cells never do.
