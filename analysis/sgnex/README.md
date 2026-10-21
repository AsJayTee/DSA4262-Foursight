# SG-NEx

Task 2 work: scoring the public SG-NEx direct RNA-seq samples with the shipped
model, and the analyses built on those scores.

The bucket `s3://sg-nex-data` is **public and needs no credentials**. Nothing
here needs an AWS key, a `.env`, or the R2 bucket.

SG-NEx has 14 cell lines with direct RNA-seq, but only 7 of them (22 samples)
have already been through m6Anet's dataprep, the step that produces the
`data.json` our model reads. Those 22 samples are what Task 2 covers; the rest
would need nanopolish eventalign and m6Anet dataprep first.

## The pipeline

Run in this order. Each step reads the previous step's output.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `score_sgnex.py` | `data.json` of each sample, from S3 | `data/sgnex_scores/` (gitignored) |
| 2 | `explore_lines.py` | `data/sgnex_scores/` | `explore/` (committed) |
| 3 | `export_dashboard_data.py` | `explore/` | `dashboard/public/data/summary.json` |

`survey.py` and `label_mapping_check.py` are checks that sit beside this
pipeline (below).

### 1. `score_sgnex.py`: score every sample

```bash
python analysis/sgnex/score_sgnex.py --line A549                   # one cell line
python analysis/sgnex/score_sgnex.py --all --jobs 3                # every line, 3 at a time
python analysis/sgnex/score_sgnex.py --line Hct116 --limit 20000   # quick test
```

Downloads each sample's `data.json` (1.5-2 GB) and scores it with
`models/final`, in numpy. For each cell line it scores every sample on its own,
then all of that line's samples **pooled**: each site's reads from every sample
combined. Pooling gives each site more reads: the median is 3 in a single
sample and 3-5 after pooling, depending on the line (`explore/summary.md`, A1).

Each site gets two scores:

- `score_rank`: what `scripts/predict.py` outputs, the site's rank within the
  file averaged over the six networks. It orders sites within one sample but
  **cannot be compared between samples**, since every file has 5% of its sites
  in its own top 5%.
- `score_raw`: the mean of the six networks' raw outputs. It is on one fixed
  scale, so it can be compared between samples and cell lines at similar read
  counts. It is still not a calibrated probability.

The run on 10 Oct took 10-15 minutes per sample and about 25 minutes per pooled
cell line on Ronin. The logs are in `logs/`.

### 2. `explore_lines.py`: compare the cell lines

```bash
python analysis/sgnex/explore_lines.py
```

Five preliminary analyses on `score_raw`, each made within a read-count band or
on the same sites, because fewer reads pull raw scores toward the middle:

- **A, trust checks:** reads per site per line; whether two samples of one line
  agree more than samples of two different lines; whether GGACU, the strongest
  known motif, scores highest everywhere.
- **B, along genes:** mean score around the stop codon and by distance to the
  nearest exon junction.
- **C, how much m6A:** the share of sites above cut-offs chosen on cell line 1,
  checked on cell line 2 (where the answer is known), then applied here.
- **D, genes:** genes that score high in every line, against genes high in one
  line only.
- **E, similarity:** correlation between lines at site and gene level.

Results are in [explore/summary.md](explore/summary.md), with the tables and
`overview.png` beside it.

### 3. `export_dashboard_data.py`: data for the dashboard

```bash
python analysis/sgnex/export_dashboard_data.py
```

Turns the committed `explore/` tables into the small JSON the dashboard loads.
It exports aggregate numbers from public SG-NEx data only: never course labels
or reads (see [dashboard/README.md](../../dashboard/README.md)).

## Checks

### `survey.py`: what is in the bucket

```bash
# what cell lines and samples exist, and which are ready to predict on
python analysis/sgnex/survey.py catalogue

# the read-depth distribution of one m6Anet-processed sample
python analysis/sgnex/survey.py depth SGNex_A549_directRNA_replicate6_run1

# ...and how our own labelled sites look inside it
python analysis/sgnex/survey.py depth SGNex_Hct116_directRNA_replicate3_run4 \
       --labels data0/data.info.labelled
```

Needs boto3: `pip install -e '.[train]'`. Every SG-NEx number in
[GAPS.md](../../GAPS.md) comes from one of these commands.

It reads each sample's `data.readcount` rather than `data.json`. That file is a
plain CSV (`transcript_id,transcript_position,n_reads`, one line per site) of
about 34 MB, against about 2 GB for `data.json`, and it answers every read-depth
question. [Read depth](../../docs/data.md#read-depth) is the most important
variable in this project.

### `label_mapping_check.py`: do the coordinates line up?

```bash
python analysis/sgnex/label_mapping_check.py
```

Our labels are matched to SG-NEx sites by transcript and position. If two
files use the same coordinates, the 7-mer at a matched site must be identical,
since a one-nucleotide offset would change it. For each scored sample, this
counts how many of dataset0's 121,838 labelled sites are present, how many have
the same 7-mer, and how many reads they have. Other cell lines are included as
a coordinate check only: dataset0's labels apply to HCT116, not to them.
Writes `label_mapping_check.csv`.
