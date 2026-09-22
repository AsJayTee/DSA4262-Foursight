# SG-NEx

Task 2 work: what is in `s3://sg-nex-data`, and what shape it is in.

The bucket is **public and needs no credentials**. Nothing here needs an AWS
key, a `.env`, or the R2 bucket — the S3 client is explicitly unsigned.

## `survey.py`

```bash
# what cell lines and samples exist, and which are ready to predict on
python analysis/sgnex/survey.py catalogue

# the read-depth distribution of one m6Anet-processed sample
python analysis/sgnex/survey.py depth SGNex_A549_directRNA_replicate6_run1

# ...and how our own labelled sites look inside it
python analysis/sgnex/survey.py depth SGNex_Hct116_directRNA_replicate3_run4 \
       --labels data0/data.info.labelled
```

Every SG-NEx number in [GAPS.md](../../GAPS.md) comes from one of those two
commands. It exists because they used to come from a crawl that was not in the
repo, so nobody could re-derive or challenge them — and when the crawl was
finally repeatable it turned out SG-NEx has **14 cell lines, not 7**. The 7 are
just the subset somebody else has already run m6Anet's dataprep over.

Needs boto3: `pip install -e '.[train]'`.

## Why it reads `data.readcount`

Each m6Anet sample directory holds four files. `data.json` is the one
`scripts/predict.py` consumes and it is ~2 GB. `data.readcount` is a plain CSV —
`transcript_id,transcript_position,n_reads`, one line per site — and about
34 MB.

Read depth is the single most important variable in this project
([docs/data.md](../../docs/data.md#read-depth)), and `data.readcount` answers
every depth question for 1/60th of the bytes. A depth distribution costs well
under a minute and no download at all.

## What it is not

Not part of the harness. It talks to the network, nothing imports it, and it is
read-only — it never writes to the bucket. It is closer to the scripts in
[../evaluation/scratch/](../evaluation/scratch/) than to `scripts/evaluate.py`,
except that it has no hardcoded paths and its numbers are meant to be quoted.

Downloading a sample's `data.json` and predicting on it is not done yet, and
does not belong here when it is — see the Task 2 section of GAPS.md.
