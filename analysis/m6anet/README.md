# m6Anet benchmark

The handout asks us to benchmark our method against m6Anet (Hendra et al.,
2022). This directory holds everything that benchmark needed: converting our
data into m6Anet's input format, running the pretrained model, retraining it on
our own folds, and scoring its predictions with the same code that scores ours.

m6Anet runs in its **own environment**, separate from the main package. Do not
add it to `pyproject.toml`.

---

## Results

| | Cell line 1 (HCT116) PR AUC | ROC AUC | Cell line 2 (HEK293T) PR AUC | ROC AUC |
|---|---:|---:|---:|---:|
| m6Anet, pretrained (`HCT116_RNA002`) | 0.503\* | 0.932\* | 0.356 | 0.819 |
| m6Anet, retrained on our gene folds of cell line 1 | 0.488 | 0.924 | — | — |

\* Not held out. The pretrained model was trained on SG-NEx HCT116 with HCT116
m6ACE-seq labels: the same cell line, assay and pipeline as our training file
(dataset0), and almost certainly many of the same sites. Its cell line 1 score
is therefore an upper bound for m6Anet, not a fair held-out number.

Retraining on our folds scored slightly below the pretrained model, so the
report uses the pretrained model as the comparator. Its cell line 2 score
closely matches the HEK293T result published for m6Anet (0.35 PR AUC / 0.83
ROC AUC), which indicates it was run as intended. The comparison with our
models, including the fair test on a cell line neither was trained on, is in
the report (Results 1) and in `report/findings.md` section 13.

---

## Files

| File | What it does |
|---|---|
| `run_ronin.sh` | The whole benchmark on one Linux instance, in stages (below) |
| `make_m6anet_index.py` | Converts `dataset0.json.gz` into the `data.json` + `data.info` pair m6Anet reads |
| `prepare_cv.py` | Lays out our gene-grouped, seed-4262 folds in the form m6Anet's trainer expects |
| `score_m6anet.py` | Scores m6Anet's `data.site_proba.csv` with `m6a.evaluation`, optionally paired against one of our runs |
| `xsrc_m6anet.py` | Puts m6Anet on the cross-source split (decisions 0029, 0032) and measures every model against it |
| `m6anet_pretrained__xsrc.json` | The committed output of `xsrc_m6anet.py` |

## Running it

```bash
bash analysis/m6anet/run_ronin.sh setup       # install m6Anet in its own venv, ~5 min
bash analysis/m6anet/run_ronin.sh convert     # build data.json + data.info, ~2 min
bash analysis/m6anet/run_ronin.sh pretrained  # pretrained inference, ~1.5 h
bash analysis/m6anet/run_ronin.sh probe       # time 2 epochs before committing to a retrain
bash analysis/m6anet/run_ronin.sh retrain     # 5 folds; hours (read the probe first)
bash analysis/m6anet/run_ronin.sh score       # metrics, logged to W&B
```

`bash analysis/m6anet/run_ronin.sh all` runs every stage unattended, scoring
the pretrained model before the retrain starts so that a late failure still
leaves half the benchmark recorded. It needs the main package installed in
`.venv/` (the helper scripts import `m6a` for the split, labels and metrics)
and the training data downloaded. Everything it generates goes under `data/`,
which is gitignored.

### The install that worked

`uv venv --python 3.8` followed by `m6anet==2.1.0`. `setup` does this for you.

- A plain `python3 -m venv` cannot give Python 3.8: it copies the system
  interpreter (3.12 on Ubuntu 24.04). `uv` downloads 3.8 itself, which is
  lighter than installing conda.
- m6Anet pins `torch==1.6.0`. Where that has no wheel, `setup` installs m6Anet
  without dependencies and adds a modern torch; this was verified on torch
  2.4.1.
- m6Anet imports `pkg_resources` and `ujson` without declaring them, so
  `setup` always installs `setuptools` and `ujson` explicitly.

---

## Two traps in m6Anet's input format

Both were found by reading m6Anet's source (`m6anet/utils/data_utils.py`), and
`make_m6anet_index.py` handles both.

**1. `data.info` is not `data.info.labelled`.** m6Anet needs a `data.info` next
to `data.json`: a byte-offset index with read counts, so it can seek to each
site. The course's `data.info.labelled` has a similar name but is the m6ACE-seq
**labels** (`gene_id,transcript_id,transcript_position,label`). Because m6Anet
seeks into the file, `data.json` must also be uncompressed (625 MB).

**2. Every read needs a trailing id column.** m6Anet treats the last value of
each read as a read identifier. Our reads are nine signal values with no id,
so the converter appends one. The nine signal values themselves already match
m6Anet's layout (dwell time, standard deviation, mean, per position) and need
no rescaling.

In m6Anet 2.0 and later, `data.info` replaced the older separate `data.index`
and `data.readcount` files, so documentation written for version 1 describes a
different layout.

## Reference

Hendra, C., Pratanwanich, P.N., Wan, Y.K. et al. *Detection of m6A from direct
RNA sequencing using a multiple instance learning framework.* Nature Methods
19, 1590–1598 (2022). <https://github.com/GoekeLab/m6anet>
