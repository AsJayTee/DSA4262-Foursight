# Foursight — m6A RNA modification prediction

Predicting N6-methyladenosine (m6A) RNA modifications from Nanopore direct
RNA-seq data, for DSA4262 (NUS / Genome Institute of Singapore).

The input is the m6Anet-processed `data.json`: for each candidate site (a
position on a transcript at a DRACH motif), the signal features of every read
aligned there. The output is a CSV with one score per site; a higher score
means the site is more likely to carry m6A.

- **Interactive results platform (Task 2):** <https://asjaytee.github.io/DSA4262-Foursight/>
- **Report:** submitted on Canvas

---

## Running the prediction script

Only prediction needs to be run to evaluate this repository; training is not
required. Choose **one** of the two options below. Neither needs a GPU or any
credentials, and both use a 1,000-site test dataset that is included in the
repository.

### Option A: Google Colab (no setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VWdilv97E05UyFfZ5eSe1ejZl9fbgV_M?usp=sharing)

Open the notebook and choose **Runtime → Run all**. It clones this repository,
installs it, runs the prediction script on the test dataset, checks the output
format and displays the predictions. This takes about a minute.

### Option B: your own machine (Ubuntu)

Tested from a fresh clone on Ubuntu 24.04 (Python 3.12) and Ubuntu 22.04
(Python 3.10).

Requires Python 3.10 or newer and git. On a fresh AWS Ubuntu instance, install
them first (Ubuntu does not include `venv` by default):

```bash
sudo apt-get update && sudo apt-get install -y git python3-venv python3-pip
```

Then:

```bash
git clone https://github.com/AsJayTee/DSA4262-Foursight.git
cd DSA4262-Foursight

python3 -m venv .venv
source .venv/bin/activate
pip install .

python scripts/predict.py \
    --model models/final \
    --input data/sample/sample.json.gz \
    --output predictions.csv
```

Expected output (the time varies by machine):

```
1,000 sites scored -> predictions.csv  (2.1s, model=site_graph_ensemble)
```

`pip install .` installs five packages (numpy, pandas, scikit-learn, lightgbm
and pyyaml). The model itself runs in plain numpy, so torch is not needed.

<details>
<summary>On macOS or Windows</summary>

The same commands work on macOS. On Windows, activate the environment with
`.venv\Scripts\activate` instead of `source .venv/bin/activate`, and use
`python` instead of `python3`. If `git clone` fails with "Filename too long",
run `git config --global core.longpaths true` and clone again.

</details>

### Checking the output

```bash
python -c "from m6a.evaluation import validate_submission; problems = validate_submission('predictions.csv', 'data/sample/sample.json.gz'); print(chr(10).join(problems) or 'OK: predictions.csv has the required format and one row per input site')"
```

This prints `OK` if the columns are exactly
`transcript_id,transcript_position,score`, every score is a number in [0, 1],
and the CSV has exactly one row for every site in the input file. Otherwise, it
lists each problem.

The first rows of `predictions.csv` look like this:

```csv
transcript_id,transcript_position,score
ENST00000000412,769,0.22266666666666665
ENST00000000412,2195,0.309
```

`transcript_id` and `transcript_position` are copied from the input. `score` is
each site's rank within the file, averaged over the six networks of the
ensemble and scaled to [0, 1]. It orders sites by how likely they are to be
modified, but it is **not a calibrated probability**: do not count modified
sites by applying a threshold to it.

The test dataset shows that the code runs; it does not measure accuracy. Its
sites are taken from the training data, and its 1,000 sites are spread over
884 transcripts, so few of them have the neighbouring sites the model uses. See
[Results](#results) for held-out performance.

### Predicting on other data

`--input` accepts any `data.json` or `data.json.gz` in the m6Anet-processed
format. For example, to predict on one full SG-NEx sample from the public
[SG-NEx bucket](https://registry.opendata.aws/sgnex/), which needs no
credentials:

```bash
curl -o sgnex.json https://sg-nex-data.s3.amazonaws.com/data/processed_data/m6Anet/SGNex_Hct116_directRNA_replicate3_run4/data.json
python scripts/predict.py --model models/final --input sgnex.json --output sgnex_predictions.csv
```

The file is 1.5 GB and holds 1,302,514 sites. On an 8-core laptop, prediction
took about 5 minutes and used 5.6 GB of memory at its peak, so use a machine
with at least 8 GB of RAM. Expect it to take longer on fewer cores.

---

## The method

The shipped model is an ensemble of three graph-network designs, trained with
two random seeds each (six networks). Every design starts the same way: a
DeepSets encoder summarises each site's reads into a vector and an "own-reads"
score. Each read is one RNA molecule, so a site's reads are a set of
independent measurements ([read depth](docs/data.md#read-depth)). The designs
then differ in how they use the other candidate sites on the same transcript
(its *neighbours*):

- **Two-head H2GCN:** an H2GCN-style graph network (Zhu et al., NeurIPS 2020)
  that keeps a site's own evidence separate from its neighbours'. It has one
  output per labelled cell line and ships their mean.
- **Residual + dropout (`res_gate`):** the site's own-reads score, plus a
  correction from its neighbours within 150 nt, weighted by distance.
- **Scalar messages + dropout (`scalar_drop`):** each neighbour passes a single
  number, its own-reads score, which corroborates or contradicts the site.

The last two were trained with neighbours randomly removed, so they do not
depend on any single neighbour; this is what lets them transfer to a cell line
they were not trained on. The model was trained on both labelled releases
(HCT116 and HEK293T). Design choices are recorded in
[decision 0033](docs/decisions/0033-ship-a-site-graph-ensemble-scored-in-numpy.md)
and [decision 0034](docs/decisions/0034-ship-constrained-corroboration-designs.md).

---

## Results

Held-out performance from 5-fold cross-validation **grouped by gene**, so that
no gene appears in both training and validation, with all models trained on
both cell lines:

| Model | HCT116 PR AUC | HCT116 ROC AUC | HEK293T PR AUC | HEK293T ROC AUC |
|---|---:|---:|---:|---:|
| Random classifier | 0.045 | 0.500 | 0.072 | 0.500 |
| Simple baseline (logistic regression) | 0.415 | 0.899 | 0.331 | 0.813 |
| Quantiles + LightGBM | 0.472 | 0.918 | 0.392 | 0.829 |
| m6Anet (pretrained on HCT116) | 0.503\* | 0.932\* | 0.356 | 0.819 |
| DeepSet (reads only, no neighbours) | 0.541 | 0.936 | 0.429 | 0.840 |
| **Final model** | **0.607** | **0.958** | **0.460** | **0.865** |

\* Not held out: m6Anet was trained on HCT116 data.

Only 4.5% (HCT116) and 7.2% (HEK293T) of sites are modified, so PR AUC is the
more informative metric; ROC AUC is high for every model.

Performance depends strongly on read depth. Every training site has at least 20
reads, while SG-NEx sites have a median of 3. With each site's reads randomly
thinned, the final model's PR AUC falls as follows:

| PR AUC | 1 read | 3 reads | 10 reads | All (≥ 20) |
|---|---:|---:|---:|---:|
| Final model, HCT116 | 0.333 | 0.451 | 0.534 | 0.607 |
| Quantiles + LightGBM, HCT116 | 0.140 | 0.221 | 0.368 | 0.472 |
| Final model, HEK293T | 0.279 | 0.341 | 0.397 | 0.460 |
| Quantiles + LightGBM, HEK293T | 0.136 | 0.200 | 0.306 | 0.392 |

The full comparison, significance tests and limitations are in the report.

---

## Training a new model

Not needed for evaluation. Training needs the training extras and access to the
labelled data, which is unpublished and held in a private bucket:

```bash
pip install -e ".[train]"
cp .env.example .env               # credentials, shared within the team
python scripts/download_data.py    # pulls the training set
python scripts/train.py --config configs/quantiles.yaml
```

See [docs/setup.md](docs/setup.md) for the full walkthrough, including remote
instances and Windows notes.

---

## Repository layout

```
src/m6a/              the package
  data.py             reading signal JSON and labels, the gene-level split
  features/           feature extractors (one file per idea)
  models/             models (one file per idea), including the shipped ensemble
  evaluation.py       metrics, calibration, the output-format check
  crossval.py         gene-grouped folds and out-of-fold predictions
  compare.py          paired comparison between two runs
  registry.py         name -> implementation lookup
scripts/              command-line entry points (predict.py, train.py, evaluate.py)
configs/              one YAML per experiment
models/final/         the model used for prediction
data/sample/          the test dataset
analysis/             SG-NEx (Task 2), the m6Anet benchmark, model comparisons
dashboard/            the interactive results platform
report/               the report and the scripts that generate its figures
tests/                tests, including an end-to-end smoke test
docs/                 setup, data dictionary, workflow, design decisions
```

---

## For the team

Work in this repo goes through a coding agent (Claude Code, Codex, or similar):
you describe an idea, the agent implements it, checks it with `make smoke`, and
hands you a command to run. [AGENTS.md](AGENTS.md) describes how the repo is
organised and the conventions the agent follows.

| File | What it gives you |
|---|---|
| [docs/setup.md](docs/setup.md) | Getting a machine working, including remote instances and Windows |
| [docs/running-experiments.md](docs/running-experiments.md) | How to run an experiment, with or without an agent |
| [GAPS.md](GAPS.md) | The backlog: what is missing, weak, or untested, with evidence |
| [docs/project-requirements.md](docs/project-requirements.md) | What the project is graded on |
| [docs/data.md](docs/data.md) | Data dictionary and measured statistics |
| [docs/decisions/](docs/decisions/) | Why the pipeline is built the way it is, one file per decision |
| [analysis/m6anet/README.md](analysis/m6anet/README.md) | The m6Anet benchmark |

| Command | What it does |
|---|---|
| `make doctor` | Check this machine is set up; says what is missing |
| `make smoke CONFIG=configs/x.yaml` | Full pipeline on 5,000 sites, no W&B |
| `make train CONFIG=configs/x.yaml` | Train and evaluate; both go to one W&B run |
| `make evaluate CONFIG=configs/x.yaml EVAL='--compare-features pooled_v1'` | Compare two runs fold by fold |
| `make predict INPUT=... OUTPUT=...` | Score a dataset |
| `make test` | Run the test suite |

---

## License and attribution

Coursework for DSA4262, National University of Singapore. The SG-NEx data is
described in Chen et al., *Nature Methods* 22, 801–812 (2025). m6Anet is
described in Hendra et al., *Nature Methods* 19, 1590–1598 (2022). H2GCN is
described in Zhu et al., *NeurIPS* 33 (2020).
