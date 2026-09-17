# Foursight — m6A RNA modification prediction

Predicting N6-methyladenosine (m6A) RNA modifications from Nanopore direct
RNA-seq signal data, for DSA4262 (NUS / Genome Institute of Singapore).

Given nanopore signal features for every read aligned to a candidate DRACH site,
the model predicts the probability that the site carries an m6A modification.

---

## Quick start: making predictions

Tested on a clean Ubuntu machine with Python 3.10+. Nothing below needs
credentials, network access, or a GPU.

```bash
git clone https://github.com/TEAM/DSA4262-Foursight.git
cd DSA4262-Foursight

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

python scripts/predict.py \
    --model models/final \
    --input data/sample/sample.json.gz \
    --output predictions.csv
```

Expected output:

```
1,000 sites scored -> predictions.csv  (0.5s, model=lightgbm_quantiles)
```

`data/sample/sample.json.gz` is a 1,000-site test dataset included in this repo
so the above runs immediately. To predict on your own data, point `--input` at
any `data.json` or `data.json.gz` in m6Anet's processed format.

> The sample is drawn from the training set, so scores on it are optimistic. It
> is there to prove the code runs, not to measure accuracy.

### Output format

```csv
transcript_id,transcript_position,score
ENST00000000412,769,0.0010114002630883252
ENST00000000412,2195,0.006718969056015129
```

One row per site in the input, with `score` the probability of m6A in `[0, 1]`.

---

## Results

Out-of-fold performance on the Hct116 training set (121,838 sites, 4.49%
positive), 5-fold cross-validation **grouped by gene** so no gene appears in
both training and validation.

| Model | Features | ROC AUC | PR AUC | vs. random |
|---|---|---:|---:|---:|
| Random classifier | — | 0.500 | 0.045 | 1.0× |
| Logistic regression (baseline) | pooled mean/std | 0.9008 | 0.4121 | 9.2× |
| LightGBM | pooled mean/std | 0.9125 | 0.4634 | 10.3× |
| **LightGBM (shipped)** | **read quantiles** | **0.9169** | **0.4759** | **10.6×** |

At 4.49% positives, **PR AUC is the metric that discriminates.** ROC AUC
flatters everything here — all three models sit above 0.90, while their PR AUCs
differ by a much wider margin. Rank experiments on PR AUC.

Reproduce any row:

```bash
python scripts/train.py --config configs/quantiles.yaml
```

**One pooled number is not enough to compare two models with.** Folds differ in
size and positive rate, so the same model scores 0.4548 on fold 0 and 0.5081 on
fold 1. `scripts/evaluate.py` keeps the per-fold numbers and compares two runs
fold by fold, which is the test that can actually tell an improvement from
noise:

```bash
python scripts/evaluate.py --config configs/quantiles.yaml --compare-features pooled_v1
```

Two things that table above does not show, and which the harness measures:

- **The scores are not calibrated.** Mean predicted probability is 0.0809
  against an actual positive rate of 0.0449 - a **1.80x overcount**. PR AUC and
  ROC AUC are rank-based and cannot see it, but any use of a score *as a
  probability* is wrong by that factor.
- **Performance collapses below 20 reads per site.** Every training site has at
  least 20 reads; SG-NEx has a median of 3. At one read the model scores 0.153,
  which is what a motif-only classifier with no signal data at all scores
  (0.1537). See [read depth](docs/data.md#read-depth) and
  `scripts/evaluate.py --depth-sweep`.

---

## Training a new model

Needs the training extras and access to the data:

```bash
pip install -e ".[train]"
cp .env.example .env               # fill in — ask in the Telegram groupchat
python scripts/download_data.py    # pulls the training set from R2
python scripts/train.py --config configs/quantiles.yaml
```

Or on a fresh Ronin instance, all of the above in one command from your laptop:

```bash
./setup_remote.sh -i ~/.ssh/yourkey.pem ubuntu@<instance-host>
```

Safe to re-run at any time — it resets the instance to a known-good state and
skips work already done.

See [docs/setup.md](docs/setup.md) for the full walkthrough, including key
permissions and the Windows notes. **On Windows, use Git Bash, not PowerShell.**

### Everyday commands

| Command | What it does |
|---|---|
| `make doctor` | Check this machine is set up; says what is missing |
| `make smoke CONFIG=configs/x.yaml` | Full pipeline on 5,000 sites, ~4s, no W&B |
| `make train CONFIG=configs/x.yaml` | Train **and** evaluate (~100s); both go to one W&B run |
| `make evaluate CONFIG=configs/x.yaml EVAL='--compare-features pooled_v1'` | Compare two runs, paired fold by fold and within each depth band |
| `... EVAL='--compare-features pooled_v1 --repeats 10'` | 50 paired observations instead of 5, when it is too close to call |
| `make predict INPUT=... OUTPUT=...` | Score a dataset |
| `make test` | Run the test suite |
| `make sample` | Rebuild `data/sample/` |

---

## For the team

**The default way to work in this repo is through a coding agent** — Claude
Code, Codex, whatever you use. You describe an idea; the agent writes it, proves
it runs with `make smoke`, and hands you a command to run on Ronin.

You do not need to read this codebase to contribute to it. To understand,
change, or extend anything, **ask your agent** — [AGENTS.md](AGENTS.md) tells it
how the repo is organised, and it can read and explain any part of it in
context. Reading the source yourself is always an option, never a prerequisite.

| File | What it gives you |
|---|---|
| [docs/setup.md](docs/setup.md) | Getting a machine working — **Ronin walkthrough, and the Windows notes** |
| [docs/running-experiments.md](docs/running-experiments.md) | How to actually run an experiment, with or without an agent |
| [GAPS.md](GAPS.md) | The backlog: what is missing, weak, or untested, with evidence |
| [docs/project-requirements.md](docs/project-requirements.md) | What the project is graded on — tasks, formats, deadlines, assessment |
| [docs/data.md](docs/data.md) | Data dictionary and measured statistics |
| [AGENTS.md](AGENTS.md) | Conventions your agent follows — and why |
| [docs/decisions/](docs/decisions/) | Why the pipeline is built the way it is, one file per decision |
| [analysis/m6anet/README.md](analysis/m6anet/README.md) | The m6Anet benchmark, and two traps waiting in it |

---

## Repository layout

```
src/m6a/              the package — production code
  data.py             reading signal JSON and labels, the gene-level split,
                      read subsampling
  features/           feature extractors (one file per idea)
  models/             models (one file per idea)
  evaluation.py       metrics, stratified metrics, calibration, submission checks
  crossval.py         the gene-grouped folds and the out-of-fold table
  compare.py          paired comparison between two runs (needs scipy)
  feature_cache.py    disk cache for extracted features, in .cache/features/
  registry.py         name -> implementation lookup
scripts/              command-line entry points
configs/              one YAML per experiment
models/final/         the model used for grading
data/sample/          the committed test dataset
analysis/             exploratory work: SG-NEx (Task 2), m6Anet benchmark
tests/                guardrails, including an end-to-end smoke test
docs/                 setup, data dictionary, workflow
  decisions/          why the shared pipeline is shaped the way it is
```

Adding an experiment means adding one file to `features/` or `models/` plus one
config YAML. Nothing existing gets edited. See [AGENTS.md](AGENTS.md).

---

## The data

Each line of `data.json` is one transcript position and every read aligned to it:

```json
{"ENST00000000233": {"244": {"AAGACCA": [[0.00299, 2.06, 125.0, ...], ...]}}}
```

Nine features per read — dwell time, signal standard deviation and mean current
— for each of the three positions in the 7-mer window (−1, 0, +1). The central
5-mer is always one of the 18 DRACH motifs.

Training set: 121,838 sites, 11.0M reads, 3,852 genes, from the Hct116 colon
cancer cell line. Labels come from m6ACE-Seq. Full dictionary in
[docs/data.md](docs/data.md).

Raw data is never committed — it lives in a private R2 bucket and is fetched by
`scripts/download_data.py`.

---

## License and attribution

Coursework for DSA4262, National University of Singapore. The SG-NEx data is
described in Chen et al., *Nature Methods* 22, 801–812 (2025). m6Anet is
described in Hendra et al., *Nature Methods* 19, 1590–1598 (2022).
