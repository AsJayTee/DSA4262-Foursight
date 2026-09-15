# Project requirements

DSA4262 Sense-making Case Analysis: Health and Medicine — Data Science in
Genomics, 2026. NUS / Genome Institute of Singapore, Dr Jonathan Göke.

This file is the authoritative record of what the project asks for. It is
transcribed from the course handout and briefing deck, which are not kept in
this repo. **Read this before planning any work.**

---

## The problem

Develop a machine learning method that identifies **m6A RNA modifications**
from **direct RNA-Seq (Nanopore) data**, then apply it to the SG-NEx datasets.

m6A (N6-methyladenosine) is one of the most frequent mRNA modifications. It
occurs at DRACH motifs, is written by METTL3/METTL14, and is implicated in
acute myeloid leukemia among other diseases. Nanopore direct RNA sequencing
reads raw electrical current as RNA passes through a pore; a methylated
adenine perturbs that current, so the modification is inferred from signal
distortion rather than observed directly.

Labels come from **m6ACE-Seq** on the **Hct116** colon cancer cell line.

---

## Task 1 — a classifier

Write a computational method that predicts m6A modification from direct
RNA-Seq data. It must be able to train a new model and to make predictions on
unseen test data.

### (1) Model training

- **Input:** direct RNA-Seq data processed by m6Anet (JSON), plus m6A labels.
- **Output:** the trained model, in any format.

### (2) Prediction

- **Input:** direct RNA-Seq data processed by m6Anet (JSON). Optionally the
  pre-trained model as an argument; it may also be accessed within the code.
- **Output:** predicted m6A sites, in the CSV format below.

**The prediction script is evaluated by other students on an AWS Ubuntu
machine.** It must run from a clean clone with no credentials.

### (3) Benchmarking against m6Anet

Benchmark the method against m6Anet, considering how the two methods differ.
The handout notes m6Anet requires Python 3.8, installed via conda rather than
pip.

### Code availability and documentation

- Accessible through a **public GitHub repository** at submission time.
- The code must be **executable**.
- The repository must be **documented**: how to install it and any
  requirements, how to run it, and **a small test dataset** to run the
  prediction script against (not required for the training script).
- Code and documentation are evaluated by other students on an AWS Ubuntu
  machine.

---

## Task 2 — application to SG-NEx

1. Predict m6A modifications in **all direct RNA-Seq datasets from SG-NEx**.
2. Describe the results and compare across cell lines. Summarise and visualise
   the observations. **Intentionally open-ended** — the brief asks for
   exploration and whatever insight can be extracted.
3. Create an **interactive data visualisation platform** for other researchers
   to explore the results. It may be hosted on any platform.

SG-NEx data is public, hosted on AWS: <https://github.com/GoekeLab/sg-nex-data>
and <https://registry.opendata.aws/sgnex/>. Release v0.6 covers 113 samples
across 13 cell lines; the deck scopes this task to 5 cancer cell lines.

---

## Data formats

### Input — `data.json`

One line per transcript position, holding every read aligned there:

```json
{"ENST00000000233": {"244": {"AAGACCA": [[0.00299, 2.06, 125.0,
                                          0.01770, 10.40, 122.0,
                                          0.00930, 10.90, 84.1], ...]}}}
```

- `ENST00000000233` — transcript ID.
- `244` — position within that transcript.
- `AAGACCA` — the 7-mer spanning positions 243–245. A 5-mer sits in the pore at
  any moment and shifts one base at a time, so this covers `AAGAC` (243),
  `AGACC` (244) and `GACCA` (245). **The middle 5-mer is always one of the 18
  DRACH motifs** (D = A/G/T, R = A/G, H = A/C/T).
- Nested lists — one per read, 9 features each: dwell time, signal standard
  deviation and mean current, for position −1, then 0, then +1.

### Labels — `data.info`

```csv
gene_id,transcript_id,transcript_position,label
ENSG00000004059,ENST00000000233,244,0
```

`label` is 1 if the position carries m6A, 0 otherwise. Join to the signal data
on `transcript_id` and `transcript_position`.

### Output — predictions

```csv
transcript_id,transcript_position,score
ENST00000005260,425,0.03794821493241728
```

- Must be CSV, comma-separated.
- `transcript_id` and `transcript_position` **must exactly match** those in the
  input `data.json`.
- `score` must be a float in `[0, 1]` — the probability the site carries m6A.

---

## Deadlines

| Date | What |
|---|---|
| **7 Oct, 23:59** | Intermediate leaderboard submission |
| **28 Oct, 23:59** | Final leaderboard, report, interactive visualisation, code |
| **6 Nov, 23:59** | Individual teamwork summary (Canvas, confidential) |

New data (JSON) is released for each leaderboard, with further information at
the time. **No questions are answered between 5 Oct 21:00 and 7 Oct**, or
between 26 Oct 21:00 and 28 Oct — ask during the classes on 5 Oct and 26 Oct.

---

## The report

Maximum **5 pages**, excluding an optional title page, optional table of
contents, and references. **Arial 11**; Arial 9–11 for figure captions, tables
and references.

Required sections:

- Title
- Introduction
- Problem statement
- Methods — the ML model used for the final submission, and the datasets
  analysed for Task 2 with how they were analysed
- **Results (1): model evaluation** — performance on the training data using
  independent training and test data; comparison against a **simple baseline
  model** and against **m6Anet**; optionally comparison against intermediate
  models to show improvement
- **Results (2): application to SG-NEx**
- Discussion
- References
- Code availability — a link to the repository, which must include complete
  documentation for predicting m6A probabilities on a test dataset

### AI use documentation

AI tools are allowed for all tasks. No code or text may be copied from a public
source without acknowledgment. Document AI use as a table:

| AI Tool used | Prompt and output | How the output was used |
|---|---|---|

In addition, document **at least two instances where an AI tool's output was
wrong, misleading, or rested on an unverified assumption**; how it was
detected; and what was done about it. A team reporting no such instances must
explain how it verified that.

---

## Assessment (65% total)

| Component | Weight |
|---|---:|
| Report, including interactive data visualisation | 50% |
| Documentation — code executable, results reproducible | 5% |
| Model performance on an unknown test dataset (ROC AUC and PR AUC), against a random classifier and a simple baseline | 5% |
| Individual component | 5% |

Ranking against other teams is **not** used for grading.

The individual component covers: in-person attendance and active contribution;
a specific contribution from each student in **report writing, model
development, and implementation**; and confidential peer reports.

**AWS overspending leads to a 5% deduction.** No-shows also lead to a
deduction.

Evaluation criteria: clarity and presentation, domain understanding,
methodology, innovativeness and creativity. The briefing highlights
"evaluating tools with rigour and critical judgement" — whether the team fully
understands the dataset and its limitations, and whether it understands and has
clearly communicated the model's limitations.

---

## Logistics

- Each team gets **US$100 AWS credit per student**, managed at team level — any
  member can spend the whole budget.
- Four hackathon sessions. For each, the team creates a shared Google document
  in a Zulip channel shared with Jonathan and Clare, stating the aims for the
  session and who is working on what; and after the session (by 22:00) what was
  achieved and what each member will do before the next one.
- A private Zulip channel `#proj-teamname` with Jonathan Göke and Clare
  Robinson.
- Final leaderboard results are revealed in the last lecture. The top 3 teams
  win a prize and certificates.
- Each student is assigned to evaluate other teams' projects.

---

## References

- Chen, Y. et al. *A systematic benchmark of Nanopore long-read RNA sequencing
  for transcript-level analysis in human cell lines.* Nature Methods 22,
  801–812 (2025). <https://doi.org/10.1038/s41592-025-02623-4>
- Hendra, C., Pratanwanich, P.N., Wan, Y.K. et al. *Detection of m6A from
  direct RNA sequencing using a multiple instance learning framework.* Nature
  Methods 19, 1590–1598 (2022). <https://doi.org/10.1038/s41592-022-01666-1>
- Koh, C.W.Q. et al. m6ACE-Seq — the protocol behind the training labels (2019).
