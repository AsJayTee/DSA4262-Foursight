# m6Anet benchmark

The handout requires us to benchmark our method against m6Anet. This is a
separate environment from the main package — do not add m6Anet to
`pyproject.toml`.

**Owner: lane C. Start this in week 1, not week 5.** Two things below can each
cost a day if discovered late.

---

## Trap 1: `data.info` is not `data.info.labelled`

m6Anet inference needs **two** files in its input directory:

| File | What it is | Do we have it? |
|---|---|---|
| `data.json` | Signal features | Yes — this is `dataset0.json.gz` |
| `data.info` | **A byte-offset index** with read counts, for fast random access | **No** |

The course gave us `data.info.labelled`, which is
`gene_id,transcript_id,transcript_position,label` — the m6ACE-Seq **labels**.
Same name prefix, completely different format and purpose. Pointing m6Anet at
it produces an unhelpful error.

The index is derivable: scan `data.json` once and record each line's byte
offset, length and read count. Write that as `scripts/make_m6anet_index.py`
here when you get to it.

In m6Anet ≥ 2.0 this single `data.info` replaced the older separate
`data.index` / `data.readcount` pair. Documentation written for v1 will
describe the old layout; `m6anet convert` bridges them.

---

## Trap 2: a venv cannot give you Python 3.8

`python3 -m venv` clones the **system** interpreter. On Ubuntu 24.04 that is
3.12, so a venv can never be 3.8. Options, in order of how little they cost:

1. **Try a modern Python first.** Upstream m6Anet claims Python 3.7+, while the
   handout says 3.8-via-conda. The handout may simply be dated. If m6Anet
   installs on 3.10+, use that and skip the rest.
2. `uv venv --python 3.8` — uv downloads the interpreter for you. Lighter than
   a full conda install, and only lane C needs it.
3. Miniforge / conda, as the handout says. The safe fallback.

Record which one worked in this file, so nobody re-derives it.

---

## If m6Anet cannot be installed at all

Say so early. The fallback is benchmarking against m6Anet's published figures
on comparable data, which is a materially weaker claim and has to be stated
honestly in the report. That is a decision for the whole team, not a silent
substitution — and it needs to be made in week 1, not the week before the
deadline.

---

## What this directory should end up containing

```
environment.yml           the working environment, once you find one
make_m6anet_index.py      builds m6Anet's data.info from data.json
run_m6anet.sh             inference over our test split
results/                  m6Anet scores, for comparison in evaluation.py
README.md                 this file — record what actually worked
```

## Reference

Hendra, C., Pratanwanich, P.N., Wan, Y.K. et al. *Detection of m6A from direct
RNA sequencing using a multiple instance learning framework.* Nature Methods
19, 1590–1598 (2022). <https://github.com/GoekeLab/m6anet>
