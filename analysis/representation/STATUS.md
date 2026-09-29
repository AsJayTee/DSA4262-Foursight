# DONE 2026-09-28: round 3 finished - no architecture beats deepset

Gate (`results/r3_gate.md`) passed nothing; no 50-observation run was needed.
Results pulled back to `results/diagnose2*`. Nothing is left running on Ronin.
Summary for the team: [docs/catch-up.md](../../docs/catch-up.md).

---

# (history) ROUND 3 running unattended on Ronin

Question: were deepset / attn_mil underfitting, and do residual streams or a
supervised Set Transformer fix it and beat deepset?

- **Fit diagnostic** (`diagnose2.py`): 6 variants x 5 folds -> `results/diagnose2.md`
  (training-site AP vs held-out AP, each paired against deepset@v1).
- **Unattended runner** (`run_r3.py`, nohup, started ~07:00 UTC): waits for the
  diagnostic, applies a gate fixed in advance (best new architecture must beat
  deepset@v1 held-out by >= +0.005 on >= 4/5 folds) -> `results/r3_gate.md`. If one
  passes: its 50-observation cross-fit (~10 h) and combinations ->
  `results_r3/r2.md`, which ends with a head-to-head against deepset.
  **Finished when `results/R3_DONE` exists.** Log: `logs/run_r3.log`.
- **Pull back before terminating:** `analysis/representation/{results,results_r3,logs}`
  and `.cache/representation/cf50/<winner>/`.
- If the runner died: rerun `nohup python -u run_r3.py > logs/run_r3.log 2>&1 &`
  from `analysis/representation/` (with the venv and `M6A_DATA_DIR` set) - finished
  jobs are skipped.

---

# UPDATE 2026-09-27: running on Ronin

Launched ~04:30 instance time (UTC), expected done ~12:00-14:00 UTC.

- **SSH:** the Ronin instance's host and key are shared in the team channel, not here
  (this repo is public).
- **Repo on the instance:** `/mnt/sdd/DSA4262` (copied over SSH, not git). venv `.venv`,
  data `data0/`, `export M6A_DATA_DIR=/mnt/sdd/DSA4262/data0`.
- **Two stages, both `nohup`, both resumable** (`analysis/representation/run_remote.py`):
  - **R1** (`run_remote.py r1`): set_masked, attn_mil, contrastive, deepset_large, and
    longer reruns of read_ae, pos_lstm_ae, pos_attn_ae (25 min/fold) -> probes ->
    `results/summary.md` -> `figures/umap.png`, `figures/tsne.png`. Done when
    `logs/run_r1.log` says `r1 done`.
  - **R2** (`run_remote.py r2`): the decisive test. 5 networks (mlp_hand control,
    deepset, read_ae_pred, deepset_large, attn_mil) x 10 repetitions x 5 folds,
    cross-fitted (`crossfit50.py`), then LightGBM combinations (`combine50.py`)
    against `hand` and against depth-augmented `everything`, also at 1 and 3 reads.
    Done when `logs/run_r2.log` says `r2 done`; answer in `results/r2.md`.
- **Check progress:**
  `ls /mnt/sdd/DSA4262/.cache/representation/cf50/*/*.npz | wc -l` (of 250), then
  `ls analysis/representation/results/r2/*.json | wc -l` (of 50).
  `grep -h FAILED analysis/representation/logs/*.log` should be empty.
- **If a stage died:** rerun the same `nohup python -u run_remote.py r1|r2 ...`
  command from `analysis/representation/` - finished work is skipped.
- **Pull results back BEFORE terminating the instance** - nothing goes to W&B:
  `analysis/representation/{results,figures,logs}` (small). The site vectors under
  `.cache/representation/` are large and optional.

Then: write REPORT.md from `results/summary.md`, `results/r2.md` and the figures.

---

# Status - stopped 2026-09-26, night

Stopped on request mid-run. Nothing is lost that is recorded below; the
in-progress steps restart cleanly.

## Done (full data, canonical split, 5 folds - a screen, not the harness's 50)

Screen protocol (PLAN.md): encoder on half A of each training fold, probes on
half B. `hand` alone, LightGBM: **0.5351**. Rotating `hand` drops the tree to
0.4617 - the axis-dependence test works.

| encoder | uses labels | alone, LightGBM | added to hand, LightGBM (screen) |
|---|---|---:|---|
| `random` (control) | no | 0.2635 | - |
| `mlp_hand` | yes | 0.4519 | +0.0092, 5/5 (label confound) |
| `deepset` | yes | 0.4791 | +0.0309, 5/5 (label confound) |
| `deepset_nokmer` (ablation) | yes | 0.4632 | +0.0236, 5/5 (label confound) |
| `read_ae` | no | 0.2293 | -0.0012, 2/5 - null |
| `read_ae_pred` | yes | 0.4616 | +0.0276, 5/5 (label confound) |
| `pos_lstm_ae` | no | 0.2687 | +0.0039, 5/5, p = 0.19 |
| `pos_attn_ae` | no | 0.2311 | +0.0020, 4/5, p = 0.72 - null |
| **`kmer_norm`** (no network) | no | 0.3731 | **+0.0168, 5/5, p = 0.025 - clean** |

Full tables: `results/summary.md`.

**Cross-fitted, every training label on both sides (removes the confound):**

| | hand | + network score | difference |
|---|---:|---:|---|
| `mlp_hand` (stacking control) | 0.5493 | 0.5507 | +0.0014, 3/5, p = 0.73 - null |
| **`deepset`** | 0.5493 | **0.5729** | **+0.0236, 5/5, p = 0.026** |
| `read_ae_pred` | fold 0 only: 0.5490 | 0.5769 | +0.028 on fold 0 (stopped) |

(`results/crossfit_deepset_run1.json` is the run quoted above.)

## Not done

- **Screen:** `set_masked`, `attn_mil`, `contrastive`, then the UMAP / t-SNE
  figures. `set_masked` was part-way through training; its partial fold
  files will be regenerated.
- **Cross-fit** of `read_ae_pred` (stopped on fold 1) and a rerun of
  `deepset` that saves its per-site scores - `crossfit.py` now saves them,
  the first run predates that.
- **`combine.py`:** do `kmer_norm`, `deepset` and `read_ae_pred` stack.

## Resume (from analysis/representation/)

```bash
PY=../../.venv/Scripts/python.exe
rm -rf ../../.cache/representation/set_masked
$PY -u run_all.py > run_all.log 2>&1        # skips every finished encoder
for m in read_ae_pred deepset; do $PY -u crossfit.py --model $m > logs/crossfit_$m.log 2>&1; done
$PY -u combine.py > logs/combine.log 2>&1
```

Budget: run_all ~3 h for the remaining three encoders plus figures; the two
cross-fits ~90 min each; combine ~15 min. Run the cross-fits in parallel with
run_all (they use 2 threads).

## Two things to fix in the write-up

- The per-read encoders' site vectors include log read count, so their
  "R^2 of depth = 1.000" in E4 is trivial, not a finding.
- `pos_lstm_ae` / `pos_attn_ae` only fit ~20 epochs in their 8-minute box,
  so "null" there may partly mean "undertrained".
