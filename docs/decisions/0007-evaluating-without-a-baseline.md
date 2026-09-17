# 0007. Instances are disposable, so W&B is the only place results exist

- **Date:** 2026-09-16
- **Status:** Accepted — built 2026-09-17
- **Affects:** `scripts/evaluate.py`, `scripts/train.py`, `src/m6a/tracking.py`, `src/m6a/report.py`, `src/m6a/figures.py`, `scripts/doctor.py`, `meta.json`, AGENTS.md, docs/running-experiments.md
- **Shipped as:** three modules rather than the one this record named — see
  [0010](0010-the-report-is-a-module-not-a-script.md), which also settles what
  still gets written to `analysis/evaluation/reports/`.

## Context

Two facts, and the second one is the one that drives this record.

**1. The harness was built around comparing two runs.** `--compare-features`,
`--compare-with`, `--compare-oof`. That is the right tool when you have a
baseline in hand — and most of the time you will not. You try an idea, you get a
number, and the thing you actually want to compare against is *every experiment
anyone has already run*.

**2. Ronin instances are created, used, and terminated, and nothing is pulled
off them first.** Anything written to local disk is destroyed. Today:

- `scripts/evaluate.py` logs **nothing** to W&B. Verified: zero references to
  wandb in the file. The entire rich half of an evaluation — strata,
  calibration, depth sweep — is written to `analysis/evaluation/reports/*.json`
  on a machine that is about to cease to exist.
- `oof.csv` is written next to the model and never uploaded. That table is the
  whole premise of [0001](0001-retain-per-fold-metrics.md) — cheap re-analysis,
  paired comparison without refitting. On a disposable instance it survives
  exactly as long as the instance does.
- W&B holds the pooled metrics, the per-fold table and the model artifact from
  `train.py`, and nothing else.

So a teammate cannot see your depth curve, you cannot compare against last
week's run on anything but pooled PR AUC, and the out-of-fold table is
recomputed from scratch every time because the copy that would have made it
cheap was deleted with the VM.

Comparison-by-filtering also only works if every run reports the same keys. If
one experiment was run with `--depth-sweep` and the next was not, they are not
comparable on the dimension that matters most, and nobody finds out until they
try.

## Decision

### 1. Nothing interesting is allowed to exist only on local disk

Every artifact an evaluation produces is uploaded to W&B:

| artifact | why it has to survive |
|---|---|
| `report.json` | the full nested report, the thing GAPS numbers are traced to |
| per-fold / per-repetition metric vectors | everything a paired test needs, at ~50 floats ([0009](0009-distributions-not-per-site-scores.md)) |
| `model.txt` / `columns.json` / `meta.json` | already uploaded by `train.py` today |
| the plots below | so a result can be read without rerunning anything |

Local files stay as a convenience for laptop work. They are no longer the
record.

### 2. Training runs evaluation, by default

`scripts/train.py` runs the standard evaluation profile as part of the run and
logs it to the same W&B run.

This reverses the position in the first draft of this record, which kept the two
apart on the grounds that training should not pay for a depth sweep. On a
disposable instance that reasoning is backwards: a training run that terminates
before anyone evaluates it has lost everything, and re-running it costs far more
than the sweep would have. **Gather everything while the machine exists.**

`--quick` skips the sweep for iteration. `scripts/evaluate.py` remains a
standalone command for comparisons and for scoring a trained model against a
labelled dataset (`--model`).

### 3. A default profile, so runs are comparable by construction

`--profile {quick,standard,full}`, default **`standard`**.

| profile | contents |
|---|---|
| `quick` | per-fold, pooled, calibration. Iteration only; not a recorded result |
| `standard` | + strata by depth and motif, + depth sweep, + all plots. **The default.** |
| `full` | + ablations (signal-only, motif-only) |

The depth sweep is in `standard` deliberately: it is the largest known risk in
the project, it is the number most likely to be wanted retrospectively, and
anything opt-in gets skipped. VM time is not the constraint worth optimising
here — a terminated instance that gathered too little is.

### 4. A flat, stable metric namespace

Nested JSON is right for the report and useless for W&B's run table. Alongside
it, flat scalars with fixed keys, so any run can be sorted, filtered and charted
against any other:

```
oof/pr_auc            oof/roc_auc           oof/pr_auc_lift
fold/pr_auc_mean      fold/pr_auc_sd        fold/{0..4}/pr_auc
depth/{1,3,5,10,20,full}/pr_auc             depth/{...}/retained
calib/ece             calib/brier           calib/count_ratio
band/{20-31,...}/pr_auc_lift
motif/{GGACT,...}/pr_auc_lift
ablation/{signal,motif}/pr_auc
compare/{name}/mean_difference   compare/{name}/p_value   compare/{name}/wins
```

**This is the answer to "what if I have no baseline".** You always have one:
every run in the project, in one filterable table.

### 5. Plots, logged not left

Every `standard` run logs:

- **ROC curve** and **PR curve** over the pooled out-of-fold scores, with the
  random-classifier line drawn on the PR plot at the 4.49% base rate, because a
  PR curve without it is unreadable.
- **PR AUC and ROC AUC against read depth** — both the depth-sweep curve (scored
  at reduced depth) and the by-band curve (true depth). These are different
  questions and are easy to confuse, so they are separate plots with the
  distinction in the caption.
- **A reliability diagram** — predicted against observed by decile, with the
  diagonal.
- For a comparison: **both arms' curves on one axis**, and the per-fold
  difference as a strip.

Plots are generated with matplotlib and logged as images. matplotlib is **not**
a base dependency and must not become one — it goes in the `train` extra and is
imported inside the plotting functions, never at module level, so that
`predict.py` cannot reach it (AGENTS.md section 4).

### 6. `evaluate.py` attaches to the training run

`train.py` records its W&B run id in `meta.json`; `evaluate.py` resumes that run
when it can find one, so one experiment is one row in the table rather than two.
Failing that it starts its own run tagged `eval`.

W&B stays non-fatal exactly as `train.py` already treats it — no key, no
package, no network, print a line and carry on. A disposable instance with a
broken W&B key should fail *loudly at the start*, though, not after 40 minutes
of compute: `doctor` should check the key works, not merely that it is set.

### 7. The agent asks, up front

Before preparing an experiment, the agent asks whether the run is **standalone**
or **compared against a specific baseline**, and builds the command accordingly:

- standalone → `train.py --config <cfg>` (standard profile, everything to W&B)
- explicit baseline → add `--compare-features <name>` or `--compare-with <cfg>`

Recorded in AGENTS.md and docs/running-experiments.md. Nobody should discover
after an instance is gone that the comparison they wanted was never made.

## Why this and not the alternatives

**Keep results local and pull them off the instance before terminating.**
Rejected: it is a manual step, and the failure mode is silent and total. The
instance is gone and so is the work.

**Make `standard` minimal and have people opt into the sweep.** Rejected.
Optional evaluation is evaluation that does not happen, and a run without depth
numbers cannot be filtered against one that has them.

**Keep `train.py` and `evaluate.py` fully separate.** The first draft's
position. Cleaner in the abstract; wrong given disposable instances.

**Always start a fresh W&B run from `evaluate.py`.** Simpler, but splits one
experiment across two rows and makes the filtering worse — the exact thing this
record exists to fix.

## Consequences

- `standard` costs more per run than today's default. That is the intended
  trade: ~5 minutes the first time a feature set is swept against losing the
  result entirely.
- **matplotlib becomes a `train`-extra dependency.** New dependency, flagged
  here per AGENTS.md section 7. It must never reach the prediction path.
- `meta.json` gains `wandb_run_id`. Additive; `predict.py` unaffected.
- The flat key names become an interface. Renaming one orphans every historical
  run in W&B, so they are a schema and change only via a new decision record.
- Runs made before this ships will lack `depth/*` and `calib/*` keys. They show
  as blanks in the run table rather than being silently wrong.
- Nothing over ~1 MB is uploaded per run. The per-site score table is not
  stored at all - see [0009](0009-distributions-not-per-site-scores.md), which
  amends 0001 and removes the `--oof` mode this record originally assumed.

## How to check it still holds

```bash
python scripts/train.py --config configs/quantiles.yaml
```

on the full training set must put all of this into one W&B run: `oof/*`,
`fold/*`, `depth/*`, `calib/*`, `band/*`, `motif/*`, the `per_fold`, `by_depth`,
`by_motif` and `depth_sweep` tables, seven `fig/*` images, and a `report`
artifact under 1 MB beside the model artifact. Verified on
[run y2lk7ilf](https://wandb.ai/dsa4262-team/dsa4262-project/runs/y2lk7ilf):
0.37 MB of figures, a 10 KB report, 4.17 MB of model.

Two runs of different configs must emit the same `oof/*`, `fold/*`, `depth/*`
and `calib/*` keys — that is what makes the run table filterable, and it is the
thing a leaner default profile would quietly break.

`tests/test_smoke.py::test_predict_path_has_no_heavy_imports` must still pass
with matplotlib installed — that is the guard that it has not leaked into the
graded path.

`python scripts/doctor.py` must **fail**, not warn, on a W&B key that does not
work.
