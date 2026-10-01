# 0030. The W&B dashboard is built from code, and every run says which evaluation produced it

- **Date:** 2026-10-01
- **Status:** Accepted
- **Affects:** `scripts/build_dashboard.py` (new), `docs/wandb-views.json` (new), `pyproject.toml` (train extra gains `wandb-workspaces`), the `eval_schema` run-config key, docs/wandb-panels.md

## Context

The W&B workspace was assembled by hand from the recipe in docs/wandb-panels.md.
It existed only as clicks, so nobody could review or reproduce it, and it had
already drifted from the guide.

It also answered the wrong question. Every panel and the default sort were
built around `oof/pr_auc` - dataset0 cross-validation - which
[0029](0029-models-are-selected-on-cross-source-gain.md) showed overstates
transferable gains by ~3x. The new decision keys (`xsrc/*`, and `ext/*` from
[0028](0028-every-run-is-also-judged-on-data1-and-data2.md)) had no panels at
all, and none of the 16 existing runs carried them, so new panels would have
mixed runs that can and cannot answer the question.

## Decision

1. **`scripts/build_dashboard.py` defines the dashboard.** Two saved views of
   the shared workspace, created on first run and updated in place after, with
   their URLs in `docs/wandb-views.json` so teammates update the same views:
   - **Decisions** - runs with `eval_schema >= 2`, sorted by `xsrc/gain_mean`:
     the selection rule, transfer (scatter of the two gains, the crossed test,
     the new-genes check), training-arm comparison, data2, held-out data1.
   - **History (CV only)** - `eval_schema == 1`, the earlier panels (depth
     collapse, calibration, thresholds, curves, fit progress), sorted by
     `oof/pr_auc` as before.
2. **`eval_schema` in every run config** says which evaluation produced the
   row: `1` = dataset0 cross-validation only (every run before 2026-09-29,
   tagged retroactively; no metric changed), `2` = also held out on data1/data2
   (0028), `3` = cross-source (0029; set by `evaluate.py --cross-source`).
3. **The first two cross-source runs were uploaded, not refitted**
   (`analysis/newdata/upload_xsrc_reports.py`): they were validated with
   `--no-wandb`, and their saved reports hold every key a live run logs.

## Why this and not the alternatives

**Keep building it by hand.** Free and flexible, and it is how the dashboard
drifted. Four people share it and it will change again once more cross-source
runs exist.

**One view with every panel, filtered per panel.** W&B filters apply to a
whole view, so one view would either hide the history or put CV-only runs into
the decision panels, where they show blank and look like failures.

**Re-run the 16 old runs instead of tagging them.** Hours of compute for
numbers we already have; they stay as history and the decision view excludes
them.

## Consequences

- **The views belong to the script.** A change made in the W&B UI is
  overwritten on the next build - change the script instead.
- A new training-side dependency, `wandb-workspaces`. It never reaches
  `predict.py`.
- `wandb-workspaces` 0.4.12 builds view URLs with a Windows path separator;
  the script normalises them. If a view ever opens to the wrong project,
  check that first.
- Runs without `eval_schema` appear in neither view until tagged.

## How to check it still holds

```bash
python scripts/build_dashboard.py --dry-run   # 14 + 11 panels, saves nothing
python scripts/build_dashboard.py             # updates the two views in place
```

`docs/wandb-views.json` must still hold the same two URLs afterwards; a new
URL means a copy was created instead of an update.
