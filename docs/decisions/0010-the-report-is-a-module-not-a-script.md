# 0010. The evaluation report is a module, and the local JSON stays

- **Date:** 2026-09-17
- **Status:** Accepted
- **Affects:** `src/m6a/report.py`, `src/m6a/tracking.py`, `src/m6a/figures.py`, `scripts/evaluate.py`, `scripts/train.py`, `analysis/evaluation/reports/`, `$M6A_REPORT_DIR`

## Context

[0007](0007-evaluating-without-a-baseline.md) decided that `train.py` runs the
evaluation inline, and named one new module for it (`tracking.py`). It did not
say where the *report* lives.

That mattered as soon as it was built. Every report section — per fold, strata,
calibration, the depth sweep, a paired comparison — was a function inside
`scripts/evaluate.py`, and `train.py` now needs all of them. A script importing
another script is the obvious shortcut and the wrong one: `scripts/` is not on
the path, the two would share a CLI's argument object rather than an interface,
and the profile logic would have nowhere sensible to sit.

0007 also left one question open that the user asked explicitly: now that W&B is
the record, does anything still get written to `analysis/evaluation/reports/`?

## Decision

### 1. Three modules, split by what they depend on

| module | what it is | its heaviest import |
|---|---|---|
| `m6a.report` | the report's sections, the profiles, and where it is written | pandas |
| `m6a.tracking` | the W&B run wrapper and the flat metric schema | wandb, lazily |
| `m6a.figures` | every matplotlib figure | matplotlib, lazily |

The split is by dependency, not by taste. `figures` is the only module that may
touch matplotlib and `tracking` the only one that may touch wandb, both inside
functions, so there is exactly one place to look when asking whether a
`train`-extra dependency has leaked toward the prediction path
([0004](0004-comparison-stats-outside-evaluation.md)). `report` imports neither
at module level and calls into both.

`scripts/evaluate.py` and `scripts/train.py` are now both thin CLIs over
`m6a.report`, which is what `evaluate.py`'s docstring already claimed it was.

### 2. The local JSON report stays, and is uploaded as well

`analysis/evaluation/reports/<name>.json` is still written by every run, and the
same file is uploaded to W&B as an artifact.

It is **not** the record — 0007 settled that — but deleting it would cost two
things for nothing. A laptop run with no W&B key has to leave something behind,
and a number quoted in GAPS.md wants a file a reader can open. It is ~10 KB.

Two guards, because the failure mode is a number nobody can trace:

- A run over a subset of the sites writes `<name>__limit5000.json`, never
  `<name>.json`. A `--smoke` run overwriting a recorded full-dataset report is
  silent and total, and it happened once while this was being built.
- `$M6A_REPORT_DIR` overrides the directory, the same shape as `$M6A_CACHE_DIR`
  and `$M6A_DATA_DIR`, so the test suite cannot write into the real one.

### 3. Profiles live in `report.py`, and the section flags still work

`--profile {quick,standard,full}` (0007 section 3) decides which sections run.
`--depth-sweep` and `--ablate` still force their section on regardless of
profile, so every command already written down in AGENTS.md, GAPS.md and
docs/running-experiments.md keeps working and keeps meaning the same thing.

## Why this and not the alternatives

**Have `train.py` import `scripts/evaluate.py`.** The shortcut. It needs a
`sys.path` insert to work at all, and it couples the two scripts through an
argparse `Namespace` — so adding a flag to one silently changes the other.

**Put the report sections in `evaluation.py`.** Rejected outright: that module
is on predict.py's import path and may not grow a single import
(AGENTS.md section 4). The report needs `m6a.compare`, which needs scipy.

**One module instead of three.** Then wandb and matplotlib are imported from the
same file that formats tables, and the "which module may import what" rule that
0004 rests on stops being checkable by looking at one file.

**Drop the local JSON entirely.** Consistent with "W&B is the record", and worse
in practice: the number in GAPS.md that says "regenerate with this command" then
has nothing to point at on the machine where the command was run.

## Consequences

- `m6a.report` is shared infrastructure now. A new report section is an edit to
  a shared file rather than an additive change, so it needs saying out loud in a
  PR (AGENTS.md section 1). Adding a *figure* does not — that is additive inside
  `figures.py`.
- Three more modules to keep off the prediction path. The existing import-graph
  tests cover this: they assert `m6a.compare` and `m6a.crossval` stay out of
  `sys.modules`, and `report`, `tracking` and `figures` are reachable only
  through those.
- The report JSON is written twice over in a comparison run — once per arm is
  *not* what happens; both arms share one report file keyed by the primary run's
  name. Retrieving the baseline arm's numbers means reading that file's
  `compare_*` section, not looking for a second file.

## How to check it still holds

```bash
python scripts/train.py --config configs/quantiles.yaml --smoke
```

writes `analysis/evaluation/reports/lightgbm_quantiles__limit5000.json` and
leaves `lightgbm_quantiles.json` alone. And the guard that matters:

```bash
python -c "import sys; sys.path.insert(0,'src'); import m6a.data, m6a.evaluation; \
from m6a import registry; registry.available('features'); registry.available('models'); \
print([m for m in ('wandb','matplotlib','scipy') if m+'.' in str(sys.modules) or m in sys.modules])"
```

`matplotlib` and `wandb` must not appear. `tests/test_smoke.py::test_predict_path_has_no_heavy_imports`
is the enforced version.
