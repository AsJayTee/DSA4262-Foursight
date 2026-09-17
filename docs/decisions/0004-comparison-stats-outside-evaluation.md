# 0004. Comparison statistics live in `m6a/compare.py`, never in `evaluation.py`

- **Date:** 2026-09-16
- **Status:** Accepted
- **Affects:** `src/m6a/compare.py`, `src/m6a/evaluation.py`, `pyproject.toml`

## Context

Comparing two runs needs a paired significance test, and a paired t-test needs
scipy.

`scripts/predict.py` is the script other students run from a clean `git clone`
plus `pip install -e .`, with no `.env`, no network, and only the base
dependencies. It imports `m6a.evaluation`.

scipy **is** importable on essentially every machine that has scikit-learn,
because scikit-learn depends on it. That is exactly the problem: it is there by
accident, not by declaration. Putting the t-test behind `evaluation.py` would
make the graded prediction path depend on somebody else's dependency tree
staying the way it is today.

## Decision

Paired comparison statistics live in a separate module, `src/m6a/compare.py`,
which imports scipy at module level. `m6a.evaluation` gains **no new imports at
all** — the stratified-metric and calibration functions added alongside this
work use only what was already imported there.

Nothing on the prediction path imports `m6a.compare`. The registry scans only
`m6a.features` and `m6a.models`, so it is never reached by discovery either.

scipy is declared in `[project.optional-dependencies] train`, with a comment
saying why a package that is already installed is being named.

## Why this and not the alternatives

**Put it in `evaluation.py` and rely on scikit-learn pulling scipy in.** The
thing this decision exists to prevent. It works today and fails silently on the
day it doesn't.

**Add scipy to the base dependencies.** Rejected: the base list is what an
evaluator installs to run `predict.py`, and every entry is something that can
fail on someone else's machine. Prediction does not need a t-test.

**Hand-roll the t-test to avoid the dependency.** Rejected. Writing your own
significance test to dodge an import is how you end up with a subtly wrong
p-value and no way to notice.

## Consequences

- `m6a.compare` raises an `ImportError` naming the fix (`pip install -e '.[train]'`)
  rather than a bare traceback, for the rare machine without scipy.
- Anyone adding a statistical test must put it here, not in `evaluation.py`.
  The rule is: **if `predict.py` can reach it, it may only use base
  dependencies.**

## How to check it still holds

`tests/test_evaluation.py::test_the_predict_path_never_imports_the_comparison_module`
imports `m6a.data` and `m6a.evaluation`, runs registry discovery, and asserts
neither `m6a.compare` nor `m6a.crossval` ended up in `sys.modules`.

Note this checks the *import graph*, not whether scipy is loaded — scikit-learn
loads scipy regardless, which is the whole point. The question is whether our
code depends on that happening.

`tests/test_smoke.py::test_predict_path_has_no_heavy_imports` remains the guard
for wandb, boto3 and torch.
