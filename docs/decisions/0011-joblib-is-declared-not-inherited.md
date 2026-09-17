# 0011. joblib is a declared base dependency, not one inherited from scikit-learn

- **Date:** 2026-09-17
- **Status:** Accepted
- **Affects:** `pyproject.toml`, `src/m6a/models/baseline.py`

## Context

`LogisticModel.save` and `.load` call `import joblib` inside the method
(`src/m6a/models/baseline.py`). joblib is not in the base dependency list. It is
importable anyway, because scikit-learn depends on it.

That is precisely the situation [0004](0004-comparison-stats-outside-evaluation.md)
exists to prevent, stated there about scipy:

> It is there by accident, not by declaration. […] It works today and fails
> silently on the day it doesn't.

And this one is worse placed than scipy was. scipy sat behind `m6a.compare`,
which the prediction path never reaches. joblib sits in a **model's `load`**, and
`scripts/predict.py` calls `load` on whatever model `models/final/meta.json`
names. Ship a logistic model as the final model — nothing stops that — and an
evaluator on a clean `pip install -e .` is one scikit-learn packaging change away
from a traceback in the graded path.

Moving the import inside the method, which is already done, does not fix this. It
defers *when* the failure happens; it does not change whether the dependency is
declared.

## Decision

`joblib>=1.3` joins the base dependency list in `pyproject.toml`, with a comment
saying why a package that is already installed is being named.

The import stays inside the methods. It is not needed to import the module, and
the registry imports every module in `models/` to discover them.

## Why this and not the alternatives

**Leave it.** The status quo. Works until it doesn't, and the day it doesn't is
during someone else's grading run, with no traceback anyone here will see.

**Replace joblib with stdlib `pickle`.** Tempting: zero new dependencies, and a
logistic pipeline is small enough that joblib's array handling buys nothing. The
reason it loses is that it does not make the model *more* portable — a pickle of
a scikit-learn `Pipeline` is exactly as version-fragile as a joblib dump of one,
so the trade is a format change and a broken `model.joblib` on disk in exchange
for a dependency that is installed on every machine that can run the model at
all. joblib is also what scikit-learn's own documentation tells you to use.

**Put it in the `train` extra.** Wrong extra: the code that imports it runs in
`predict.py`. The rule from 0004 is "if `predict.py` can reach it, it may only
use base dependencies" — that is a rule about the base list, so the fix is to add
it to the base list.

**Stop shipping non-LightGBM models.** Solves this instance and not the class.
The handout requires the simple baseline for comparison, and the next model
someone adds will reach for joblib too.

## Consequences

- The base list grows by one, which AGENTS.md section 7 says to say out loud:
  every entry is something that can fail on someone else's machine. The
  mitigation is that this one cannot realistically fail while scikit-learn is
  installable, because scikit-learn requires it.
- It stops being a coincidence that `pip install -e .` gives you a working
  `LogisticModel.load`. That is the whole point.
- A model that needs a *heavy* dependency still does not get to do this. The rule
  is unchanged: torch goes in the `mil` extra, imported inside methods, and a
  model that needs it is not the model we ship.

## How to check it still holds

`grep -rn "^import \|^from \|    import " src/m6a/models/ src/m6a/features/ src/m6a/data.py src/m6a/evaluation.py src/m6a/registry.py scripts/predict.py`
— every module named there may import only what is in `[project] dependencies`.

`tests/test_smoke.py::test_predict_path_has_no_heavy_imports` catches wandb,
boto3 and torch, and `tests/test_evaluation.py::test_the_predict_path_never_imports_the_comparison_module`
catches scipy's route in. Neither catches a *base-list omission* like this one,
which is why it survived — the check above is by hand.
