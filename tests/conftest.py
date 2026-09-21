"""Test-session setup that keeps the suite out of the repo's own directories.

There is one rule here and it exists because it was broken twice: **a test run
must never write where a real run writes.**

`scripts/train.py` evaluates inline and drops a JSON report in
`analysis/evaluation/reports/<config name>.json` (docs/decisions/0007). The
end-to-end tests run the scripts against a ~300-site synthetic dataset using the
*real* config files - so without a guard, `make test` silently replaced the
recorded `baseline_logistic.json` and `lightgbm_pooled.json` with synthetic
results carrying the same names.

The second time was worse. `evaluate.py` gained W&B logging and one test invoked
it without `--no-wandb`, so every `pytest` run published a `lightgbm_pooled` run
to the shared project scoring **PR AUC 0.9999 and ROC AUC 1.0** - the synthetic
data has a signal injected into it deliberately, so a model fits it almost
perfectly. Those sat in the run table next to real experiments looking like a
breakthrough.

Neither failure raises. Someone opens a report or a run to check a number and it
is quietly a different experiment. Both are fixed here rather than flag-by-flag
at each call site, because the next test anyone writes will forget the flag too.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _reports_go_somewhere_temporary(tmp_path_factory):
    """Point M6A_REPORT_DIR at a temporary directory for the whole session.

    Session-scoped and autouse, so it covers subprocess invocations of train.py
    and evaluate.py too: they inherit os.environ, and the tests that build an
    explicit `env` dict copy os.environ into it.
    """
    directory = tmp_path_factory.mktemp("reports")
    previous = os.environ.get("M6A_REPORT_DIR")
    os.environ["M6A_REPORT_DIR"] = str(directory)
    yield directory
    if previous is None:
        os.environ.pop("M6A_REPORT_DIR", None)
    else:
        os.environ["M6A_REPORT_DIR"] = previous


@pytest.fixture(autouse=True, scope="session")
def _no_test_run_reaches_wandb():
    """Make it impossible for the suite to publish to the shared W&B project.

    Belt and braces on purpose. `m6a.tracking.start` returns a disabled tracker
    when `WANDB_API_KEY` is absent, so clearing it is enough for our own code
    and avoids importing wandb at all; `WANDB_MODE=disabled` covers anything
    that reaches wandb by another route.

    This is deliberately **not** a `--no-wandb` flag on each call site. One test
    already forgot it, and the cost of forgetting is a run in the shared project
    scoring 1.0 next to real experiments.
    """
    saved = {key: os.environ.get(key) for key in ("WANDB_API_KEY", "WANDB_MODE")}
    os.environ.pop("WANDB_API_KEY", None)
    os.environ["WANDB_MODE"] = "disabled"
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
