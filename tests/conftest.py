"""Test-session setup that keeps the suite out of the repo's own directories.

There is one rule here and it exists because it was broken: **a test run must
never write where a real run writes.** `scripts/train.py` now evaluates inline
and drops a JSON report in `analysis/evaluation/reports/<config name>.json`
(docs/decisions/0007). The end-to-end tests run `train.py` against a 240-site
synthetic dataset using the *real* config files - so without this, running
`make test` silently replaced the recorded `baseline_logistic.json` and
`lightgbm_pooled.json` with synthetic results carrying the same names.

Nothing fails when that happens. The numbers in GAPS.md say "regenerate with
this command", someone opens the report to check one, and it is quietly a
different experiment. That is the same failure mode the feature cache has and
the reason `M6A_REPORT_DIR` exists at all.
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
