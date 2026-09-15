"""Minimal .env loading.

Hand-rolled rather than depending on python-dotenv so that `make doctor` works
before the optional extras are installed. The first thing a stuck teammate runs
should never itself be the thing that is broken.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env", override: bool = False) -> bool:
    """Load KEY=VALUE lines into os.environ. Returns False if there is no file.

    Existing environment variables win unless override=True, so a value set on
    the command line is not silently replaced by the file.
    """
    path = Path(path)
    if not path.exists():
        return False

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return True
