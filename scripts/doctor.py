#!/usr/bin/env python
"""Check this machine is set up correctly, and say plainly what is not.

    python scripts/doctor.py      (or: make doctor)

Run this first whenever anything is wrong. Every line is either OK or names the
command that fixes it. Exits non-zero only if something essential is broken —
warnings are for things you may not need yet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK = "  OK  "
WARN = " WARN "
FAIL = " FAIL "

results: list[tuple[str, str, str]] = []


def check(status: str, label: str, detail: str = "") -> None:
    results.append((status, label, detail))


def report() -> int:
    width = max(len(label) for _, label, _ in results) + 3
    print()
    for status, label, detail in results:
        print(f"[{status}] {label.ljust(width)}{detail}")

    failures = sum(1 for status, _, _ in results if status == FAIL)
    warnings = sum(1 for status, _, _ in results if status == WARN)
    print()
    if failures:
        print(f"{failures} problem(s) must be fixed before anything will run.")
        return 1
    if warnings:
        print(f"Ready, with {warnings} warning(s) above. Training and prediction will work.")
    else:
        print("Ready.")
    print("Next: make smoke CONFIG=configs/lightgbm.yaml")
    return 0


def main() -> int:
    from m6a.env import load_env

    found_env = load_env(ROOT / ".env")

    version = sys.version_info
    if version >= (3, 10):
        check(OK, f"python {version.major}.{version.minor}.{version.micro}")
    else:
        check(FAIL, f"python {version.major}.{version.minor}", "need >= 3.10")

    try:
        import m6a  # noqa: F401
        from m6a import registry

        check(OK, "m6a package imports")

        features = registry.available("features")
        models = registry.available("models")
        check(OK, f"{len(features)} feature set(s)", ", ".join(features))
        check(OK, f"{len(models)} model(s)", ", ".join(models))

        for kind in ("features", "models"):
            for module, error in registry.failures(kind).items():
                # Expected for optional extras (mil_torch with no torch installed).
                # A surprise here means a genuinely broken module.
                check(WARN, f"{kind}/{module} did not load", error)
    except Exception as exc:  # noqa: BLE001
        check(FAIL, "m6a package imports", f"{type(exc).__name__}: {exc}")
        return report()

    if found_env:
        check(OK, ".env found")
    else:
        check(WARN, "no .env file", "cp .env.example .env - ask on Telegram for the values")

    r2_keys = ["R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
    missing_r2 = [key for key in r2_keys if not os.environ.get(key)]
    if missing_r2:
        check(WARN, "R2 not configured", f"missing {', '.join(missing_r2)}")
    else:
        check(OK, "R2 configured", os.environ.get("R2_BUCKET", ""))

    if os.environ.get("WANDB_API_KEY"):
        entity = os.environ.get("WANDB_ENTITY", "?")
        project = os.environ.get("WANDB_PROJECT", "?")
        check(OK, "W&B configured", f"{entity}/{project}")
    else:
        check(WARN, "WANDB_API_KEY not set", "training works, but nothing is tracked")

    from m6a.data import resolve_data_dir

    data_dir = resolve_data_dir()
    expected = ["dataset0.json.gz", "data.info.labelled"]
    present = [name for name in expected if (data_dir / name).exists()]
    if len(present) == len(expected):
        size_mb = sum((data_dir / name).stat().st_size for name in present) / 1e6
        check(OK, f"training data in {data_dir}/", f"{size_mb:.0f} MB")
    else:
        check(
            WARN,
            f"training data missing from {data_dir}/",
            "run: python scripts/download_data.py   (or set M6A_DATA_DIR)",
        )

    if (ROOT / "data" / "sample" / "sample.json.gz").exists():
        check(OK, "sample test data present")
    else:
        check(WARN, "no data/sample/sample.json.gz", "run: python scripts/make_sample.py")

    if (ROOT / "models" / "final" / "meta.json").exists():
        check(OK, "models/final present")
    else:
        check(WARN, "no models/final", "train one, then copy it to models/final/ to ship it")

    for module, why in (("wandb", "experiment tracking"), ("boto3", "R2 downloads")):
        try:
            __import__(module)
            check(OK, f"{module} installed")
        except ImportError:
            check(WARN, f"{module} not installed", f"{why} - pip install -e .[train]")

    return report()


if __name__ == "__main__":
    raise SystemExit(main())
