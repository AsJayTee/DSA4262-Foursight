"""Put the first two cross-source evaluations into W&B without refitting them.

    python analysis/newdata/upload_xsrc_reports.py

Both were run with --no-wandb while the mode was being validated (2026-10-01).
Their full reports are in analysis/newdata/xsrc/. This logs each one exactly as
a live `evaluate.py --cross-source` run would have: the same flat keys
(tracking.flat_metrics), the same run config including eval_schema 3 and the
dataset fingerprint (so --compare-run can pair against it), job_type evaluate.
The run notes say it was uploaded, and from which file.

Idempotent: a report whose run name already exists in the project is skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a import crosssource, report as reporting, tracking  # noqa: E402
from m6a.config import Config, describe_train_depths  # noqa: E402
from m6a.data import resolve_data_dir  # noqa: E402
from m6a.env import load_env  # noqa: E402

HERE = Path(__file__).resolve().parent / "xsrc"
REPORTS = [  # (report, config, baseline config)
    ("lightgbm_quantiles__xsrc.json", "configs/quantiles.yaml", "configs/lightgbm.yaml"),
    ("everything__xsrc.json", "configs/everything.yaml", "configs/quantiles.yaml"),
]


def existing_names() -> set[str]:
    import os
    import wandb
    path = f"{os.environ.get('WANDB_ENTITY') or wandb.Api().default_entity}/" \
           f"{os.environ.get('WANDB_PROJECT', 'dsa4262-project')}"
    return {r.name for r in wandb.Api().runs(path)}


def main() -> None:
    load_env(ROOT / ".env")
    data_dir = resolve_data_dir()
    present = existing_names()
    for filename, config_path, baseline_path in REPORTS:
        config, baseline = Config.load(ROOT / config_path), Config.load(ROOT / baseline_path)
        name = f"{config.name}__xsrc"
        if name in present:
            print(f"skip {name}: already in W&B")
            continue
        path = HERE / filename
        data = json.loads(path.read_text())
        fingerprint = tracking.dataset_fingerprint(
            data_dir / "dataset0.json.gz", data_dir / "data.info.labelled", config, None)
        tracker = tracking.start(
            name, config={"profile": "cross-source", "eval_schema": crosssource.EVAL_SCHEMA,
                          "baseline": baseline.name, "arms": ",".join(crosssource.ARMS),
                          "features": config.features, "model": config.model,
                          "train_depths": describe_train_depths(config.train_depths),
                          **fingerprint},
            notes=f"Uploaded from analysis/newdata/{path.relative_to(HERE.parent).as_posix()}: "
                  "run with --no-wandb on 2026-10-01 while --cross-source was validated; "
                  "numbers identical to that run.",
            tags=[tracking.EVAL_TAG, "xsrc", "uploaded"], job_type="evaluate")
        if not tracker.enabled:
            raise SystemExit("W&B is not available - check WANDB_API_KEY in .env.")
        try:
            reporting.publish(reporting.Report(profile=reporting.profile("standard"), data=data),
                              tracker, path, name=name)
        finally:
            tracker.finish()
        print(f"uploaded {name}")


if __name__ == "__main__":
    main()
