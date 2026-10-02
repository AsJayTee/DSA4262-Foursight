"""Re-judge the project's main configs with the cross-source rule (decision 0029).

    python analysis/newdata/run_xsrc_batch.py

Runs `evaluate.py --cross-source` for each config below against
configs/quantiles.yaml, logging each to W&B so it appears in the Decisions view
(decision 0030). Skips any config whose `<name>__xsrc` run already exists, so
it can be stopped and restarted. One log per config in analysis/newdata/xsrc/logs/.

Python rather than sh: a shell script read incrementally breaks if edited
mid-run (analysis/evaluation/run_batch.sh records how).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.config import Config  # noqa: E402
from m6a.env import load_env  # noqa: E402

LOGS = Path(__file__).resolve().parent / "xsrc" / "logs"
# (config, arms). The logistic and MLP models take no per-row weights, so they
# run only the arms that need none.
UNWEIGHTED = "dataset0,data1,ensemble"
BATCH = [
    ("configs/quantiles_depth_augmented.yaml", None),
    ("configs/quantiles_flank.yaml", None),
    ("configs/quantiles_flank_depth_augmented.yaml", None),
    ("configs/coupling.yaml", None),
    ("configs/crosssite_robust.yaml", None),
    ("configs/final_candidate.yaml", None),
    ("configs/baseline.yaml", UNWEIGHTED),
    ("configs/mlp.yaml", UNWEIGHTED),
]


def existing() -> set[str]:
    import wandb
    project = os.environ.get("WANDB_PROJECT", "dsa4262-project")
    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
    # Finished only: a run killed mid-way (the laptop slept on 2026-10-01) has
    # no results and must not block its own rerun.
    return {r.name for r in wandb.Api().runs(f"{entity}/{project}") if r.state == "finished"}


ABLATIONS = [  # decision 0029 step 1: everything minus one family at a time
    ("configs/everything_no_crosssite.yaml", None),
    ("configs/everything_no_flank.yaml", None),
    ("configs/everything_no_readcount.yaml", None),
    ("configs/everything_fulldepth.yaml", None),
]


def main() -> None:
    load_env(ROOT / ".env")
    LOGS.mkdir(parents=True, exist_ok=True)
    done = existing()
    batch = ABLATIONS if "--ablations" in sys.argv[1:] else BATCH
    for config_path, arms in batch:
        name = f"{Config.load(ROOT / config_path).name}__xsrc"
        if name in done:
            print(f"skip {name}: already in W&B", flush=True)
            continue
        command = [sys.executable, "-u", "scripts/evaluate.py", "--config", config_path,
                   "--cross-source", "--baseline", "configs/quantiles.yaml"]
        if arms:
            command += ["--arms", arms]
        started = time.time()
        print(f"run  {name} ...", flush=True)
        with (LOGS / f"{name}.log").open("w") as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else 'FAILED'} {name} ({(time.time() - started) / 60:.0f} min)",
              flush=True)
    # A run started before its process picked up the curve code has no
    # curve/xsrc/* series; add them so it draws on the Decisions view.
    subprocess.run([sys.executable, "analysis/newdata/backfill_xsrc_curves.py"], cwd=ROOT)
    # Line styles in the Decisions view are keyed by run id, so new runs need a rebuild.
    subprocess.run([sys.executable, "scripts/build_dashboard.py"], cwd=ROOT)
    print("batch finished", flush=True)


if __name__ == "__main__":
    main()
