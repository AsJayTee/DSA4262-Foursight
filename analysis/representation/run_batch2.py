"""The 2026-10-09 batch (approved list, jobs 1-3): seeds for the shipped-ensemble
comparison and the controls, and scalar messages with neighbour dropout.

    nohup python -u analysis/representation/run_batch2.py > analysis/representation/logs/batch2.log 2>&1 &

  seed 1, then seed 2: h2gcn_twohead_aux (both files only) - job 1; the four
                       controls - job 2; scalar_drop - job 3
  seed 0:              scalar_drop
  then convergence.py and summarize_day.py (-> results/day_summary.md)

Every stage runs even if an earlier one fails; rerunning resumes (`all` skips
finished networks). Jobs 4-5 (scoring only) are separate scripts.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
NETS = [PY, "-u", str(HERE / "xsrc_nets.py"), "all", "--threads", "4", "--minutes", "720", "--jobs", "16"]
SEEDED = "h2gcn_twohead_aux,fk_band_shuffled,res_nogate,res_nodrop,scalar_random,scalar_drop"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(name: str, args: list[str]) -> None:
    log(f"start {name}")
    with open(HERE / "logs" / f"batch2_{name}.log", "w") as out:
        code = subprocess.run(args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"{'done' if code == 0 else 'FAILED'} {name} (exit {code})")


def main() -> None:
    for seed in (1, 2):
        run(f"seed{seed}", [*NETS, "--model", SEEDED, "--seed", str(seed)])
    run("seed0", [*NETS, "--model", "scalar_drop", "--seed", "0"])
    run("convergence", [PY, "-u", str(HERE / "convergence.py")])
    run("summary", [PY, "-u", str(HERE / "summarize_day.py")])
    log("batch 2 finished")


if __name__ == "__main__":
    main()
