"""Premise test (approved 9 Oct): h2gcn_aux and gps trained on cell line 2 only.

    nohup python -u analysis/representation/run_data1_arm.py > analysis/representation/logs/data1_arm.log 2>&1 &

Waits for run_batch2.py to finish (so the two never compete for the CPUs),
then fits 2 models x 5 folds, seed 0, arm data1, ten at a time, and runs
data1_arm_eval.py -> results/data1_arm.csv.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> None:
    log("waiting for run_batch2.py")
    while subprocess.run(["pgrep", "-f", "run_batch2.py"], capture_output=True).returncode == 0:
        time.sleep(60)
    log("batch 2 gone; fitting the data1 arm")
    procs = []
    for model in ("h2gcn_aux", "gps"):
        for fold in range(5):
            out = open(HERE / "logs" / f"data1_arm_{model}_{fold}.log", "w")
            procs.append(subprocess.Popen(
                [PY, "-u", str(HERE / "xsrc_nets.py"), "fit", "--model", model, "--fold", str(fold),
                 "--arm", "data1", "--threads", "6", "--minutes", "720"],
                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    log(f"fits done, exit codes {codes}")
    code = subprocess.run([PY, "-u", str(HERE / "data1_arm_eval.py")], cwd=ROOT).returncode
    log(f"{'done' if code == 0 else 'FAILED'} data1_arm_eval (exit {code})")


if __name__ == "__main__":
    main()
