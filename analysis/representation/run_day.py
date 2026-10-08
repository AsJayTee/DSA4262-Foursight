"""The 2026-10-08 unattended batch: constrained corroboration + a GraphGPS control.

    nohup python -u analysis/representation/run_day.py > analysis/representation/logs/day.log 2>&1 &

Order puts one seed of EVERY design first, so a stop part-way still leaves a
complete single-seed comparison; seeds 1 and 2 follow. Confound analyses run
alongside the first stage. Every stage runs even if an earlier one fails, and
`xsrc_nets.py all` skips finished networks, so rerunning resumes.

  stage 0  confound analyses (gain by neighbour count / transcript size / ROC;
           band ablation with 0-10 and 10-20 nt split out) - in parallel
  stage 1  seed 0 of every new design and control, both arms
  stage 2  seed 1 of the main designs + h2gcn_aux and h2gcn_aux_r100 (matched baselines)
  stage 3  seed 2 of the same
  stage 4  convergence check + summarize_day.py -> results/day_summary.md
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
NETS = [PY, "-u", str(HERE / "xsrc_nets.py"), "all", "--threads", "4", "--minutes", "720", "--jobs", "16"]
MAIN = "fk_band,fk_kernel,res_gate,scalar_msg,gps"
CONTROLS = "fk_band_shuffled,res_nogate,res_nodrop,scalar_random"
BASELINES = "h2gcn_aux,h2gcn_aux_r100"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(name: str, args: list[str], wait: bool = True):
    log(f"start {name}")
    p = subprocess.Popen(args, cwd=ROOT, stdout=open(HERE / "logs" / f"day_{name}.log", "w"),
                         stderr=subprocess.STDOUT)
    if wait:
        code = p.wait()
        log(f"{'done' if code == 0 else 'FAILED'} {name} (exit {code})")
    return p


def main() -> None:
    # Stage 0 alongside stage 1. Keep the earlier band ablation (coarser bands).
    old = HERE / "results" / "band_ablation.csv"
    if old.exists() and not (HERE / "results" / "band_ablation_v1.csv").exists():
        shutil.copy(old, HERE / "results" / "band_ablation_v1.csv")
    confound = run("confounds", [PY, "-c",
                                 "import subprocess,sys;"
                                 f"subprocess.run([sys.executable,'-u',r'{HERE / 'gain_decomposition.py'}']);"
                                 f"subprocess.run([sys.executable,'-u',r'{HERE / 'band_ablation.py'}'])"], wait=False)
    run("seed0", [*NETS, "--model", f"{MAIN},{CONTROLS}", "--seed", "0"])
    for seed in (1, 2):
        run(f"seed{seed}", [*NETS, "--model", f"{MAIN},{BASELINES}", "--seed", str(seed)])
    log(f"confounds {'done' if confound.wait() == 0 else 'FAILED'}")
    run("convergence", [PY, "-u", str(HERE / "convergence.py")])
    run("summary", [PY, "-u", str(HERE / "summarize_day.py")])
    log("day batch finished")


if __name__ == "__main__":
    main()
