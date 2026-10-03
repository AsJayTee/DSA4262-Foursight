"""The last pre-leaderboard batch (agreed 2026-10-03), unattended on Ronin.

    nohup python -u analysis/representation/run_final.py > analysis/representation/logs/final.log 2>&1 &

1. new variants, both arms (20 networks):  h2gcn_local_deep, h2gcn_aux
   alongside seed 1 of the finalists, both-files arm (15 networks):
   h2gcn, h2gcn_local, h2gcn_twohead
2. seed 2 of the finalists (15 networks)
3. depth_rescore.py - every saved finalist and variant from 1, 3 and 10 reads
4. ensemble.py - rank-averaged combinations, including the 3-seed averages

Each stage is `xsrc_nets.py all`, which skips finished networks, so rerunning
this resumes. Seeds other than 0 are not logged to W&B (their reports are).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
NETS = [PY, "-u", str(HERE / "xsrc_nets.py"), "all", "--threads", "4", "--minutes", "720"]
FINALISTS = "h2gcn,h2gcn_local,h2gcn_twohead"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def launch(name: str, args: list[str]) -> subprocess.Popen:
    log(f"start {name}")
    out = open(HERE / "logs" / f"final_{name}.log", "w")
    return subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)


def wait(jobs: dict) -> None:
    for name, job in jobs.items():
        code = job.wait()
        log(f"{'done' if code == 0 else 'FAILED'} {name} (exit {code})")


def main() -> None:
    wait({"variants": launch("variants", [*NETS, "--model", "h2gcn_local_deep,h2gcn_aux", "--jobs", "10"]),
          "seed1": launch("seed1", [*NETS, "--model", FINALISTS, "--seed", "1", "--arms", "pooled_both",
                                    "--jobs", "6"])})
    wait({"seed2": launch("seed2", [*NETS, "--model", FINALISTS, "--seed", "2", "--arms", "pooled_both",
                                    "--jobs", "16"])})
    wait({"depth": launch("depth", [PY, "-u", str(HERE / "depth_rescore.py"), "--models",
                                    f"{FINALISTS},h2gcn_local_deep,h2gcn_aux"])})
    wait({"ensemble": launch("ensemble", [PY, "-u", str(HERE / "ensemble.py")])})
    log("final batch finished")


if __name__ == "__main__":
    main()
