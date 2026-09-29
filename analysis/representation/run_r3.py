"""Round 3, unattended: fit diagnostic -> gate -> 50-observation confirmation.

    nohup python -u run_r3.py > logs/run_r3.log 2>&1 &

Written to run on the instance with nobody watching. Every step writes its
result to disk; nothing needs the laptop.

1. Wait for the 30 diagnose2.py jobs; write results/diagnose2.md.
2. GATE, fixed before seeing the results: the best new architecture
   (res_deepset, res_attn_mil, set_transformer; all recipe v2) must beat
   deepset@v1 on held-out AP by >= +0.005 on average AND on >= 4 of 5 folds.
   The decision and its numbers go to results/r3_gate.md whichever way it goes.
3. If one passes: its cross-fitted scores over 10 repetitions x 5 folds
   (crossfit50.py - same halves and folds as the deepset scores already on
   disk), then LightGBM combinations (combine_r3.py) against `everything`,
   `everything + deepset` and at 1 / 3 reads -> results_r3/r3.md.
4. results/R3_DONE.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
LOGS = HERE / "logs"
RESULTS = HERE / "results"
CANDIDATES = ["res_deepset@v2", "res_attn_mil@v2", "set_transformer@v2"]
MIN_GAIN, MIN_WINS = 0.005, 4


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def env(threads: int) -> dict:
    import os
    e = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        e[key] = str(threads)
    return e


def run_parallel(script: str, arg_flag: str, items: list[str], workers: int, threads: int,
                 name: str, extra: list[str] = ()) -> None:
    procs = []
    for i in range(workers):
        shard = items[i::workers]
        if not shard:
            continue
        fh = (LOGS / f"{name}{i}.log").open("a")
        procs.append(subprocess.Popen([PY, "-u", script, *extra, arg_flag, *shard], cwd=HERE,
                                      stdout=fh, stderr=subprocess.STDOUT, env=env(threads)))
    for p in procs:
        p.wait()


def gate() -> str | None:
    recs = [json.loads(p.read_text()) for p in (RESULTS / "diagnose2").glob("*.json")]
    by: dict = {}
    for r in recs:
        by.setdefault(f"{r['model']}@{r['recipe']}", {})[r["fold"]] = r["heldout_ap"]
    base = by["deepset@v1"]
    lines = ["# Round-3 gate", "",
             f"Rule, fixed before the results: mean held-out gain over deepset@v1 >= "
             f"+{MIN_GAIN} AND wins >= {MIN_WINS}/5.", "",
             "| candidate | mean gain | wins | passes |", "|---|---:|---:|---|"]
    passing = []
    for c in CANDIDATES + ["deepset@v2", "attn_mil@v2"]:
        if c not in by:
            continue
        d = np.array([by[c][f] - base[f] for f in sorted(base) if f in by[c]])
        ok = c in CANDIDATES and d.mean() >= MIN_GAIN and (d > 0).sum() >= MIN_WINS
        lines.append(f"| {c} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | "
                     f"{'yes' if ok else 'no' if c in CANDIDATES else '(recipe control)'} |")
        if ok:
            passing.append((d.mean(), c.split("@")[0]))
    winner = max(passing)[1] if passing else None
    lines += ["", f"**Decision:** {'confirm ' + winner + ' at 50 observations' if winner else 'no candidate passes; stop here - deepset stays.'}"]
    (RESULTS / "r3_gate.md").write_text("\n".join(lines) + "\n")
    log("gate: " + (winner or "none"))
    return winner


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    while len(list((RESULTS / "diagnose2").glob("*.json"))) < 30:
        time.sleep(60)
    log("diagnostic complete")
    subprocess.run([PY, "diagnose2.py", "--aggregate"], cwd=HERE,
                   stdout=(LOGS / "diagnose2_aggregate.log").open("a"), stderr=subprocess.STDOUT)
    winner = gate()
    if winner:
        jobs = [f"{winner}:{r}:{f}" for r in range(10) for f in range(5)]
        run_parallel("crossfit50.py", "--jobs", jobs, workers=16, threads=4, name="r3_cf",
                     extra=["--threads", "4"])
        log("cross-fits done")
        pairs = [f"{r}:{f}" for r in range(10) for f in range(5)]
        run_parallel("combine_r3.py", "--pairs", pairs, workers=12, threads=4, name="r3_combine",
                     extra=["--winner", winner])
        subprocess.run([PY, "combine_r3.py", "--winner", winner, "--aggregate"], cwd=HERE,
                       stdout=(LOGS / "r3_aggregate.log").open("a"), stderr=subprocess.STDOUT)
    (RESULTS / "R3_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC\n", time.gmtime()))
    log("round 3 done")


if __name__ == "__main__":
    main()
