"""The overnight queue of 2026-10-02, in order, unattended on Ronin.

    nohup python -u analysis/representation/run_overnight.py > analysis/representation/logs/overnight.log 2>&1 &

1. waits for the running slate (`xsrc_nets.py all`, no --model) to finish
2. the k-NN read graphs: knn_static, knn_dynamic, knn_random
3. ideas 1 and 2: h2gcn_twohead, h2gcn_noisy (both files only),
   h2gcn_local, h2gcn_transcript
4. idea 4, h2gcn_knn - only if the gate below passes

**The gate was fixed before any k-NN result existed:** h2gcn_knn runs only if
knn_static beats knn_random (the shuffled-graph control) on data1 PR AUC in
both training arms, pooled out of fold. If the read graph's structure does not
help on its own, putting it inside h2gcn is not worth the night.

Python, not sh: a shell script read incrementally breaks if edited mid-run.
Each stage is `xsrc_nets.py all`, which skips finished networks, so rerunning
this resumes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NETS = [sys.executable, "-u", str(HERE / "xsrc_nets.py"), "all", "--jobs", "16", "--threads", "4"]


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def slate_running() -> bool:
    out = subprocess.run(["pgrep", "-af", "xsrc_nets.py all"], capture_output=True, text=True).stdout
    return any("--model" not in line and "pgrep" not in line for line in out.splitlines())


def stage(name: str, models: str) -> None:
    log(f"start {name}: {models}")
    with open(HERE / "logs" / f"xsrc_nets_{name}.log", "w") as out:
        code = subprocess.run([*NETS, "--model", models], cwd=ROOT, stdout=out,
                              stderr=subprocess.STDOUT).returncode
    log(f"{'done' if code == 0 else 'FAILED'} {name} (exit {code})")


def knn_gate() -> bool:
    sys.path.insert(0, str(HERE))
    from sklearn.metrics import average_precision_score
    import xsrc_nets as X

    sources, _ = X.load(1.0)
    static, random = X.candidate("knn_static", sources, 1.0), X.candidate("knn_random", sources, 1.0)
    if static is None or random is None:
        log("gate: k-NN results incomplete - h2gcn_knn skipped")
        return False
    y = sources["data1"].y
    passed = True
    for arm in X.ARMS:
        a = average_precision_score(y, static[arm]["data1"])
        b = average_precision_score(y, random[arm]["data1"])
        log(f"gate: {arm:12s} data1 PR AUC knn_static {a:.4f} vs knn_random {b:.4f} ({a - b:+.4f})")
        passed &= a > b
    log(f"gate: {'PASS - running h2gcn_knn' if passed else 'FAIL - h2gcn_knn skipped'}")
    return passed


def main() -> None:
    while slate_running():
        time.sleep(60)
    log("slate finished")
    stage("knn", "knn_static,knn_dynamic,knn_random")
    stage("variants", "h2gcn_twohead,h2gcn_noisy,h2gcn_local,h2gcn_transcript")
    if knn_gate():
        stage("h2gcn_knn", "h2gcn_knn")
    log("overnight queue finished")


if __name__ == "__main__":
    main()
