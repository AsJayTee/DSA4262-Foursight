"""10 Oct (approved): seeds 1 and 2 of the report-figure rows that had seed 0 only.

    nohup python -u analysis/representation/run_batch3.py > analysis/representation/logs/batch3.log 2>&1 &

DeepSet, Set Transformer, GCN, GAT and H2GCN at 400 nt, both arms, so every
network row of report_models.py's distribution figure has 15 points (5 folds x
3 seeds). Logistic regression is deterministic and m6Anet is a fixed pretrained
model: more seeds add nothing for them.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NETS = [sys.executable, "-u", str(HERE / "xsrc_nets.py"), "all", "--threads", "4", "--minutes", "720", "--jobs", "16"]
MODELS = "set_transformer,gat,gcn,h2gcn_aux_r400,deepset"     # slowest first


def main() -> None:
    for seed in (1, 2):
        print(time.strftime("%H:%M:%S"), f"start seed{seed}", flush=True)
        with open(HERE / "logs" / f"batch3_seed{seed}.log", "w") as out:
            code = subprocess.run([*NETS, "--model", MODELS, "--seed", str(seed)], cwd=ROOT,
                                  stdout=out, stderr=subprocess.STDOUT).returncode
        print(time.strftime("%H:%M:%S"), f"{'done' if code == 0 else 'FAILED'} seed{seed} (exit {code})", flush=True)
    print(time.strftime("%H:%M:%S"), "batch 3 finished", flush=True)


if __name__ == "__main__":
    main()
