"""Run the whole screen: learn -> probe -> summarise, encoder by encoder.

    python analysis/representation/run_all.py

Python rather than sh on purpose: a shell script reads itself incrementally and
breaks if edited mid-run (analysis/evaluation/run_batch.sh records how). An
encoder whose five fold files already exist is not retrained; one that fails is
logged and skipped so the rest still run. Results appear after every encoder,
so a run stopped early still leaves a usable partial summary.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
PY = sys.executable
LOGS = HERE / "logs"
ORDER = ["hand", "hand_pca", "random", "mlp_hand", "deepset", "read_ae", "read_ae_pred",
         "pos_lstm_ae", "pos_attn_ae", "set_masked", "attn_mil", "contrastive"]


def run(args: list[str], log: Path) -> bool:
    with log.open("a") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        return subprocess.run([PY, *args], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode == 0


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    failed = []
    for name in ORDER:
        started = time.time()
        done = all(common.embedding_path(name, f).exists() for f in range(5))
        ok = done or run(["learn.py", "--model", name], LOGS / f"learn_{name}.log")
        ok = ok and run(["probe.py", "--models", name], LOGS / f"probe_{name}.log")
        if not ok:
            failed.append(name)
        run(["summarise.py"], LOGS / "summarise.log")
        print(f"{name}: {'ok' if ok else 'FAILED'} ({(time.time() - started) / 60:.1f} min)", flush=True)
    finished = [n for n in ORDER if n not in failed]
    run(["visualise.py", "--models", *finished], LOGS / "visualise.log")
    print(f"all done; failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
