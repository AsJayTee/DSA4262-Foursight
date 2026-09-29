"""Drive R1 (finish the screen) and R2 (the 50-observation test) on a big box.

    nohup python run_remote.py r2 > logs/run_r2.log 2>&1 &
    nohup python run_remote.py r1 > logs/run_r1.log 2>&1 &

Both resume: finished encoders, finished cross-fit jobs and finished
(repetition, fold) combinations are skipped. Python, not sh, so it can be
edited while running without the shell-offset problem run_batch.sh records.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
LOGS = HERE / "logs"

# R1: the three unfinished encoders, the larger deepset, and longer reruns of
# the reconstruction encoders that only fit ~20 epochs in their first 8-minute box.
R1 = ["set_masked", "attn_mil", "contrastive", "deepset_large",
      "read_ae", "pos_lstm_ae", "pos_attn_ae"]
R1_MINUTES = 25
# R2: priority order - the stacking control and deepset decide the question,
# so they go first; the rest follow.
R2_NETWORKS = ["mlp_hand", "deepset", "read_ae_pred", "deepset_large", "attn_mil"]


def env(threads: int) -> dict:
    e = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        e[key] = str(threads)
    return e


def launch(args: list[str], log: Path, threads: int) -> subprocess.Popen:
    fh = log.open("a")
    return subprocess.Popen([PY, "-u", *args], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                            env=env(threads))


def wait_all(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.wait()


def r1(threads: int = 3) -> None:
    import common
    # Every probe's incremental test (E3) reads the hand vectors.
    if not all(common.embedding_path("hand", f).exists() for f in range(5)):
        subprocess.run([PY, "learn.py", "--model", "hand"], cwd=HERE, env=env(8),
                       stdout=(LOGS / "r1_hand.log").open("a"), stderr=subprocess.STDOUT)
    procs = []
    for name in R1:
        # A rerun replaces the first run's vectors; its probe JSON is overwritten too.
        chain = (f"'{PY}' -u learn.py --model {name} --minutes {R1_MINUTES} && "
                 f"'{PY}' -u probe.py --models {name}")
        fh = (LOGS / f"r1_{name}.log").open("a")
        procs.append(subprocess.Popen(["sh", "-c", chain], cwd=HERE, stdout=fh,
                                      stderr=subprocess.STDOUT, env=env(threads)))
    wait_all(procs)
    subprocess.run([PY, "summarise.py"], cwd=HERE, env=env(8),
                   stdout=(LOGS / "summarise.log").open("a"), stderr=subprocess.STDOUT)
    names = [n for n in ["hand", "hand_pca", "random", "mlp_hand", "deepset", "deepset_large",
                         "read_ae", "read_ae_pred", "pos_lstm_ae", "pos_attn_ae", "set_masked",
                         "attn_mil", "contrastive", "deepset_nokmer", "kmer_norm"]
             if all(common.embedding_path(n, f).exists() for f in range(5))]
    subprocess.run([PY, "visualise.py", "--models", *names], cwd=HERE, env=env(16),
                   stdout=(LOGS / "visualise.log").open("a"), stderr=subprocess.STDOUT)
    print("r1 done", flush=True)


def r2(workers: int = 14, threads: int = 3, combiners: int = 12) -> None:
    jobs = [f"{m}:{r}:{f}" for m in R2_NETWORKS for r in range(10) for f in range(5)]
    # Round-robin so every worker gets its share of the high-priority jobs first.
    shards = [jobs[i::workers] for i in range(workers)]
    started = time.time()
    wait_all([launch(["crossfit50.py", "--threads", str(threads), "--jobs", *shard],
                     LOGS / f"r2_worker{i}.log", threads) for i, shard in enumerate(shards)])
    print(f"cross-fits done ({(time.time() - started) / 3600:.1f} h)", flush=True)
    pairs = [f"{r}:{f}" for r in range(10) for f in range(5)]
    wait_all([launch(["combine50.py", "--pairs", *pairs[i::combiners]],
                     LOGS / f"r2_combine{i}.log", 4) for i in range(combiners)])
    subprocess.run([PY, "combine50.py", "--aggregate"], cwd=HERE,
                   stdout=(LOGS / "r2_aggregate.log").open("a"), stderr=subprocess.STDOUT)
    print(f"r2 done ({(time.time() - started) / 3600:.1f} h)", flush=True)


if __name__ == "__main__":
    LOGS.mkdir(exist_ok=True)
    {"r1": r1, "r2": r2}[sys.argv[1]]()
