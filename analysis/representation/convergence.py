"""Did each xsrc_nets.py network converge, or did the time cap stop it early?

    python analysis/representation/convergence.py            # every model
    python analysis/representation/convergence.py --model gat

Reads analysis/representation/logs/xsrc_nets_fit_<model>_<fold>.log. Per
network (model x arm x fold):

  stop       "patience" - 60 epochs without a better validation AP, so it had
             peaked (early stopping restores the best epoch: past-peak
             overfitting never reaches the scores); "cap" - the time limit
  best/ran   the epoch with the best validation AP, and epochs run
  late gain  validation AP at the best epoch minus the best over the first 75%
             of epochs run: what the last quarter still bought

UNDERTRAINED = stopped by the cap with its best epoch in the last 10% of the
run, or a late gain above 0.005. Such a model is not given a fair comparison:
rerun it with a longer cap and a schedule that completes inside it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

LOGS = Path(__file__).resolve().parent / "logs"
HEADER = re.compile(r"^\[(\S+) (\S+) fold (\d+)(?: seed (\d+))?\]$")
EPOCH = re.compile(r"epoch (\d+):.*?val_ap ([0-9.]+)")


def parse(path: Path) -> list[dict]:
    runs, current = [], None
    for line in path.read_text(errors="ignore").splitlines():
        if m := HEADER.match(line.strip()):
            current = {"model": m[1], "arm": m[2], "fold": int(m[3]), "seed": int(m[4] or 0),
                       "aps": [], "stop": "cap", "done": False, "tag": line.strip()}
            runs.append(current)
        elif current is None:
            continue
        elif m := EPOCH.search(line):
            current["aps"].append(float(m[2]))
        elif "stopping: no validation gain" in line:
            current["stop"] = "patience"
        elif line.startswith(f"{current['tag']} done"):
            current["done"] = True
    return runs


def verdict(run: dict) -> dict:
    aps = run["aps"]
    ran = len(aps)
    best = max(range(ran), key=aps.__getitem__) + 1 if ran else 0
    early = aps[: max(1, int(ran * 0.75))]
    late = (max(aps) - max(early)) if ran else 0.0
    # Only a finished network has a verdict: a running one has not stopped yet.
    under = run["done"] and run["stop"] == "cap" and (best > 0.9 * ran or late > 0.005)
    return {**run, "ran": ran, "best": best, "best_ap": max(aps) if aps else float("nan"),
            "late": late, "undertrained": under}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    rows = [verdict(r) for p in sorted(LOGS.glob("xsrc_nets_fit_*.log"))
            if "genes" not in p.name for r in parse(p)
            if not args.model or r["model"] == args.model]
    print(f"{'model':16s} {'arm':12s} fold  stop      best/ran  best AP  late gain")
    for r in sorted(rows, key=lambda r: (r["model"], r["arm"], r["seed"], r["fold"])):
        flag = "UNDERTRAINED" if r["undertrained"] else ("" if r["done"] else "(running)")
        name = r["model"] + (f" s{r['seed']}" if r["seed"] else "")
        print(f"{name:16s} {r['arm']:12s} {r['fold']:4d}  {r['stop']:8s} {r['best']:4d}/{r['ran']:<4d} "
              f"{r['best_ap']:.4f}  {r['late']:+.4f}  {flag}")
    finished = [r for r in rows if r["done"]]
    for model in sorted({r["model"] for r in finished}):
        mine = [r for r in finished if r["model"] == model]
        n = sum(r["undertrained"] for r in mine)
        print(f"{model}: {n}/{len(mine)} finished networks undertrained"
              + (" - rerun with a longer budget before comparing" if n else ""))


if __name__ == "__main__":
    main()
