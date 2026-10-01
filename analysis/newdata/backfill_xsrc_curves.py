"""Add the cross-source curves to W&B runs logged before they existed.

    python analysis/newdata/backfill_xsrc_curves.py

`report.cross_source_series` turns a run's cross-source block into the
`curve/xsrc/*` series the Decisions view draws (docs/decisions/0031). Runs
logged before 2026-10-01 have the block only as flat `xsrc/*` summary keys; this
rebuilds the block from those, and appends the curves to the run. Runs that
already have the curves are skipped, so it is safe to re-run.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.env import load_env  # noqa: E402
from m6a.report import ARM_ORDER, cross_source_series  # noqa: E402

STATS = ("pr_auc", "gain", "ci_low", "ci_high")


def block_from_summary(s: dict) -> dict:
    arms = {}
    for arm in ARM_ORDER:
        if f"xsrc/{arm}/gain_mean" not in s:
            continue
        arms[arm] = {src: {stat: s[f"xsrc/{arm}/{src}/{stat}"] for stat in STATS}
                     for src in ("dataset0", "data1")}
        arms[arm].update(gain_mean=s[f"xsrc/{arm}/gain_mean"])
    block = {"arms": arms, "headline_arm": "dataset0"}
    crossed = {c: {stat: s[f"xsrc/crossed/{c}/{stat}"] for stat in STATS[1:]}
               for c in "ABCD" if f"xsrc/crossed/{c}/gain" in s}
    if crossed:
        block["crossed"] = crossed
    if "xsrc/data1_new_genes/gain" in s:
        block["data1_new_genes"] = {stat: s[f"xsrc/data1_new_genes/{stat}"] for stat in STATS[1:]}
    return block


def main() -> None:
    load_env(ROOT / ".env")
    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    project = os.environ.get("WANDB_PROJECT", "dsa4262-project")
    for run in api.runs(f"{entity}/{project}", filters={"config.eval_schema": {"$gte": 3}}):
        if run.state != "finished":
            print(f"skip {run.name}: {run.state}")
            continue
        if list(run.scan_history(keys=["curve/xsrc/transfer/gain"], page_size=10)):
            print(f"skip {run.name}: already has curves")
            continue
        series = cross_source_series(block_from_summary(dict(run.summary)))
        live = wandb.init(entity=entity, project=project, id=run.id, resume="must")
        for x_key, (_, ys) in series.items():
            live.define_metric(x_key, summary="none")
            for y_key in ys:
                live.define_metric(y_key, step_metric=x_key, summary="none")
        # Steps can only move forward on a resumed run, so the curves go after
        # whatever it already logged; the panels plot against x, not step.
        # Not live.step: on a resumed run it reads 0, and a row logged at a step
        # already used is silently dropped.
        start = run.lastHistoryStep + 1
        longest = max(len(xs) for xs, _ in series.values())
        for i in range(longest):
            row = {}
            for x_key, (xs, ys) in series.items():
                if i < len(xs):
                    row[x_key] = float(xs[i])
                    # NaN marks a reference-line row with no data point.
                    row.update({k: float(v[i]) for k, v in ys.items() if math.isfinite(v[i])})
            live.log(row, step=start + i)
        live.finish()
        print(f"done {run.name}")


if __name__ == "__main__":
    main()
