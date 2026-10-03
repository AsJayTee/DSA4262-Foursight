"""m6Anet under cross-source evaluation (decisions 0029, 0032), and every model
measured against it.

    python analysis/m6anet/xsrc_m6anet.py            # needs the two site_proba files below

The pretrained HCT116_RNA002 model scores every site of both files; those
scores are put on the cross-source union split and compared, held-out gene by
held-out gene, with:

  - quantiles + LightGBM (the baseline every Decisions row is measured against)
  - the networks of analysis/representation/xsrc_nets.py, trained on dataset0

**Read the m6Anet numbers as an upper bound for m6Anet.** It was trained on
HCT116 m6ACE-seq sites - very likely the source of dataset0's labels - so it
may have seen the answers to every dataset0 site and to the 74% of data1 sites
shared with it. A model of ours that beats it anyway beats it fairly; one that
loses to it may only be losing to that head start.

Inputs (from analysis/m6anet/run_ronin.sh `pretrained`, and the same on data1):
  data0/m6anet_out/data.site_proba.csv
  data/m6anet/data1_pretrained_out/data.site_proba.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("M6A_CACHE_DIR", str(ROOT / ".cache" / "features"))

from m6a import crosssource as xs  # noqa: E402
from m6a import report as reporting, tracking  # noqa: E402
from m6a.data import SUBSAMPLE_SEED, resolve_data_dir  # noqa: E402
from m6a.env import load_env  # noqa: E402

NETS = ROOT / ".cache" / "representation" / "xsrc_nets"
COMPARE = ("h2gcn", "h2gcn_local", "deepset", "attn_mil", "set_transformer")


def m6anet_scores(path: Path, index: pd.MultiIndex) -> np.ndarray:
    """m6Anet's site probabilities in `index` order; NaN where it gave none."""
    site = pd.read_csv(path).set_index(["transcript_id", "transcript_position"])["probability_modified"]
    scores = site.reindex(index).to_numpy()
    missing = int(np.isnan(scores).sum())
    # m6Anet dropped one transcript's 10 sites of data1's 90,810 (2026-10-03);
    # a handful is tolerable, a September-style partial run (28k of 122k) is not.
    if missing > 0.001 * len(index):
        raise SystemExit(f"{path}: {missing:,} of {len(index):,} labelled sites have no m6Anet score - "
                         "was inference run on the full file? (run_ronin.sh pretrained)")
    return scores


def keep_scored(sources: dict, scores: dict, *others: dict) -> tuple:
    """Drop the sites m6Anet did not score from the sources AND every model's
    scores, so every comparison is on identical sites."""
    keep = {s: ~np.isnan(scores["dataset0"][s]) for s in xs.SOURCES}
    for s in xs.SOURCES:
        if (~keep[s]).any():
            print(f"  {s}: {int((~keep[s]).sum())} sites without an m6Anet score left out of every comparison")
    thin = {s: xs.Source(s, src.index[keep[s]], {d: X.iloc[keep[s]] for d, X in src.X.items()},
                         src.y[keep[s]], src.genes[keep[s]], src.folds[keep[s]])
            for s, src in sources.items()}
    cut = lambda block: {arm: {s: v[keep[s]] for s, v in by.items()} for arm, by in block.items()}  # noqa: E731
    return (thin, cut(scores), *(None if o is None else cut(o) for o in others))


def network(model: str, sources) -> dict | None:
    """A network's out-of-fold scores, trained on dataset0 (the arm that ships)."""
    n0 = len(sources["dataset0"])
    paths = [NETS / f"{model}_dataset0_fold{f}.npy" for f in range(5)]
    if not all(p.exists() for p in paths):
        return None
    total = np.nanmax(np.stack([np.load(p) for p in paths]), 0)
    return {"dataset0": {"dataset0": total[:n0], "data1": total[n0:]}}


def row(name: str, block: dict) -> dict:
    a = block["arms"]["dataset0"]
    g = block.get("data1_new_genes", {})
    return {"model": name, "dataset0 gain": a["dataset0"]["gain"], "data1 gain": a["data1"]["gain"],
            "data1 95% CI": f"[{a['data1']['ci_low']:+.4f}, {a['data1']['ci_high']:+.4f}]",
            "worst gain": a["gain_worst"], "new genes gain": g.get("gain", np.nan),
            "new genes 95% CI": f"[{g.get('ci_low', np.nan):+.4f}, {g.get('ci_high', np.nan):+.4f}]"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not data0/m6anet_out: the September run there scored 28,357 of 121,838 sites.
    ap.add_argument("--dataset0", default="data/m6anet/dataset0_pretrained_out/data.site_proba.csv")
    ap.add_argument("--data1", default="data/m6anet/data1_pretrained_out/data.site_proba.csv")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    load_env(ROOT / ".env")
    sources = xs.load("quantiles_v1", [None], resolve_data_dir(), log=lambda *a: None)
    m6a = {"dataset0": {"dataset0": m6anet_scores(ROOT / args.dataset0, sources["dataset0"].index),
                        "data1": m6anet_scores(ROOT / args.data1, sources["data1"].index)}}
    stored = np.load(NETS / "baseline_lgbm.npz")
    lgbm = {"dataset0": {s: stored[f"dataset0__{s}"] for s in xs.SOURCES}}
    nets = {model: network(model, sources) for model in COMPARE}
    sources, m6a, lgbm, *cut_nets = keep_scored(sources, m6a, lgbm, *nets.values())
    nets = dict(zip(nets, cut_nets))

    report = reporting.new("standard", subsample_seed=SUBSAMPLE_SEED)
    # m6Anet as a candidate against the usual baseline, so it gets a Decisions row.
    block = xs.summarise(sources, m6a, lgbm)
    reporting.cross_source(report, block, "m6Anet (pretrained HCT116)", "lightgbm_quantiles")
    raw = {s: block["arms"]["dataset0"][s]["pr_auc"] for s in xs.SOURCES}

    # Then every model against m6Anet: m6Anet as the baseline.
    rows = [row("quantiles + LightGBM", xs.summarise(sources, lgbm, m6a))]
    for model, cand in nets.items():
        if cand is not None:
            rows.append(row(f"{model} (network)", xs.summarise(sources, cand, m6a)))
    table = pd.DataFrame(rows).set_index("model")
    report.heading("Every model against m6Anet (trained on dataset0; gain = model minus m6Anet)")
    report.show(table, floats="%+.4f")
    report.log(f"m6Anet PR AUC: dataset0 held-out {raw['dataset0']:.4f}, data1 held-out {raw['data1']:.4f}.\n"
               "m6Anet's HCT116 training data very likely includes dataset0's sites and labels - "
               "these are its best case.")
    report.data["vs_m6anet"] = json.loads(table.reset_index().to_json(orient="records"))
    path = reporting.write(report, ROOT / "analysis" / "m6anet", "m6anet_pretrained__xsrc")

    tracker = tracking.start(
        "m6anet_pretrained__xsrc", enabled=not args.no_wandb, job_type="evaluate",
        tags=[tracking.EVAL_TAG],
        config={"eval_schema": xs.EVAL_SCHEMA, "baseline": "lightgbm_quantiles", "arms": "dataset0",
                "features": "m6Anet (pretrained HCT116_RNA002)", "model": "m6anet",
                "train_depths": "n/a (pretrained)", "script": "analysis/m6anet/xsrc_m6anet.py"},
        notes="Pretrained on HCT116 - likely saw dataset0's labels; an upper bound for m6Anet.")
    try:
        reporting.publish(report, tracker, path, name="m6anet_pretrained__xsrc")
    finally:
        tracker.finish()
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
