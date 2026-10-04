"""Export the final-fit networks to the numpy files predict.py loads.

    python analysis/representation/export_final.py                     # -> models/final/
    python analysis/representation/export_final.py --out /tmp/check --genes 0.05   # smoke

Reads .cache/representation/final/<model>_s<seed>.pt (final_fit.py) and writes,
per network, an .npz of its weights plus its read standardisation and
constructor arguments, and a meta.json naming the ensemble for
m6a.models.site_graph (docs/decisions/0033). A LightGBM model found in <out>
is moved aside to <out>/lightgbm/ rather than overwritten; the placeholder
that shipped until 2026-10-04 was then deleted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import common
import final_fit
import xsrc_nets as X


def export_network(saved: dict, path: Path) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in saved["state_dict"].items()}
    np.savez(path, **arrays, _read_mean=np.asarray(saved["read_mean"], np.float32),
             _read_scale=np.asarray(saved["read_scale"], np.float32),
             _meta=json.dumps({"model": saved["model"], "seed": saved.get("seed"),
                               "kwargs": saved["kwargs"], "best_epoch": saved.get("best_epoch"),
                               "best_val_ap": saved.get("best_val_ap")}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(common.ROOT / "models" / "final"))
    ap.add_argument("--genes", type=float, default=1.0, help="export the smoke fits instead")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old_meta = out / "meta.json"
    if old_meta.exists() and json.loads(old_meta.read_text()).get("model") == "lightgbm":
        keep = out / "lightgbm"
        keep.mkdir(exist_ok=True)
        for f in ("meta.json", "model.txt", "columns.json"):
            if (out / f).exists():
                shutil.move(str(out / f), keep / f)
        print(f"previous LightGBM model kept at {keep}")
    networks = []
    for model, seed in final_fit.ENSEMBLE:
        src = final_fit.OUT / f"{model}_s{seed}{X.suffix(args.genes)}.pt"
        saved = torch.load(src, weights_only=False)
        name = f"{model}_s{seed}.npz"
        export_network(saved, out / name)
        networks.append(name)
        print(f"{src.name} -> {name} (best epoch {saved['best_epoch']}, validation AP {saved['best_val_ap']:.4f})")
    (out / "meta.json").write_text(json.dumps({
        "name": "site_graph_ensemble",
        "model": "site_graph_ensemble",
        "networks": networks,
        "combine": "mean of within-file ranks",
        "trained_on": "dataset0 + data1, every labelled site (pooled_both)",
        "evaluation": "analysis/representation/results/ (cross-source, decision 0032)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"meta.json -> {out}")


if __name__ == "__main__":
    main()
