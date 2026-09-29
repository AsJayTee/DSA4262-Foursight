"""How do models trained on dataset0 score on the course's later releases?

    python analysis/newdata/score_new_data.py

Each model is trained on ALL of dataset0, exactly as it would be shipped
(configs/quantiles.yaml, configs/everything.yaml, and everything without its
seven cross-site columns), then scores data1 and data2. Nothing here tunes on
the new data.

data1 is reported in slices, because 74% of its sites are also in dataset0 and
the model has seen those sites (with dataset0's labels, which differ on 5.75%):

  new sites         not in dataset0 - the honest external test
  new transcripts   not even on a dataset0 transcript - the strictest slice
  shared sites      seen in training; shown only for contrast

data2's labels are the fraction of molecules modified per transcript, so it is
reported as mean score per fraction and a rank correlation, not PR AUC.

Writes analysis/newdata/results.md and results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402
from m6a.config import Config  # noqa: E402
from m6a.crossval import build_datasets, stack_rows  # noqa: E402
from m6a.evaluation import metrics  # noqa: E402
from m6a.feature_cache import extract  # noqa: E402

DATA = ROOT / "data0"
HERE = Path(__file__).resolve().parent
CROSS = ("nbr_", "tx_")


def train(config_path: str, drop_cross: bool = False):
    config = Config.load(ROOT / config_path)
    depths = list(dict.fromkeys(config.train_depths + [None]))
    sets = build_datasets(DATA / "dataset0.json.gz", DATA / "data.info.labelled", config.features,
                          depths, log=lambda *a: None)
    columns = [c for c in sets[None].columns if not (drop_cross and c.startswith(CROSS))]
    X, y, _ = stack_rows([sets[d] for d in config.train_depths], columns)
    model = registry.get("models", config.model)(**config.model_params)
    model.fit(X, y)
    return model, config.features, columns


def score(model, features, columns, json_path):
    frame = extract(json_path, features, [None], log=print)[None].features
    return pd.Series(model.predict_proba(frame[columns]), index=frame.index)


def main() -> None:
    train_keys = pd.read_csv(DATA / "data.info.labelled").set_index(
        ["transcript_id", "transcript_position"]).index
    train_tx = set(train_keys.get_level_values(0))
    d1 = pd.read_csv(DATA / "data1" / "data.info").set_index(["transcript_id", "transcript_position"])
    d2 = pd.read_csv(DATA / "data2" / "data.info").set_index(["transcript_id", "transcript_position"])
    shared = d1.index.isin(train_keys)
    new_tx = ~d1.index.get_level_values(0).isin(train_tx)
    slices = {"all data1": np.ones(len(d1), bool), "new sites": ~shared,
              "new transcripts": new_tx, "shared sites (seen)": shared}

    models = {
        "quantiles (0.4759 in CV)": ("configs/quantiles.yaml", False),
        "everything (0.5408 in CV)": ("configs/everything.yaml", False),
        "everything minus cross-site": ("configs/everything.yaml", True),
    }
    out, lines = {}, []
    for name, (cfg, drop) in models.items():
        print(f"== {name}: training on all of dataset0", flush=True)
        model, features, columns = train(cfg, drop)
        s1 = score(model, features, columns, DATA / "data1" / "dataset1.json.gz").reindex(d1.index)
        s2 = score(model, features, columns, DATA / "data2" / "dataset2.json.gz").reindex(d2.index)
        res = {}
        for label, mask in slices.items():
            y = d1.label.to_numpy()[mask]
            m = metrics(y, s1.to_numpy()[mask])
            res[label] = {"n": int(mask.sum()), "positive_rate": float(y.mean()), **m}
        by_frac = pd.DataFrame({"fraction": d2.label, "score": s2}).groupby("fraction")["score"]
        rho = spearmanr(d2.label, s2).correlation
        full_vs_none = metrics((d2.label[d2.label.isin([0.0, 1.0])] == 1.0).to_numpy().astype(int),
                               s2[d2.label.isin([0.0, 1.0])].to_numpy())
        res["data2"] = {"mean_score_by_fraction": by_frac.mean().round(4).to_dict(),
                        "spearman_site_score_vs_fraction": float(rho),
                        "roc_auc_100pct_vs_0pct": full_vs_none["roc_auc"]}
        out[name] = res
        print(json.dumps(res, indent=1, default=float), flush=True)

    lines.append("# Models trained on dataset0, scored on the later releases\n")
    lines.append("## data1 (PR AUC / ROC AUC; lift = PR AUC over the slice's positive rate)\n")
    lines.append("| model | " + " | ".join(f"{k} (n={out[next(iter(out))][k]['n']:,}, "
                                         f"{out[next(iter(out))][k]['positive_rate']:.1%} pos)"
                                         for k in slices) + " |")
    lines.append("|---|" + "---|" * len(slices))
    for name, res in out.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{res[k]['pr_auc']:.4f} / {res[k]['roc_auc']:.4f} ({res[k]['pr_auc_lift']:.1f}x)"
            for k in slices) + " |")
    lines.append("\n## data2 (mean score per fraction of molecules modified)\n")
    fracs = sorted(next(iter(out.values()))["data2"]["mean_score_by_fraction"])
    lines.append("| model | " + " | ".join(f"{f:.0%}" for f in fracs) +
                 " | Spearman (site score vs fraction) | ROC AUC 100% vs 0% |")
    lines.append("|---|" + "---:|" * (len(fracs) + 2))
    for name, res in out.items():
        d = res["data2"]
        lines.append(f"| {name} | " + " | ".join(f"{d['mean_score_by_fraction'][f]:.3f}" for f in fracs)
                     + f" | {d['spearman_site_score_vs_fraction']:.3f} | {d['roc_auc_100pct_vs_0pct']:.3f} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
