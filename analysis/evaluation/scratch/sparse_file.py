"""How does the best model score on a file where most sites are alone on their transcript?

`data/sample/` - the file other students run predict.py on - is 1,000 random
sites over 884 transcripts, 778 of them alone. The cross-site columns
(docs/decisions/0025) are computed from whatever else is in the file, so on a
file like that they are mostly zeros, a regime ~1% of training sites are in.

Fits `configs/everything.yaml` exactly (depth-augmented, same params) on the
canonical split, and scores each held-out fold with its cross-site columns
recomputed as if the file were thinned. Beside it, the same model with the
cross-site columns removed - the fallback whose score cannot depend on the file.

    python analysis/evaluation/scratch/sparse_file.py
"""
import hashlib
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from m6a import registry
from m6a.config import Config
from m6a.crossval import build_datasets
from m6a.feature_cache import extract

config = Config.load("configs/everything.yaml")
DATA, LABELS = "data0/dataset0.json.gz", "data0/data.info.labelled"
depths = list(dict.fromkeys(config.train_depths + [None]))
datasets = build_datasets(DATA, LABELS, config.features, depths, log=lambda *a: None)
full = datasets[None]
extraction = extract(DATA, config.features, [None], log=lambda *a: None)[None]
extractor = registry.get("features", config.features)()

cross = [c for c in full.columns if c.startswith(("nbr_", "tx_"))]
base = extraction.features.drop(columns=cross)
meta = extraction.sites
assert extractor.finalise(base, meta)[extraction.features.columns].equals(extraction.features)


def singleton_view() -> pd.DataFrame:
    """Every site alone on its transcript: give each one a transcript of its own."""
    alone = pd.MultiIndex.from_arrays(
        [[f"{t}#{p}" for t, p in base.index], base.index.get_level_values(1)],
        names=base.index.names)
    view = extractor.finalise(base.set_axis(alone), meta.set_axis(alone))
    return view.set_axis(base.index)


def random_view(d: float) -> pd.DataFrame:
    """Every site, seeing a random fraction d of its neighbours (see density_screen.py)."""
    u = np.array([int.from_bytes(hashlib.blake2b(f"4262:r{d}:{t}:{p}".encode(),
                  digest_size=8).digest(), "big") / 2.0**64 for t, p in base.index])
    home = np.floor(u / d).astype(int)
    parts = []
    for j in range(int(np.ceil(1 / d))):
        inside = ((u - j * d) % 1.0) < d
        view = extractor.finalise(base.loc[inside], meta.loc[inside])
        parts.append(view.loc[home[inside] == j])
    return pd.concat(parts).loc[base.index]


t0 = time.time()
views = {"normal file": extraction.features}
for d in (0.10, 0.05, 0.01):
    views[f"random {d:.0%} of sites"] = random_view(d)
views["every site alone"] = singleton_view()
for name, v in views.items():
    v = v.loc[full.X.index, full.columns]
    alone = (v["tx_n_reads_loo_mean"] == 0).mean()
    views[name] = v
    print(f"  {name:>22}: {alone:.0%} of sites have no other site on their transcript")
print(f"views built in {time.time() - t0:.0f}s\n")

ARMS = {"everything": list(full.columns),
        "everything minus cross-site": [c for c in full.columns if c not in cross]}
model_class = registry.get("models", config.model)
per = {(arm, v): [] for arm in ARMS for v in views}
pooled = {key: np.zeros(len(full.y)) for key in per}
for fold in range(5):
    test = full.folds == fold
    for arm, columns in ARMS.items():
        t0 = time.time()
        model = model_class(**config.model_params)
        model.fit(pd.concat([datasets[d].X.loc[~test, columns] for d in config.train_depths]),
                  np.tile(full.y[~test], len(config.train_depths)))
        for v, X in views.items():
            scores = model.predict_proba(X.loc[test, columns])
            pooled[(arm, v)][test] = scores
            per[(arm, v)].append(average_precision_score(full.y[test], scores))
        print(f"  fold {fold} {arm}: {time.time() - t0:.0f}s", flush=True)

print(f"\n{'scored on':>24}" + "".join(f"  {a:>28}" for a in ARMS))
for v in views:
    line = f"{v:>24}"
    for arm in ARMS:
        line += (f"  PR {average_precision_score(full.y, pooled[(arm, v)]):.4f}"
                 f" ROC {roc_auc_score(full.y, pooled[(arm, v)]):.4f}    ")
    print(line)

print("\nPer fold, everything minus the fallback (positive = keeping cross-site helps):")
for v in views:
    diff = np.array(per[("everything", v)]) - np.array(per[("everything minus cross-site", v)])
    print(f"{v:>24}  {diff.mean():+.4f}  wins {(diff > 0).sum()}/5  "
          f"[{', '.join(f'{x:+.4f}' for x in diff)}]")
