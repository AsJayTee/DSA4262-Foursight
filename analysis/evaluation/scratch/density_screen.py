"""Screen for train_densities before building it: which thinning, and does it help?

density_mismatch.py measured what a *randomly* thinned evaluation file costs a
cross-site model. But the training file is SG-NEx Hct116 filtered to >= 20
reads per site, so a file from a shallower (or deeper) run is thinned by
COVERAGE, not at random: low-coverage sites drop out first, and they are not
spread evenly along a transcript. This screen measures both perturbations, and
whether training on thinned copies protects against each.

One canonical split (seed 4262), five folds, full-depth training rows only, so
it runs in minutes. Per-fold numbers are printed so the arms can be read
paired. It is a screen, not a result: the harness run is what gets quoted.

    python analysis/evaluation/scratch/density_screen.py
"""
import hashlib
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from m6a import registry
from m6a.data import align_to_features, assign_folds, load_labels
from m6a.feature_cache import extract

DATA = "data0/dataset0.json.gz"
LABELS = "data0/data.info.labelled"
FEATURES = "quantiles_all_v1"
PARAMS = dict(learning_rate=0.05, num_leaves=63, n_estimators=600,
              min_child_samples=40, feature_fraction=0.7)
FLOOR = 20   # the read-count filter the training file was built with

extraction = extract(DATA, FEATURES, [None])[None]
full, meta = extraction.features, extraction.sites
extractor = registry.get("features", FEATURES)()
cross = [c for c in full.columns if c.startswith(("nbr_", "tx_"))]
base = full.drop(columns=cross)
# The derivation of the pre-finalise frame is only valid if re-finalising it
# reproduces the cached table exactly.
assert extractor.finalise(base, meta)[full.columns].equals(full), "base frame is wrong"

joined = align_to_features(full, load_labels(LABELS))
y = joined["label"].to_numpy()
folds = assign_folds(joined[["gene_id"]].reset_index(drop=True), 4262, 5, "gene_id").to_numpy()
index = joined.index
columns = list(full.columns)
n_reads = meta["n_reads"].to_numpy()
print(f"{len(index):,} sites, {y.sum():,} positive, {len(columns)} columns, "
      f"{len(cross)} cross-site: {cross}\n")


def uniform(density_key: str) -> np.ndarray:
    out = np.empty(len(base))
    for i, (t, p) in enumerate(base.index):
        h = hashlib.blake2b(f"4262:{density_key}:{t}:{p}".encode(), digest_size=8).digest()
        out[i] = int.from_bytes(h, "big") / 2.0**64
    return out


def random_view(d: float) -> pd.DataFrame:
    """Every site, each seeing a random `d` of its neighbours plus itself.

    Circular windows [j*d, j*d + d) mod 1 over a per-site uniform: a site is
    finalised inside window floor(u/d), which contains it, and every OTHER site
    falls in that window independently with probability exactly d.
    """
    u = uniform(f"r{d}")
    home = np.floor(u / d).astype(int)
    out = []
    for j in range(int(np.ceil(1 / d))):
        lo = j * d
        inside = ((u - lo) % 1.0) < d
        view = extractor.finalise(base.loc[inside], meta.loc[inside])
        mine = home[inside] == j
        out.append(view.loc[mine])
    return pd.concat(out).loc[base.index, columns]


def coverage_view(f: float) -> pd.DataFrame:
    """The file a run at fraction `f` of our coverage would produce: only sites
    whose expected depth n*f still clears the floor exist, and their cross-site
    columns are computed from each other."""
    kept = n_reads * f >= FLOOR
    return extractor.finalise(base.loc[kept], meta.loc[kept])[columns]


t0 = time.time()
views = {"full": full[columns]}
for d in (0.75, 0.5, 0.25):
    views[f"rand{d}"] = random_view(d)
for f in (0.75, 0.5, 0.25):
    views[f"cov{f}"] = coverage_view(f)
print(f"views built in {time.time() - t0:.0f}s")
for name, v in views.items():
    per_tx = v.groupby(level=0).size()
    print(f"  {name:>8}: {len(v):>7,} sites ({len(v) / len(full):.0%}), "
          f"{per_tx.median():.0f} per transcript (median)")
print()

ARMS = {
    "no augmentation": ["full"],
    "+ random 75/50/25": ["full", "rand0.75", "rand0.5", "rand0.25"],
    "+ coverage 75/50/25": ["full", "cov0.75", "cov0.5", "cov0.25"],
}
SCORED = ["full", "rand0.75", "rand0.5", "rand0.25", "cov0.75", "cov0.5", "cov0.25"]

results = {}
for arm, sources in ARMS.items():
    t0 = time.time()
    per = {s: [] for s in SCORED}
    ref = {s: [] for s in SCORED}   # the same rows scored with full-density features
    for fold in range(5):
        test = folds == fold
        train_idx, test_idx = index[~test], index[test]
        Xs, ys = [], []
        for s in sources:
            v = views[s]
            rows = train_idx.intersection(v.index)
            Xs.append(v.loc[rows])
            ys.append(joined.loc[rows, "label"].to_numpy())
        model = registry.get("models", "lightgbm")(**PARAMS)
        model.fit(pd.concat(Xs), np.concatenate(ys))
        for s in SCORED:
            rows = test_idx.intersection(views[s].index)
            truth = joined.loc[rows, "label"].to_numpy()
            per[s].append(average_precision_score(truth, model.predict_proba(views[s].loc[rows])))
            ref[s].append(average_precision_score(truth, model.predict_proba(full.loc[rows, columns])))
    results[arm] = (per, ref)
    print(f"[{arm}] {time.time() - t0:.0f}s")
    print(f"  {'scored on':>9}  {'PR AUC':>7}  {'same rows, full-density':>24}  {'cost':>8}")
    for s in SCORED:
        a, b = np.mean(per[s]), np.mean(ref[s])
        print(f"  {s:>9}  {a:>7.4f}  {b:>24.4f}  {a - b:>+8.4f}")
    print()

print("Paired against no augmentation, per-fold mean PR AUC on the thinned file:")
print(f"  {'scored on':>9}" + "".join(f"  {arm:>22}" for arm in list(ARMS)[1:]))
for s in SCORED:
    line = f"  {s:>9}"
    for arm in list(ARMS)[1:]:
        diff = np.array(results[arm][0][s]) - np.array(results["no augmentation"][0][s])
        line += f"  {diff.mean():>+12.4f} ({(diff > 0).sum()}/5)    "
    print(line)
