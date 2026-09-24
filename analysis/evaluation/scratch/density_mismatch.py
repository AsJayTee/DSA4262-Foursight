"""Train on a dense candidate file, predict on a sparse one.

Bias in sd units says the features MOVE. It does not say what that costs in
PR AUC, and that is the number a shipping decision needs. This simulates the
deployment mismatch directly: fold models are fitted on full-density features,
exactly as every recorded run fits them, and then the held-out fold is scored
twice - once with the features it was trained to expect, and once with the
features it would actually receive if the evaluation file were thinner.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from m6a import registry
from m6a.feature_cache import extract
from m6a.data import load_labels, align_to_features, assign_folds

FEATURES = "quantiles_flank_crosssite_v1"
PARAMS = dict(learning_rate=0.05, num_leaves=63, n_estimators=600,
              min_child_samples=40, feature_fraction=0.7)

base = extract("data0/dataset0.json.gz", "quantiles_v1", [None])[None]
extractor = registry.get("features", FEATURES)()

full_frame = extract("data0/dataset0.json.gz", FEATURES, [None])[None].features
joined = align_to_features(full_frame, load_labels("data0/data.info.labelled"))
y = joined["label"].to_numpy()
folds = assign_folds(joined[["gene_id"]].reset_index(drop=True), 4262, 5, "gene_id").to_numpy()
X = joined.drop(columns=["gene_id", "label"])
print(f"{len(X):,} sites, {y.sum():,} positives, {X.shape[1]} columns\n")

# Thinned-file features: recompute the cross-site pass over a subset of sites,
# then keep the rows we still have labels for. The per-site columns are
# untouched; only the neighbourhood changes, which is the whole point.
flank_only = registry.get("features", "quantiles_flank_v1")()
per_site = full_frame[[c for c in full_frame.columns
                       if not c.startswith(("nbr_", "tx_"))]]
sites_meta = extract("data0/dataset0.json.gz", FEATURES, [None])[None].sites

print(f"{'kept':>6}  {'trained-on features':>19}  {'thinned features':>17}  {'cost':>8}")
rng = np.random.default_rng(4262)
for keep in (1.00, 0.75, 0.50, 0.25):
    if keep == 1.0:
        thinned = full_frame
    else:
        mask = rng.random(len(per_site)) < keep
        sub = per_site.loc[mask]
        thinned = extractor.finalise(sub.copy(), sites_meta.loc[mask])
    shared = X.index.intersection(thinned.index)
    Xthin = thinned.loc[shared, X.columns]

    dense_scores, thin_scores = {}, {}
    for f in range(5):
        test = folds == f
        model = registry.get("models", "lightgbm")(**PARAMS)
        model.fit(X.loc[~test], y[~test])
        here = X.index[test].intersection(shared)
        dense_scores.update(zip(here, model.predict_proba(X.loc[here])))
        thin_scores.update(zip(here, model.predict_proba(Xthin.loc[here])))

    idx = list(dense_scores)
    truth = joined.loc[idx, "label"].to_numpy()
    a = average_precision_score(truth, [dense_scores[i] for i in idx])
    b = average_precision_score(truth, [thin_scores[i] for i in idx])
    print(f"{keep:>5.0%}  {a:>19.4f}  {b:>17.4f}  {b - a:>+8.4f}")
