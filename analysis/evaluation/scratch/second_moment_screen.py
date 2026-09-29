"""Screen B's premise before paying for it: is there second-moment signal left?

The proposed MIL change appends the 36 pairwise products of the nine per-read
measurements to each read, so mean pooling recovers every second moment. That
only helps if second moments carry signal the model does not already have -
and `quantiles_all_v1` already carries 12 of the 36 correlations through
`coupling.py` (every same-measurement position pair, plus three centre
cross-channel pairs). This measures the other 24, and the nine variances,
for SIGNAL (univariate abs(AUC - 0.5)) and for REDUNDANCY (largest abs(r)
against any column the best model already has). A univariate screen predicts
marginal signal, not incremental value - see GAPS.md - so a strong column here
is necessary, not sufficient.

    python analysis/evaluation/scratch/second_moment_screen.py
"""
import sys
from itertools import combinations

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from m6a.data import iter_sites, load_labels
from m6a.feature_cache import extract

N_SITES = 40_000
NAMES = ["dwell_m1", "sd_m1", "mean_m1", "dwell_0", "sd_0", "mean_0",
         "dwell_p1", "sd_p1", "mean_p1"]
# Pairs coupling.py already emits (as correlations), by read-column index.
COVERED = {(0, 3), (3, 6), (0, 6), (1, 4), (4, 7), (1, 7), (2, 5), (5, 8), (2, 8),
           (3, 4), (4, 5), (3, 5)}

rows, keys = [], []
for site in iter_sites("data0/dataset0.json.gz", limit=N_SITES):
    reads = np.asarray(site.reads, dtype=np.float64)
    # log dwell and log sd: both are heavy-tailed positive quantities and a
    # Pearson r on the raw scale is dominated by a handful of reads.
    reads[:, [0, 1, 3, 4, 6, 7]] = np.log(np.maximum(reads[:, [0, 1, 3, 4, 6, 7]], 1e-6))
    centred = reads - reads.mean(axis=0)
    cov = centred.T @ centred / (len(reads) - 1)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(np.where(sd > 0, sd, 1), np.where(sd > 0, sd, 1))
    row = {f"var_{NAMES[i]}": cov[i, i] for i in range(9)}
    for i, j in combinations(range(9), 2):
        row[f"r_{NAMES[i]}__{NAMES[j]}"] = corr[i, j]
    rows.append(row)
    keys.append(site.key)

frame = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys))
labels = load_labels("data0/data.info.labelled").set_index(
    ["transcript_id", "transcript_position"])["label"]
y = labels.reindex(frame.index).to_numpy()
keep = ~np.isnan(y)
frame, y = frame[keep], y[keep].astype(int)

existing = extract("data0/dataset0.json.gz", "quantiles_all_v1", [None])[None].features
existing = existing.loc[frame.index]
E = (existing - existing.mean()) / existing.std().replace(0, 1)

out = []
for column in frame.columns:
    x = frame[column].to_numpy()
    signal = abs(roc_auc_score(y, x) - 0.5)
    z = (x - x.mean()) / (x.std() or 1)
    r = np.abs(E.to_numpy().T @ z / len(z))
    best = int(np.nanargmax(r))
    covered = False
    if column.startswith("r_"):
        i, j = (NAMES.index(p) for p in column[2:].split("__"))
        covered = (i, j) in COVERED
    out.append({"column": column, "signal": signal, "max_abs_r": r[best],
                "nearest_existing": existing.columns[best], "in_coupling": covered})

table = pd.DataFrame(out).sort_values("signal", ascending=False)
print(f"{len(frame):,} labelled sites, {y.sum():,} positive\n")
pd.set_option("display.width", 160)
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nNot already in coupling.py, sorted by signal:")
print(table[~table["in_coupling"]].head(12).to_string(index=False,
      float_format=lambda v: f"{v:.4f}"))
