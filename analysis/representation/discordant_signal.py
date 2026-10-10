"""Model-free check: do the raw reads differ where the two cell lines' labels disagree? (10 Oct)

    python analysis/representation/discordant_signal.py

discordant_sites.py found that no trained model ranks a site higher in the cell
line whose label says "modified" (which-line AUC ~0.5). Either the labels
disagree without the molecules differing (labelling noise / thresholds), or the
molecules differ below what those models resolve. This asks the reads directly,
with no network:

  per site, per file: mean and sd over its reads of the 9 measurements (dwell
  and current sd logged, as the networks see them), centred on the file's mean
  for that 7-mer (so sequence effects and file-wide shifts cancel)

  1. a "modification direction": logistic regression on those features within
     ONE file, fitted on sites NOT shared by both files (no leakage into the
     test), checked by its within-file ROC AUC on the shared sites
  2. at each shared site, the change in that score from cell line 1's reads to
     cell line 2's. If the molecules carry the label difference, sites modified
     only in cell line 2 shift up and sites modified only in cell line 1 shift
     down: which-line AUC well above 0.5
  3. the same for the single most telling raw measurement (current at the
     centre position), no fitting at all

Writes results/discordant_signal.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common
from m6a import crosssource as xs
from m6a import external, feature_cache
from m6a.config import Config
from m6a.crossval import _reads_for
from m6a.data import SUBSAMPLE_SEED

BASELINE = common.ROOT / "configs" / "quantiles.yaml"
LOG_COLUMNS = [0, 1, 3, 4, 6, 7]
NAMES = [f"{m} {p}" for p in ("-1", "0", "+1") for m in ("dwell", "current sd", "current mean")]


def quiet(*_a, **_k) -> None:
    pass


def site_features(name: str, src, cfg) -> pd.DataFrame:
    path = common.DATA if name == "dataset0" else external.paths(common.DATA_DIR, "data1")[0]
    ext = feature_cache.extract(path, cfg.features, [None], log=quiet)[None]
    reads = _reads_for(feature_cache.extract_reads(path, log=quiet), ext, src.index, None, SUBSAMPLE_SEED)
    v = reads.values.astype(np.float64).copy()
    v[:, LOG_COLUMNS] = np.log(np.maximum(v[:, LOG_COLUMNS], 1e-6))
    counts = np.diff(reads.offsets)
    mean = np.add.reduceat(v, reads.offsets[:-1], axis=0) / counts[:, None]
    sd = np.sqrt(np.add.reduceat((v - np.repeat(mean, counts, axis=0)) ** 2, reads.offsets[:-1], axis=0)
                 / counts[:, None])
    f = pd.DataFrame(np.c_[mean, sd], index=src.index,
                     columns=[f"mean {n}" for n in NAMES] + [f"sd {n}" for n in NAMES])
    kmer = ext.sites.loc[src.index, "kmer"].to_numpy()
    return f - f.groupby(kmer).transform("mean")          # centre on the file's 7-mer mean


def main() -> None:
    cfg = Config.load(BASELINE)
    sources = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed,
                      n_folds=cfg.split.n_folds, log=quiet)
    feats = {n: site_features(n, sources[n], cfg) for n in ("dataset0", "data1")}
    y = {n: pd.Series(sources[n].y, index=sources[n].index) for n in feats}
    shared = feats["dataset0"].index.intersection(feats["data1"].index)
    y0, y1 = y["dataset0"][shared].to_numpy(), y["data1"][shared].to_numpy()
    groups = {"modified in cell line 2 only": (y0 == 0) & (y1 == 1),
              "modified in cell line 1 only": (y0 == 1) & (y1 == 0),
              "modified in both": (y0 == 1) & (y1 == 1), "modified in neither": (y0 == 0) & (y1 == 0)}
    disc = groups["modified in cell line 2 only"] | groups["modified in cell line 1 only"]
    rows = []
    for fit_on in ("dataset0", "data1"):
        train = feats[fit_on].index.difference(shared)
        scaler = StandardScaler().fit(feats[fit_on].loc[train])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            scaler.transform(feats[fit_on].loc[train]), y[fit_on][train])
        score = {n: clf.decision_function(scaler.transform(feats[n].loc[shared])) for n in feats}
        row = {"direction fitted on": f"{'cell line 1' if fit_on == 'dataset0' else 'cell line 2'} (unshared sites)",
               "within-file AUC, cell line 1": roc_auc_score(y0, score["dataset0"]),
               "within-file AUC, cell line 2": roc_auc_score(y1, score["data1"])}
        delta = score["data1"] - score["dataset0"]
        row["which-line AUC (discordant sites)"] = roc_auc_score(y1[disc], delta[disc])
        for g, m in groups.items():
            row[f"mean shift, {g}"] = delta[m].mean()
        rows.append(row)
    # No fitting at all: current at the centre position, per site, cell line 2 minus cell line 1.
    raw = feats["data1"].loc[shared, "mean current mean 0"].to_numpy() - feats["dataset0"].loc[shared, "mean current mean 0"].to_numpy()
    sep = roc_auc_score(y0, feats["dataset0"].loc[shared, "mean current mean 0"])
    row = {"direction fitted on": "none: centre-position current mean",
           "within-file AUC, cell line 1": sep,
           "within-file AUC, cell line 2": roc_auc_score(y1, feats["data1"].loc[shared, "mean current mean 0"]),
           "which-line AUC (discordant sites)": roc_auc_score(y1[disc], raw[disc] * (1 if sep > 0.5 else -1))}
    for g, m in groups.items():
        row[f"mean shift, {g}"] = raw[m].mean()
    rows.append(row)
    table = pd.DataFrame(rows)
    print(f"shared sites {len(shared):,}; " + ", ".join(f"{g} {int(m.sum()):,}" for g, m in groups.items()))
    pd.set_option("display.width", 250)
    print(table.round(3).T.to_string())
    table.to_csv(common.RESULTS / "discordant_signal.csv", index=False)


if __name__ == "__main__":
    main()
