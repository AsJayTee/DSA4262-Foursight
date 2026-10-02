"""How should k be chosen for a mutual k-NN graph over a site's reads?

    python analysis/representation/knn_k_analysis.py

Two measurements, before any k-NN model is trained (discussed 2026-10-02):

1. **Read-level truth from data2.** Each of its 189 positions was sequenced in
   a 0% sample (every read unmodified) and a 100% sample (every read
   modified). Mixing reads from the two at a known fraction builds a synthetic
   site whose read labels are known. Per k: adjusted homophily (share of edges
   joining reads of the same class, rescaled so 0 = random edges and 1 =
   perfect; Lim et al. 2021), and how many modified / unmodified reads the
   mutual graph leaves isolated.
   Control: the same mixing between the 95% and 100% samples - both nearly all
   modified, so any structure there is the sequencing run, not modification.

2. **Site-level signal in dataset0.** Without read labels: does the mutual
   graph's shape (share of isolated reads, components per read) differ between
   positive and negative sites? ROC AUC per k.

Reads are transformed and standardised exactly as the networks see them
(common.transform_values, then a z-score fitted on the pooled reads).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score

import common
from m6a import external, feature_cache
from m6a.data import load_labels

KS = (2, 3, 4, 5, 6, 8, 10, 12, 16)
DEPTHS = (20, 47, 150)          # dataset0 minimum, median, and well above
FRACTIONS = (0.1, 0.3, 0.5)
REPEATS = 10
RNG = np.random.default_rng(4262)


def mutual_knn(x: np.ndarray, ks) -> dict[int, np.ndarray]:
    """{k: boolean adjacency} of the mutual k-NN graph, for every k at once."""
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, 1)
    out = {}
    for k in ks:
        if k >= len(x):
            continue
        a = np.zeros(d.shape, dtype=bool)
        np.put_along_axis(a, order[:, :k], True, 1)
        out[k] = a & a.T
    return out


def adjusted_homophily(adj: np.ndarray, label: np.ndarray) -> float:
    i, j = np.nonzero(np.triu(adj, 1))
    if len(i) == 0:
        return np.nan
    same = (label[i] == label[j]).mean()
    p = label.mean()
    chance = p ** 2 + (1 - p) ** 2
    return (same - chance) / (1 - chance)


def standardiser(values: np.ndarray):
    t = common.transform_values(values)
    mean, sd = t.mean(0), t.std(0)
    return lambda v: (common.transform_values(v) - mean) / np.where(sd > 0, sd, 1)


def site_reads(json_path, index_filter=None):
    ext = feature_cache.extract(json_path, "quantiles_v1", [None], log=lambda *a: None)[None]
    blocks = feature_cache.extract_reads(json_path, log=lambda *a: None)
    return ext.features.index, blocks


def data2_analysis() -> pd.DataFrame:
    json_path, info_path = external.paths(common.DATA_DIR, "data2")
    info = pd.read_csv(info_path)
    fraction = info.set_index(["transcript_id", "transcript_position"])["label"]
    index, blocks = site_reads(json_path)
    frac = fraction.reindex(index).to_numpy()
    pos = index.get_level_values(1).to_numpy()
    scale = standardiser(blocks.values)
    reads = {(f, p): blocks.site(i) for i, (f, p) in enumerate(zip(frac, pos))}
    rows = []
    for name, modified, unmodified in (("0% vs 100%", 1.0, 0.0), ("control: 95% vs 100%", 1.0, 0.95)):
        for p in np.unique(pos):
            a, b = reads[(modified, p)], reads[(unmodified, p)]
            for n in DEPTHS:
                for f in FRACTIONS:
                    m = max(1, round(n * f))
                    # Never with replacement: a duplicated read is its own
                    # nearest neighbour and fakes structure.
                    if m > len(a) or n - m > len(b):
                        continue
                    for _ in range(REPEATS):
                        x = np.concatenate([a[RNG.choice(len(a), m, replace=False)],
                                            b[RNG.choice(len(b), n - m, replace=False)]])
                        label = np.r_[np.ones(m), np.zeros(n - m)]
                        for k, adj in mutual_knn(scale(x), KS).items():
                            deg = adj.sum(1)
                            rows.append({"pairing": name, "position": p, "n": n, "f": f, "k": k,
                                         "homophily": adjusted_homophily(adj, label),
                                         "isolated_mod": (deg[label == 1] == 0).mean(),
                                         "isolated_unmod": (deg[label == 0] == 0).mean()})
    return pd.DataFrame(rows)


def dataset0_analysis(n_sites: int = 2000) -> pd.DataFrame:
    index, blocks = site_reads(common.DATA)
    labels = load_labels(common.LABELS).set_index(["transcript_id", "transcript_position"])["label"]
    y = labels.reindex(index).to_numpy()
    pick = np.r_[RNG.choice(np.flatnonzero(y == 1), n_sites, replace=False),
                 RNG.choice(np.flatnonzero(y == 0), n_sites, replace=False)]
    scale = standardiser(blocks.values[RNG.choice(blocks.total_reads, 500_000, replace=False)])
    stats = []
    for i in pick:
        x = blocks.site(i)
        if len(x) > 150:      # same cost for every site; depth is a confound we do not want
            x = x[RNG.choice(len(x), 150, replace=False)]
        for k, adj in mutual_knn(scale(x), KS).items():
            n_comp = connected_components(csr_matrix(adj), directed=False)[0]
            stats.append({"site": i, "y": y[i], "k": k, "isolated": (adj.sum(1) == 0).mean(),
                          "components": n_comp / len(x)})
    frame = pd.DataFrame(stats)
    out = []
    for k, g in frame.groupby("k"):
        out.append({"k": k, **{f"AUC {s}": roc_auc_score(g.y, g[s]) for s in ("isolated", "components")},
                    "isolated (neg sites)": g[g.y == 0].isolated.mean(),
                    "isolated (pos sites)": g[g.y == 1].isolated.mean()})
    return pd.DataFrame(out).set_index("k")


def main() -> None:
    pd.set_option("display.width", 200)
    d2 = data2_analysis()
    print("data2 - adjusted homophily (0 = random edges, 1 = perfect), mean over 189 positions:\n")
    for name, g in d2.groupby("pairing"):
        print(name)
        print(g.pivot_table(index=["n", "f"], columns="k", values="homophily").round(3), "\n")
    real = d2[d2.pairing == "0% vs 100%"]
    print("Isolated reads in the mutual graph (modified / unmodified), f = 0.3:")
    iso = real[real.f == 0.3].groupby(["n", "k"])[["isolated_mod", "isolated_unmod"]].mean().round(3)
    print(iso.unstack("k"), "\n")
    # Positions where modification moves the signal at all: the homophily a
    # well-separated position reaches at n = 150, f = 0.5, k = 8.
    sep = real[(real.n == 150) & (real.f == 0.5) & (real.k == 8)].groupby("position").homophily.mean()
    top = sep[sep >= sep.quantile(0.75)].index
    print("Same, top quarter of positions by separability (where the signal is):")
    print(real[real.position.isin(top)].pivot_table(index=["n", "f"], columns="k",
                                                    values="homophily").round(3), "\n")
    print("dataset0 - does the mutual graph's shape separate positive from negative sites?")
    print(dataset0_analysis().round(3))
    d2.to_csv(common.RESULTS / "knn_k_data2.csv", index=False)


if __name__ == "__main__":
    main()
