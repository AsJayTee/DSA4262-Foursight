"""Where does the graph network beat the read-only network? (why-the-GNN-works, 1)

    python analysis/representation/gain_decomposition.py

h2gcn_aux (reads + neighbours within 50 nt) against deepset (the same kind of
read encoder, no graph), out of fold, on identical held-out sites, both arms.
The gain in PR AUC is computed INSIDE strata of sites, each with a paired
gene-resampled interval:

  neighbour   has a MODIFIED site within 50 nt / has only unmodified ones /
              has none (labels used for this analysis only - the model never
              sees a neighbour's label)
  depth       the site's own read count
  zone        >= 100 nt from an exon junction or not; transcript region

Corroboration predicts the gain concentrates at sites with a modified
neighbour; normalisation predicts it is spread over sites with any neighbour;
borrowing strength predicts it is largest at low depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import xsrc_nets as X
from m6a import crosssource as xs

CONTEXT = common.ROOT / "analysis" / "newdata" / "annotation_context" / "site_context.csv.gz"
DEPTH_BINS = [20, 30, 50, 100, 10**6]


def neighbour_status(index: pd.MultiIndex, y: np.ndarray, radius: int = 50) -> np.ndarray:
    frame = pd.DataFrame({"t": index.get_level_values(0), "p": index.get_level_values(1), "y": y})
    out = np.empty(len(frame), dtype=object)
    for _, g in frame.groupby("t"):
        p, lab = g.p.to_numpy(), g.y.to_numpy()
        near = (np.abs(p[:, None] - p[None, :]) <= radius) & ~np.eye(len(p), dtype=bool)
        has_any, has_pos = near.any(1), (near & (lab[None, :] == 1)).any(1)
        out[g.index] = np.where(has_pos, "a modified neighbour", np.where(has_any, "unmodified neighbours only",
                                                                          "no neighbour"))
    return out


def main() -> None:
    sources, bundle = X.load(1.0)
    n0 = len(sources["dataset0"])
    ctx = pd.read_csv(CONTEXT)
    rows = []
    for arm in X.ARMS:
        graphnet, deepset = X.candidate("h2gcn_aux", sources, 1.0), X.candidate("deepset", sources, 1.0)
        for src, sl in (("dataset0", slice(0, n0)), ("data1", slice(n0, None))):
            s = sources[src]
            cell = "dataset0" if src == "dataset0" else "data1"
            c = ctx[ctx.cell_line == cell].set_index(["transcript_id", "transcript_position"]).reindex(s.index)
            strata = {
                "neighbour within 50 nt": neighbour_status(s.index, s.y),
                "own read count": pd.cut(bundle.reads.counts[sl], DEPTH_BINS, right=False).astype(str),
                "exon junction": np.where(c.dist_junction.to_numpy() >= 100, ">= 100 nt", "< 100 nt"),
                "region": c.region.fillna("unknown").to_numpy(),
            }
            for kind, labels in strata.items():
                for level in pd.unique(labels):
                    m = labels == level
                    if m.sum() < 500 or s.y[m].sum() < 20:
                        continue
                    g = xs.paired_gain(s.y[m], graphnet[arm][src][m], deepset[arm][src][m], s.genes[m], n=300)
                    rows.append({"trained on": arm, "scored on": src, "stratum": kind, "level": level,
                                 "sites": int(m.sum()), "positive %": 100 * s.y[m].mean(),
                                 "graph PR AUC": g["pr_auc"], "deepset PR AUC": g["baseline_pr_auc"],
                                 "gain": g["gain"], "ci_low": g["ci_low"], "ci_high": g["ci_high"]})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.to_csv(common.RESULTS / "gain_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
