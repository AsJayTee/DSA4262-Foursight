"""The row keys of every held-out prediction array (Task 2 site-type analysis).

    python analysis/representation/export_site_keys.py

xsrc_nets.py writes held-out scores as bare arrays in bundle order (dataset0's
sites, then data1's). Analyses that only read those arrays - the site-type
performance and top-k shortlist tables (report/task2_plan.md, section 4) - need
to know which site each row is, without loading every read again.

Writes .cache/representation/xsrc_nets/site_keys.csv.gz (gitignored: it holds
the course labels, which never leave the machine or enter git).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import xsrc_deepset
import xsrc_nets as X
from m6a import external, feature_cache
from m6a.config import Config

import common


def main() -> None:
    sources, bundle = X.load(1.0)
    cfg = Config.load(xsrc_deepset.BASELINE)
    paths = {"dataset0": common.DATA, "data1": external.paths(common.DATA_DIR, "data1")[0]}
    kmers = []
    for name in ("dataset0", "data1"):
        ext = feature_cache.extract(paths[name], cfg.features, [None], log=lambda *a: None)[None]
        kmers.append(ext.sites.loc[sources[name].index, "kmer"].to_numpy())
    idx = [sources["dataset0"].index, sources["data1"].index]
    keys = pd.DataFrame({
        "file": np.where(bundle.file == 0, "dataset0", "data1"),
        "transcript_id": np.concatenate([i.get_level_values(0) for i in idx]),
        "transcript_position": bundle.position,
        "gene_id": bundle.genes,
        "fold": bundle.folds,
        "kmer": np.concatenate(kmers),
        "n_reads": np.diff(bundle.reads.offsets),
        "graph_id": bundle.graph_id,
        "label": bundle.y_own.astype(int),
    })
    out = X.OUT / "site_keys.csv.gz"
    keys.to_csv(out, index=False)
    print(f"{len(keys):,} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
