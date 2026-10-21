"""Per-site held-out scores on the two other SG-NEx HCT116 runs (Task 2 site-type analysis).

    python analysis/representation/score_hct116_runs_sites.py

score_hct116_runs.py asked "does a new run of the same cells score as well?" and
kept summary numbers only, for an earlier ensemble. The Task 2 shortlist analysis
(report/task2_plan.md, 4.1) needs the per-site scores of the shipped designs -
h2gcn_twohead_aux, res_gate, scalar_drop, seeds 0-2, trained on both files - so
the top-k hit rate can be measured on real, independently sequenced molecules
rather than only on thinned ones.

Same rule as score_hct116_runs.py: every site is scored by the fold model that
held out its gene, with graphs over the labelled sites present in that run.

Writes .cache/representation/xsrc_nets/hct116_runs_sites/<run>.npz (gitignored): site keys,
read count in that run, and one score array per network.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

import common
import graph
import xsrc_nets as X
from score_hct116_runs import RUNS, download, read_run, strip

DESIGNS = ["h2gcn_twohead_aux", "res_gate", "scalar_drop"]
SEEDS = [0, 1, 2]
ARM = "pooled_both"


def main() -> None:
    torch.set_num_threads(16)
    sources, bundle = X.load(1.0)
    s0 = sources["dataset0"]
    n0 = len(s0)
    ref = pd.DataFrame({"transcript": [strip(t) for t in s0.index.get_level_values(0)],
                        "position": s0.index.get_level_values(1).astype(int),
                        "fold": bundle.folds[:n0], "row0": np.arange(n0)})
    wanted = ref.groupby("transcript").position.apply(set).to_dict()
    out_dir = X.OUT / "hct116_runs_sites"
    out_dir.mkdir(exist_ok=True)
    for run in RUNS:
        started = time.time()
        sites = read_run(download(run), wanted).merge(ref, on=["transcript", "position"])
        counts = sites.reads.map(len).to_numpy()
        values = np.concatenate(sites.reads.to_list())
        offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        kmer = common.onehot(sites.kmer.to_list(), 7)
        pos = sites.position.to_numpy(np.int64)
        gid = pd.factorize(sites.transcript)[0]
        print(f"{run}: {len(sites):,} of {n0:,} dataset0 sites, median depth {np.median(counts):.0f} "
              f"({time.time() - started:.0f}s)", flush=True)
        scores = {}
        for model in DESIGNS:
            for seed in SEEDS:
                v = np.full(len(sites), np.nan)
                for fold in range(5):
                    held = np.flatnonzero(sites.fold.to_numpy() == fold)
                    suffix = "" if seed == 0 else f"_s{seed}"
                    saved = torch.load(X.OUT / "weights" / f"{model}_{ARM}_fold{fold}{suffix}.pt",
                                       weights_only=False)
                    net = graph.GraphNet(**X.GRAPH_MODELS[model])
                    net.load_state_dict(saved["state_dict"])
                    std = common.Standardiser(saved["read_mean"], saved["read_scale"])(values)
                    got = graph.score(net, graph.graphs_of(held, gid, pos), std, offsets, kmer, pos)
                    v[list(got)] = list(got.values())
                if np.isnan(v).any():
                    raise SystemExit(f"{model} seed {seed}: {np.isnan(v).sum()} sites unscored")
                scores[f"{model}_s{seed}"] = v
                print(f"  {model} s{seed} ({time.time() - started:.0f}s)", flush=True)
        np.savez(out_dir / f"{run}.npz", transcript=sites.transcript.to_numpy().astype(str),
                 position=pos, n_reads=counts, row0=sites.row0.to_numpy(), **scores)
        print(f"-> {out_dir / run}.npz", flush=True)


if __name__ == "__main__":
    main()
