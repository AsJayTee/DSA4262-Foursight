"""Other sequencing runs of the same cell line (job 5 of the 9 Oct batch). No training.

    python analysis/representation/score_hct116_runs.py

dataset0 is SG-NEx Hct116 replicate 3 run 1 (analysis/newdata/sgnex_source_match.py).
SG-NEx has two more Hct116 direct-RNA runs, processed by m6Anet exactly like the
course data, in the public bucket. Same cell line, so dataset0's m6ACE-seq
labels apply; different molecules and a different read depth at every site.
That is "a new run of the same biology" - most likely what an Hct116 test file
looks like - with real depth variation instead of subsampling.

For each run: every dataset0 labelled site present in it (any depth), graphs
over those sites per transcript, each site scored by the seed-0 fold model that
held out its gene (so no gene is ever scored by a model that saw it). Compared
with the same models' out-of-fold scores on dataset0 itself, on identical sites.

Writes results/hct116_runs.csv (PR AUC, ROC AUC by run, model and read depth).
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import common
import graph
import xsrc_nets as X

RUNS = ["SGNex_Hct116_directRNA_replicate3_run4", "SGNex_Hct116_directRNA_replicate4_run3"]
BUCKET = "https://sg-nex-data.s3.amazonaws.com/data/processed_data/m6Anet/"
EXT = common.ROOT / "data" / "sgnex_runs"   # data/ is gitignored: never committed
MODELS = [("dataset0", "h2gcn_aux"), ("dataset0", "res_gate"), ("dataset0", "scalar_msg"), ("dataset0", "gps"),
          ("pooled_both", "h2gcn_twohead_aux"), ("pooled_both", "h2gcn_aux"),
          ("pooled_both", "res_gate"), ("pooled_both", "scalar_msg")]
ENSEMBLES = {"shipped (twohead + h2gcn_aux)": ["h2gcn_twohead_aux", "h2gcn_aux"],
             "twohead + res_gate + scalar_msg": ["h2gcn_twohead_aux", "res_gate", "scalar_msg"]}
DEPTHS = [1, 5, 10, 20, 50, 10**6]


def strip(t: str) -> str:
    return t.split(".")[0]


def download(run: str):
    path = EXT / run / "data.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {run}/data.json ...", flush=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(f"{BUCKET}{run}/data.json") as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 24):
                f.write(chunk)
        tmp.rename(path)
    return path


def read_run(path, wanted: dict) -> pd.DataFrame:
    """The wanted sites of one run: (transcript, position) -> reads, 7-mer."""
    rows = []
    with open(path) as fh:
        for line in fh:
            t = line[2:line.index('"', 2)]
            if strip(t) not in wanted:
                continue
            rec = json.loads(line)[t]
            p = next(iter(rec))
            if int(p) not in wanted[strip(t)]:
                continue
            kmer = next(iter(rec[p]))
            rows.append((strip(t), int(p), kmer, np.asarray(rec[p][kmer], dtype=np.float32)))
    return pd.DataFrame(rows, columns=["transcript", "position", "kmer", "reads"])


def main() -> None:
    torch.set_num_threads(8)
    sources, bundle = X.load(1.0)
    s0 = sources["dataset0"]
    n0 = len(s0)
    ref = pd.DataFrame({"transcript": [strip(t) for t in s0.index.get_level_values(0)],
                        "position": s0.index.get_level_values(1).astype(int),
                        "y": s0.y, "fold": bundle.folds[:n0], "row0": np.arange(n0)})
    wanted = ref.groupby("transcript").position.apply(set).to_dict()

    # Out-of-fold scores on dataset0 itself (replicate 3 run 1), seed 0, for the paired comparison.
    def oof(model, arm):
        return np.nanmax(np.stack([np.load(X.OUT / f"{model}_{arm}_fold{f}.npy") for f in range(5)]), 0)[:n0]

    rows = []
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
        new, old = {}, {}
        for arm, model in MODELS:
            v = np.full(len(sites), np.nan)
            for fold in range(5):
                held = np.flatnonzero(sites.fold.to_numpy() == fold)
                saved = torch.load(X.OUT / "weights" / f"{model}_{arm}_fold{fold}.pt", weights_only=False)
                net = graph.GraphNet(**X.GRAPH_MODELS[model])
                net.load_state_dict(saved["state_dict"])
                std = common.Standardiser(saved["read_mean"], saved["read_scale"])(values)
                out = graph.score(net, graph.graphs_of(held, gid, pos), std, offsets, kmer, pos)
                v[list(out)] = list(out.values())
            new[(arm, model)] = v
            old[(arm, model)] = oof(model, arm)[sites.row0.to_numpy()]
        for name, parts in ENSEMBLES.items():
            for d in (new, old):
                d[("pooled_both", name)] = np.mean([rankdata(d[("pooled_both", m)]) for m in parts], 0)
        y = sites.y.to_numpy()
        for (arm, model), v in new.items():
            for lo, hi in [(1, 10**6)] + list(zip(DEPTHS[:-1], DEPTHS[1:])):
                m = (counts >= lo) & (counts < hi)
                if m.sum() < 500 or y[m].sum() < 20:
                    continue
                rows.append({"run": run, "trained on": arm, "model": model,
                             "depth in this run": "all" if hi == 10**6 and lo == 1 else f"{lo}-{hi - 1}",
                             "sites": int(m.sum()), "positive %": 100 * y[m].mean(),
                             "pr_auc this run": average_precision_score(y[m], v[m]),
                             "pr_auc dataset0 run, same sites": average_precision_score(y[m], old[(arm, model)][m]),
                             "roc_auc this run": roc_auc_score(y[m], v[m])})
        print(pd.DataFrame(rows)[lambda t: (t.run == run) & (t["depth in this run"] == "all")]
              .round(4).to_string(index=False), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(common.RESULTS / "hct116_runs.csv", index=False)


if __name__ == "__main__":
    main()
