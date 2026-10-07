"""What do neighbours give the graph network? Swap them, at prediction time.
(why-the-GNN-works, 2 and 3)

    python analysis/representation/ablate_graph.py --arm dataset0 --fold 0   # one network
    python analysis/representation/ablate_graph.py --all                     # all ten, then the table

Trained h2gcn_aux networks (neighbours within 50 nt, no transcript node; the
local half of the shipped ensemble), out of fold, nothing retrained. Every site
is both a scored site and someone's neighbour, so scoring runs TWO tracks:
the scored site always keeps its own inputs (the "focal" track), while the
messages it receives are computed from a "neighbour world" in which only the
neighbours' inputs are changed. With both tracks unchanged this reproduces the
network's own output exactly (checked: "as trained").

Swaps of what neighbours provide (2):
  a  reads    each neighbour's reads -> those of a random site with the same
              5-mer on another transcript (keeps position, sequence, count)
  b  7-mer    each neighbour's 7-mer -> a random site's (keeps its reads)
  c  random   each neighbour -> a random site from another transcript, reads
              and 7-mer (keeps only the graph's shape and distances)
  d  baseline each neighbour's reads with its transcript's mean signal removed
              (keeps differences between sites, removes the transcript level)
  none        no neighbours at all

Depth asymmetry (3): thin the scored site to 1/3/10 reads (the keyed subsets
of m6a.data.subsample_blocks) while neighbours keep all reads / are thinned
too / are absent.

These are inputs the network never saw in training: they measure what it
RELIES on, not what an ideal model would do.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common
import graph
import xsrc_nets as X
from m6a import crosssource as xs
from m6a.data import ReadBlocks, subsample_blocks

MODEL = "h2gcn_aux"
DEPTHS = (1, 3, 10)
OUT = common.RESULTS / "ablate_graph"
RNG_SEED = 4262


def decode_kmers(onehot: np.ndarray) -> np.ndarray:
    idx = onehot.reshape(len(onehot), 7, 4).argmax(-1)
    return np.array(["".join("ACGT"[i] for i in row) for row in idx])


@torch.no_grad()
def two_track(net, focal, world, gid, pos, n_graphs, no_neighbours=False):
    """h2gcn (local edges only, single output) with the scored site's own track
    separate from the track its neighbours' messages come from."""
    s_f, s_n = net.sites(*focal), net.sites(*world)
    delta = (pos[:, None] - pos[None, :]).abs()
    adj = ((gid[:, None] == gid[None, :]) & (delta <= net.window)).float()
    adj.fill_diagonal_(0.0)
    if no_neighbours:
        adj.zero_()
    deg = adj.sum(1, keepdim=True)
    flag = (deg > 0).float()
    outs = [s_f]
    for layer in net.layers:
        s_f_next = F.relu(layer(torch.cat([s_f, (adj @ s_n) / deg.clamp(min=1), flag], -1)))
        s_n = F.relu(layer(torch.cat([s_n, (adj @ s_n) / deg.clamp(min=1), flag], -1)))
        s_f = s_f_next
        outs.append(s_f)
    return net.head(torch.cat(outs, -1)).squeeze(-1)


def pad(blocks: ReadBlocks, rows: np.ndarray):
    counts = blocks.counts[rows]
    width = int(counts.max())
    reads = np.zeros((len(rows), width, blocks.values.shape[1]), dtype=np.float32)
    mask = np.zeros((len(rows), width), dtype=bool)
    for k, r in enumerate(rows):
        n = counts[k]
        reads[k, :n] = blocks.values[blocks.offsets[r]:blocks.offsets[r + 1]]
        mask[k, :n] = True
    return torch.from_numpy(reads), torch.from_numpy(mask)


def score(net, graphs, focal_blocks, focal_kmer, world_blocks, world_kmer, position, no_neighbours=False):
    out, batch, size = {}, [], 0
    budget = 400_000

    def flush():
        rows = np.concatenate(batch)
        gid = torch.from_numpy(np.repeat(np.arange(len(batch)), [len(b) for b in batch]))
        pos = torch.from_numpy(position[rows].astype(np.float32))
        fr, fm = pad(focal_blocks, rows)
        wr, wm = pad(world_blocks, rows)
        logit = two_track(net, (fr, fm, torch.from_numpy(focal_kmer[rows])),
                          (wr, wm, torch.from_numpy(world_kmer[rows])), gid, pos, len(batch), no_neighbours)
        out.update(zip(rows.tolist(), logit.numpy()))

    for g in graphs:
        width = max(int(focal_blocks.counts[g].max()), int(world_blocks.counts[g].max()))
        if batch and (size + len(g)) * width > budget:
            flush()
            batch, size = [], 0
        batch.append(g)
        size += len(g)
    if batch:
        flush()
    return out


def run_one(arm: str, fold: int) -> None:
    torch.set_num_threads(6)
    rng = np.random.default_rng(RNG_SEED + fold)
    sources, bundle = X.load(1.0)
    saved = torch.load(X.OUT / "weights" / f"{MODEL}_{arm}_fold{fold}.pt", weights_only=False)
    net = graph.GraphNet(**X.GRAPH_MODELS[MODEL]).eval()
    net.load_state_dict(saved["state_dict"])
    std = common.Standardiser(saved["read_mean"], saved["read_scale"])
    full = ReadBlocks(std(bundle.reads.values), bundle.reads.offsets)
    kmer = bundle.kmer_onehot
    n = len(bundle.genes)
    held = np.flatnonzero(bundle.folds == fold)
    graphs = graph.graphs_of(held, bundle.graph_id, bundle.position)

    # Replacement maps for the neighbour world.
    five = np.array([k[1:6] for k in decode_kmers(kmer)])
    same_five = np.empty(n, dtype=np.int64)
    for _, rows in pd.Series(np.arange(n)).groupby(five):
        rows = rows.to_numpy()
        pick = rows[rng.integers(0, len(rows), len(rows))]
        clash = bundle.graph_id[pick] == bundle.graph_id[rows]          # same transcript: redraw once
        pick[clash] = rows[rng.integers(0, len(rows), int(clash.sum()))]
        same_five[rows] = pick
    random_site = rng.integers(0, n, n)
    reads_a = full.take(same_five)
    kmer_b = kmer[rng.integers(0, n, n)]
    reads_c, kmer_c = full.take(random_site), kmer[random_site]
    # d: remove each transcript's mean read signal, keep the global mean.
    read_gid = np.repeat(bundle.graph_id, full.counts)
    sums = np.zeros((bundle.graph_id.max() + 1, full.values.shape[1]))
    np.add.at(sums, read_gid, full.values)
    tx_mean = sums / np.bincount(read_gid)[:, None]
    reads_d = ReadBlocks((full.values - tx_mean[read_gid] + full.values.mean(0)).astype(np.float32), full.offsets)

    thinned = {d: ReadBlocks(std(b.values), b.offsets) for d, b in
               ((d, subsample_blocks(bundle.reads, list(sources["dataset0"].index) + list(sources["data1"].index), d))
                for d in DEPTHS)}

    configs = {"as trained": (full, kmer, full, kmer, False),
               "none": (full, kmer, full, kmer, True),
               "a: neighbours' reads from same-5-mer sites": (full, kmer, reads_a, kmer, False),
               "b: neighbours' 7-mer randomised": (full, kmer, full, kmer_b, False),
               "c: neighbours = random sites": (full, kmer, reads_c, kmer_c, False),
               "d: neighbours' transcript baseline removed": (full, kmer, reads_d, kmer, False)}
    for d in DEPTHS:
        configs[f"site {d} reads, neighbours full"] = (thinned[d], kmer, full, kmer, False)
        configs[f"site {d} reads, neighbours {d} reads"] = (thinned[d], kmer, thinned[d], kmer, False)
        configs[f"site {d} reads, no neighbours"] = (thinned[d], kmer, thinned[d], kmer, True)

    # Sanity: the two-track scorer reproduces the network's own scoring.
    reference = graph.score(net, graphs, full.values, full.offsets, kmer, bundle.position)
    out = {}
    for name, (fb, fk, wb, wk, none) in configs.items():
        started = time.time()
        got = score(net, graphs, fb, fk, wb, wk, bundle.position, none)
        out[name] = np.array([got[r] for r in held])
        if name == "as trained":
            ref = np.array([reference[r] for r in held])
            gap = float(np.abs(out[name] - ref).max())
            if gap > 1e-3:
                raise SystemExit(f"two-track scorer disagrees with the network by {gap:.2e} - aborting")
            print(f"  sanity: two-track = network scoring (max difference {gap:.1e})", flush=True)
        print(f"  {name} ({time.time() - started:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{arm}_fold{fold}.npz", held=held, **{k: v for k, v in out.items()})


def combine() -> None:
    sources, bundle = X.load(1.0)
    n0 = len(sources["dataset0"])
    rows = []
    for arm in X.ARMS:
        scores = {}
        for fold in range(5):
            z = np.load(OUT / f"{arm}_fold{fold}.npz")
            for k in z.files:
                if k != "held":
                    scores.setdefault(k, np.full(len(bundle.genes), np.nan))[z["held"]] = z[k]
        for src, sl in (("dataset0", slice(0, n0)), ("data1", slice(n0, None))):
            s = sources[src]
            base = scores["as trained"][sl]
            for name, v in scores.items():
                ref_name = name.split(",")[0] + ", neighbours full" if name.startswith("site ") else "as trained"
                ref = scores[ref_name][sl] if ref_name in scores else base
                g = xs.paired_gain(s.y, v[sl], ref, s.genes, n=300)
                rows.append({"trained on": arm, "scored on": src, "configuration": name, "pr_auc": g["pr_auc"],
                             "vs": ref_name, "change": g["gain"] if name != ref_name else 0.0,
                             "ci_low": g["ci_low"], "ci_high": g["ci_high"]})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table.to_csv(common.RESULTS / "ablate_graph.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=X.ARMS)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    if args.all:
        logs = common.ROOT / "analysis" / "representation" / "logs"
        jobs = [subprocess.Popen([sys.executable, "-u", __file__, "--arm", a, "--fold", str(f)],
                                 stdout=open(logs / f"ablate_graph_{a}_{f}.log", "w"), stderr=subprocess.STDOUT)
                for a in X.ARMS for f in range(5)]
        failed = [j.args for j in jobs if j.wait() != 0]
        if failed:
            raise SystemExit(f"failed: {failed} - see logs/ablate_graph_*.log")
        combine()
    elif args.combine:
        combine()
    else:
        run_one(args.arm, args.fold)


if __name__ == "__main__":
    main()
