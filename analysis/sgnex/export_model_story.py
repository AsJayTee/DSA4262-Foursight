"""Data for the dashboard's "How it works" page: the shipped model, step by step, on MYC.

    curl -o data/sgnex_runs/SGNex_MCF7_directRNA_replicate4_run1/data.json \
      https://sg-nex-data.s3.amazonaws.com/data/processed_data/m6Anet/SGNex_MCF7_directRNA_replicate4_run1/data.json
    python analysis/sgnex/export_model_story.py

Public SG-NEx data only (one MCF7 sample, no credentials), scored with models/final in
numpy. The example gene is chosen by name, never by the course labels. The page's
interactive parts are precomputed here (option A in the dashboard plan):

  - the example site's reads (mean current and dwell at the three pore positions),
    beside a site with the same 7 letters that the model ranks low;
  - the example site's rank when only k of its reads are seen (25 random draws per k,
    its neighbours kept whole): what "few reads" does to the evidence;
  - every MYC site's rank with its neighbours and alone: what corroboration adds;
  - the six networks' ranks for the example site: the ensemble.

Ranks are as predict.py makes them: each network's score placed among that network's
scores over the whole sample (the saved per-sample file), the six percentiles averaged.
The script first re-scores MYC at full depth and checks it reproduces the saved scores.

Performance numbers are copied from committed results (sources in the JSON).

Writes dashboard/public/data/model_story.json.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.data import iter_sites  # noqa: E402
from m6a.models.site_graph import SiteGraphEnsemble, dense, kmer_onehot, relu, transform_reads  # noqa: E402

SAMPLE = "SGNex_MCF7_directRNA_replicate4_run1"
JSON = ROOT / "data" / "sgnex_runs" / SAMPLE / "data.json"
SCORES = ROOT / "data" / "sgnex_scores" / f"{SAMPLE}.csv.gz"
GTF = ROOT / "data" / "annotation" / "gtf_91.gtf.gz"
RESULTS = ROOT / "analysis" / "site_types" / "results"
OUT = ROOT / "dashboard" / "public" / "data" / "model_story.json"
CDNA = ROOT / "data" / "annotation" / "cdna_91.fa.gz"
GENE = "MYC"
KS = [1, 2, 3, 5, 10, 20, 40]
DRAWS = 25
SHOW_READS = 40
# The network whose layers the page draws: a res_gate design has an own-reads head.
SHOWN_NET = "res_gate_s0"
TABLE_SITES = 4      # sites shown with their read tables under the RNA text
SHOWN_UNITS = 10     # hidden units drawn per layer (of 64)


def transcript_sequence(tx: str) -> str:
    seq, take = [], False
    with gzip.open(CDNA, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if take:
                    break
                take = line[1:].split(".")[0].split()[0] == tx
            elif take:
                seq.append(line.strip())
    return "".join(seq)


def layer_story(net, site_reads: np.ndarray, kmer: str, own_ref: np.ndarray) -> dict:
    """What one network computes for one site, read by read (real activations).

    Per read: its nine standardised inputs and the most variable units of both read
    layers (phi). Per number of reads seen: the own-reads score as a rank in the sample,
    and the site network's (rho) strongest units - so the page can show the evidence
    accumulating."""
    x = transform_reads(site_reads, net.read_mean, net.read_scale)
    onehot = kmer_onehot([kmer])
    inp = np.concatenate([x, np.repeat(onehot, len(x), 0)], 1)
    h1 = relu(dense(inp, net.w, "phi.0"))
    h2 = relu(dense(h1, net.w, "phi.2"))
    pick = lambda h: np.argsort(-h.std(0))[:SHOWN_UNITS]  # noqa: E731
    u1, u2 = pick(h1), pick(h2)
    full = net.site_vectors(site_reads, np.array([0, len(site_reads)]), onehot)[0]
    u3 = np.argsort(-np.abs(full))[:8]
    prefix_rank, prefix_site = [], []
    for k in range(1, len(site_reads) + 1):
        s = net.site_vectors(site_reads[:k], np.array([0, k]), onehot)
        own = float(dense(s, net.w, "aux_head")[0, 0])
        prefix_rank.append(round(float((np.searchsorted(own_ref, own) + 0.5) / len(own_ref)), 4))
        prefix_site.append([round(float(v), 3) for v in s[0, u3]])
    r = lambda a: [[round(float(v), 3) for v in row] for row in a]  # noqa: E731
    return {"inputs": r(x), "h1": r(h1[:, u1]), "h2": r(h2[:, u2]), "site": prefix_site,
            "own_rank": prefix_rank, "units": {"h1": len(u1), "h2": len(u2), "of": int(h1.shape[1])}}


def gene_transcripts(name: str) -> set[str]:
    out = set()
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if f'gene_name "{name}";' in line and "\ttranscript\t" in line:
                out.add(re.search(r'transcript_id "([^"]+)"', line).group(1))
    return out


def load_sites(keep) -> list:
    return [s for s in iter_sites(JSON) if keep(s.transcript_id.split(".")[0], s)]


class Scorer:
    """The six networks on one transcript, ranked against the whole sample."""

    def __init__(self):
        self.model = SiteGraphEnsemble.load(ROOT / "models" / "final")
        self.names = [n.meta["model"] + f"_s{n.meta.get('seed', 0)}" for n in self.model.networks]
        cols = [f"raw_{n}" for n in self.names]
        ref = pd.read_csv(SCORES, usecols=cols)
        self.ref = {n: np.sort(ref[f"raw_{n}"].to_numpy(np.float64)) for n in self.names}

    def pct(self, name: str, v: np.ndarray) -> np.ndarray:
        r = self.ref[name]
        # Average rank of v among the sample's scores, as pandas' rank(method="average") / n.
        return (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right") + 1) / 2 / len(r)

    def logits(self, reads: list[np.ndarray], kmers: list[str], pos: np.ndarray) -> dict:
        counts = np.array([len(r) for r in reads])
        offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        values = np.concatenate(reads).astype(np.float32)
        onehot = kmer_onehot(kmers)
        out = {}
        for name, net in zip(self.names, self.model.networks):
            s = net.site_vectors(values, offsets, onehot)
            out[name] = net.logits(s, pos.astype(np.float32), counts)
        return out

    def rank(self, logits: dict) -> np.ndarray:
        return np.mean([self.pct(n, v) for n, v in logits.items()], 0)


def main() -> None:
    txs = gene_transcripts(GENE)
    sites = load_sites(lambda t, s: t in txs)
    by_tx: dict[str, list] = {}
    for s in sites:
        by_tx.setdefault(s.transcript_id.split(".")[0], []).append(s)
    tx = max(by_tx, key=lambda t: (len(by_tx[t]), sum(len(s.reads) for s in by_tx[t])))
    sites = sorted(by_tx[tx], key=lambda s: s.position)
    reads = [np.asarray(s.reads, dtype=np.float32) for s in sites]
    kmers = [s.kmer for s in sites]
    pos = np.array([s.position for s in sites])
    print(f"{GENE}: {len(by_tx)} transcripts in {SAMPLE}; using {tx}, {len(sites)} sites")

    sc = Scorer()
    full = sc.logits(reads, kmers, pos)
    rank_full = sc.rank(full)
    saved = pd.read_csv(SCORES, usecols=["transcript_id", "transcript_position", "score_rank"])
    saved = saved[saved.transcript_id.str.split(".").str[0] == tx].set_index("transcript_position").score_rank
    gap = np.abs(saved.reindex(pos).to_numpy() - rank_full).max()
    print(f"re-scored vs saved score_rank: max difference {gap:.5f}")
    if gap > 0.002:
        raise SystemExit("Re-scoring does not reproduce the saved scores; check models/final and the sample.")

    # Each site alone: no neighbours, full reads.
    alone = np.array([sc.rank(sc.logits([reads[i]], [kmers[i]], pos[i:i + 1]))[0] for i in range(len(sites))])

    # The example: MYC's top site with plenty of reads (signal and ensemble panels).
    order = np.argsort(-rank_full)
    ex = next(i for i in order if len(reads[i]) >= 40)
    # The slider site: the well-covered site its neighbours lift most, so one slider shows
    # both lessons - few reads weaken a site's own evidence; nearby sites can make up for it.
    # Only sites that end in the sample's top 5%: a lift for a site that looks unmodified
    # on its own reads would read as a flaw, not as corroboration.
    well = [i for i in range(len(sites)) if len(reads[i]) >= 40 and rank_full[i] >= 0.95]
    sl = max(well, key=lambda i: rank_full[i] - alone[i])
    rng = np.random.default_rng(4262)
    thinning = []
    for k in KS + [len(reads[sl])]:
        if k > len(reads[sl]):
            continue
        row = {"reads": k}
        for mode in ("with_neighbours", "alone"):
            draws = []
            for _ in range(DRAWS if k < len(reads[sl]) else 1):
                kept = reads[sl][rng.choice(len(reads[sl]), k, replace=False)]
                if mode == "alone":
                    draws.append(float(sc.rank(sc.logits([kept], [kmers[sl]], pos[sl:sl + 1]))[0]))
                else:
                    r = list(reads)
                    r[sl] = kept
                    draws.append(float(sc.rank(sc.logits(r, kmers, pos))[sl]))
            row[mode] = {"median": round(float(np.median(draws)), 4), "low": round(float(np.percentile(draws, 10)), 4),
                         "high": round(float(np.percentile(draws, 90)), 4), "draws": [round(d, 4) for d in draws]}
        thinning.append(row)
        print(f"  {k:3d} reads: with neighbours {row['with_neighbours']['median']:.3f}, alone {row['alone']['median']:.3f}"
              f" [{row['alone']['low']:.3f}, {row['alone']['high']:.3f}]")

    # A contrast: same 7 letters, plenty of reads, ranked low by the model (anywhere in the sample).
    saved_all = pd.read_csv(SCORES, usecols=["transcript_id", "transcript_position", "kmer", "n_reads", "score_rank"])
    cand = saved_all[(saved_all.kmer == kmers[ex]) & (saved_all.n_reads >= 40)].sort_values("score_rank")
    low = cand.iloc[0]
    low_tx = low.transcript_id.split(".")[0]
    contrast = next(s for s in iter_sites(JSON)
                    if s.transcript_id.split(".")[0] == low_tx and s.position == low.transcript_position)
    print(f"contrast: {low_tx}:{low.transcript_position} {low.kmer}, rank {low.score_rank:.3f}")

    def read_summary(r: np.ndarray) -> list:
        r = r[np.random.default_rng(0).permutation(len(r))[:SHOW_READS]]
        # Columns per pore position (-1, 0, +1): dwell, signal sd, mean current.
        return [[round(float(v), 4) for v in (row[2], row[5], row[8], row[0], row[3], row[6])] for row in r]

    nets = {n: float(sc.pct(n, full[n][ex:ex + 1])[0]) for n in sc.names}
    own = {}
    for name, net in zip(sc.names, sc.model.networks):
        if net.v2:
            s = net.site_vectors(reads[ex], np.array([0, len(reads[ex])]), kmer_onehot([kmers[ex]]))
            own[name] = float(dense(s, net.w, "aux_head")[0, 0])

    # The RNA as text, and the read tables of its first few candidate sites.
    seq = transcript_sequence(tx)
    for p, k in zip(pos, kmers):
        assert seq[p - 3:p + 4] == k, f"sequence and site letters disagree at {p}"
    net = sc.model.networks[sc.names.index(SHOWN_NET)]
    shown = list(range(TABLE_SITES))
    start, end = max(0, int(pos[shown[0]]) - 40), min(len(seq), int(pos[shown[-1]]) + 41)
    tables = [{"position": int(pos[i]), "kmer": kmers[i], "depth": len(reads[i]),
               "rows": [[round(float(v), 2) for v in row]
                        for row in transform_reads(reads[i], net.read_mean, net.read_scale)]} for i in shown]

    # The read network on the slider site, molecules in a fixed random order.
    own_ref = np.sort(pd.read_csv(SCORES, usecols=[f"own_{SHOWN_NET}"])[f"own_{SHOWN_NET}"].to_numpy())
    order_sl = np.random.default_rng(1).permutation(len(reads[sl]))
    layers = layer_story(net, reads[sl][order_sl], kmers[sl], own_ref)

    shortlist = pd.read_csv(RESULTS / "shortlist.csv")
    by_type = pd.read_csv(RESULTS / "by_type.csv")
    hit = lambda where, k, m: {r.cell_line.split(" ")[0]: round(r.hit_rate, 3) for r in  # noqa: E731
                               shortlist[(shortlist["evaluated on"] == where) & (shortlist.k == k)
                                         & (shortlist.method == m)].itertuples()}
    depth = by_type[by_type.family == "all"]
    motif = by_type[(by_type.family == "motif") & (by_type.depth == "full")]
    worse = motif.groupby("type").pair_acc.min().dropna().sort_values(ascending=False)

    story = {
        "gene": GENE, "sample": SAMPLE, "cell_line": "MCF7", "transcript": tx,
        "text": {"start": start, "letters": seq[start:end], "length": len(seq), "tables": tables,
                 "features": ["dwell −1", "spread −1", "current −1", "dwell 0", "spread 0", "current 0",
                              "dwell +1", "spread +1", "current +1"]},
        "network": {"name": SHOWN_NET, **layers},
        "kernel_nt": round(float(net.w["kernel_nt"]), 1), "radius_nt": 150,
        "sites": [{"position": int(p), "kmer": k, "reads": len(r), "rank": round(float(a), 4),
                   "alone": round(float(b), 4)} for p, k, r, a, b in zip(pos, kmers, reads, rank_full, alone)],
        "example": {"index": int(ex), "position": int(pos[ex]), "kmer": kmers[ex], "reads": len(reads[ex]),
                    "rank": round(float(rank_full[ex]), 4), "alone": round(float(alone[ex]), 4),
                    "signal": read_summary(reads[ex]), "networks": nets, "own_logits": own},
        "slider": {"index": int(sl), "position": int(pos[sl]), "kmer": kmers[sl], "reads": len(reads[sl]),
                   "rank": round(float(rank_full[sl]), 4), "alone": round(float(alone[sl]), 4),
                   "networks": {n: round(float(sc.pct(n, full[n][sl:sl + 1])[0]), 4) for n in sc.names},
                   "thinning": thinning},
        "contrast": {"transcript": low_tx, "position": int(low.transcript_position), "kmer": low.kmer,
                     "reads": int(low.n_reads), "rank": round(float(low.score_rank), 4),
                     "signal": read_summary(np.asarray(contrast.reads, dtype=np.float32))},
        "performance": {
            "source": "analysis/site_types/results/ (held-out genes, shipped designs, 3 seeds); "
                      "baselines from README.md Results",
            "pr_auc_by_reads": {cl.split(" ")[0]: dict(zip(g.depth, g.pr_auc.round(3)))
                                for cl, g in depth.groupby("cell_line")},
            "positive_rate": {"HCT116": 0.045, "HEK293T": 0.072},
            "baselines_pr_auc": [
                {"model": "Random", "HCT116": 0.045, "HEK293T": 0.072},
                {"model": "Logistic regression (simple baseline)", "HCT116": 0.415, "HEK293T": 0.331},
                {"model": "Quantiles + LightGBM", "HCT116": 0.472, "HEK293T": 0.392},
                {"model": "m6Anet (pretrained on HCT116)", "HCT116": 0.503, "HEK293T": 0.356,
                 "note": "HCT116 not held out: m6Anet was trained on it"},
                {"model": "Our model", "HCT116": 0.607, "HEK293T": 0.460}],
            "top3_hit": {m: hit("full reads", 3, m) for m in ("model", "motif rule", "random")},
            "top3_hit_1read": {m: hit("1 reads (thinned, 5 draws)", 3, m) for m in ("model", "motif rule", "random")},
            "motif_pair_accuracy": {k: round(float(v), 3) for k, v in worse.items()},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(story, separators=(",", ":")))
    print(f"-> {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
