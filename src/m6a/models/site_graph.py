"""The shipped site-graph ensemble, scored in plain numpy (docs/decisions/0033, 0034).

The networks are trained with torch (analysis/representation/: graph.py,
final_fit.py) and exported to .npz by analysis/representation/export_final.py.
This module re-implements their forward pass - dense layers, ReLU, averages
over reads and over neighbouring sites - so predict.py needs no torch.
tests/test_site_graph.py checks it against the torch model on the same sites.

**Inference only.** It cannot be trained here: `fit` raises. It also does not
score a feature table like every other model - it needs each site's reads,
7-mer and transcript position, to group sites into transcript graphs - so it
sets CONSUMES_SITES and predict.py hands it the sites themselves.

How one network scores a site (graph.GraphNet, kind="h2gcn"):

  1. each read: its 9 values (dwell, sd, mean at -1/0/+1; dwell and sd logged,
     then standardised as in training) next to the site's 7-mer one-hot
     -> 2 dense layers -> 64 numbers
  2. the site: mean and sd of those over its reads, log read count, 7-mer
     -> 2 dense layers -> 64 numbers
  3. two graph layers: each site's vector beside the mean of its neighbours on
     the same transcript within `window` nt (and a has-neighbour flag), and/or
     the mean of the transcript's OTHER sites -> dense layer
  4. the head reads the site's vector from all three stages. A two-head network
     has one output per labelling and ships their mean.

The constrained designs (0034: res_gate, scalar_drop) keep stages 1-2 and add
an own-reads score s = aux_head(site vector). Their neighbours are every site
within 150 nt, weighted exp(-d / lambda) with lambda stored in the network:
  res_gate     s + g x (stages 3-4 output) x has-neighbour,
               g = sigmoid(a log(1 + reads) + b |s| + c)
  scalar_drop  s + c_mlp([s, log(1 + reads), weighted mean of neighbours' s,
               best neighbour s within 75 nt, weighted mean of their log reads,
               has-neighbour]) - neighbours pass one number, never a vector
Dropout was a training device only; nothing is dropped here.

The ensemble ranks each network's scores within the file and averages the
ranks - the combination that was evaluated (analysis/representation/ensemble.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m6a.models.base import BaseModel
from m6a.registry import register

# Mirrors analysis/representation/common.py, which the networks were trained
# with: dwell and current sd are positive and heavy-tailed, so they are logged.
LOG_COLUMNS = [0, 1, 3, 4, 6, 7]
BASES = "ACGT"
# graph.FK_RADIUS: the constrained designs' neighbourhood; and the scalar
# messages' "best neighbour" distance.
KERNEL_RADIUS = 150
BEST_RADIUS = 75


def transform_reads(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    out[:, LOG_COLUMNS] = np.log(np.maximum(out[:, LOG_COLUMNS], 1e-6))
    return ((out - mean) / scale).astype(np.float32)


def kmer_onehot(kmers: list[str]) -> np.ndarray:
    lookup = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(kmers), 7 * 4), dtype=np.float32)
    for row, kmer in enumerate(kmers):
        for pos, base in enumerate(kmer):
            out[row, pos * 4 + lookup[base]] = 1.0
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def dense(x: np.ndarray, w: dict, name: str) -> np.ndarray:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def mlp2(x: np.ndarray, w: dict, name: str) -> np.ndarray:
    """encoders.mlp([a, b, c]): Linear, ReLU, Linear, ReLU."""
    return relu(dense(relu(dense(x, w, f"{name}.0")), w, f"{name}.2"))


class Network:
    """One exported GraphNet."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as stored:
            self.w = {k: stored[k] for k in stored.files if not k.startswith("_")}
            self.meta = json.loads(str(stored["_meta"]))
            self.read_mean, self.read_scale = stored["_read_mean"], stored["_read_scale"]
        kw = self.meta["kwargs"]
        if kw.get("kind") != "h2gcn" or kw.get("reader", "mean") != "mean" or kw.get("deep", 0) \
                or kw.get("pool", "meansd") != "meansd" or kw.get("gps") or kw.get("nbr", "mean") == "bands":
            raise ValueError(f"{path.name}: only plain-read-encoder h2gcn networks (mean or kernel "
                             f"neighbours, residual or scalar messages) can be scored in numpy, got "
                             f"{kw}. Re-export a supported model.")
        # graph.GraphNet.v2: the constrained designs take a different forward pass.
        self.v2 = (kw.get("nbr", "mean") != "mean" or kw.get("residual", False) or kw.get("scalar", False)
                   or bool(kw.get("edge_drop")) or bool(kw.get("node_drop")))
        self.kernel = kw.get("nbr", "mean") == "kernel" or kw.get("scalar", False)
        self.residual, self.gate = kw.get("residual", False), kw.get("gate", True)
        self.scalar = kw.get("scalar", False)
        self.window = kw.get("window", 200)
        self.local, self.transcript = kw.get("local", True), kw.get("transcript", True)
        self.twohead = kw.get("output", "single") == "twohead"
        self.n_layers = len([k for k in self.w if k.startswith("layers.") and k.endswith(".weight")])

    def site_vectors(self, reads: np.ndarray, offsets: np.ndarray, kmers: np.ndarray) -> np.ndarray:
        """Stages 1-2 for every site at once; reads are concatenated, `offsets` (n+1)."""
        counts = np.diff(offsets)
        x = transform_reads(reads, self.read_mean, self.read_scale)
        h = mlp2(np.concatenate([x, np.repeat(kmers, counts, axis=0)], 1), self.w, "phi")
        mean = np.add.reduceat(h, offsets[:-1], axis=0) / counts[:, None]
        # Two-pass variance over each site's reads, as the torch model computes it.
        var = np.add.reduceat((h - np.repeat(mean, counts, axis=0)) ** 2, offsets[:-1], axis=0) / counts[:, None]
        sd = np.sqrt(var + 1e-6)
        log_count = (np.log1p(counts) / 5.0)[:, None].astype(np.float32)
        return mlp2(np.concatenate([mean, sd, log_count, kmers], 1), self.w, "site_in")

    def logits(self, s: np.ndarray, positions: np.ndarray, counts: np.ndarray | None = None) -> np.ndarray:
        """Stages 3-4 for ONE transcript's sites (`counts`: reads per site, needed by v2)."""
        if self.v2:
            return self._logits_v2(s, positions, counts)
        delta = np.abs(positions[:, None] - positions[None, :])
        adj = (delta <= self.window).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        deg = adj.sum(1, keepdims=True)
        n = len(s)
        outs = [s]
        for i in range(self.n_layers):
            parts = [s]
            if self.local:
                parts.append((adj @ s) / np.maximum(deg, 1.0))
            if self.transcript:
                parts.append((s.sum(0, keepdims=True) - s) / max(n - 1, 1))
            if self.local:
                parts.append((deg > 0).astype(np.float32))   # the order GraphNet was trained with
            s = relu(dense(np.concatenate(parts, 1), self.w, f"layers.{i}"))
            outs.append(s)
        out = dense(np.concatenate(outs, 1), self.w, "head")
        return out.mean(1) if self.twohead else out[:, 0]

    def _logits_v2(self, s: np.ndarray, positions: np.ndarray, counts: np.ndarray | None) -> np.ndarray:
        """graph.GraphNet.forward_v2 in eval mode, for ONE transcript's sites."""
        if counts is None:
            raise ValueError("the constrained designs need each site's read count")
        own = dense(s, self.w, "aux_head")[:, 0]
        logn = np.log1p(np.asarray(counts, dtype=np.float32))
        delta = np.abs(positions[:, None] - positions[None, :]).astype(np.float32)
        adj = delta <= (KERNEL_RADIUS if self.kernel else self.window)
        np.fill_diagonal(adj, False)
        a = adj.astype(np.float32)
        has = (a.sum(1) > 0).astype(np.float32)
        if self.kernel:
            kern = a * np.exp(-delta / float(self.w["kernel_nt"]))
            weights = kern / np.maximum(kern.sum(1, keepdims=True), 1e-9)
        else:
            weights = a / np.maximum(a.sum(1, keepdims=True), 1.0)
        if self.scalar:
            near = adj & (delta <= BEST_RADIUS)
            best = np.where(near.any(1), np.where(near, own[None, :], -1e9).max(1), 0.0)
            feats = np.stack([own, logn, weights @ own, best, weights @ logn, has], -1).astype(np.float32)
            return own + dense(relu(dense(feats, self.w, "c_mlp.0")), self.w, "c_mlp.2")[:, 0]
        outs = [s]
        for i in range(self.n_layers):
            s = relu(dense(np.concatenate([s, weights @ s, has[:, None]], 1), self.w, f"layers.{i}"))
            outs.append(s)
        correction = dense(np.concatenate(outs, 1), self.w, "head")[:, 0]
        if not self.residual:
            return correction
        ga, gb, gc = self.w["gate_params"]
        g = 1.0 / (1.0 + np.exp(-(ga * logn + gb * np.abs(own) + gc))) if self.gate else np.ones_like(own)
        return own + g * correction * has


@register("models", "site_graph_ensemble")
class SiteGraphEnsemble(BaseModel):
    name = "site_graph_ensemble"
    CONSUMES_SITES = True

    def __init__(self, networks: list[Network] | None = None):
        self.networks = networks or []

    def fit(self, X, y, groups=None):  # noqa: D401
        raise NotImplementedError(
            "site_graph_ensemble is inference-only. Train with "
            "analysis/representation/final_fit.py, export with export_final.py.")

    def predict_proba(self, X):
        raise NotImplementedError("site_graph_ensemble scores sites, not a feature table: "
                                  "use predict_sites (predict.py does).")

    def save(self, directory: str | Path) -> None:
        raise NotImplementedError("Written by analysis/representation/export_final.py.")

    @classmethod
    def load(cls, directory: str | Path) -> "SiteGraphEnsemble":
        meta = json.loads((Path(directory) / "meta.json").read_text())
        return cls([Network(Path(directory) / f) for f in meta["networks"]])

    def predict_sites(self, sites) -> pd.DataFrame:
        """Score every site of an iterable of m6a.data.Site; returns
        transcript_id, transcript_position, score (the averaged within-file rank)."""
        ids, positions, kmers, chunks, counts = [], [], [], [], []
        for site in sites:
            ids.append(site.transcript_id)
            positions.append(site.position)
            kmers.append(site.kmer)
            chunks.append(np.asarray(site.reads, dtype=np.float32))
            counts.append(len(site.reads))
        if not ids:
            return pd.DataFrame(columns=["transcript_id", "transcript_position", "score"])
        frame = pd.DataFrame({"transcript_id": ids, "transcript_position": positions})
        reads = np.concatenate(chunks)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        onehot = kmer_onehot(kmers)
        pos = frame["transcript_position"].to_numpy().astype(np.float32)
        groups = frame.groupby("transcript_id", sort=False).indices
        ranks = []
        for net in self.networks:
            # Stages 1-2 in chunks of sites: memory scales with reads, not sites.
            s = np.concatenate([
                net.site_vectors(reads[offsets[a]:offsets[b]], offsets[a:b + 1] - offsets[a], onehot[a:b])
                for a, b in _chunks(offsets, 2_000_000)])
            logit = np.empty(len(frame), dtype=np.float64)
            n_reads = np.asarray(counts)
            for rows in groups.values():
                logit[rows] = net.logits(s[rows], pos[rows], n_reads[rows])
            ranks.append(pd.Series(logit).rank(method="average").to_numpy() / len(logit))
        frame["score"] = np.mean(ranks, 0)
        return frame


def _chunks(offsets: np.ndarray, max_reads: int):
    """(start, stop) site ranges holding at most ~max_reads reads each."""
    start, n = 0, len(offsets) - 1
    while start < n:
        stop = int(np.searchsorted(offsets, offsets[start] + max_reads, side="right")) - 1
        stop = max(stop, start + 1)
        yield start, min(stop, n)
        start = min(stop, n)
