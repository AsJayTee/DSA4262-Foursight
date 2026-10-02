"""Graph networks over reads -> sites -> transcript (xsrc_nets.py, discussed 2026-10-02).

Every site is a node. Its own reads are pooled into it (mean + sd, or gated
attention); sites on the same transcript within WINDOW nt are linked; a
transcript node (the mean of its sites) is a further neighbour. No gene level:
test files carry no gene ids and predict.py runs offline.

Labels never enter the graph - only other sites' *reads* do - and the
cross-source split keeps whole genes, so every neighbour of a held-out site is
held out too. The three variants differ only in the neighbour step:

  gcn    typical: average self, neighbours and transcript, then transform. It
         assumes homophily at whatever strength it learned (dataset0 clusters
         ~2x more than data1 - the neighbour features that did not transfer).
  h2gcn  heterophily-aware (Zhu et al. 2020): self, neighbour mean and the
         transcript mean excluding self kept as separate channels, and the
         head sees every layer - so it learns how far to trust neighbours.
  gat    attention: gated attention over a site's reads, then multi-head
         attention over its neighbours and transcript with a learned bias by
         distance - it chooses which reads and which neighbours count.

Plain torch (dense per-batch adjacency), no torch_geometric.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import common
from encoders import KMER, N_VALUES, log_count, masked_mean_sd, mlp, with_kmer

KINDS = ("gcn", "h2gcn", "gat")
WINDOW = 200          # nt; label clustering is 5-10x within 50 nt, ~6-13x within 200
TRAIN_SITES = 1024    # sites per training batch
SCORE_BUDGET = 400_000  # sites x widest site, per scoring batch


def t(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


class GraphNet(nn.Module):
    def __init__(self, kind: str, width: int = 64, d: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind, self.heads = kind, heads
        self.phi = mlp([N_VALUES + KMER, width, width])
        if kind == "gat":
            self.attend = nn.Sequential(nn.Linear(width, 32), nn.Tanh())
            self.gate = nn.Sequential(nn.Linear(width, 32), nn.Sigmoid())
            self.weigh = nn.Linear(32, 1)
            pooled = width
        else:
            pooled = 2 * width
        self.site_in = mlp([pooled + 1 + KMER, d, d])
        if kind == "gcn":
            self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.head = nn.Linear(d, 1)
        elif kind == "h2gcn":
            self.layers = nn.ModuleList([nn.Linear(3 * d + 1, d) for _ in range(layers)])
            self.head = nn.Linear(d * (layers + 1), 1)
        else:
            self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            # Bias per head from log distance; the transcript key gets its own.
            self.dist = nn.ModuleList([mlp([1, 16, heads], last_activation=False) for _ in range(layers)])
            self.t_bias = nn.Parameter(torch.zeros(layers, heads))
            self.norm1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.norm2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.ff = nn.ModuleList([mlp([d, 2 * d, d], last_activation=False) for _ in range(layers)])
            self.head = nn.Linear(d, 1)

    def sites(self, reads, mask, kmer):
        h = self.phi(with_kmer(reads, kmer))
        if self.kind == "gat":
            scores = self.weigh(self.attend(h) * self.gate(h)).squeeze(-1)
            w = torch.softmax(scores.masked_fill(~mask, float("-inf")), 1).unsqueeze(-1)
            pooled = (h * w).sum(1)
        else:
            pooled = torch.cat(masked_mean_sd(h, mask), -1)
        return self.site_in(torch.cat([pooled, log_count(mask), kmer], -1))

    def forward(self, reads, mask, kmer, gid, pos, n_graphs):
        s = self.sites(reads, mask, kmer)
        delta = (pos[:, None] - pos[None, :]).abs()
        adj = (gid[:, None] == gid[None, :]) & (delta <= WINDOW)
        adj.fill_diagonal_(False)
        deg = adj.sum(1, keepdim=True).float()
        count = torch.zeros(n_graphs).index_add_(0, gid, torch.ones(len(gid)))[gid].unsqueeze(1)

        def transcript_sum(h):
            return torch.zeros(n_graphs, h.shape[1]).index_add_(0, gid, h)[gid]

        if self.kind == "gcn":
            for layer in self.layers:
                mixed = (s + adj.float() @ s + transcript_sum(s) / count) / (deg + 2)
                s = F.relu(layer(mixed))
            return self.head(s).squeeze(-1)

        if self.kind == "h2gcn":
            outs = [s]
            for layer in self.layers:
                neighbours = (adj.float() @ s) / deg.clamp(min=1)
                others = (transcript_sum(s) - s) / (count - 1).clamp(min=1)
                s = F.relu(layer(torch.cat([s, neighbours, others, (deg > 0).float()], -1)))
                outs.append(s)
            return self.head(torch.cat(outs, -1)).squeeze(-1)

        n, h, dh = len(s), self.heads, s.shape[1] // self.heads
        keys = adj | torch.eye(n, dtype=torch.bool)
        logd = torch.log1p(delta.float()).unsqueeze(-1)
        for i in range(len(self.q)):
            x = self.norm1[i](s)
            q, k, v = (f(x).view(n, h, dh) for f in (self.q[i], self.k[i], self.v[i]))
            tmean = transcript_sum(x) / count
            kt, vt = self.k[i](tmean).view(n, h, dh), self.v[i](tmean).view(n, h, dh)
            # Only pairs inside one transcript's window get a score; the dense
            # distance bias is computed for those and masked elsewhere.
            scores = torch.einsum("ihd,jhd->hij", q, k) / dh ** 0.5
            scores = scores + self.dist[i](logd).permute(2, 0, 1)
            scores = scores.masked_fill(~keys, float("-inf"))
            st = ((q * kt).sum(-1) / dh ** 0.5 + self.t_bias[i]).T.unsqueeze(-1)   # (h, n, 1)
            attn = torch.softmax(torch.cat([scores, st], -1), -1)
            out = torch.einsum("hij,jhd->ihd", attn[..., :n], v) + attn[..., n:].permute(1, 0, 2) * vt
            s = s + self.o[i](out.reshape(n, -1))
            s = s + self.ff[i](self.norm2[i](s))
        return self.head(s).squeeze(-1)


# ------------------------------------------------------------------ batching

def graphs_of(rows: np.ndarray, graph_id: np.ndarray, position: np.ndarray) -> list[np.ndarray]:
    """Rows grouped into graphs (one per file x transcript), each sorted by position."""
    rows = np.asarray(rows)
    order = np.lexsort((position[rows], graph_id[rows]))
    rows = rows[order]
    cuts = np.flatnonzero(np.diff(graph_id[rows])) + 1
    return np.split(rows, cuts)


def _batch(graphs, picks, values, kmer, graph_rows_pos):
    width = max(len(p) for p in picks)
    reads = np.zeros((len(picks), width, values.shape[1]), dtype=np.float32)
    mask = np.zeros((len(picks), width), dtype=bool)
    for i, p in enumerate(picks):
        reads[i, :len(p)] = values[p]
        mask[i, :len(p)] = True
    rows = np.concatenate(graphs)
    gid = np.repeat(np.arange(len(graphs)), [len(g) for g in graphs])
    return (t(reads), t(mask), t(kmer[rows]), t(gid), t(graph_rows_pos[rows].astype(np.float32)),
            len(graphs)), rows


def score(model, graphs, values, offsets, kmer, position) -> dict:
    """Logit for every row of `graphs`, from all of each site's reads."""
    model.eval()
    out, batch, width, size = {}, [], 0, 0
    counts = np.diff(offsets)

    def flush():
        picks = [np.arange(offsets[r], offsets[r + 1]) for g in batch for r in g]
        inputs, rows = _batch(batch, picks, values, kmer, position)
        with torch.no_grad():
            out.update(zip(rows.tolist(), model(*inputs).numpy()))

    for g in graphs:
        new_width, new_size = max(width, int(counts[g].max())), size + len(g)
        if batch and (new_size * new_width > SCORE_BUDGET or new_size > 2048):
            flush()
            batch, new_width, new_size = [], int(counts[g].max()), len(g)
        batch.append(g)
        width, size = new_width, new_size
    if batch:
        flush()
    return out


def train(model, train_graphs, val_graphs, values, offsets, kmer, position, target, y_val,
          minutes: float, epochs: int, seed: int, log) -> list[dict]:
    """Recipe v2 of learn.train (AdamW, warm-up + cosine, patience 60), over
    batches of whole transcripts; early stopping on validation AP."""
    rng = np.random.default_rng(seed)
    batcher = common.SetBatcher(values, offsets, seed=seed)
    rows = np.concatenate(train_graphs)
    positives = target[rows].sum()
    pos_weight = torch.tensor([(len(rows) - positives) / max(positives, 1)])
    per_epoch = max(1, -(-len(rows) // TRAIN_SITES))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warm, total = 3 * per_epoch, max(epochs * per_epoch, 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(
        (s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(s, total) / total))))
    val_rows = np.concatenate(val_graphs)
    history, best, best_ap, started = [], None, -1.0, time.time()
    for epoch in range(epochs):
        model.train()
        losses, batch = [], []
        order = rng.permutation(len(train_graphs))
        for gi in order:
            batch.append(train_graphs[gi])
            if sum(len(b) for b in batch) < TRAIN_SITES and gi != order[-1]:
                continue
            picks = [batcher._subset(r) for b in batch for r in b]
            inputs, brows = _batch(batch, picks, values, kmer, position)
            loss = F.binary_cross_entropy_with_logits(model(*inputs), t(target[brows]),
                                                      pos_weight=pos_weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            losses.append(loss.item())
            batch = []
            if time.time() - started > minutes * 60:
                break
        scores = score(model, val_graphs, values, offsets, kmer, position)
        ap = float(average_precision_score(y_val[val_rows], [scores[r] for r in val_rows]))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ap": ap,
                        "minutes": round((time.time() - started) / 60, 2)})
        log(f"    epoch {epoch + 1}: loss {history[-1]['loss']:.4f}, val_ap {ap:.4f}, "
            f"minutes {history[-1]['minutes']}")
        if ap > best_ap:
            best_ap, best = ap, copy.deepcopy(model.state_dict())
        if epoch + 1 - max(i for i, h in enumerate(history, 1) if h["val_ap"] == best_ap) >= 60:
            log("    stopping: no validation gain in 60 epochs")
            break
        if time.time() - started > minutes * 60:
            break
    model.load_state_dict(best)
    model.eval()
    return history
