"""A k-NN graph over a site's reads as the read encoder (discussed 2026-10-02).

A site's reads are a point cloud in the 9 measured dimensions. Deepset encodes
each read alone and pools (PointNet); this lets reads exchange information
with their *nearest* reads first (EdgeConv, DGCNN - Wang et al. 2019), so a
read inside a tight group of shifted reads - a modified subpopulation - can be
told apart from a lone outlier. Sparse on purpose: distant reads never talk.

The graph is **mutual** k-NN: i and j are linked only if each is among the
other's k nearest, so outliers are left isolated, and each read's degree is an
input. Every read also has a self-loop, so an isolated read still has itself.

  static   the graph is built once, on the standardised measured values
  dynamic  rebuilt in each layer's learned space (DGCNN's variant)
  random   the control: the static graph with its reads shuffled across its
           nodes - identical degrees and shape, but no longer tied to which
           reads are alike. If it matches `static`, the similarity structure
           is not what helps.

Interface as encoders.DeepSet (padded reads, mask, k-mer), so learn.train and
learn.site_logits drive it unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from encoders import KMER, N_VALUES, log_count, masked_mean_sd, mlp, with_kmer

MODES = ("static", "dynamic", "random")


def mutual_knn(x: torch.Tensor, mask: torch.Tensor, k: int):
    """(neighbour index (B, R, k), valid (B, R, k)) of the mutual k-NN graph."""
    d = torch.cdist(x, x)
    big = torch.finfo(d.dtype).max
    invalid = ~(mask[:, :, None] & mask[:, None, :])
    d = d.masked_fill(invalid, big)
    d.diagonal(dim1=1, dim2=2).fill_(big)
    k = min(k, x.shape[1] - 1)
    if k < 1:
        return torch.zeros(*x.shape[:2], 0, dtype=torch.long), torch.zeros(*x.shape[:2], 0, dtype=torch.bool)
    nearest = d.topk(k, -1, largest=False).indices                        # (B, R, k)
    knn = torch.zeros(d.shape, dtype=torch.bool).scatter_(-1, nearest, True) & ~invalid
    knn.diagonal(dim1=1, dim2=2).fill_(False)
    mutual = knn & knn.transpose(1, 2)
    return nearest, mutual.gather(-1, nearest)


def gather_rows(h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """h (B, R, d), index (B, R, k) -> (B, R, k, d): each read's neighbours' rows."""
    b, r, k = index.shape
    return torch.gather(h, 1, index.reshape(b, r * k, 1).expand(-1, -1, h.shape[-1])).view(b, r, k, -1)


class EdgeConv(nn.Module):
    """h_i' = max over j in N(i) + {i} of MLP([h_i, h_j - h_i, |x_j - x_i|])."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.edge = mlp([2 * d_in + 1, d_out, d_out])

    def forward(self, h, neighbours, valid, dist):
        b, r, k = neighbours.shape
        own = self.edge(torch.cat([h, torch.zeros_like(h), torch.zeros(b, r, 1)], -1))
        if k == 0:
            return own
        hj, hi = gather_rows(h, neighbours), h.unsqueeze(2)
        messages = self.edge(torch.cat([hi.expand_as(hj), hj - hi, dist.unsqueeze(-1)], -1))
        messages = messages.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        return torch.maximum(own, messages.max(2).values)


class KNNDeepSet(nn.Module):
    QUADRATIC = True   # scoring batches are sized by reads^2 (learn.site_logits)

    def __init__(self, mode: str, k: int, width: int = 64, z: int = 32, layers: int = 2):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.k = mode, k
        self.read_in = mlp([N_VALUES + KMER + 1, width])
        self.convs = nn.ModuleList([EdgeConv(width, width) for _ in range(layers)])
        self.rho = mlp([2 * width + 1 + KMER, width, z])
        self.head = nn.Linear(z, 1)

    def _graph(self, space, reads, mask):
        """(neighbours, valid, distance in measured space) for every read."""
        neighbours, valid = mutual_knn(space, mask, self.k)
        b, r, k = neighbours.shape
        if k == 0:
            return neighbours, valid, None
        dist = (gather_rows(reads, neighbours) - reads.unsqueeze(2)).norm(dim=-1)
        if self.mode == "random":
            # Move every read to a random node of its own site's graph: the
            # edge (u, v) now joins reads perm[u] and perm[v]. Shape, degrees
            # and edge distances are kept; which reads are linked is not.
            # Padding sorts last and maps to itself (pad() puts reads first).
            keys = torch.rand(b, r).masked_fill(~mask, 2.0) + (~mask) * torch.arange(r)
            perm = keys.argsort(1)
            moved = torch.gather(perm, 1, neighbours.flatten(1)).view_as(neighbours)
            at = perm.unsqueeze(-1).expand(-1, -1, k)
            neighbours = torch.empty_like(neighbours).scatter_(1, at, moved)
            valid = torch.empty_like(valid).scatter_(1, at, valid)
            dist = torch.empty_like(dist).scatter_(1, at, dist)
        return neighbours, valid, dist

    def embed(self, reads, mask, kmer, windows=None):
        neighbours, valid, dist = self._graph(reads, reads, mask)
        degree = torch.log1p(valid.sum(-1, keepdim=True).float())
        h = self.read_in(torch.cat([with_kmer(reads, kmer), degree], -1))
        for i, conv in enumerate(self.convs):
            if self.mode == "dynamic" and i > 0:
                neighbours, valid, dist = self._graph(h, reads, mask)
            h = conv(h, neighbours, valid, dist)
        mean, sd = masked_mean_sd(h, mask)
        return self.rho(torch.cat([mean, sd, log_count(mask), kmer], -1))

    def forward(self, reads, mask, kmer, windows=None):
        return self.head(self.embed(reads, mask, kmer)).squeeze(-1)
