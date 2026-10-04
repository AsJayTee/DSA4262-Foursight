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

Variants of h2gcn (discussed 2026-10-02; every option defaults to the model above):

  window / local / transcript   which neighbours it sees: local edges within
         `window` nt, the transcript summary, or both - where its gain lives
  output="twohead"  one head per labelling (dataset0's, data1's); ships the mean
  output="noisy"    one hidden "is m6A" probability, read through a learned
         true-/false-positive rate per labelling (the two files as two noisy
         annotators of one truth); ships the hidden probability
  reader="knn"      a site's reads encoded by the mutual k-NN read graph
         (readgraph.py) instead of mean + sd: reads -> site -> transcript
  deep=N            N pre-norm residual blocks after the read encoder and after
         the site encoder (encoders.ResBlock): depth where it has not been
         tried inside the graph model; the graph itself stays 2 layers
  pool="quantile"   reads pooled into the site by mean, sd AND the 10/25/50/75/90th
         percentile of each read-encoding dimension - the shape of the read
         distribution (a shifted subpopulation sits in the tails), as the
         quantile features that first helped the trees
  frac=True         a per-read head estimates P(read modified); its mean over the
         site's reads (the modified fraction) and that mean's standard error
         join the site description
  aux=w             a second head scores each site from its own reads alone,
         before the graph, trained with weight w beside the main loss (deep
         supervision): the read encoder must keep evidence that works without
         neighbours. The shipped score is the main head's

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
from encoders import KMER, N_VALUES, ResBlock, log_count, masked_mean_sd, mlp, with_kmer
from learn import Warmup
import readgraph

KINDS = ("gcn", "h2gcn", "gat")
WINDOW = 200          # nt; label clustering is 5-10x within 50 nt, ~6-13x within 200
TRAIN_SITES = 1024    # sites per training batch
SCORE_BUDGET = 400_000  # sites x widest site, per scoring batch
QUADRATIC_BUDGET = 20_000_000  # sites x widest site^2, for the k-NN reader
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def masked_quantiles(h: torch.Tensor, mask: torch.Tensor, levels=QUANTILES) -> torch.Tensor:
    """(sites, reads, d) -> (sites, len(levels) * d): per-dimension quantiles over each
    site's valid reads, linearly interpolated like numpy's default. Differentiable
    through the sort. With one read every quantile is that read."""
    n = mask.sum(1).clamp(min=1).float()                                  # (sites,)
    ordered = h.masked_fill(~mask.unsqueeze(-1), float("inf")).sort(dim=1).values
    pos = (n[:, None] - 1) * torch.tensor(levels)[None, :]                # (sites, Q)
    lo, hi = pos.floor().long(), pos.ceil().long()
    w = (pos - lo.float()).unsqueeze(-1)
    take = lambda i: ordered.gather(1, i.unsqueeze(-1).expand(-1, -1, h.shape[-1]))  # noqa: E731
    return ((1 - w) * take(lo) + w * take(hi)).flatten(1)
OUTPUTS = ("single", "twohead", "noisy")


def t(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


class GraphNet(nn.Module):
    def __init__(self, kind: str, width: int = 64, d: int = 64, layers: int = 2, heads: int = 4,
                 window: int = WINDOW, local: bool = True, transcript: bool = True,
                 output: str = "single", reader: str = "mean", deep: int = 0, aux: float = 0.0,
                 pool: str = "meansd", frac: bool = False):
        super().__init__()
        if kind not in KINDS or output not in OUTPUTS or reader not in ("mean", "knn"):
            raise ValueError(f"kind {KINDS}, output {OUTPUTS}, reader mean|knn")
        if (not local or not transcript or output != "single" or reader != "mean" or deep or aux
                or pool != "meansd" or frac) and kind != "h2gcn":
            raise ValueError("the neighbour, output and reader variants are defined for h2gcn only")
        self.kind, self.heads = kind, heads
        self.window, self.local, self.transcript = window, local, transcript
        self.output, self.reader = output, reader
        self.quadratic = reader == "knn"
        if reader == "knn":
            self.knn = readgraph.KNNDeepSet("static", k=4)
            self.site_in = mlp([32, d, d])
        else:
            self.phi = mlp([N_VALUES + KMER, width, width])
        if reader == "knn":
            pass
        elif kind == "gat":
            self.attend = nn.Sequential(nn.Linear(width, 32), nn.Tanh())
            self.gate = nn.Sequential(nn.Linear(width, 32), nn.Sigmoid())
            self.weigh = nn.Linear(32, 1)
            pooled = width
        else:
            pooled = 2 * width
        self.pool, self.frac = pool, frac
        if pool == "quantile":
            pooled += len(QUANTILES) * width
        if frac:
            self.read_mod = nn.Linear(width, 1)
            pooled += 2
        if reader != "knn":
            self.site_in = mlp([pooled + 1 + KMER, d, d])
        if deep:
            # Residual stacks; the final LayerNorm keeps the scale the graph
            # layers were designed for. deep=0 leaves the recorded models intact.
            self.phi = nn.Sequential(self.phi, *[ResBlock(width) for _ in range(deep)], nn.LayerNorm(width))
            self.site_in = nn.Sequential(self.site_in, *[ResBlock(d) for _ in range(deep)], nn.LayerNorm(d))
        self.aux = aux
        if aux:
            # With two heads the auxiliary head has two as well, one per labelling.
            self.aux_head = nn.Linear(d, 2 if output == "twohead" else 1)
        if kind == "gcn":
            self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.head = nn.Linear(d, 1)
        elif kind == "h2gcn":
            n_in = d * (1 + local + transcript) + local
            self.layers = nn.ModuleList([nn.Linear(n_in, d) for _ in range(layers)])
            self.head = nn.Linear(d * (layers + 1), 2 if output == "twohead" else 1)
            if output == "noisy":
                # Per labelling (0 = dataset0, 1 = data1): P(called positive |
                # truly m6A) and P(called positive | not). Start near a clean
                # annotator so the hidden output starts out meaning "m6A".
                self.tpr = nn.Parameter(torch.full((2,), 2.2))     # sigmoid ~ 0.90
                self.fpr = nn.Parameter(torch.full((2,), -3.9))    # sigmoid ~ 0.02
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
        if self.reader == "knn":
            return self.site_in(self.knn.embed(reads, mask, kmer))
        h = self.phi(with_kmer(reads, kmer))
        if self.kind == "gat":
            scores = self.weigh(self.attend(h) * self.gate(h)).squeeze(-1)
            w = torch.softmax(scores.masked_fill(~mask, float("-inf")), 1).unsqueeze(-1)
            pooled = (h * w).sum(1)
        else:
            parts = list(masked_mean_sd(h, mask))
            if getattr(self, "pool", "meansd") == "quantile":
                parts.append(masked_quantiles(h, mask))
            if getattr(self, "frac", False):
                m = mask.float()
                n = m.sum(1, keepdim=True).clamp(min=1)
                f = (torch.sigmoid(self.read_mod(h)).squeeze(-1) * m).sum(1, keepdim=True) / n
                parts += [f, torch.sqrt(f * (1 - f) / n + 1e-6)]
            pooled = torch.cat(parts, -1)
        return self.site_in(torch.cat([pooled, log_count(mask), kmer], -1))

    def forward(self, reads, mask, kmer, gid, pos, n_graphs):
        s = self.sites(reads, mask, kmer)
        if self.aux:
            # Read by loss(); the shipped score never uses it.
            self._aux_logit = self.aux_head(s).squeeze(-1)
        delta = (pos[:, None] - pos[None, :]).abs()
        adj = (gid[:, None] == gid[None, :]) & (delta <= self.window)
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
                parts = [s]
                if self.local:
                    parts += [(adj.float() @ s) / deg.clamp(min=1), (deg > 0).float()]
                if self.transcript:
                    parts.append((transcript_sum(s) - s) / (count - 1).clamp(min=1))
                if self.local and self.transcript:
                    # The order every recorded h2gcn was trained with.
                    parts = [parts[0], parts[1], parts[3], parts[2]]
                s = F.relu(layer(torch.cat(parts, -1)))
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


    def predict_logit(self, *inputs):
        """The one score a site ships with."""
        out = self(*inputs)
        return out.mean(-1) if self.output == "twohead" else out

    def loss(self, out, rows, target, pos_weight):
        """Single output: BCE on `target` (the arm's target, as every recorded
        run). Two-head / noisy: each site's labels kept apart - its own file's,
        and the other file's where the site is in both - via `self.obs`
        (y_own, y_other with NaN where unshared, file), set by xsrc_nets.py.
        A shared site's copy carries each label at weight 1/2, so a physical
        site counts once over its two copies, as pooled_both weighs it."""
        if self.output == "single":
            loss = F.binary_cross_entropy_with_logits(out, t(target[rows]), pos_weight=pos_weight)
            if self.aux:
                loss = loss + self.aux * F.binary_cross_entropy_with_logits(
                    self._aux_logit, t(target[rows]), pos_weight=pos_weight)
            return loss
        y_own, y_other, file = (a[rows] for a in self.obs)
        shared = ~np.isnan(y_other)
        labels = torch.cat([t(y_own), t(np.nan_to_num(y_other[shared]))]).float()
        who = torch.cat([t(file), t(1 - file[shared])]).long()
        node = torch.cat([torch.arange(len(rows)), t(np.flatnonzero(shared))])
        weight = torch.cat([t(np.where(shared, 0.5, 1.0)), torch.full((int(shared.sum()),), 0.5)]).float()
        if self.output == "twohead":
            logits = out[node, who]
            per = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
            if self.aux:
                per = per + self.aux * F.binary_cross_entropy_with_logits(
                    self._aux_logit[node, who], labels, pos_weight=pos_weight, reduction="none")
        else:
            p = torch.sigmoid(out[node])
            q = (p * torch.sigmoid(self.tpr[who]) + (1 - p) * torch.sigmoid(self.fpr[who])).clamp(1e-6, 1 - 1e-6)
            # Up-weighting positives (as every other model here does) distorts
            # q away from a probability, and the learned rates then stop meaning
            # anything: h2gcn_noisy estimated data1 calls 18% of unmodified
            # sites positive, against a 7.3% positive rate (2026-10-03).
            w = 1.0 if getattr(self, "unweighted", False) else pos_weight
            per = -(w * labels * torch.log(q) + (1 - labels) * torch.log(1 - q))
        return (per * weight).sum() / weight.sum()


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
            predict = getattr(model, "predict_logit", model)
            out.update(zip(rows.tolist(), predict(*inputs).numpy()))

    for g in graphs:
        new_width, new_size = max(width, int(counts[g].max())), size + len(g)
        # The k-NN reader compares every pair of a site's reads.
        quadratic = getattr(model, "quadratic", False)
        cost = new_size * new_width ** 2 if quadratic else new_size * new_width
        if batch and (cost > (QUADRATIC_BUDGET if quadratic else SCORE_BUDGET) or new_size > 2048):
            flush()
            batch, new_width, new_size = [], int(counts[g].max()), len(g)
        batch.append(g)
        width, size = new_width, new_size
    if batch:
        flush()
    return out


def train(model, train_graphs, val_graphs, values, offsets, kmer, position, target, y_val,
          minutes: float, epochs: int, seed: int, log) -> list[dict]:
    """Recipe v3 of learn.train (AdamW, warm-up, halve on plateau, patience 25), over
    batches of whole transcripts; early stopping on validation AP."""
    rng = np.random.default_rng(seed)
    batcher = common.SetBatcher(values, offsets, seed=seed)
    rows = np.concatenate(train_graphs)
    positives = target[rows].sum()
    pos_weight = torch.tensor([(len(rows) - positives) / max(positives, 1)])
    per_epoch = max(1, -(-len(rows) // TRAIN_SITES))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # learn.py's recipe v3, so every model converges under the same rule.
    schedule = Warmup(optimiser, 3 * per_epoch)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, "max", factor=0.5, patience=8)
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
            loss = model.loss(model(*inputs), brows, target, pos_weight)
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
        plateau.step(ap)
        log(f"    epoch {epoch + 1}: loss {history[-1]['loss']:.4f}, val_ap {ap:.4f}, "
            f"minutes {history[-1]['minutes']}, lr {optimiser.param_groups[0]['lr']:.2e}")
        if ap > best_ap:
            best_ap, best = ap, copy.deepcopy(model.state_dict())
        if epoch + 1 - max(i for i, h in enumerate(history, 1) if h["val_ap"] == best_ap) >= 25:
            log("    stopping: no validation gain in 25 epochs")
            break
        if time.time() - started > minutes * 60:
            break
    model.load_state_dict(best)
    model.eval()
    return history
