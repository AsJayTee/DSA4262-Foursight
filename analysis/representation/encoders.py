"""The encoders from PLAN.md section 2.

Two kinds:

- **set** encoders see a site's reads as a padded (sites, reads, 9) batch and
  return one vector per site directly;
- **read** encoders code each read on its own; the site vector is a summary of
  those codes (and of reconstruction error) computed in `common.pool_codes`.

Every module takes standardised reads, a boolean read mask, the 7-mer one-hot
(28) and, for the position models, the three 5-mer windows (3 x 20). The read
count a module is shown is derived from the mask, so a site embedded from 3
reads is told it has 3.

Research code: torch is imported at module level here, which would be wrong in
`src/m6a/` (AGENTS.md section 4) and is fine in analysis/.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_VALUES = 9
KMER = 28
WINDOW = 20


def mlp(sizes: list[int], last_activation: bool = True, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or last_activation:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def log_count(mask: torch.Tensor) -> torch.Tensor:
    # log1p(991) = 6.9, so /5 keeps it near unit scale.
    return torch.log1p(mask.sum(1, keepdim=True).float()) / 5.0


def masked_mean_sd(h: torch.Tensor, mask: torch.Tensor):
    m = mask.unsqueeze(-1).float()
    n = m.sum(1).clamp(min=1.0)
    mean = (h * m).sum(1) / n
    var = (((h - mean.unsqueeze(1)) ** 2) * m).sum(1) / n
    # +1e-6 inside the root: a one-read site has zero variance, and the gradient
    # of sqrt at exactly zero is infinite.
    return mean, torch.sqrt(var + 1e-6)


def with_kmer(reads: torch.Tensor, kmer: torch.Tensor) -> torch.Tensor:
    return torch.cat([reads, kmer[:, None, :].expand(-1, reads.shape[1], -1)], -1)


# ------------------------------------------------------------------ set encoders

class DeepSet(nn.Module):
    """phi per read -> masked mean and sd -> rho -> z. Sum and max are avoided:
    both grow with the number of reads (PLAN.md section 1)."""

    def __init__(self, width: int = 64, z: int = 32, read_kmer: bool = True):
        super().__init__()
        # read_kmer=False is the mechanism ablation: each read is encoded
        # WITHOUT its sequence, which only joins after pooling.
        self.read_kmer = read_kmer
        self.phi = mlp([N_VALUES + (KMER if read_kmer else 0), width, width])
        self.rho = mlp([2 * width + 1 + KMER, width, z])
        self.head = nn.Linear(z, 1)

    def embed(self, reads, mask, kmer, windows=None):
        x = with_kmer(reads, kmer) if self.read_kmer else reads
        mean, sd = masked_mean_sd(self.phi(x), mask)
        return self.rho(torch.cat([mean, sd, log_count(mask), kmer], -1))

    def forward(self, reads, mask, kmer, windows=None):
        return self.head(self.embed(reads, mask, kmer)).squeeze(-1)


class AttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al. 2018), as src/m6a/models/mil.py."""

    def __init__(self, width: int = 64, gate: int = 32, z: int = 32):
        super().__init__()
        self.phi = mlp([N_VALUES + KMER, width, width])
        self.attend = nn.Sequential(nn.Linear(width, gate), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(width, gate), nn.Sigmoid())
        self.weigh = nn.Linear(gate, 1)
        self.rho = mlp([width + 1 + KMER, width, z])
        self.head = nn.Linear(z, 1)

    def embed(self, reads, mask, kmer, windows=None):
        h = self.phi(with_kmer(reads, kmer))
        scores = self.weigh(self.attend(h) * self.gate(h)).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), 1).unsqueeze(-1)
        return self.rho(torch.cat([(h * weights).sum(1), log_count(mask), kmer], -1))

    def forward(self, reads, mask, kmer, windows=None):
        return self.head(self.embed(reads, mask, kmer)).squeeze(-1)


class SetMasked(nn.Module):
    """Transformer across a site's reads, trained to fill in hidden values.

    Hiding whole reads would teach it only the site mean: every hidden read
    would get the same input and the best guess is the average. Hiding 30% of
    each read's *values* instead means predicting a read's hidden current from
    its visible dwell and sd **and the other reads at the site** - which is
    the within-site joint structure (coupling) as a self-supervised task.

    z = [the 7-mer token's output, which attends over every read ; the mean of
    the read tokens' outputs].
    """

    def __init__(self, d: int = 32, heads: int = 4, layers: int = 2, hide: float = 0.3):
        super().__init__()
        self.hide = hide
        self.read_in = nn.Linear(2 * N_VALUES, d)
        self.site_in = nn.Linear(KMER + 1, d)
        block = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, N_VALUES)

    def _run(self, reads, mask, kmer, keep):
        tokens = self.read_in(torch.cat([reads * keep, keep], -1))
        site = self.site_in(torch.cat([kmer, log_count(mask)], -1)).unsqueeze(1)
        padding = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        return self.body(torch.cat([site, tokens], 1), src_key_padding_mask=padding)

    def loss(self, reads, mask, kmer, windows=None):
        keep = (torch.rand_like(reads) >= self.hide).float()
        out = self.out(self._run(reads, mask, kmer, keep)[:, 1:])
        hidden = (1.0 - keep) * mask.unsqueeze(-1).float()
        return (((out - reads) ** 2) * hidden).sum() / hidden.sum().clamp(min=1.0)

    def embed(self, reads, mask, kmer, windows=None):
        h = self._run(reads, mask, kmer, torch.ones_like(reads))
        mean, _ = masked_mean_sd(h[:, 1:], mask)
        return torch.cat([h[:, 0], mean], -1)


class Contrastive(nn.Module):
    """A DeepSet trained so two disjoint read subsets of one site agree (InfoNCE)."""

    def __init__(self, z: int = 32, temperature: float = 0.1):
        super().__init__()
        self.encoder = DeepSet(z=z)
        self.project = mlp([z, z, z], last_activation=False)
        self.temperature = temperature

    def embed(self, reads, mask, kmer, windows=None):
        return self.encoder.embed(reads, mask, kmer)

    def pair_loss(self, first, second, kmer):
        a = F.normalize(self.project(self.embed(*first, kmer)), dim=-1)
        b = F.normalize(self.project(self.embed(*second, kmer)), dim=-1)
        logits = a @ b.T / self.temperature
        target = torch.arange(len(a))
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


class HandMLP(nn.Module):
    """A network over the hand features: the requested baseline NN on what we have."""

    def __init__(self, n_in: int, z: int = 32):
        super().__init__()
        self.body = mlp([n_in, 128, z], dropout=0.1)
        self.head = nn.Linear(z, 1)

    def embed(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


# ------------------------------------------------------------------ read encoders

class ReadAE(nn.Module):
    """Per-read autoencoder, bottleneck 4, sequence-conditioned on both sides.

    Without the 7-mer the bottleneck would spend itself on which bases are in
    the pore - the dominant source of variation in current - and 'badly
    reconstructed' would just mean 'rare sequence'.

    With `predict`, a site head on the pooled codes adds a supervised loss: the
    'bottleneck AE with goal to predict'.
    """

    def __init__(self, code: int = 4, width: int = 64, predict: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(mlp([N_VALUES + KMER, width, width]), nn.Linear(width, code))
        self.decoder = nn.Sequential(mlp([code + KMER, width, width]), nn.Linear(width, N_VALUES))
        self.predict = predict
        if predict:
            self.head = mlp([2 * code + 2 + 1 + KMER, 32, 1], last_activation=False)

    def codes(self, reads, kmer_per_read):
        return self.encoder(torch.cat([reads, kmer_per_read], -1))

    def reconstruct(self, reads, kmer_per_read):
        c = self.codes(reads, kmer_per_read)
        recon = self.decoder(torch.cat([c, kmer_per_read], -1))
        return c, ((recon - reads) ** 2).mean(-1)

    def loss(self, reads, mask, kmer, windows=None, y=None, pos_weight=None, weight=1.0):
        k = kmer[:, None, :].expand(-1, reads.shape[1], -1)
        c, err = self.reconstruct(reads, k)
        m = mask.float()
        reconstruction = (err * m).sum() / m.sum()
        if not self.predict:
            return reconstruction
        mean, sd = masked_mean_sd(c, mask)
        err_mean, err_sd = masked_mean_sd(err.unsqueeze(-1), mask)
        logit = self.head(torch.cat([mean, sd, err_mean, err_sd, log_count(mask), kmer], -1))
        bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y, pos_weight=pos_weight)
        return reconstruction + weight * bce

    def site_logit(self, reads, mask, kmer, windows=None):
        k = kmer[:, None, :].expand(-1, reads.shape[1], -1)
        c, err = self.reconstruct(reads, k)
        mean, sd = masked_mean_sd(c, mask)
        err_mean, err_sd = masked_mean_sd(err.unsqueeze(-1), mask)
        return self.head(torch.cat([mean, sd, err_mean, err_sd, log_count(mask), kmer], -1)).squeeze(-1)

    def read_codes(self, reads, kmer, windows):
        """Flat reads -> (codes, reconstruction error). kmer is per read here."""
        return self.reconstruct(reads, kmer)


def _steps(reads, windows):
    """(N, 9) values + (N, 3, 20) windows -> (N, 3, 23): one step per pore position.
    The value order is dwell, sd, mean at -1, then 0, then +1 (docs/data.md)."""
    return torch.cat([reads.view(-1, 3, 3), windows], -1)


class _PositionDecoder(nn.Module):
    """Rebuilds the 3 x 3 readings from the code, told which 5-mer was in the pore."""

    def __init__(self, code: int, hidden: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(code + WINDOW, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 3)

    def forward(self, code, windows):
        x = torch.cat([code[:, None, :].expand(-1, 3, -1), windows], -1)
        return self.out(self.rnn(x)[0]).reshape(-1, N_VALUES)


class PositionLSTMAE(nn.Module):
    """LSTM over the three pore positions of one read; final hidden -> code(4)."""

    def __init__(self, code: int = 4, hidden: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(3 + WINDOW, hidden, batch_first=True)
        self.to_code = nn.Linear(hidden, code)
        self.decoder = _PositionDecoder(code, hidden)

    def encode(self, reads, windows):
        _, (h, _) = self.rnn(_steps(reads, windows))
        return self.to_code(h[-1])

    def read_codes(self, reads, kmer, windows):
        c = self.encode(reads, windows)
        return c, ((self.decoder(c, windows) - reads) ** 2).mean(-1)

    def loss(self, reads, mask, kmer, windows=None, **_):
        flat = reads[mask]
        w = windows[:, None].expand(-1, reads.shape[1], -1, -1)[mask]
        return self.read_codes(flat, None, w)[1].mean()


class PositionAttentionAE(PositionLSTMAE):
    """PositionLSTMAE with self-attention over the three positions instead of the
    LSTM encoder. Same decoder, so any difference is the encoder's."""

    def __init__(self, code: int = 4, d: int = 32, heads: int = 4, layers: int = 2):
        super().__init__(code, d)
        self.rnn = None
        self.step_in = nn.Linear(3 + WINDOW, d)
        self.position = nn.Parameter(torch.zeros(1, 4, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        block = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def encode(self, reads, windows):
        tokens = self.step_in(_steps(reads, windows))
        x = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], 1) + self.position
        return self.to_code(self.body(x)[:, 0])


# ------------------------------------------------------------------ residual variants
#
# Added after the fit diagnostic (results/diagnose_fit.json): deepset and
# attn_mil fit their OWN training sites no better than unseen ones (AP ~0.47
# vs ~0.46-0.52), with training loss still falling when early stopping ended
# them - an underfitting signature. Pre-norm residual blocks are the standard
# fix for networks that fit slowly (tabular residual MLPs; the transformer
# block), so these test whether better optimisation lifts the ceiling.

class ResBlock(nn.Module):
    """Pre-norm residual MLP block: x + W2(GELU(W1(LayerNorm(x))))."""

    def __init__(self, d: int, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.body = nn.Sequential(nn.Linear(d, expand * d), nn.GELU(), nn.Linear(expand * d, d))

    def forward(self, x):
        return x + self.body(self.norm(x))


class _ResidualSetBase(nn.Module):
    def __init__(self, width: int, z: int, read_blocks: int, site_blocks: int, pooled: int):
        super().__init__()
        self.read_in = nn.Linear(N_VALUES + KMER, width)
        self.read_blocks = nn.Sequential(*[ResBlock(width) for _ in range(read_blocks)],
                                         nn.LayerNorm(width))
        self.site_in = nn.Linear(pooled + 1 + KMER, width)
        self.site_blocks = nn.Sequential(*[ResBlock(width) for _ in range(site_blocks)],
                                         nn.LayerNorm(width))
        self.to_z = nn.Sequential(nn.Linear(width, z), nn.GELU())
        self.head = nn.Linear(z, 1)

    def encode_reads(self, reads, kmer):
        return self.read_blocks(self.read_in(with_kmer(reads, kmer)))

    def encode_site(self, pooled, mask, kmer):
        h = self.site_blocks(self.site_in(torch.cat([pooled, log_count(mask), kmer], -1)))
        return self.to_z(h)

    def forward(self, reads, mask, kmer, windows=None):
        return self.head(self.embed(reads, mask, kmer)).squeeze(-1)


class ResidualDeepSet(_ResidualSetBase):
    """deepset with residual per-read and site networks; mean + sd pooling."""

    def __init__(self, width: int = 128, z: int = 32, read_blocks: int = 4, site_blocks: int = 2):
        super().__init__(width, z, read_blocks, site_blocks, 2 * width)

    def embed(self, reads, mask, kmer, windows=None):
        mean, sd = masked_mean_sd(self.encode_reads(reads, kmer), mask)
        return self.encode_site(torch.cat([mean, sd], -1), mask, kmer)


class ResidualAttentionMIL(_ResidualSetBase):
    """attn_mil with residual per-read and site networks; gated attention pooling."""

    def __init__(self, width: int = 128, gate: int = 64, z: int = 32, read_blocks: int = 4,
                 site_blocks: int = 2):
        super().__init__(width, z, read_blocks, site_blocks, width)
        self.attend = nn.Sequential(nn.Linear(width, gate), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(width, gate), nn.Sigmoid())
        self.weigh = nn.Linear(gate, 1)

    def embed(self, reads, mask, kmer, windows=None):
        h = self.encode_reads(reads, kmer)
        scores = self.weigh(self.attend(h) * self.gate(h)).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), 1).unsqueeze(-1)
        return self.encode_site((h * weights).sum(1), mask, kmer)


class _AttentionBlock(nn.Module):
    """Pre-norm transformer block over a site's reads, padding masked."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = ResBlock(d)

    def forward(self, x, padding):
        h = self.norm(x)
        x = x + self.attn(h, h, h, key_padding_mask=padding, need_weights=False)[0]
        return self.ff(x)


class SetTransformer(nn.Module):
    """Supervised set transformer (Lee et al. 2019, SAB + PMA).

    Reads attend to each other, so the model can represent read-to-read
    structure - e.g. a subgroup of molecules that agree with each other, which
    is what a partly modified site looks like - which every per-read-then-pool
    model here cannot. A learned query then attends over the reads to pool
    them, alongside mean + sd so it starts no worse than deepset.
    """

    QUADRATIC = True   # scoring batches are sized by reads^2 (learn.site_logits)

    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2, z: int = 32):
        super().__init__()
        self.read_in = nn.Linear(N_VALUES + KMER, d)
        self.blocks = nn.ModuleList([_AttentionBlock(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.site_in = nn.Linear(3 * d + 1 + KMER, d)
        self.site_blocks = nn.Sequential(ResBlock(d), ResBlock(d), nn.LayerNorm(d))
        self.to_z = nn.Sequential(nn.Linear(d, z), nn.GELU())
        self.head = nn.Linear(z, 1)

    def embed(self, reads, mask, kmer, windows=None):
        x = self.read_in(with_kmer(reads, kmer))
        padding = ~mask
        for block in self.blocks:
            x = block(x, padding)
        x = self.norm(x)
        query = self.seed.expand(len(x), -1, -1)
        attended = self.pool(query, x, x, key_padding_mask=padding,
                             need_weights=False)[0].squeeze(1)
        mean, sd = masked_mean_sd(x, mask)
        h = self.site_in(torch.cat([attended, mean, sd, log_count(mask), kmer], -1))
        return self.to_z(self.site_blocks(h))

    def forward(self, reads, mask, kmer, windows=None):
        return self.head(self.embed(reads, mask, kmer)).squeeze(-1)
