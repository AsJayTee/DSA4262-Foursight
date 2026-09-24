"""How strongly the pore positions move *together* across the reads at a site.

This is the feature set `quantiles_grid_v1` should have been, and the reason it
was not is worth keeping, because it is a general trap.

## What the grid attempt got wrong

`quantiles_grid_v1` took per-read contrasts along the position axis -
centre-surround and gradient - and quantiled them across reads. It screened
beautifully (univariate abs(AUC-0.5) = 0.1198, second only to `mean_m1_q05`) and
came back at **+0.0032, 29/50, corrected p = 0.4070**. Measured afterwards, every
one of its 21 columns sat at **abs(r) = 0.86 to 0.97** against a column the model
already had.

The mechanism: **for near-independent X and Y, the distribution of X - Y is
determined by the two marginals.** And the positions *are* near-independent
across reads - measured within-site correlation is r = 0.13 (mean current),
0.00 (dwell), 0.00 (sd). So a quantile of a per-read difference is a marginal
summary of a derived variable, not a joint statistic, and the marginals already
pinned it down. The per-read pairing was real and the summary threw it away.

## What actually carries the signal

The **correlation itself**, which is the one second-moment quantity the
marginals cannot produce: `r = E[(X-ux)(Y-uy)] / (sx sy)` needs `E[XY]`, and
quantiles of X and of Y never give you that.

Measured on 30,000 sites, within-site correlation between mean current at the
three positions, across reads:

| | negatives | positives | abs(AUC-0.5) |
|---|---:|---:|---:|
| `posr_mean` | 0.134 | **0.244** | **0.2183** |
| `posr_dwell` | 0.005 | -0.004 | 0.0325 |
| `posr_sd` | -0.001 | 0.004 | 0.0128 |

**0.2183 is stronger than every column in `quantiles_v1`** (the best is
`mean_m1_q05` at 0.1625), and its largest correlation with any existing column
is **0.226** - against 0.86-0.97 for the grid columns. Strong *and* new, which
is the combination the grid lacked.

## Why it should work, physically

The three positions are three overlapping 5-mer windows that all contain the
candidate base. A modification perturbs the pore while that base is inside it,
so on a **modified molecule all three readings shift together**; on an
unmodified one they vary independently. A site where some fraction of molecules
is modified therefore shows *elevated correlation across positions* - and that
is a direct signature of the Multiple Instance Learning structure
(docs/data.md#read-depth: only some copies of a modified site carry the mark).

Mean current is where it shows and dwell and sd are where it does not, which is
consistent: current is the quantity the base identity actually sets.

## This also explains why `quantiles_joint_v1` was null

That feature set computed each read's Mahalanobis distance from the site's own
centroid using **the site's own correlation matrix** as the whitening. Whitening
by the correlation is precisely *removing* the correlation. It conditioned away
the thing that carries the information, which is why it came back at +0.0013.

## The limitation, stated up front

A correlation needs several reads. At depth 1 and 2 it is undefined and these
columns are 0.0; at 20 reads the sampling noise on r is about 0.24, against an
effect of ~0.11. So this should help most at **high depth**, which is the
regime the graded evaluation is in and the opposite of where the grid was
predicted (wrongly) to help. Do not expect anything from it on the depth sweep.
"""

from __future__ import annotations

import numpy as np

from m6a.data import Site
from m6a.features.quantiles import QuantileFeatures
from m6a.registry import register

# Column indices of each measurement at pore positions -1, 0, +1.
POSITION_COLUMNS: dict[str, tuple[int, int, int]] = {
    "dwell": (0, 3, 6),
    "sd": (1, 4, 7),
    "mean": (2, 5, 8),
}

# The three measurements at the centre position, for the cross-channel block:
# dwell_0, sd_0, mean_0 in READ_FEATURE_NAMES order.
CENTRE_COLUMNS: dict[str, int] = {"dwell": 3, "sd": 4, "mean": 5}

# Below this many reads a correlation is noise, and at one or two reads it does
# not exist. Reported as 0.0, with n_reads already in the row to disambiguate.
MIN_READS_FOR_CORRELATION = 5

_PAIRS = (("m1", "0", 0, 1), ("0", "p1", 1, 2), ("m1", "p1", 0, 2))

#: Every column this feature set adds to `quantiles_v1`, in emission order.
COUPLING_COLUMNS = tuple(
    [
        f"coupling_{m}_{a}_{b}"
        for m in POSITION_COLUMNS
        for a, b, _, _ in _PAIRS
    ]
    + [f"coupling_{m}_mean" for m in POSITION_COLUMNS]
    + ["coupling_chan_dwell_sd", "coupling_chan_sd_mean", "coupling_chan_dwell_mean"]
)


def _safe_corrcoef(block: np.ndarray) -> np.ndarray:
    """Correlation matrix of the columns of `block`, with zeros where undefined.

    A column that never varies has no correlation with anything; numpy would
    return NaN and warn. Zero is the honest value and keeps every row finite.
    """
    spread = block.std(axis=0)
    varying = spread > 0
    size = block.shape[1]
    out = np.zeros((size, size))
    if varying.sum() < 2:
        return out
    kept = block[:, varying]
    centred = kept - kept.mean(axis=0)
    covariance = centred.T @ centred
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    index = np.flatnonzero(varying)
    out[np.ix_(index, index)] = np.clip(correlation, -1.0, 1.0)
    return out


def coupling_features(site: Site) -> dict[str, float]:
    """Within-site correlations: along the position axis, and across measurements."""
    reads = np.asarray(site.reads, dtype=np.float64)
    out = {name: 0.0 for name in COUPLING_COLUMNS}
    if reads.shape[0] < MIN_READS_FOR_CORRELATION:
        return out

    for measurement, columns in POSITION_COLUMNS.items():
        correlation = _safe_corrcoef(reads[:, list(columns)])
        values = []
        for first, second, i, j in _PAIRS:
            value = float(correlation[i, j])
            out[f"coupling_{measurement}_{first}_{second}"] = value
            values.append(value)
        # The summary that screened strongest: how much the three positions move
        # together overall, on this measurement.
        out[f"coupling_{measurement}_mean"] = float(np.mean(values))

    centre = _safe_corrcoef(
        reads[:, [CENTRE_COLUMNS["dwell"], CENTRE_COLUMNS["sd"], CENTRE_COLUMNS["mean"]]]
    )
    out["coupling_chan_dwell_sd"] = float(centre[0, 1])
    out["coupling_chan_sd_mean"] = float(centre[1, 2])
    out["coupling_chan_dwell_mean"] = float(centre[0, 2])
    return out


@register("features", "quantiles_coupling_v1")
class QuantileCouplingFeatures(QuantileFeatures):
    """`quantiles_v1` plus the within-site correlation structure.

    Subclassed rather than copied so the two cannot drift, and so
    `--compare-features quantiles_v1` stays a test of the added columns alone.
    """

    def site_features(self, site: Site) -> dict[str, float]:
        out = super().site_features(site)
        out.update(coupling_features(site))
        return out
