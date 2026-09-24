"""The 3 x 3 measurement grid inside each read, which every feature set so far
has flattened away.

The nine numbers per read are not nine arbitrary features. They are three
measurements (dwell time, signal standard deviation, mean current) at three
**ordered, adjacent** positions (-1, 0, +1):

|  | dwell | sd | mean |
|---|---|---|---|
| **-1** | `dwell_m1` | `sd_m1` | `mean_m1` |
| **0**  | `dwell_0`  | `sd_0`  | `mean_0`  |
| **+1** | `dwell_p1` | `sd_p1` | `mean_p1` |

A nanopore holds a 5-mer at any instant and the RNA ratchets through one base
at a time, so -1, 0 and +1 are three *consecutive pore states of the same
molecule*, in time order - which is why the 7-mer is the union of three
overlapping 5-mers ([docs/data.md](../../../docs/data.md)). The position axis is
therefore genuinely ordered; the measurement-type axis is not, and is treated as
channels.

`pooled_v1` and `quantiles_v1` compute statistics per column across reads, so
the nine columns are interchangeable labels to them and the grid is invisible.
`attention_mil`'s read encoder is `Linear(9 -> 64)`, which is equally
structure-blind.

## Why the pairing is the point

A modification perturbs the pore, and because the window slides, the
perturbation is smeared across -1, 0 and +1 rather than sitting only on 0. The
informative quantity is the **contrast** between the centre and its flanks.

A gradient-boosted tree cannot recover that from what it currently has, and not
merely because it is axis-aligned: **the quantile of a difference is not the
difference of the quantiles.** Measured on 40,000 sites, univariate
abs(AUC - 0.5) against the label:

| quantity | abs(AUC - 0.5) |
|---|---:|
| `mean` centre-surround as a **difference of quantiles** (derivable today) | 0.0008 |
| `mean` abs(gradient) as a **quantile of per-read differences** (this module) | **0.1198** |
| `sd` centre-surround, 5th percentile of per-read values (this module) | 0.1128 |

The first is useless and the second is the second-strongest single feature in
the whole table, behind only `mean_m1_q50` at 0.1320. Same nominal contrast; the
per-read pairing is doing all of the work.

## How this differs from `quantiles_joint_v1`, which came back null

Both are "joint within-read" feature sets, and [that one found nothing]
(+0.0013, corrected p = 0.7140), so the family deserves suspicion. The
difference is that `joint` asks an **undirected, symmetric** question - is this
read an outlier across all nine measurements at once - while this asks a
**directed, structured** one along a physically ordered axis. The univariate
screen above says they are not the same question.

## Behaviour at low depth

Unlike `quantiles_joint_v1`, which necessarily collapses to zero at one read
(a single read is its own centroid), **these features are defined at depth 1**:
one molecule still has three positions. That makes this the only joint feature
family in the repo that can help in the SG-NEx regime.

This is `quantiles_v1` plus these columns and nothing else, so

    python scripts/evaluate.py --config configs/quantiles_grid.yaml \
           --compare-features quantiles_v1

varies exactly one thing.
"""

from __future__ import annotations

import numpy as np

from m6a.data import Site
from m6a.features.quantiles import QuantileFeatures
from m6a.registry import register

# Column indices of each measurement at positions -1, 0, +1. Reading these off
# READ_FEATURE_NAMES rather than hardcoding 0..8 in the arithmetic keeps the
# grid's meaning visible at the one place it matters.
POSITION_COLUMNS: dict[str, tuple[int, int, int]] = {
    "dwell": (0, 3, 6),
    "sd": (1, 4, 7),
    "mean": (2, 5, 8),
}

# Quantiles taken across reads of each per-read contrast. The tails carry the
# "a minority of molecules is modified" signal that a mean washes out, which is
# the same reasoning quantiles.py is built on.
CONTRAST_QUANTILES = (0.05, 0.50, 0.95)

_SUFFIXES = (
    "cs_q05", "cs_q50", "cs_q95",      # centre-surround, kernel [-0.5, 1, -0.5]
    "grad_q05", "grad_q95",            # signed gradient, kernel [-1, 0, 1]
    "absgrad_q50", "absgrad_q95",      # unsigned gradient - magnitude, not direction
)

#: Every column this feature set adds to `quantiles_v1`, in emission order.
GRID_COLUMNS = tuple(
    f"grid_{measurement}_{suffix}"
    for measurement in POSITION_COLUMNS
    for suffix in _SUFFIXES
)


def grid_features(site: Site) -> dict[str, float]:
    """Per-read contrasts along the ordered position axis, summarised across reads.

    Defined for any read count including one, because the contrast is computed
    inside a read rather than between reads.
    """
    reads = np.asarray(site.reads, dtype=np.float64)
    out: dict[str, float] = {}

    for measurement, (minus, centre, plus) in POSITION_COLUMNS.items():
        before, here, after = reads[:, minus], reads[:, centre], reads[:, plus]

        # The two length-3 convolutions worth having. Anything wider than this
        # does not exist - the axis is three long.
        centre_surround = here - 0.5 * (before + after)
        gradient = after - before

        cs_q05, cs_q50, cs_q95 = np.quantile(centre_surround, CONTRAST_QUANTILES)
        grad_q05, _, grad_q95 = np.quantile(gradient, CONTRAST_QUANTILES)
        absgrad_q50, absgrad_q95 = np.quantile(np.abs(gradient), (0.50, 0.95))

        prefix = f"grid_{measurement}_"
        out[prefix + "cs_q05"] = float(cs_q05)
        out[prefix + "cs_q50"] = float(cs_q50)
        out[prefix + "cs_q95"] = float(cs_q95)
        out[prefix + "grad_q05"] = float(grad_q05)
        out[prefix + "grad_q95"] = float(grad_q95)
        out[prefix + "absgrad_q50"] = float(absgrad_q50)
        out[prefix + "absgrad_q95"] = float(absgrad_q95)

    return out


@register("features", "quantiles_grid_v1")
class QuantileGridFeatures(QuantileFeatures):
    """`quantiles_v1` plus the within-read position-axis contrasts.

    Subclassed rather than copied so the two cannot drift: whatever
    `quantiles_v1` computes, this computes too, and the paired comparison stays
    a test of the added columns alone.
    """

    def site_features(self, site: Site) -> dict[str, float]:
        out = super().site_features(site)
        out.update(grid_features(site))
        return out
