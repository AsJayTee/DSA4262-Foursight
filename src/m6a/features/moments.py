"""Every within-site correlation `coupling.py` does not already carry.

## Why this is a feature set and not a network

The proposed next architecture was attention-MIL with the 36 pairwise products
of the nine measurements appended to each read, so that mean pooling recovers
every second moment. A network that does that computes, at best, the site's
correlation matrix and hands it to an MLP head - and a plain MLP on site
features tied LightGBM (-0.0021, 21/50, GAPS.md). So the premise can be tested
on the tree for the price of a feature set: **if the tree cannot use the full
correlation matrix, a network built to rediscover it has nothing new to find.**

`coupling.py` already emits 12 of the 36 pairs - every same-measurement pair
across positions, plus the three centre cross-channel pairs. This adds the other
24: pairs of *different* measurements at *different* positions, and the
cross-channel pairs at -1 and +1.

## What the screen said (40,000 sites, analysis/evaluation/scratch/second_moment_screen.py)

| column | abs(AUC - 0.5) | largest abs(r) vs `quantiles_all_v1` |
|---|---:|---:|
| `moments_sd_0__mean_p1` | 0.1393 | 0.30 |
| `moments_sd_m1__mean_0` | 0.1300 | 0.37 |
| `moments_sd_m1__mean_p1` | 0.1110 | 0.23 |
| `moments_dwell_m1__sd_m1` | 0.0848 | 0.19 |

Strong and not redundant - the profile `coupling` had before it won (0.2183,
0.23), and not the profile `grid` had before it did not (0.1198, 0.86-0.97). The
screen also checked itself: every pair `coupling.py` covers came back at
abs(r) = 0.90-1.00 against its existing column. A univariate screen predicts
marginal signal, not incremental value, so this is a reason to run it and not a
prior on the result.

## Log scale for dwell and sd

Both are positive and heavy-tailed (sd runs 0.04-206 in a single column), so a
Pearson r on the raw scale is set by a handful of reads. The screen measured on
log dwell and log sd, and this computes what the screen measured. Mean current
is roughly symmetric and stays raw.

Below `MIN_READS_FOR_CORRELATION` reads every column is 0.0, as in coupling.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from m6a.data import Site
from m6a.features.coupling import MIN_READS_FOR_CORRELATION, _safe_corrcoef
from m6a.features.neighbourhood import EverythingFeatures
from m6a.registry import register

READ_NAMES = ("dwell_m1", "sd_m1", "mean_m1", "dwell_0", "sd_0", "mean_0",
              "dwell_p1", "sd_p1", "mean_p1")
LOG_COLUMNS = [0, 1, 3, 4, 6, 7]

# The 12 pairs coupling.py emits, by read-column index. Excluded here so the
# new columns are new, and so a paired comparison against quantiles_all_v1
# measures them alone.
_IN_COUPLING = {
    (0, 3), (3, 6), (0, 6),   # dwell across positions
    (1, 4), (4, 7), (1, 7),   # sd across positions
    (2, 5), (5, 8), (2, 8),   # mean across positions
    (3, 4), (4, 5), (3, 5),   # the three channels at the centre
}
PAIRS = tuple(p for p in combinations(range(9), 2) if p not in _IN_COUPLING)
MOMENT_COLUMNS = tuple(f"moments_{READ_NAMES[i]}__{READ_NAMES[j]}" for i, j in PAIRS)


def moment_features(site: Site) -> dict[str, float]:
    reads = np.asarray(site.reads, dtype=np.float64)
    if reads.shape[0] < MIN_READS_FOR_CORRELATION:
        return {name: 0.0 for name in MOMENT_COLUMNS}
    reads[:, LOG_COLUMNS] = np.log(np.maximum(reads[:, LOG_COLUMNS], 1e-6))
    correlation = _safe_corrcoef(reads)
    return {name: float(correlation[i, j]) for name, (i, j) in zip(MOMENT_COLUMNS, PAIRS)}


@register("features", "quantiles_all_moments_v1")
class EverythingPlusMomentsFeatures(EverythingFeatures):
    """`quantiles_all_v1` plus the 24 correlations coupling does not carry.

    Subclassed so `--compare-with configs/everything.yaml` varies these columns
    and nothing else. `finalise` is inherited unchanged, so the cross-site
    columns are exactly the tested ones.
    """

    def site_features(self, site: Site) -> dict[str, float]:
        out = super().site_features(site)
        out.update(moment_features(site))
        return out
