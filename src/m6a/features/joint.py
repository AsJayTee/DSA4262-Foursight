"""Joint within-read structure: was the *same molecule* extreme on several
measurements at once?

Every feature set in this repo so far is **marginal**. `mean_0_q95` is the 95th
percentile of one measurement taken across reads and `sd_0_q95` is the 95th
percentile of another, computed independently; nothing in the representation can
express that one molecule was unusual on both. GAPS.md records that as the best
remaining lead, and the literature review reaches the same conclusion by another
route - the field's migration to per-read neural encoders (m6Anet, DENA, CHEUI,
RedNano) is precisely about preserving that coupling, and a learned read encoder
is the expensive way to buy it.

**So this is the cheap test of the same hypothesis.** A learned per-read encoder
is ~30 GPU-hours; this is a feature set and runs in the time any other one does.
If a hand-crafted joint feature moves nothing, the neural version is probably
chasing noise, and that is worth knowing before anyone books an instance.

Depth is a count of distinct RNA molecules, not repeated readings of one
([docs/data.md](../../../docs/data.md#read-depth)), which is what makes "the
same molecule was extreme on two measurements" a statement about one physical
RNA copy passing the pore - and therefore a statement the modification could
plausibly have caused.

Two blocks, both pure numpy inside `site_features(site)`.

## 1. Mahalanobis distance from the site's own read centroid

Standardise each measurement by *this site's* own spread, estimate the
correlation between measurements from *this site's* own reads, and ask how far
each read sits from the middle once that correlation is accounted for. A read
that is mildly high on two measurements that usually move together is
unremarkable; one that is mildly high on two that usually do not is exactly what
a marginal quantile cannot see.

The correlation matrix is shrunk toward the identity by `SHRINKAGE`, for two
reasons. With nine measurements and as few as twenty reads the sample
correlation is noisy and can be near-singular. And the shrinkage target is
meaningful rather than merely convenient: at a shrinkage of 1 the distance
collapses to the mean squared z-score, which is exactly "extreme on several
measurements, treating them as independent". What the feature adds over that
floor is the correlation structure itself.

Distances are reported as `d^2 / k` over the `k` measurements that varied at
this site, so a typical read sits near 1 whatever `k` is.

## 2. Co-extremeness counts

For each measurement, mark the reads in this site's own top and bottom decile.
Then count, per read, how many measurements it is extreme on, and report how
many reads clear two, three and four. Under independence those fractions have a
fixed expectation, and subtracting a constant cannot help a tree that is already
free to split anywhere, so the raw fraction is logged and the excess over
independence is what it carries.

`_up` is the one-sided version, because a modification shifts the pore current
in a direction rather than merely scattering it. `_central` restricts the count
to `dwell_0`, `sd_0` and `mean_0` - the position the candidate base itself sits
at.

## Behaviour at low depth

The depth sweep calls this at one read per site, so every branch here has to
produce a finite number there rather than a NaN. It does, and the number is zero
throughout: a single read *is* its own centroid, so its distance from it is
zero, and it sits in every decile of a one-read distribution at once, so no tail
test can separate it. That is the honest answer - joint structure across reads
does not exist at depth 1 - and `tests/test_joint_features.py` pins it.

This is `quantiles_v1` plus these columns and nothing else, so

    python scripts/evaluate.py --config configs/quantiles_joint.yaml \
           --compare-features quantiles_v1

varies exactly one thing.
"""

from __future__ import annotations

import numpy as np

from m6a.data import Site
from m6a.features.quantiles import QuantileFeatures
from m6a.registry import register

# How far the per-site correlation matrix is shrunk toward the identity before
# it is inverted. Nine measurements estimated from as few as twenty reads is a
# noisy correlation matrix and sometimes a singular one; at 1.0 the distance
# degrades to the mean squared z-score, which is the independent-measurements
# reading of the same question. So this interpolates between "joint" and
# "marginal" rather than between "works" and "breaks".
SHRINKAGE = 0.1

# Which tail of this site's own read distribution counts as extreme, per
# measurement. 0.10 puts roughly two of the twenty reads at the shallowest
# training site in each tail, which is the least that can support a count.
TAIL = 0.10

# dwell_0, sd_0, mean_0 - the three measurements at the position the candidate
# base itself occupies. Indices into m6a.data.READ_FEATURE_NAMES.
CENTRAL = (3, 4, 5)

_MAHALANOBIS_KEYS = (
    "joint_maha_mean", "joint_maha_std", "joint_maha_q50", "joint_maha_q75",
    "joint_maha_q90", "joint_maha_q95", "joint_maha_max",
    "joint_maha_q95_over_q50", "joint_maha_frac_gt2", "joint_maha_frac_gt3",
)
_COEXTREME_KEYS = (
    "joint_coex_mean", "joint_coex_max", "joint_coex_frac_ge2",
    "joint_coex_frac_ge3", "joint_coex_frac_ge4", "joint_coex_up_frac_ge2",
    "joint_coex_central_frac_ge2",
)

#: Every column this feature set adds to `quantiles_v1`, in emission order.
JOINT_COLUMNS = _MAHALANOBIS_KEYS + _COEXTREME_KEYS


def _mahalanobis(reads: np.ndarray) -> np.ndarray:
    """Each read's squared Mahalanobis distance from the site's centroid, per dimension.

    One value per read, normalised by the number of measurements that varied at
    this site so a typical read sits near 1 regardless of how many that was. All
    zeros when nothing varied, which is the depth-1 case.
    """
    n_reads = reads.shape[0]
    spread = reads.std(axis=0)
    varying = spread > 0
    k = int(varying.sum())
    if k == 0:
        # One read, or every measurement identical across reads. A point has no
        # distance from itself, and there is no distribution to be extreme in.
        return np.zeros(n_reads)

    kept = reads[:, varying]
    z = (kept - kept.mean(axis=0)) / spread[varying]
    # z is standardised, so this is the correlation matrix with 1s on the
    # diagonal - which is what makes a scale-free shrinkage target correct.
    correlation = (z.T @ z) / n_reads
    shrunk = (1.0 - SHRINKAGE) * correlation + SHRINKAGE * np.eye(k)

    try:
        solved = np.linalg.solve(shrunk, z.T)
    except np.linalg.LinAlgError:
        # Shrinkage should make this unreachable; if floating point manages it
        # anyway, fall back to the shrinkage-1 reading rather than failing a
        # whole run over one site.
        solved = z.T

    squared = np.einsum("ij,ji->i", z, solved)
    # A quadratic form in a positive-definite matrix cannot be negative, so this
    # clips rounding rather than a real value.
    return np.maximum(squared, 0.0) / k


def _coextreme(reads: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per read: measurements it is extreme on, extreme-high on, centrally extreme on."""
    low = np.quantile(reads, TAIL, axis=0)
    high = np.quantile(reads, 1.0 - TAIL, axis=0)
    upper = reads > high
    extreme = upper | (reads < low)
    return (
        extreme.sum(axis=1),
        upper.sum(axis=1),
        extreme[:, CENTRAL].sum(axis=1),
    )


def joint_features(site: Site) -> dict[str, float]:
    """The joint-structure columns for one site. Always the same keys, always finite."""
    # float64: the nine measurements span 1e-3 to 2e2, and a correlation matrix
    # built from float32 at that spread loses digits the inversion needs.
    reads = np.asarray(site.reads, dtype=np.float64)

    distance = _mahalanobis(reads)
    q50, q75, q90, q95 = np.quantile(distance, [0.50, 0.75, 0.90, 0.95])
    out: dict[str, float] = {
        "joint_maha_mean": float(distance.mean()),
        "joint_maha_std": float(distance.std()),
        "joint_maha_q50": float(q50),
        "joint_maha_q75": float(q75),
        "joint_maha_q90": float(q90),
        "joint_maha_q95": float(q95),
        "joint_maha_max": float(distance.max()),
        # Scale-free shape: a heavy upper tail relative to the middle is what a
        # minority of jointly-odd reads looks like. Reported as 0.0 rather than
        # as an infinity when the middle is zero, and 0.0 cannot be mistaken for
        # a real ratio, which is always >= 1.
        "joint_maha_q95_over_q50": float(q95 / q50) if q50 > 0 else 0.0,
        "joint_maha_frac_gt2": float((distance > 2.0).mean()),
        "joint_maha_frac_gt3": float((distance > 3.0).mean()),
    }

    count, up_count, central_count = _coextreme(reads)
    out.update({
        "joint_coex_mean": float(count.mean()),
        "joint_coex_max": float(count.max()),
        "joint_coex_frac_ge2": float((count >= 2).mean()),
        "joint_coex_frac_ge3": float((count >= 3).mean()),
        "joint_coex_frac_ge4": float((count >= 4).mean()),
        "joint_coex_up_frac_ge2": float((up_count >= 2).mean()),
        "joint_coex_central_frac_ge2": float((central_count >= 2).mean()),
    })
    return out


@register("features", "quantiles_joint_v1")
class QuantileJointFeatures(QuantileFeatures):
    """`quantiles_v1` plus the joint within-read columns.

    Subclassed rather than copied so the two cannot drift: whatever
    `quantiles_v1` computes, this computes too, and the paired comparison stays
    a test of the added columns alone.
    """

    def site_features(self, site: Site) -> dict[str, float]:
        out = super().site_features(site)
        out.update(joint_features(site))
        return out
