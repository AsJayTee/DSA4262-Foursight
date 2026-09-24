"""Guardrails for `quantiles_joint_v1` (src/m6a/features/joint.py).

Three guarantees, each protecting a failure that would otherwise be silent.

**It measures something marginal features cannot.** That is the entire premise -
if the joint columns are a re-reading of the quantiles already in the row, the
experiment answers nothing and a null result would be blamed on the hypothesis
rather than on the implementation. The test builds two sites with *identical*
per-measurement distributions and different joint structure, and pins that
`quantiles_v1` cannot tell them apart while these columns can.

**It is finite at depth 1.** The depth sweep calls every extractor at one read
per site. A NaN there does not raise: LightGBM accepts missing values, so the
run completes and the depth curve is quietly wrong.

**The superset property.** `--compare-features quantiles_v1` claims exactly one
thing varies. If this stops being `quantiles_v1` plus its own columns, the
comparison measures something else and still prints a p-value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402
from m6a.data import Site, subsample_reads  # noqa: E402
from m6a.features.joint import JOINT_COLUMNS  # noqa: E402

# What scripts/train.py sweeps, plus full depth. Depth 1 is the one that matters.
SWEEP_DEPTHS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]


def a_site(reads: np.ndarray) -> Site:
    return Site("ENST00000000233", 244, "AAGACCA", np.asarray(reads, dtype=np.float32))


def co_occurring_and_spread(n_reads: int = 40) -> tuple[Site, Site]:
    """Two sites with the same nine marginal distributions and different coupling.

    Each measurement is the same set of values in both sites - only *which read*
    carries which value changes. In the first, every measurement's extremes sit
    on the same handful of reads, which is what a minority of modified molecules
    would look like. In the second they are spread across different reads.

    Because each column is a permutation of the other's, every mean, standard
    deviation and quantile is identical by construction. Any feature that
    separates these two sites is reading the joint structure and nothing else.
    """
    column = np.arange(n_reads, dtype=np.float32)
    co_occurring = np.column_stack([column] * 9)
    spread = np.column_stack([np.roll(column, 4 * k) for k in range(9)])
    return a_site(co_occurring), a_site(spread)


def test_the_joint_columns_separate_sites_that_marginal_features_cannot():
    plain = registry.get("features", "quantiles_v1")()
    joint = registry.get("features", "quantiles_joint_v1")()
    together, apart = co_occurring_and_spread()

    # The premise: to the existing feature set these two sites are one site.
    marginal_together, marginal_apart = plain.site_features(together), plain.site_features(apart)
    assert marginal_together == marginal_apart

    rich_together, rich_apart = joint.site_features(together), joint.site_features(apart)
    differing = [k for k in JOINT_COLUMNS if rich_together[k] != rich_apart[k]]
    assert differing, "no joint column reacted to a purely joint difference"

    # The sharpest of them, with the direction the hypothesis predicts: when the
    # extremes land on the same molecules, some molecule is extreme on many
    # measurements at once.
    assert rich_together["joint_coex_max"] > rich_apart["joint_coex_max"]
    assert rich_together["joint_coex_frac_ge4"] > rich_apart["joint_coex_frac_ge4"]

    # And the control that proves these are not the marginals in disguise: the
    # *mean* number of extreme measurements per read is fixed by the marginal
    # distributions, so it must be identical. Only its spread across reads is
    # joint information.
    assert rich_together["joint_coex_mean"] == rich_apart["joint_coex_mean"]


def test_one_read_reports_no_joint_structure_rather_than_a_nan():
    joint = registry.get("features", "quantiles_joint_v1")()
    row = joint.site_features(a_site(np.arange(9, dtype=np.float32).reshape(1, 9)))

    # A single read is its own centroid and sits in every decile of a one-read
    # distribution at once. Zero is the honest answer; NaN would survive into
    # LightGBM as a missing value and quietly bend the depth-1 row of the sweep.
    assert all(row[k] == 0.0 for k in JOINT_COLUMNS)


def test_joint_features_stay_finite_at_every_depth_the_sweep_uses():
    joint = registry.get("features", "quantiles_joint_v1")()
    rng = np.random.default_rng(4262)

    for n_reads in (20, 47, 200):
        # Correlated measurements on wildly different scales, which is what the
        # real nine look like and what makes a raw covariance ill-conditioned.
        latent = rng.normal(size=(n_reads, 3))
        reads = np.column_stack([
            latent[:, k % 3] * scale + offset
            for k, (scale, offset) in enumerate(
                [(1e-3, 1e-2), (5.0, 20.0), (3.0, 100.0)] * 3
            )
        ])
        site = a_site(reads)

        for depth in SWEEP_DEPTHS:
            row = joint.site_features(subsample_reads(site, depth))
            values = np.array([row[k] for k in JOINT_COLUMNS], dtype=np.float64)
            assert np.isfinite(values).all(), f"non-finite at depth {depth}, n={n_reads}"


def test_a_constant_measurement_does_not_make_the_distance_undefined():
    """A column that never varies carries no within-site information.

    It also makes the covariance singular, which is the obvious way a
    Mahalanobis implementation blows up on real data.
    """
    joint = registry.get("features", "quantiles_joint_v1")()
    rng = np.random.default_rng(7)
    reads = rng.normal(size=(30, 9))
    reads[:, 4] = 3.5  # sd_0 identical on every read

    row = joint.site_features(a_site(reads))
    values = np.array([row[k] for k in JOINT_COLUMNS], dtype=np.float64)
    assert np.isfinite(values).all()
    assert row["joint_maha_mean"] > 0.0


def test_the_joint_set_is_quantiles_plus_its_own_columns_and_changes_nothing_else():
    plain = registry.get("features", "quantiles_v1")()
    joint = registry.get("features", "quantiles_joint_v1")()
    rng = np.random.default_rng(11)
    site = a_site(rng.normal(size=(50, 9)) * 10.0 + 50.0)

    base, extended = plain.site_features(site), joint.site_features(site)

    assert set(extended) - set(base) == set(JOINT_COLUMNS)
    # Bit-identical, not merely close.
    assert all(extended[k] == base[k] for k in base)
