"""Guardrails for `quantiles_grid_v1` (src/m6a/features/grid.py).

The feature set exists to exploit one fact: **the quantile of a per-read
difference is not the difference of the quantiles.** Measured univariately, the
first carries 0.1198 of discriminative power and the second 0.0008. If an
implementation ever computes the second by accident - by quantiling each column
first and subtracting - the columns still populate, the run still finishes, and
the result reads as "the grid carries nothing". That is the failure these tests
exist to catch, and it is silent.

The other guarantee worth pinning is that these features are **live at depth 1**,
which is what distinguishes this family from `quantiles_joint_v1` (which
necessarily collapses to zero with one read). The depth sweep scores at one read
per site, and a feature set that silently zeroes there cannot help in the
regime the project most needs help in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402
from m6a.data import Site, subsample_reads  # noqa: E402
from m6a.features.grid import GRID_COLUMNS, POSITION_COLUMNS  # noqa: E402

SWEEP_DEPTHS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]


def a_site(reads: np.ndarray) -> Site:
    return Site("ENST00000000233", 244, "AAGACCA", np.asarray(reads, dtype=np.float32))


def test_the_grid_reads_the_quantile_of_differences_not_the_difference_of_quantiles():
    """Two sites whose per-column distributions are identical and whose per-read
    contrasts are not.

    Column by column these sites are the same multiset, so every marginal
    statistic - and therefore every difference between marginal statistics - is
    identical. Only the per-read pairing separates them.
    """
    grid = registry.get("features", "quantiles_grid_v1")()
    plain = registry.get("features", "quantiles_v1")()

    n = 40
    ramp = np.arange(n, dtype=np.float64)
    reads_aligned = np.zeros((n, 9))
    reads_shuffled = np.zeros((n, 9))
    for minus, centre, plus in POSITION_COLUMNS.values():
        # Aligned: -1 and +1 move together with 0, so the contrast is ~flat.
        reads_aligned[:, minus] = ramp
        reads_aligned[:, centre] = ramp
        reads_aligned[:, plus] = ramp
        # Shuffled: identical column multisets, but the centre is paired with
        # different flank values per read, so the contrast is large.
        reads_shuffled[:, minus] = ramp
        reads_shuffled[:, centre] = np.roll(ramp, 13)
        reads_shuffled[:, plus] = np.roll(ramp, 26)

    aligned, shuffled = a_site(reads_aligned), a_site(reads_shuffled)

    # The premise: the existing feature set cannot tell these apart.
    assert plain.site_features(aligned) == plain.site_features(shuffled)

    a, s = grid.site_features(aligned), grid.site_features(shuffled)
    differing = [k for k in GRID_COLUMNS if a[k] != s[k]]
    assert differing, "no grid column reacted to a purely per-read-pairing difference"

    # The aligned site has essentially no centre-surround signal; the shuffled
    # one has a lot. Direction, not just difference.
    assert abs(s["grid_mean_cs_q95"]) > abs(a["grid_mean_cs_q95"])
    assert s["grid_mean_absgrad_q50"] > a["grid_mean_absgrad_q50"]


def test_the_contrasts_are_the_kernels_they_claim_to_be():
    """Checked against the arithmetic by hand on a site with one read.

    With a single read every quantile is that read's own value, so the columns
    are the raw kernels and an index mix-up between positions cannot hide.
    """
    grid = registry.get("features", "quantiles_grid_v1")()
    reads = np.zeros((1, 9))
    # dwell at (-1, 0, +1) = (1, 5, 2);  sd = (10, 20, 40);  mean = (100, 90, 130)
    reads[0, [0, 3, 6]] = [1.0, 5.0, 2.0]
    reads[0, [1, 4, 7]] = [10.0, 20.0, 40.0]
    reads[0, [2, 5, 8]] = [100.0, 90.0, 130.0]
    row = grid.site_features(a_site(reads))

    for name, (before, here, after) in {
        "dwell": (1.0, 5.0, 2.0),
        "sd": (10.0, 20.0, 40.0),
        "mean": (100.0, 90.0, 130.0),
    }.items():
        assert row[f"grid_{name}_cs_q50"] == here - 0.5 * (before + after)
        assert row[f"grid_{name}_grad_q95"] == after - before
        assert row[f"grid_{name}_absgrad_q50"] == abs(after - before)


def test_the_grid_is_still_live_at_one_read():
    """The property that separates this family from quantiles_joint_v1.

    A single molecule still has three pore positions, so the contrast exists.
    `joint` has to return zeros here; this must not.
    """
    grid = registry.get("features", "quantiles_grid_v1")()
    reads = np.zeros((1, 9))
    reads[0, [2, 5, 8]] = [100.0, 130.0, 95.0]  # a real centre-surround on mean
    row = grid.site_features(a_site(reads))

    assert row["grid_mean_cs_q50"] != 0.0
    assert row["grid_mean_absgrad_q50"] != 0.0


def test_grid_features_stay_finite_at_every_depth_the_sweep_uses():
    grid = registry.get("features", "quantiles_grid_v1")()
    rng = np.random.default_rng(4262)
    for n_reads in (20, 47, 200):
        reads = np.column_stack([
            rng.normal(loc, scale, n_reads)
            for loc, scale in [(0.01, 0.005), (20.0, 8.0), (100.0, 6.0)] * 3
        ])
        site = a_site(reads)
        for depth in SWEEP_DEPTHS:
            row = grid.site_features(subsample_reads(site, depth))
            values = np.array([row[k] for k in GRID_COLUMNS], dtype=np.float64)
            assert np.isfinite(values).all(), f"non-finite at depth {depth}, n={n_reads}"


def test_the_grid_set_is_quantiles_plus_its_own_columns_and_changes_nothing_else():
    plain = registry.get("features", "quantiles_v1")()
    grid = registry.get("features", "quantiles_grid_v1")()
    rng = np.random.default_rng(11)
    site = a_site(rng.normal(size=(50, 9)) * 10.0 + 50.0)

    base, extended = plain.site_features(site), grid.site_features(site)
    assert set(extended) - set(base) == set(GRID_COLUMNS)
    assert all(extended[k] == base[k] for k in base)
