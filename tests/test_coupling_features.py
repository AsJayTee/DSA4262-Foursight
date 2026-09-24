"""Guardrails for `quantiles_coupling_v1` (src/m6a/features/coupling.py).

The feature set exists because a **correlation is the one second moment the
marginals cannot produce**. `r = E[(X-ux)(Y-uy)] / (sx sy)` needs `E[XY]`, and
no collection of quantiles of X and of Y supplies it.

That is also the exact property an implementation can lose without anything
looking wrong. Compute the correlation of the wrong pair of columns, or
accidentally normalise it away, and the columns still populate and the run still
finishes - it just measures nothing, and reads as "coupling does not help". The
sibling failure is on record: `quantiles_grid_v1` shipped, screened at 0.1198,
and turned out to be 0.86-0.97 correlated with columns the model already had.

So these tests pin: the arithmetic against hand-computed correlations, the
independence of the statistic from the marginals, and the low-depth floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402
from m6a.data import Site  # noqa: E402
from m6a.features.coupling import (  # noqa: E402
    COUPLING_COLUMNS,
    MIN_READS_FOR_CORRELATION,
    POSITION_COLUMNS,
)

SWEEP_DEPTHS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]


def a_site(reads: np.ndarray) -> Site:
    return Site("ENST00000000233", 244, "AAGACCA", np.asarray(reads, dtype=np.float32))


def test_the_coupling_columns_are_the_correlations_they_claim_to_be():
    """Checked against numpy's own corrcoef on the exact column pairs."""
    rng = np.random.default_rng(4262)
    reads = rng.normal(size=(200, 9)) * 5.0 + 50.0
    row = registry.get("features", "quantiles_coupling_v1")().site_features(a_site(reads))

    for measurement, (minus, centre, plus) in POSITION_COLUMNS.items():
        expected = np.corrcoef(reads[:, [minus, centre, plus]].T)
        assert row[f"coupling_{measurement}_m1_0"] == _approx(expected[0, 1])
        assert row[f"coupling_{measurement}_0_p1"] == _approx(expected[1, 2])
        assert row[f"coupling_{measurement}_m1_p1"] == _approx(expected[0, 2])
        assert row[f"coupling_{measurement}_mean"] == _approx(
            np.mean([expected[0, 1], expected[1, 2], expected[0, 2]])
        )


def test_coupling_separates_sites_the_marginals_cannot_tell_apart():
    """The premise, as a controlled pair.

    Both sites have the *same* per-column distributions. In one, the three
    positions move together on every read - what a modified molecule does, since
    all three pore windows contain the modified base. In the other, each column
    is independently shuffled. Every marginal is identical; only the coupling
    differs.
    """
    plain = registry.get("features", "quantiles_v1")()
    coupling = registry.get("features", "quantiles_coupling_v1")()
    rng = np.random.default_rng(7)

    n = 120
    together = np.zeros((n, 9))
    apart = np.zeros((n, 9))
    for minus, centre, plus in POSITION_COLUMNS.values():
        shared = rng.normal(size=n) * 4.0 + 100.0
        for column in (minus, centre, plus):
            values = shared + rng.normal(size=n) * 0.3
            together[:, column] = values
            # Same multiset, coupling destroyed.
            apart[:, column] = rng.permutation(values)

    coupled_row = coupling.site_features(a_site(together))
    apart_row = coupling.site_features(a_site(apart))

    # Coupled positions read near 1; independently shuffled ones read near 0.
    assert coupled_row["coupling_mean_mean"] > 0.8
    assert abs(apart_row["coupling_mean_mean"]) < 0.4
    # ... while the marginal feature set sees two sites it cannot separate on
    # the columns those permutations touched.
    marginal_coupled = plain.site_features(a_site(together))
    marginal_apart = plain.site_features(a_site(apart))
    for measurement in ("mean_0_q95", "mean_0_q50", "mean_m1_q50"):
        assert marginal_coupled[measurement] == _approx(marginal_apart[measurement])


def test_a_constant_measurement_gives_zero_not_a_nan():
    """numpy's corrcoef returns NaN for a zero-variance column and warns."""
    rng = np.random.default_rng(3)
    reads = rng.normal(size=(40, 9))
    reads[:, 5] = 7.0  # mean_0 identical on every read
    row = registry.get("features", "quantiles_coupling_v1")().site_features(a_site(reads))

    values = np.array([row[k] for k in COUPLING_COLUMNS], dtype=np.float64)
    assert np.isfinite(values).all()
    assert row["coupling_mean_0_p1"] == 0.0


def test_too_few_reads_reports_zero_rather_than_a_correlation_of_noise():
    """A correlation from two or three points is not a measurement.

    Stated as a limitation in the module docstring and pinned here: the depth
    sweep runs to one read, and a spurious +/-1.0 from three points would be
    worse than an honest zero.
    """
    coupling = registry.get("features", "quantiles_coupling_v1")()
    rng = np.random.default_rng(5)
    for n_reads in range(1, MIN_READS_FOR_CORRELATION):
        row = coupling.site_features(a_site(rng.normal(size=(n_reads, 9))))
        assert all(row[k] == 0.0 for k in COUPLING_COLUMNS), n_reads

    enough = coupling.site_features(
        a_site(rng.normal(size=(MIN_READS_FOR_CORRELATION, 9)))
    )
    assert any(enough[k] != 0.0 for k in COUPLING_COLUMNS)


def test_every_coupling_column_stays_in_range_and_finite_at_every_sweep_depth():
    coupling = registry.get("features", "quantiles_coupling_v1")()
    rng = np.random.default_rng(4262)
    from m6a.data import subsample_reads

    for n_reads in (20, 47, 200):
        reads = np.column_stack([
            rng.normal(loc, scale, n_reads)
            for loc, scale in [(0.01, 0.005), (20.0, 8.0), (100.0, 6.0)] * 3
        ])
        site = a_site(reads)
        for depth in SWEEP_DEPTHS:
            row = coupling.site_features(subsample_reads(site, depth))
            values = np.array([row[k] for k in COUPLING_COLUMNS], dtype=np.float64)
            assert np.isfinite(values).all(), f"depth {depth}, n={n_reads}"
            assert (np.abs(values) <= 1.0 + 1e-9).all(), "a correlation left [-1, 1]"


def test_the_coupling_set_is_quantiles_plus_its_own_columns_and_nothing_else():
    plain = registry.get("features", "quantiles_v1")()
    coupling = registry.get("features", "quantiles_coupling_v1")()
    rng = np.random.default_rng(11)
    site = a_site(rng.normal(size=(50, 9)) * 10.0 + 50.0)

    base, extended = plain.site_features(site), coupling.site_features(site)
    assert set(extended) - set(base) == set(COUPLING_COLUMNS)
    assert all(extended[k] == base[k] for k in base)


def _approx(value, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(float(other) - float(value)) <= tol * max(1.0, abs(float(value)))

        def __repr__(self):
            return f"~{value}"

    return _A()
