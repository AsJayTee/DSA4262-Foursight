"""Guardrails for `quantiles_all_moments_v1` (src/m6a/features/moments.py).

A correlation of the wrong column pair still populates and the run still
finishes; it just measures nothing and reads as "second moments do not help".
So the arithmetic is pinned against numpy, and the no-overlap-with-coupling
claim - which is what makes the paired comparison attributable - is checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402
from m6a.data import Site  # noqa: E402
from m6a.features.coupling import COUPLING_COLUMNS  # noqa: E402
from m6a.features.moments import LOG_COLUMNS, MOMENT_COLUMNS, PAIRS  # noqa: E402


def a_site(reads: np.ndarray) -> Site:
    return Site("ENST00000000233", 244, "AAGACCA", np.asarray(reads, dtype=np.float32))


def test_the_moment_columns_are_the_correlations_they_claim_to_be():
    rng = np.random.default_rng(4262)
    reads = np.exp(rng.normal(size=(300, 9)))  # positive, like dwell and sd
    reads[:, 5] += 0.5 * np.log(reads[:, 4])   # a real sd_0 / mean_0 coupling
    row = registry.get("features", "quantiles_all_moments_v1")().site_features(a_site(reads))

    transformed = reads.astype(np.float32).astype(np.float64)
    transformed[:, LOG_COLUMNS] = np.log(transformed[:, LOG_COLUMNS])
    expected = np.corrcoef(transformed.T)
    for name, (i, j) in zip(MOMENT_COLUMNS, PAIRS):
        assert row[name] == pytest.approx(expected[i, j], abs=1e-6), name


def test_it_adds_exactly_the_24_pairs_coupling_does_not_carry():
    assert len(PAIRS) == 36 - 12
    assert len(set(PAIRS)) == len(PAIRS)
    extractor = registry.get("features", "quantiles_all_moments_v1")()
    base = registry.get("features", "quantiles_all_v1")()
    reads = np.random.default_rng(1).lognormal(size=(40, 9))
    added = set(extractor.site_features(a_site(reads))) - set(base.site_features(a_site(reads)))
    assert added == set(MOMENT_COLUMNS)
    assert not added & set(COUPLING_COLUMNS)


def test_below_the_read_floor_every_column_is_zero_and_finite():
    extractor = registry.get("features", "quantiles_all_moments_v1")()
    for depth in (1, 2, 4):
        row = extractor.site_features(a_site(np.random.default_rng(depth).lognormal(size=(depth, 9))))
        assert all(row[name] == 0.0 for name in MOMENT_COLUMNS)
    row = extractor.site_features(a_site(np.ones((30, 9))))  # nothing varies
    assert all(np.isfinite(row[name]) for name in MOMENT_COLUMNS)
