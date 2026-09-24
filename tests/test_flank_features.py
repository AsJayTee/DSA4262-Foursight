"""Guardrails for `quantiles_flank_v1` (src/m6a/features/flank.py).

Two things here are easy to break and would not announce themselves.

**An off-by-one in the flank slice.** `kmer[1:6]` is the motif, so the flanks are
`kmer[0]` and `kmer[6]`. Read `kmer[1]` or `kmer[5]` by mistake and the columns
still populate, the run still finishes, and the result reads as "the flanking
bases do not help" when what was actually encoded was the motif's own first and
last base - which the motif one-hot already carries.

**The superset property.** The whole claim of
`--compare-features quantiles_v1` is that exactly one thing varies. If this
feature set ever stops being `quantiles_v1` plus eight columns, the comparison
silently starts measuring something else and still prints a p-value.
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
from m6a.features.flank import BASES  # noqa: E402


def a_site(kmer: str = "AAGACCA", n_reads: int = 40) -> Site:
    reads = np.arange(n_reads * 9, dtype=np.float32).reshape(n_reads, 9)
    return Site("ENST00000000233", 244, kmer, reads)


def flank_of(row: dict[str, float], side: str) -> str:
    """Which base the one-hot for `side` actually names."""
    hot = [base for base in BASES if row[f"{side}_{base}"] == 1.0]
    assert len(hot) == 1, f"{side} one-hot set {len(hot)} columns, expected exactly 1"
    return hot[0]


def test_the_flank_columns_read_the_outer_bases_of_the_seven_mer():
    extractor = registry.get("features", "quantiles_flank_v1")()

    # Every base distinct, so reading the wrong index cannot coincidentally
    # produce the right answer - which is exactly how an off-by-one survives.
    row = extractor.site_features(a_site("CAGACTG"))

    assert flank_of(row, "left") == "C"
    assert flank_of(row, "right") == "G"
    # And the motif the run already had is untouched in the middle.
    assert row["motif_AGACT"] == 1.0


@pytest.mark.parametrize("kmer", ["AAGACCA", "TGGACTC", "GAAACAT", "CTAACCG"])
def test_the_flanks_track_the_kmer_rather_than_being_constant(kmer: str):
    extractor = registry.get("features", "quantiles_flank_v1")()
    row = extractor.site_features(a_site(kmer))

    assert flank_of(row, "left") == kmer[0]
    assert flank_of(row, "right") == kmer[-1]
    assert row[f"motif_{kmer[1:6]}"] == 1.0


def test_the_flank_set_is_quantiles_plus_eight_columns_and_changes_nothing_else():
    plain = registry.get("features", "quantiles_v1")()
    flank = registry.get("features", "quantiles_flank_v1")()
    site = a_site("CAGACTG")

    base, extended = plain.site_features(site), flank.site_features(site)
    added = set(extended) - set(base)

    assert added == {f"{side}_{b}" for side in ("left", "right") for b in BASES}
    # Bit-identical, not merely close: anything else and --compare-features
    # quantiles_v1 is no longer a one-variable comparison.
    assert all(extended[k] == base[k] for k in base)


def test_a_kmer_without_flanking_bases_is_refused_rather_than_encoded_as_nothing():
    extractor = registry.get("features", "quantiles_flank_v1")()

    with pytest.raises(ValueError, match="quantiles_v1"):
        extractor.site_features(a_site("AGACT"))
