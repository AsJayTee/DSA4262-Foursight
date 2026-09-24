"""Guardrails for the cross-site feature sets and the `finalise` hook (0025).

The hook lets a feature set see every site at once. That makes a useful family
writable and it makes one catastrophic mistake possible, so the tests are mostly
about the mistake.

**A neighbour's label must never reach a feature.** m6A clusters along
transcripts at 6.54x, so a neighbour-label feature would be enormously
predictive out of fold and completely unusable in `predict.py`, where every site
of a held-out gene is held out together and no label exists. The design prevents
it by not handing `finalise` the labels; the first test asserts that is actually
true of what gets passed, rather than true of the intention.

**Aggregates must be leave-one-out**, or a window mean re-reads the centre
site's own features under a new name.

**A feature set that does not override `finalise` must be untouched**, because
every existing feature set relies on that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry  # noqa: E402
from m6a.data import Site  # noqa: E402
from m6a.features.base import FeatureExtractor  # noqa: E402

FORBIDDEN = {"label", "gene_id", "fold", "y"}


def a_site(transcript: str, position: int, value: float) -> Site:
    reads = np.full((24, 9), value, dtype=np.float32)
    reads += np.linspace(0, 1, 24, dtype=np.float32)[:, None]
    return Site(transcript, position, "AAGACCA", reads)


def a_transcript(transcript: str, positions, values) -> list[Site]:
    return [a_site(transcript, p, v) for p, v in zip(positions, values)]


def test_finalise_never_receives_a_label_or_a_gene():
    """The guard that makes cross-site features safe at all (0025 section 1)."""
    seen: dict[str, list[str]] = {}

    class Spy(FeatureExtractor):
        def site_features(self, site):
            return {"x": float(site.reads[:, 0].mean())}

        def finalise(self, frame, sites):
            seen["frame"] = list(frame.columns)
            seen["sites"] = list(sites.columns)
            seen["index"] = list(frame.index.names)
            return frame

    Spy().transform(iter(a_transcript("ENST1", [10, 50, 90], [1.0, 2.0, 3.0])))

    assert not FORBIDDEN & set(seen["frame"]), seen["frame"]
    assert not FORBIDDEN & set(seen["sites"]), seen["sites"]
    # It does get identity, which is what the neighbourhood is built from.
    assert seen["index"] == ["transcript_id", "transcript_position"]


def test_a_feature_set_that_does_not_override_finalise_is_unchanged():
    """The default must be a true no-op, byte for byte."""
    sites = a_transcript("ENST1", [10, 50, 90, 400], [1.0, 2.0, 3.0, 4.0])
    plain = registry.get("features", "quantiles_v1")()

    frame = plain.transform(iter(sites))
    rebuilt = pd.DataFrame(
        [plain.site_features(s) for s in sites], dtype=np.float32
    )
    assert np.array_equal(frame.to_numpy(), rebuilt.to_numpy())


def test_neighbour_aggregates_exclude_the_site_itself():
    """Leave-one-out, checked against the arithmetic by hand.

    Three sites 40 nt apart on one transcript. With a +/-50 nt window the middle
    site sees both others and each outer site sees only the middle one, so every
    mean is computable without reference to the implementation.
    """
    extractor = registry.get("features", "quantiles_nbr_signal_v1")()
    # Deliberately NOT evenly spaced in value: with 10/20/30 the middle site's
    # leave-one-out mean equals its own value by coincidence, and a broken
    # implementation that included the centre would pass.
    sites = a_transcript("ENST1", [100, 140, 180], [10.0, 20.0, 80.0])
    frame = extractor.transform(iter(sites))

    own = frame["mean_0_q95"].to_numpy()
    got = frame["nbrsig_mean_0_q95_w50_mean"].to_numpy()

    assert list(frame["nbrsig_n_w50"]) == [1.0, 2.0, 1.0]
    # first site: only the middle one. middle: the mean of the two outer ones.
    assert got[0] == pytest_approx(own[1])
    assert got[1] == pytest_approx((own[0] + own[2]) / 2)
    assert got[2] == pytest_approx(own[1])
    # and never its own value
    assert got[1] != pytest_approx(own[1])


def test_a_site_with_no_neighbour_reports_zero_and_says_so():
    """No sentinel a reader has to decode: the count column carries the meaning."""
    extractor = registry.get("features", "quantiles_nbr_signal_v1")()
    frame = extractor.transform(iter(a_transcript("ENST1", [100], [10.0])))

    assert frame["nbrsig_n_w50"].iloc[0] == 0.0
    assert frame["nbrsig_mean_0_q95_w50_mean"].iloc[0] == 0.0
    assert np.isfinite(frame.to_numpy()).all()


def test_the_neighbourhood_does_not_reach_across_transcripts():
    """Two transcripts whose positions overlap numerically must not see each other.

    Positions are per transcript, so 100 on one transcript and 100 on another are
    unrelated. Sorting on position without grouping first would silently join them.
    """
    extractor = registry.get("features", "quantiles_nbr_signal_v1")()
    sites = (
        a_transcript("ENST1", [100], [10.0])
        + a_transcript("ENST2", [100, 120], [50.0, 60.0])
    )
    frame = extractor.transform(iter(sites))
    lonely = frame.xs("ENST1", level="transcript_id")

    assert lonely["nbrsig_n_w50"].iloc[0] == 0.0
    assert frame.xs("ENST2", level="transcript_id")["nbrsig_n_w50"].tolist() == [1.0, 1.0]


def test_structural_distances_are_per_transcript_and_finite_when_alone():
    extractor = registry.get("features", "quantiles_nbr_struct_v1")()
    sites = a_transcript("ENST1", [100, 130], [1.0, 2.0]) + a_transcript(
        "ENST2", [700], [3.0]
    )
    frame = extractor.transform(iter(sites))

    pair = frame.xs("ENST1", level="transcript_id")
    assert pair["nbr_dist_nearest"].tolist() == [30.0, 30.0]
    alone = frame.xs("ENST2", level="transcript_id")
    assert np.isfinite(alone.to_numpy()).all()
    assert alone["nbr_n_on_transcript"].iloc[0] == 1.0


def test_transcript_aggregates_are_leave_one_out():
    extractor = registry.get("features", "quantiles_transcript_v1")()
    # Asymmetric for the same reason as the window test above.
    sites = a_transcript("ENST1", [10, 50, 90], [10.0, 20.0, 80.0])
    frame = extractor.transform(iter(sites))

    own = frame["mean_0_q95"].to_numpy()
    got = frame["tx_mean_0_q95_loo_mean"].to_numpy()
    assert got[0] == pytest_approx((own[1] + own[2]) / 2)
    assert got[1] == pytest_approx((own[0] + own[2]) / 2)
    assert list(frame["tx_n_sites"]) == [3.0, 3.0, 3.0]


def test_row_order_does_not_change_the_neighbourhood():
    """The frame arrives in file order, which is not promised to be sorted."""
    extractor = registry.get("features", "quantiles_nbr_signal_v1")()
    forward = a_transcript("ENST1", [10, 50, 90], [1.0, 2.0, 3.0])
    shuffled = [forward[2], forward[0], forward[1]]

    a = extractor.transform(iter(forward))
    b = extractor.transform(iter(shuffled))
    assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy())


def pytest_approx(value, tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - float(value)) <= tol * max(1.0, abs(float(value)))

        def __ne__(self, other):
            return not self.__eq__(other)

        def __repr__(self):
            return f"~{value}"

    return _Approx()
