"""What the *neighbouring* candidate sites on the same transcript look like.

m6A clusters along transcripts, and it is not a small effect. Measured on the
training set over 116,505 adjacent same-transcript pairs
([0025](../../../docs/decisions/0025-a-feature-set-may-see-every-site.md)):

| | P(neighbour positive) | lift vs the 4.49% base |
|---|---:|---:|
| this site negative | 0.0337 | 0.75x |
| this site **positive** | **0.2937** | **6.54x** |

which splits into a **transcript-level** effect (3.60x - a random other site on
the same transcript, and positives-per-transcript carries 4.68x the binomial
variance) and **local adjacency** on top of it (1.82x, decaying from 10.9x at
11-25 nt to 1.66x past 250 nt). It survives conditioning on the neighbour's
motif in all eighteen.

## The thing that cannot be done, and why this module is shaped around avoiding it

**That 6.54x is a label-label correlation and it is unusable directly.** The
split groups by gene, so at prediction time every site of a held-out gene is
held out together and *no neighbour label is ever available*. A feature set
reaching for one would score beautifully out of fold and be worthless in
`predict.py`.

What is usable: if a neighbour is genuinely modified, **its reads carry that
signature**, so its measurements are a noisy observation of its latent state and
aggregating them is evidence about the region. That is what `finalise` is handed
- features and coordinates, never labels (0025 section 1).

## Three feature sets, because they answer three different questions

Splitting them costs three runs and buys attribution - it says *which* graph is
worth building, which is the whole point of doing this before a GNN:

| name | adds | the question |
|---|---|---|
| `quantiles_nbr_struct_v1` | where the neighbours are | is candidate *density* alone informative? |
| `quantiles_nbr_signal_v1` | what the neighbours *measure*, +/-50 and +/-200 nt | the real message-passing proxy |
| `quantiles_transcript_v1` | the whole transcript, leave-one-out | is it regional at all, or just "this transcript is methylated"? |

If `signal` wins and `struct` does not, the payload is in neighbour
measurements and a chain model over sites is justified. If only `transcript`
wins, a transcript-level random effect is all there is and a per-site graph is
overkill. If none wins, the GNN premise does not survive contact with a model,
however strong the label correlation looks.

**A transcript is one-dimensional**, so these edges form a *path*, not a general
graph - which is why the architecture this licenses is a dilated 1-D CNN or a
Bi-GRU over the ordered site chain rather than message passing.

## Two implementation rules, both load-bearing

**Every aggregate is leave-one-out.** A window mean that includes the centre
site re-reads that site's own features under a new name.

**No neighbours means the aggregate is 0.0 and the count column says so.** The
count is always emitted beside the aggregate, so a tree can branch on "I had no
neighbours" rather than being handed a sentinel it has to guess the meaning of.
Nothing here is ever NaN or infinite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m6a.features.flank import QuantileFlankFeatures
from m6a.features.quantiles import QuantileFeatures
from m6a.registry import register

# Windows in nucleotides, each side. The measured decay puts almost all of the
# local effect inside 100 nt; 200 is carried as the "regional" scale to sit
# between local adjacency and the whole transcript.
WINDOWS = (50, 200)

# Distance windows for the structural counts.
COUNT_WINDOWS = (20, 50, 100)

# Reported when a site has no neighbour on its transcript at all. Finite and far
# outside the real range (the largest observed gap is a few thousand nt), so a
# tree can isolate it with one split.
NO_NEIGHBOUR = 100_000.0

# Which of the 101 quantiles_v1 columns get aggregated over neighbours. Chosen
# from a univariate screen on 40,000 sites rather than by taste: mean_m1_q50 is
# the strongest single feature in the whole table (|AUC-0.5| = 0.1320) and the
# three centre-position q95s are the tail statistics that carry the "a minority
# of molecules is modified" signal quantiles.py exists for. Aggregating all 101
# would add 400+ columns to test one idea.
AGGREGATED = ("mean_m1_q50", "mean_0_q95", "sd_0_q95", "dwell_0_q95")


def _ordered(frame: pd.DataFrame):
    """Row order sorted by (transcript, position), plus the sorted keys.

    The frame arrives in file order, which happens to be sorted on this dataset
    and is not promised to be on any other, so this sorts rather than assuming.
    """
    transcripts = frame.index.get_level_values(0).to_numpy()
    positions = frame.index.get_level_values(1).to_numpy().astype(np.int64)
    order = np.lexsort((positions, transcripts))
    return order, transcripts[order], positions[order]


def _transcript_slices(sorted_transcripts: np.ndarray):
    """(start, stop) for each run of one transcript in the sorted order."""
    if sorted_transcripts.size == 0:
        return []
    boundaries = np.flatnonzero(sorted_transcripts[1:] != sorted_transcripts[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [sorted_transcripts.size]])
    return list(zip(starts.tolist(), stops.tolist()))


def _blank(frame: pd.DataFrame, columns) -> dict[str, np.ndarray]:
    return {name: np.zeros(len(frame), dtype=np.float64) for name in columns}


def _structural(frame: pd.DataFrame) -> pd.DataFrame:
    order, transcripts, positions = _ordered(frame)
    names = (
        ["nbr_dist_prev", "nbr_dist_next", "nbr_dist_nearest"]
        + [f"nbr_count_{w}" for w in COUNT_WINDOWS]
        + ["nbr_n_on_transcript", "nbr_rel_position", "nbr_rank_on_transcript"]
    )
    out = _blank(frame, names)

    for start, stop in _transcript_slices(transcripts):
        pos = positions[start:stop]
        n = pos.size
        rows = order[start:stop]

        previous = np.full(n, NO_NEIGHBOUR)
        following = np.full(n, NO_NEIGHBOUR)
        if n > 1:
            previous[1:] = (pos[1:] - pos[:-1]).astype(np.float64)
            following[:-1] = (pos[1:] - pos[:-1]).astype(np.float64)
        out["nbr_dist_prev"][rows] = previous
        out["nbr_dist_next"][rows] = following
        out["nbr_dist_nearest"][rows] = np.minimum(previous, following)

        for window in COUNT_WINDOWS:
            lo = np.searchsorted(pos, pos - window, side="left")
            hi = np.searchsorted(pos, pos + window, side="right")
            out[f"nbr_count_{window}"][rows] = (hi - lo - 1).astype(np.float64)

        out["nbr_n_on_transcript"][rows] = float(n)
        span = float(pos[-1] - pos[0]) if n > 1 else 0.0
        # Relative position is bounded by the first and last CANDIDATE, not by
        # the transcript's true length, which nothing here knows. A proxy, and
        # a coarse one on transcripts with few candidates.
        out["nbr_rel_position"][rows] = (
            (pos - pos[0]) / span if span > 0 else np.zeros(n)
        )
        out["nbr_rank_on_transcript"][rows] = (
            np.arange(n) / (n - 1) if n > 1 else np.zeros(n)
        )

    return pd.DataFrame(out, index=frame.index)


def _neighbour_signal(frame: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in AGGREGATED if c in frame.columns]
    order, transcripts, positions = _ordered(frame)
    names = [f"nbrsig_n_w{w}" for w in WINDOWS] + [
        f"nbrsig_{c}_w{w}_{stat}"
        for c in available for w in WINDOWS for stat in ("mean", "max")
    ]
    out = _blank(frame, names)
    values = {c: frame[c].to_numpy(dtype=np.float64)[order] for c in available}

    for start, stop in _transcript_slices(transcripts):
        pos = positions[start:stop]
        n = pos.size
        rows = order[start:stop]
        if n < 2:
            continue  # no neighbours; every aggregate stays 0.0 and the count says so

        block = {c: v[start:stop] for c, v in values.items()}
        for window in WINDOWS:
            lo = np.searchsorted(pos, pos - window, side="left")
            hi = np.searchsorted(pos, pos + window, side="right")
            counts = (hi - lo - 1).astype(np.float64)
            out[f"nbrsig_n_w{window}"][rows] = counts

            for column, series in block.items():
                prefix = np.concatenate([[0.0], np.cumsum(series)])
                # Leave-one-out: the window sum minus this site's own value.
                totals = prefix[hi] - prefix[lo] - series
                means = np.divide(
                    totals, counts, out=np.zeros(n), where=counts > 0
                )
                out[f"nbrsig_{column}_w{window}_mean"][rows] = means

                highest = np.zeros(n)
                for i in range(n):
                    left, right = lo[i], hi[i]
                    if right - left <= 1:
                        continue
                    neighbourhood = np.concatenate(
                        [series[left:i], series[i + 1:right]]
                    )
                    highest[i] = neighbourhood.max()
                out[f"nbrsig_{column}_w{window}_max"][rows] = highest

    return pd.DataFrame(out, index=frame.index)


def _transcript_level(frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in AGGREGATED if c in frame.columns]
    order, transcripts, _ = _ordered(frame)
    names = ["tx_n_sites", "tx_n_reads_loo_mean"] + [
        f"tx_{c}_loo_mean" for c in available
    ]
    out = _blank(frame, names)

    columns = {c: frame[c].to_numpy(dtype=np.float64)[order] for c in available}
    columns["__n_reads"] = sites["n_reads"].to_numpy(dtype=np.float64)[order]

    for start, stop in _transcript_slices(transcripts):
        rows = order[start:stop]
        n = stop - start
        out["tx_n_sites"][rows] = float(n)
        if n < 2:
            continue
        for column, series in columns.items():
            block = series[start:stop]
            loo = (block.sum() - block) / (n - 1)
            key = (
                "tx_n_reads_loo_mean" if column == "__n_reads"
                else f"tx_{column}_loo_mean"
            )
            out[key][rows] = loo

    return pd.DataFrame(out, index=frame.index)


@register("features", "quantiles_nbr_struct_v1")
class NeighbourStructureFeatures(QuantileFeatures):
    """`quantiles_v1` + where the neighbouring candidate sites are.

    Tests whether candidate *density and spacing* alone carry signal, with no
    reference to what the neighbours measured. If this wins and
    `quantiles_nbr_signal_v1` does not, the effect is about genomic context
    rather than about regional methylation.
    """

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([frame, _structural(frame).astype(np.float32)], axis=1)


@register("features", "quantiles_nbr_signal_v1")
class NeighbourSignalFeatures(QuantileFeatures):
    """`quantiles_v1` + what the neighbours measured, leave-one-out.

    The real proxy for message passing: this is roughly what one round of
    mean/max aggregation over a transcript chain would compute, for the price of
    a feature set rather than a GPU.
    """

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([frame, _neighbour_signal(frame).astype(np.float32)], axis=1)


@register("features", "quantiles_transcript_v1")
class TranscriptLevelFeatures(QuantileFeatures):
    """`quantiles_v1` + a leave-one-out summary of the whole transcript.

    The super-node, not the chain. Positives-per-transcript carries 4.68x the
    binomial variance, so a transcript-level random effect may be most of what
    the clustering is - and if so, this cheap set captures it and no graph is
    needed.
    """

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        extra = _transcript_level(frame, sites).astype(np.float32)
        return pd.concat([frame, extra], axis=1)


# The count column is the only transcript-level feature that a sparser input
# file biases. Everything else here is a mean, and a mean is an unbiased
# estimator of a transcript's level however many of its sites you sample.
_DENSITY_COLUMNS = ("tx_n_sites",)


@register("features", "quantiles_transcript_robust_v1")
class RobustTranscriptFeatures(QuantileFeatures):
    """`quantiles_transcript_v1` minus the one column that does not transfer.

    **The problem this exists to solve.** A `finalise` feature set is computed
    from whatever sites are in the input file, and `scripts/predict.py` runs on
    an evaluation file nobody here has seen. Simulated by dropping half the
    sites at random and re-extracting, then measured against the full-file
    values in units of each column's own standard deviation:

    | feature set | median bias | worst |
    |---|---:|---:|
    | `quantiles_nbr_signal_v1` | 0.638 | 1.172 (`nbrsig_n_w200`) |
    | `quantiles_nbr_struct_v1` | 0.202 | 0.929 (`nbr_count_100`) |
    | `quantiles_transcript_v1` | **0.020** | 0.927 (`tx_n_sites`) |

    **Bias is the thing that breaks a model; noise only costs accuracy.** A
    model trained on `nbr_count_100 = 12` and shown `nbr_count_100 = 6` for the
    same site is being fed a different variable. The split above is not luck:

    - a **count** is proportional to candidate density, so halving the file
      halves it;
    - a **max** over fewer neighbours is systematically smaller, which is why
      every `nbrsig_*_max` is biased by 0.76 or more;
    - a **mean** is unbiased at any sample size, which is why the four
      `tx_*_loo_mean` columns come in at 0.008 to 0.085.

    So this drops `tx_n_sites` and keeps the means. It should retain most of
    `quantiles_transcript_v1`'s +0.0286 while being the only cross-site set that
    can honestly be shipped through `predict.py`.

    It is still not free of the concern - the aggregates get noisier on a
    sparser file, and nothing here measures what that costs in PR AUC. What it
    removes is the systematic shift.
    """

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        extra = _transcript_level(frame, sites)
        extra = extra.drop(columns=[c for c in _DENSITY_COLUMNS if c in extra.columns])
        return pd.concat([frame, extra.astype(np.float32)], axis=1)


# The cross-site columns that a sparser input file does NOT bias, measured by
# dropping half the training sites at random and re-extracting (bias in units of
# each column's own standard deviation):
#
#   nbr_rel_position        0.006   nbr_rank_on_transcript  0.007
#   tx_dwell_0_q95_loo_mean 0.008   tx_sd_0_q95_loo_mean    0.015
#   tx_mean_0_q95_loo_mean  0.025   tx_mean_m1_q50_loo_mean 0.085
#   tx_n_reads_loo_mean     <0.01
#
# against 0.43-0.93 for every count and 0.69-1.17 for every windowed max. Two
# rules, both from first principles rather than from the table: a normalised
# POSITION is preserved under random thinning, and a MEAN is an unbiased
# estimator at any sample size. A count and a max are neither.
_ROBUST_STRUCTURAL = ("nbr_rel_position", "nbr_rank_on_transcript")


@register("features", "quantiles_crosssite_robust_v1")
class RobustCrossSiteFeatures(QuantileFeatures):
    """Every cross-site column that survives a differently-built input file.

    The cross-site family is the largest effect in the project - up to +0.0286,
    50/50, corrected p = 0.0000 - and most of it **cannot be shipped**, because
    `scripts/predict.py` runs on an evaluation file whose candidate density
    nobody here knows. This is the subset that can.

    Two families, kept for different reasons:

    - **Position along the transcript.** The strongest structural column in the
      screen and the most robust of any cross-site feature (bias 0.006, noise
      0.262). It is also the one with a mechanism: the positive rate climbs
      monotonically from 2.54% in the first tenth of a transcript's candidate
      span to 6.67% in the ninth, then falls to 4.24% in the last - a 2.6x
      swing, and the shape m6A's known enrichment near the stop codon would
      produce. Nothing here has the annotation to confirm that reading.
    - **Transcript-level leave-one-out means.** Unbiased at any sample size,
      which is the whole reason they are means and not maxes or counts.

    Dropped, with the measurement that condemns each: `nbr_count_*` and
    `tx_n_sites` and `nbr_n_on_transcript` (bias 0.43-0.93, they scale with
    density), every `nbrsig_*_max` (bias 0.76+, a max over fewer neighbours is
    smaller), `nbr_dist_prev`/`_next` (bias ~0.20), and `nbr_dist_nearest`
    (unbiased at 0.067 but noise 1.488, which buys nothing).

    **This is a robustness claim, not an accuracy claim.** What it costs in
    PR AUC against the unrestricted `quantiles_transcript_v1` is what
    `configs/crosssite_robust.yaml` measures. It is still sensitive to a sparser
    file through *noise*; what it removes is the systematic shift.
    """

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        structural = _structural(frame)[list(_ROBUST_STRUCTURAL)]
        transcript = _transcript_level(frame, sites)
        transcript = transcript.drop(
            columns=[c for c in _DENSITY_COLUMNS if c in transcript.columns]
        )
        extra = pd.concat([structural, transcript], axis=1).astype(np.float32)
        return pd.concat([frame, extra], axis=1)


@register("features", "quantiles_flank_crosssite_v1")
class FlankAndRobustCrossSiteFeatures(QuantileFlankFeatures, RobustCrossSiteFeatures):
    """Everything that has been established, in one feature set.

    `quantiles_v1`, plus the 7-mer flanking bases (+0.0109, 48/50, corrected
    p = 0.0006, and the same +0.0113 again with depth augmentation already on),
    plus the cross-site columns that survive a differently-built input file.

    Paired with `train_depths: [1, 3, 5, 10, null]` in
    `configs/final_candidate.yaml`, this is the full stack of everything that
    has independently earned its place, and the only combination of them that
    can honestly be shipped through `predict.py`.

    Composed by inheritance rather than by delegation, so neither half can
    drift from the version that was tested on its own. The MRO does the work:
    `site_features` resolves to `QuantileFlankFeatures`, whose `super()` call
    then reaches `QuantileFeatures`, while `finalise` resolves past the flank
    class to `RobustCrossSiteFeatures`. Calling
    `QuantileFlankFeatures.site_features(self, ...)` explicitly does **not**
    work - that module uses the zero-argument `super()`, which requires `self`
    to be an instance of the class the call is written in.
    """
