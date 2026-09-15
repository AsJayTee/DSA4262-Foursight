"""Mean/std pooling across reads — the baseline feature set.

This is the obvious approach and the briefing calls it out as weak: averaging
throws away the fact that only *some* reads at a modified site carry the
modification. It exists to be the number every other feature set has to beat.
See quantiles.py for the MIL-flavoured alternative.
"""

from __future__ import annotations

import numpy as np

from m6a.data import READ_FEATURE_NAMES, Site
from m6a.features.base import FeatureExtractor, motif_onehot
from m6a.registry import register


@register("features", "pooled_v1")
class PooledFeatures(FeatureExtractor):
    def site_features(self, site: Site) -> dict[str, float]:
        reads = site.reads
        out: dict[str, float] = {}

        means = reads.mean(axis=0)
        stds = reads.std(axis=0)
        for i, feature in enumerate(READ_FEATURE_NAMES):
            out[f"{feature}_mean"] = float(means[i])
            out[f"{feature}_std"] = float(stds[i])

        out["n_reads"] = float(site.n_reads)
        out["log_n_reads"] = float(np.log1p(site.n_reads))
        out.update(motif_onehot(site))
        return out
