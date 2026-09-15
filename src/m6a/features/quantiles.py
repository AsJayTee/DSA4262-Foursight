"""Read-distribution quantiles.

The modelling problem here is Multiple Instance Learning: the site carries the
label, individual reads do not, and only a fraction of reads at a positive site
are actually modified. A mean washes that fraction out — a shifted tail does
not. Quantiles capture "some reads look unusual" in a form a gradient-boosted
tree can split on, without the dependency cost of an attention-MIL network.
"""

from __future__ import annotations

import numpy as np

from m6a.data import READ_FEATURE_NAMES, Site
from m6a.features.base import FeatureExtractor, motif_onehot
from m6a.registry import register

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@register("features", "quantiles_v1")
class QuantileFeatures(FeatureExtractor):
    def site_features(self, site: Site) -> dict[str, float]:
        reads = site.reads
        out: dict[str, float] = {}

        qs = np.quantile(reads, QUANTILES, axis=0)
        means = reads.mean(axis=0)
        stds = reads.std(axis=0)

        for i, feature in enumerate(READ_FEATURE_NAMES):
            out[f"{feature}_mean"] = float(means[i])
            out[f"{feature}_std"] = float(stds[i])
            for j, q in enumerate(QUANTILES):
                out[f"{feature}_q{int(q * 100):02d}"] = float(qs[j, i])
            # Tail spread: how far the extremes sit from the middle. A site
            # where a minority of reads are modified should show a wider tail
            # than its mean suggests.
            out[f"{feature}_iqr"] = float(qs[3, i] - qs[1, i])
            out[f"{feature}_tail"] = float(qs[4, i] - qs[0, i])

        out["n_reads"] = float(site.n_reads)
        out["log_n_reads"] = float(np.log1p(site.n_reads))
        out.update(motif_onehot(site))
        return out
