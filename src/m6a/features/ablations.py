"""`quantiles_all_v1` with one feature family removed at a time.

Decision 0029 step 1. Cross-source evaluation found that about half of
`everything`'s gain over `quantiles` is tied to dataset0's labelling (+0.063
under dataset0's labels on shared sites, +0.029 under data1's). These sets find
which family carries that half: each is `everything` minus one family, so
`evaluate.py --config <cfg> --cross-source` on each says what the family is
worth under each labelling.

| feature set | removes | family's suspected risk |
|---|---|---|
| `quantiles_flank_coupling_v1` | cross-site (neighbour) columns | learns how one labelling's positives cluster along transcripts |
| `quantiles_coupling_crosssite_v1` | the 7-mer's two flanking bases | sequence priors a labelling may share |
| `quantiles_all_noreadcount_v1` | read count as a predictor | encodes run, expression and gene population |

Coupling removed is `quantiles_flank_crosssite_v1` (configs/final_candidate.yaml),
which already exists. Depth augmentation removed is a config change only
(configs/everything_fulldepth.yaml).

Composed by inheritance from the tested classes, like `quantiles_all_v1`
itself, so no family can drift from the version that was measured on its own.
"""

from __future__ import annotations

import pandas as pd

from m6a.data import Site
from m6a.features.coupling import QuantileCouplingFeatures
from m6a.features.flank import QuantileFlankFeatures
from m6a.features.neighbourhood import EverythingFeatures, RobustCrossSiteFeatures
from m6a.registry import register

# Read count enters three ways: the site's own count (twice) and the
# leave-one-out mean over the transcript's other sites.
READ_COUNT_COLUMNS = ("n_reads", "log_n_reads", "tx_n_reads_loo_mean")


@register("features", "quantiles_flank_coupling_v1")
class FlankCouplingFeatures(QuantileCouplingFeatures, QuantileFlankFeatures):
    """`everything` without the cross-site columns: no `finalise` pass at all."""


@register("features", "quantiles_coupling_crosssite_v1")
class CouplingCrossSiteFeatures(QuantileCouplingFeatures, RobustCrossSiteFeatures):
    """`everything` without the 7-mer's flanking bases."""


@register("features", "quantiles_all_noreadcount_v1")
class EverythingNoReadCountFeatures(EverythingFeatures):
    """`everything` without read count as a predictor.

    Depth augmentation still works - it subsamples reads before features are
    computed - so this removes the count as an *input*, not the model's
    exposure to sparse sites.
    """

    def site_features(self, site: Site) -> dict[str, float]:
        out = super().site_features(site)
        for column in READ_COUNT_COLUMNS:
            out.pop(column, None)
        return out

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        out = super().finalise(frame, sites)
        return out.drop(columns=[c for c in READ_COUNT_COLUMNS if c in out.columns])
