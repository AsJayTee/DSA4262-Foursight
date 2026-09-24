"""The FeatureExtractor contract.

To add a feature set: create one new module in this package, subclass
FeatureExtractor, decorate it with @register("features", "your_name"), and
reference that name from a config YAML. Do not edit any existing file.

    @register("features", "quantiles_v1")
    class Quantiles(FeatureExtractor):
        def site_features(self, site): return {"mean_0_p50": ...}
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from m6a.data import Site

# The 18 DRACH motifs: D in {A,G,T}, R in {A,G}, A, C, H in {A,C,T}.
# The central 5-mer of every site in this dataset is one of these, so the set
# is fixed and column order is stable between training and prediction.
DRACH_MOTIFS: list[str] = [
    "".join(m) for m in product("AGT", "AG", "A", "C", "ACT")
]


class FeatureExtractor:
    """Turns a stream of Sites into one row of float features per site.

    Subclasses implement `site_features`. The base class handles assembling the
    frame and its (transcript_id, transcript_position) index, so every extractor
    produces the same shape and predict.py can rely on it.
    """

    name: str = "base"

    def site_features(self, site: Site) -> dict[str, float]:
        raise NotImplementedError

    def finalise(self, frame: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
        """Add columns computed from every site at once. Default: change nothing.

        `site_features` sees one site, which is the right default and makes one
        family of features unwritable: anything about a site's *neighbours*.
        m6A clusters strongly along transcripts (docs/decisions/0025), so that
        family is worth having.

        `frame` is indexed by (transcript_id, transcript_position) and `sites`
        carries kmer, motif and n_reads on the same index. **Neither contains
        the label, the gene, or the fold** - this runs during extraction, before
        the label join and before the split. That is deliberate and it is the
        guard: the neighbour *label* correlation is unusable at prediction time,
        because every site of a held-out gene is held out together, so a feature
        set reaching for one would score well out of fold and be worthless in
        predict.py. It is prevented by what this method is handed.

        Called by `transform` below and by `feature_cache.extract`, which are the
        only two things that build a feature table. Overriding `transform`
        instead would reach only one of them.
        """
        return frame

    def transform(self, sites: Iterable[Site]) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        index: list[tuple[str, int]] = []
        kmers: list[str] = []
        n_reads: list[int] = []

        for site in sites:
            rows.append(self.site_features(site))
            index.append(site.key)
            kmers.append(site.kmer)
            n_reads.append(site.n_reads)

        if not rows:
            raise ValueError("No sites to transform — is the input file empty?")

        frame = pd.DataFrame(rows, dtype=np.float32)
        frame.index = pd.MultiIndex.from_tuples(
            index, names=["transcript_id", "transcript_position"]
        )
        # Same three columns feature_cache.extract builds, so finalise sees the
        # same thing on both paths. A test asserts the two agree.
        described = pd.DataFrame(
            {"kmer": kmers, "motif": [k[1:6] for k in kmers], "n_reads": n_reads},
            index=frame.index,
        )
        return self.finalise(frame, described)


def motif_onehot(site: Site) -> dict[str, float]:
    """One-hot over the 18 DRACH motifs. Fixed order, so train/predict align."""
    motif = site.motif
    return {f"motif_{m}": float(m == motif) for m in DRACH_MOTIFS}
