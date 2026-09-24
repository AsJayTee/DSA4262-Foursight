"""The 7-mer's flanking bases, which `motif_onehot` throws away.

`Site.kmer` is seven characters and `motif_onehot` encodes only `kmer[1:6]`, so
the first and last base are parsed out of the file and then discarded at
extraction. They are not decoration. The 7-mer is exactly the union of the three
overlapping 5-mers that produced our three measurement positions — a 5-mer sits
in the pore at any moment and shifts one base at a time, so `AAGAC`, `AGACC` and
`GACCA` are what generated the -1, 0 and +1 columns
([docs/data.md](../../../docs/data.md)). Those two outer bases are therefore part
of the sequence context that physically shaped every number we model.

Measured on the training set before this was written:

- the right flank alone moves the positive rate from **1.98% (A) to 6.88% (G)**,
- and it survives conditioning on the motif the model already sees — within
  `GGACT` the rate is 14.3% under a right-flank A and 27.8% under a G, within
  `AGACT` it is 2.7% against 11.6%,
- mean logloss is 0.15474 for a motif-only rate table against 0.15010 for one
  keyed on the full 7-mer, so the two bases recover about a sixth of what the
  motif itself contributes over the base rate.

**Encoded as 4 + 4 indicators, not as a 288-way 7-mer one-hot.** All 288
combinations do occur (18 motifs x 4 x 4, every cell populated), but 5,475
positives spread over 288 categories leaves some of them with almost no positive
example, and a tree would be fitting noise in the thin ones. The effect is also
largely *additive* — C and G on the right raise the rate in nearly every motif,
A lowers it — which is the pattern two 4-way indicators capture cheaply and 288
indicators fragment.

This is `quantiles_v1` plus those eight columns and nothing else, so

    python scripts/evaluate.py --config configs/quantiles_flank.yaml \
           --compare-features quantiles_v1

varies exactly one thing.
"""

from __future__ import annotations

from m6a.data import Site
from m6a.features.quantiles import QuantileFeatures
from m6a.registry import register

# Only ACGT occurs at either flank in this dataset, checked across all 121,838
# sites. Fixed order, so the column layout is stable between training and
# prediction the same way DRACH_MOTIFS is.
BASES = ("A", "C", "G", "T")

KMER_LENGTH = 7


def flank_onehot(site: Site) -> dict[str, float]:
    """One-hot over the 7-mer's outer bases: 4 for the left, 4 for the right."""
    kmer = site.kmer
    if len(kmer) != KMER_LENGTH:
        # motif_onehot slices kmer[1:6] and would quietly return an all-zero
        # row here rather than failing; say so instead. A feature set that
        # silently encodes nothing looks like a model that learned nothing.
        raise ValueError(
            f"{site.transcript_id}:{site.position} has a {len(kmer)}-mer "
            f"({kmer!r}), and quantiles_flank_v1 needs the {KMER_LENGTH}-mer to "
            "read its flanking bases. Use features: quantiles_v1 for data "
            "without them."
        )
    left, right = kmer[0], kmer[-1]
    out = {f"left_{base}": float(base == left) for base in BASES}
    out.update({f"right_{base}": float(base == right) for base in BASES})
    return out


@register("features", "quantiles_flank_v1")
class QuantileFlankFeatures(QuantileFeatures):
    """`quantiles_v1` plus the two flanking bases of the 7-mer.

    Subclassed rather than copied so the two feature sets cannot drift: whatever
    `quantiles_v1` computes, this computes too, and the paired comparison stays
    a test of the eight added columns alone.
    """

    def site_features(self, site: Site) -> dict[str, float]:
        out = super().site_features(site)
        out.update(flank_onehot(site))
        return out
