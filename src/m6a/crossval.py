"""Gene-grouped cross-validation, and the out-of-fold table it produces.

This is shared infrastructure (AGENTS.md section 8): `scripts/train.py` and
`scripts/evaluate.py` both go through it, so the two cannot disagree about which
sites are in which fold or about how a model was fitted.

The out-of-fold table is the point of this module. One row per site, carrying
the score, the label, the fold that produced it, and enough about the site
(gene, motif, true read depth) to slice by afterwards:

    transcript_id, transcript_position, gene_id, motif, n_reads, fold, label, score

Per-fold metrics, stratified metrics, calibration and the paired comparison
between two runs are all computed from it during a run.

**It is never written to disk.** It was, until train.py started running the
evaluation inline: the table existed so a finished run could be re-analysed
cheaply, and there is much less to re-analyse once every question is answered at
the moment the run happens. What survives instead is the metric *vector* - five
PR AUCs, or fifty under repeated cross-validation - which is everything a paired
test needs, at fifty floats rather than nine megabytes. The cost is real and is
recorded in docs/decisions/0009: a question you did not think to ask during a
run now means running it again.

Uses numpy, pandas and the registry. Not imported by predict.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from m6a.data import (
    SUBSAMPLE_SEED,
    align_to_features,
    assign_folds,
    load_labels,
)
from m6a.evaluation import metrics

OOF_COLUMNS = [
    "transcript_id",
    "transcript_position",
    "gene_id",
    "motif",
    "n_reads",
    "fold",
    "label",
    "score",
]

COLUMN_SUBSETS = ("all", "signal", "motif")


@dataclass(slots=True)
class Dataset:
    """Features, labels, folds and site metadata, all in one row order."""

    X: pd.DataFrame
    y: np.ndarray
    folds: np.ndarray
    sites: pd.DataFrame  # gene_id, motif, n_reads - indexed like X
    columns: list[str]
    depth: int | None = None

    def __len__(self) -> int:
        return len(self.y)


@dataclass(slots=True)
class CVResult:
    """Everything one cross-validated run produced, including the fold models."""

    oof: pd.DataFrame
    per_fold: list[dict]
    pooled: dict
    models: list = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    label: str = ""

    @property
    def per_fold_pr_auc(self) -> np.ndarray:
        return np.array([fold["pr_auc"] for fold in self.per_fold])

    @property
    def fold_ids(self) -> list[int]:
        return [int(fold["fold"]) for fold in self.per_fold]


def column_subset(columns: list[str], which: str = "all") -> list[str]:
    """The motif one-hots, everything else, or both.

    The two ablations the report needs are exactly these subsets of an existing
    feature set, so they cost a column filter rather than a new feature module:
    `motif` is the sequence-only floor a model must beat to have learned
    anything from the signal, and `signal` says what the motif is worth on top.
    """
    if which not in COLUMN_SUBSETS:
        raise ValueError(f"Unknown column subset {which!r}; expected one of {COLUMN_SUBSETS}")
    if which == "all":
        return list(columns)
    motif = [c for c in columns if c.startswith("motif_")]
    chosen = motif if which == "motif" else [c for c in columns if not c.startswith("motif_")]
    if not chosen:
        raise ValueError(
            f"Column subset {which!r} selected nothing out of {len(columns)} columns. "
            "Motif columns are the ones named motif_*; a feature set without them "
            "cannot be ablated this way."
        )
    return chosen


def build_datasets(
    json_path: str | Path,
    labels_path: str | Path,
    features: str,
    depths: Sequence[int | None] = (None,),
    *,
    seed: int = 4262,
    n_folds: int = 5,
    group_by: str = "gene_id",
    subsample_seed: int = SUBSAMPLE_SEED,
    limit: int | None = None,
    use_cache: bool = True,
    log=print,
) -> dict[int | None, Dataset]:
    """One Dataset per requested read depth, reading the input file once.

    `depth` subsamples every site's reads before features are computed, which is
    how the sweep builds a low-depth copy of the same sites. The labels come from
    m6ACE-Seq and not from the nanopore reads, so they stay valid under
    subsampling: dropping reads changes how much evidence the model has, not what
    is true about the site. `None` means full depth.

    Every returned Dataset shares one row order and one fold assignment, so a
    model fitted on the full-depth one can score a subsampled one fold by fold.
    """
    from m6a import feature_cache

    depths = list(dict.fromkeys(depths))
    extractions = feature_cache.extract(
        json_path,
        features,
        depths,
        seed=subsample_seed,
        limit=limit,
        use_cache=use_cache,
        log=log,
    )

    labels = load_labels(labels_path)
    out: dict[int | None, Dataset] = {}
    reference: Dataset | None = None

    for depth in depths:
        extraction = extractions[depth]
        joined = align_to_features(extraction.features, labels)

        columns = list(extraction.features.columns)
        X = joined[columns]
        y = joined["label"].to_numpy()
        folds = assign_folds(
            joined.reset_index(), seed=seed, n_folds=n_folds, group_by=group_by
        ).to_numpy()

        # Site metadata is taken at true depth even for a subsampled dataset, so
        # a depth-1 score can still be sliced by the depth the site really had.
        sites = extraction.sites.loc[X.index, ["motif", "n_reads"]].copy()
        sites["gene_id"] = joined["gene_id"].to_numpy()

        dataset = Dataset(X=X, y=y, folds=folds, sites=sites, columns=columns, depth=depth)
        if reference is None:
            reference = dataset
        elif not dataset.X.index.equals(reference.X.index) or not (
            dataset.folds == reference.folds
        ).all():
            raise RuntimeError(
                "Subsampled datasets came out in a different row or fold order from "
                "the full-depth one. Scoring fold models across depths depends on "
                "them matching, so this is a bug rather than a configuration error."
            )
        out[depth] = dataset

    return out


def build_dataset(
    json_path: str | Path,
    labels_path: str | Path,
    features: str,
    *,
    depth: int | None = None,
    **kwargs,
) -> Dataset:
    """Extract features, join labels, assign folds. One dataset, ready to fit."""
    return build_datasets(json_path, labels_path, features, [depth], **kwargs)[depth]


def cross_validate(
    dataset: Dataset,
    model_class,
    model_params: dict | None = None,
    *,
    columns: list[str] | None = None,
    label: str = "",
    keep_models: bool = True,
    log=print,
) -> CVResult:
    """Fit one model per fold and score the sites it never saw.

    Every site is scored by a model that saw no transcript of its gene, which is
    the only number worth comparing. Per-fold metrics are *kept*: collapsing them
    to one pooled figure throws away the only thing that can tell a real
    improvement from fold noise.
    """
    model_params = dict(model_params or {})
    use = list(columns) if columns is not None else list(dataset.columns)
    X, y, folds = dataset.X, dataset.y, dataset.folds

    oof = np.zeros(len(y), dtype=float)
    models: list = []
    per_fold: list[dict] = []

    for fold in sorted(np.unique(folds)):
        holdout = folds == fold
        model = model_class(**model_params)
        model.fit(X.loc[~holdout, use], y[~holdout])
        oof[holdout] = model.predict_proba(X.loc[holdout, use])
        per_fold.append({"fold": int(fold), **metrics(y[holdout], oof[holdout])})
        if keep_models:
            models.append(model)
        log(
            f"  fold {fold}: trained on {int((~holdout).sum()):,}, "
            f"scored {int(holdout.sum()):,}, PR AUC {per_fold[-1]['pr_auc']:.4f}"
        )

    return CVResult(
        oof=oof_table(dataset, oof),
        per_fold=per_fold,
        pooled=metrics(y, oof),
        models=models,
        columns=use,
        label=label,
    )


def score_folds(models: list, dataset: Dataset, columns: list[str] | None = None) -> np.ndarray:
    """Score `dataset` with fold models fitted elsewhere, fold by fold.

    The depth sweep's whole question is what happens when training depth and test
    depth differ, so the models come from the full-depth dataset and the rows
    come from a subsampled one. Both describe the same sites in the same order,
    so fold k's model still scores exactly the sites it was held out from - no
    gene it trained on appears in what it scores here either.
    """
    use = list(columns) if columns is not None else list(dataset.columns)
    scores = np.zeros(len(dataset), dtype=float)
    for position, fold in enumerate(sorted(np.unique(dataset.folds))):
        holdout = dataset.folds == fold
        scores[holdout] = models[position].predict_proba(dataset.X.loc[holdout, use])
    return scores


def oof_table(dataset: Dataset, scores: np.ndarray) -> pd.DataFrame:
    """The per-site out-of-fold table, in OOF_COLUMNS order."""
    index = dataset.X.index
    table = pd.DataFrame(
        {
            "transcript_id": index.get_level_values(0),
            "transcript_position": index.get_level_values(1),
            "gene_id": dataset.sites["gene_id"].to_numpy(),
            "motif": dataset.sites["motif"].to_numpy(),
            "n_reads": dataset.sites["n_reads"].to_numpy(),
            "fold": dataset.folds,
            "label": dataset.y,
            "score": scores,
        }
    )
    return table[OOF_COLUMNS]


# --------------------------------------------------------------------------
# repeated cross-validation
# --------------------------------------------------------------------------

def repetition_seeds(n_repeats: int, base: int = 4262) -> list[int]:
    """Split seeds for `n_repeats` repetitions. Element 0 is `base` itself.

    Repetition 0 **is** the canonical split, not a derivative of it. AGENTS.md
    section 3 freezes seed 4262, every per-fold number in GAPS.md was computed on
    it, and `SeedSequence(4262).spawn(n)[0]` is 3903649664, not 4262 - so
    deriving repetition 0 from the chain would quietly stop reproducing the
    numbers this repo claims to reproduce. Repetitions 1 onward are spawned
    children, which are independent streams rather than the correlated ones
    consecutive integers can give you.

    `spawn(n)[i]` does not depend on `n`, so asking for 10 repetitions yields the
    same first 5 as asking for 5. Two runs with different `--repeats` can
    therefore still be paired on the repetitions they share - the same property
    the keyed subsample draw provides (docs/decisions/0003), for the same reason.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if n_repeats == 1:
        return [base]
    children = np.random.SeedSequence(base).spawn(n_repeats - 1)
    return [base] + [int(child.generate_state(1)[0]) for child in children]


def refold(dataset: Dataset, seed: int, n_folds: int = 5, group_by: str = "gene_id") -> Dataset:
    """The same rows and features, re-split under a different seed.

    Fold assignment depends on the labels and the seed; it does not touch feature
    extraction. So a repetition costs five model fits and nothing else, which is
    the only reason ten of them are affordable.
    """
    if group_by not in dataset.sites.columns:
        raise ValueError(
            f"Cannot regroup by {group_by!r}; the dataset carries "
            f"{list(dataset.sites.columns)}. Repeated CV needs the grouping "
            "column that build_datasets attached."
        )
    folds = assign_folds(
        dataset.sites.reset_index(), seed=seed, n_folds=n_folds, group_by=group_by
    ).to_numpy()
    return Dataset(
        X=dataset.X, y=dataset.y, folds=folds, sites=dataset.sites,
        columns=dataset.columns, depth=dataset.depth,
    )


@dataclass(slots=True)
class RepeatedResult:
    """Every repetition's per-fold metrics, plus repetition 0 on its own.

    `canonical` is repetition 0 - the seed-4262 split - and is what the pooled
    numbers, the strata, the calibration and the depth sweep are all computed
    from, so a repeated run reports the same headline as a plain one. The extra
    repetitions are additive evidence for the comparison, nothing else.
    """

    canonical: CVResult
    observations: list[dict]   # one per (repetition, fold), each a metrics dict
    seeds: list[int]
    # Per repetition: the out-of-fold score vector and the fold assignment that
    # produced it, both in the dataset's row order. Two arrays of 121,838 floats
    # per repetition is about 2 MB - cheap enough to keep, and the only way to
    # ask a question *inside a stratum* afterwards without refitting. The
    # per-site scores still never reach disk (docs/decisions/0009); this is the
    # in-memory vector that record explicitly preserves.
    scores: list[np.ndarray] = field(default_factory=list)
    fold_ids: list[np.ndarray] = field(default_factory=list)

    @property
    def n_repeats(self) -> int:
        return len(self.seeds)

    def vector(self, metric: str = "pr_auc") -> np.ndarray:
        return np.array([o[metric] for o in self.observations], dtype=float)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.observations).set_index(["repetition", "fold"])

    def summary(self, metric: str = "pr_auc") -> dict:
        values = self.vector(metric)
        return {
            "n_repeats": self.n_repeats,
            "n_folds": len(self.observations) // self.n_repeats,
            "n_observations": len(values),
            "seeds": list(self.seeds),
            f"{metric}_mean": float(values.mean()),
            f"{metric}_sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            f"{metric}_min": float(values.min()),
            f"{metric}_max": float(values.max()),
        }


def repeated_cross_validate(
    dataset: Dataset,
    model_class,
    model_params: dict | None = None,
    *,
    n_repeats: int = 1,
    seed: int = 4262,
    n_folds: int = 5,
    group_by: str = "gene_id",
    columns: list[str] | None = None,
    label: str = "",
    keep_models: bool = True,
    log=print,
) -> RepeatedResult:
    """Cross-validate `n_repeats` times over independently seeded splits.

    Turns five paired observations into fifty, which is what makes a comparison
    between two models something other than a coin flip with four degrees of
    freedom. It does **not** make them independent: fifty splits of one dataset
    are still one dataset, the training sets still overlap, and the corrected
    t-test in `m6a.compare` is what keeps the resulting p-value honest about
    that. See docs/decisions/0006 and 0012.

    Repetition 0 uses `seed` unchanged, so the canonical result is a strict
    subset of a repeated one and nothing recorded against the plain split moves.
    """
    seeds = repetition_seeds(n_repeats, seed)
    observations: list[dict] = []
    scores: list[np.ndarray] = []
    fold_ids: list[np.ndarray] = []
    canonical: CVResult | None = None

    for repetition, rep_seed in enumerate(seeds):
        if n_repeats > 1:
            log(f"  repetition {repetition} of {n_repeats} (split seed {rep_seed})")
        split = dataset if repetition == 0 else refold(dataset, rep_seed, n_folds, group_by)
        result = cross_validate(
            split, model_class, model_params, columns=columns, label=label,
            # Only repetition 0's models are kept: the depth sweep scores with
            # them, and holding fifty boosters in memory buys nothing.
            keep_models=keep_models and repetition == 0,
            log=log if n_repeats == 1 else (lambda *_: None),
        )
        if repetition == 0:
            canonical = result
        scores.append(result.oof["score"].to_numpy())
        fold_ids.append(split.folds)
        for fold in result.per_fold:
            observations.append({"repetition": repetition, "split_seed": rep_seed, **fold})
        if n_repeats > 1:
            pr = result.per_fold_pr_auc
            log(f"    PR AUC mean {pr.mean():.4f}  range [{pr.min():.4f}, {pr.max():.4f}]")

    assert canonical is not None
    return RepeatedResult(
        canonical=canonical, observations=observations, seeds=seeds,
        scores=scores, fold_ids=fold_ids,
    )


def per_fold_frame(per_fold: list[dict]) -> pd.DataFrame:
    """Per-fold metrics as a table, with the pooled-vs-noise summary attached."""
    frame = pd.DataFrame(per_fold).set_index("fold")
    return frame[["n", "n_positive", "positive_rate", "roc_auc", "pr_auc", "pr_auc_lift"]]


def assert_same_folds(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> None:
    """Refuse to pair two runs that were not scored on the same sites and folds.

    A paired test is only valid because fold difficulty cancels, and that only
    holds if both runs really did use the same split. Two runs on different
    seeds produce two plausible-looking numbers and no warning, which is the
    failure AGENTS.md section 3 exists to prevent - so check rather than trust.
    """
    key = ["transcript_id", "transcript_position"]
    left = a.set_index(key)["fold"]
    right = b.set_index(key)["fold"]
    if len(left) != len(right) or not left.index.equals(right.index):
        raise SystemExit(
            f"{name_a} and {name_b} were scored on different sites "
            f"({len(left):,} vs {len(right):,}), so a paired comparison is not valid. "
            "Both runs have to see the same dataset."
        )
    if not (left.to_numpy() == right.to_numpy()).all():
        raise SystemExit(
            f"{name_a} and {name_b} disagree about fold assignment, so a paired "
            "comparison is not valid. Check that both configs use the same "
            "split: seed 4262, grouped by gene_id (AGENTS.md section 3)."
        )
