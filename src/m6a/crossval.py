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
    # The raw reads behind each row, ragged, in the same row order as X - or
    # None, which is the normal case. Only a model that declares CONSUMES_READS
    # ever sees them, and only `build_datasets(with_reads=True)` loads them,
    # because they are ~397 MB on the full training set against 49 MB for X.
    # See docs/decisions/0023.
    reads: "object | None" = None

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
    # One entry per fold, each a list of per-iteration dicts - empty for a model
    # that does not train iteratively, which is most of them. See
    # `m6a.models.base.BaseModel.REPORTS_TRAINING_CURVE` and docs/decisions/0021.
    histories: list[list[dict]] = field(default_factory=list)

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
    with_reads: bool = False,
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
    # Read-level data is depth-independent on disk: every depth is a hash-chosen
    # subset of the full-depth reads, so one extraction serves the whole sweep
    # and each depth is derived from it in memory. See docs/decisions/0023.
    all_reads = None
    if with_reads:
        all_reads = feature_cache.extract_reads(
            json_path, limit=limit, use_cache=use_cache, log=log
        )

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

        reads = None
        if all_reads is not None:
            reads = _reads_for(all_reads, extraction, X.index, depth, subsample_seed)

        dataset = Dataset(X=X, y=y, folds=folds, sites=sites, columns=columns,
                          depth=depth, reads=reads)
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


def _reads_for(all_reads, extraction, index, depth, subsample_seed):
    """Reindex the full-depth reads onto a dataset's rows, then thin them.

    Two things happen between extraction and a `Dataset` that the reads have to
    survive: the label join is an **inner** join, so rows can be dropped, and it
    can reorder them. `ReadBlocks` carries no site ids of its own - it is a pair
    of arrays whose meaning is "row i here is row i there" - so this is the one
    place the correspondence is established, and it is established by position
    lookup rather than by trusting the two to have stayed parallel.
    """
    from m6a.data import subsample_blocks

    positions = extraction.features.index.get_indexer(index)
    if (positions < 0).any():
        raise RuntimeError(
            "A labelled site is missing from the extracted read blocks. The "
            "feature table and the reads came from different files or different "
            "site limits; rebuild with --no-cache."
        )
    # The join usually keeps every site in file order, and `take` copies ~400 MB
    # on the full training set. Skip it when there is nothing to do: identity is
    # cheap to check and the copy is not.
    identity = (
        positions.size == len(all_reads)
        and bool(np.array_equal(positions, np.arange(positions.size)))
    )
    blocks = all_reads if identity else all_reads.take(positions)
    # Checked rather than assumed: `sites.n_reads` is the true depth recorded
    # during extraction, and if it disagrees with the reads we actually have,
    # some row is carrying another site's reads.
    expected = extraction.sites["n_reads"].to_numpy()[positions]
    if not np.array_equal(blocks.counts, expected):
        raise RuntimeError(
            "Read counts do not match the extracted site table, so the reads "
            "and the feature rows are out of step. This would train a model on "
            "one site's reads under another site's label."
        )
    if depth is None:
        return blocks
    return subsample_blocks(blocks, list(index), depth, subsample_seed)


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
    record_history: bool = True,
    train_on: Sequence[Dataset] = (),
    log=print,
) -> CVResult:
    """Fit one model per fold and score the sites it never saw.

    Every site is scored by a model that saw no transcript of its gene, which is
    the only number worth comparing. Per-fold metrics are *kept*: collapsing them
    to one pooled figure throws away the only thing that can tell a real
    improvement from fold noise.

    `record_history` hands each fold's held-out rows to a model that declares
    `REPORTS_TRAINING_CURVE`, so it can report how its fit progressed. It costs
    extra time inside every fit - the model scores the held-out set at every
    iteration - which is why repeated cross-validation turns it off for every
    repetition but the one whose curve is logged. See docs/decisions/0021.

    `train_on` is where the **training** rows come from, and it defaults to
    `dataset` itself. Passing several read depths of the same sites stacks one
    copy of every training site per depth, which is how a site-level model is
    trained at the depth it will be tested at. Scoring never changes: the
    held-out rows always come from `dataset`, so the headline number means the
    same thing as every other run's. See docs/decisions/0022.
    """
    model_params = dict(model_params or {})
    use = list(columns) if columns is not None else list(dataset.columns)
    X, y, folds = dataset.X, dataset.y, dataset.folds

    sources = list(train_on) or [dataset]
    for source in sources:
        # The fold mask is positional, so a training set whose rows are in a
        # different order would quietly train on the wrong sites and produce a
        # plausible number. build_datasets guarantees this already; checking it
        # here means an externally built Dataset cannot slip through.
        if not source.X.index.equals(X.index):
            raise RuntimeError(
                "A training set passed to cross_validate has different rows from "
                "the dataset being scored. They must be the same sites in the "
                "same order - build them together with crossval.build_datasets."
            )

    oof = np.zeros(len(y), dtype=float)
    models: list = []
    per_fold: list[dict] = []
    histories: list[list[dict]] = []

    for fold in sorted(np.unique(folds)):
        holdout = folds == fold
        train_X, train_y, train_reads = stack_rows(sources, use, ~holdout)
        model = model_class(**model_params)

        # Two opt-in capabilities, each passed **only** to a model that declared
        # it. Handing an argument to a model that did not ask for it either
        # raises on a signature that never expected it or - far worse - is
        # silently ignored, and a model quietly scoring without the reads it was
        # supposed to use looks exactly like a model that is simply worse. The
        # flags are what make that loud.
        fit_kwargs: dict = {}
        score_kwargs: dict = {}
        if getattr(model, "CONSUMES_READS", False):
            if train_reads is None or dataset.reads is None:
                raise RuntimeError(
                    f"{model_class.__name__} declares CONSUMES_READS but this "
                    "dataset carries no reads. Build it with "
                    "crossval.build_datasets(..., with_reads=True)."
                )
            fit_kwargs["reads"] = train_reads
            score_kwargs["reads"] = dataset.reads.take(holdout)

        if record_history and getattr(model, "REPORTS_TRAINING_CURVE", False):
            # `validation` is (X, y) - plus the held-out reads for a read-level
            # model, which is the same pair of flags agreeing with each other
            # rather than a third convention.
            held_out = (X.loc[holdout, use], y[holdout])
            if score_kwargs:
                held_out = (*held_out, score_kwargs["reads"])
            fit_kwargs["validation"] = held_out
        model.fit(train_X, train_y, **fit_kwargs)
        oof[holdout] = model.predict_proba(X.loc[holdout, use], **score_kwargs)
        per_fold.append({"fold": int(fold), **metrics(y[holdout], oof[holdout])})
        histories.append(list(getattr(model, "history", []) or []))
        if keep_models:
            models.append(model)
        log(
            f"  fold {fold}: trained on {len(train_X):,}, "
            f"scored {int(holdout.sum()):,}, PR AUC {per_fold[-1]['pr_auc']:.4f}"
        )

    return CVResult(
        oof=oof_table(dataset, oof),
        per_fold=per_fold,
        pooled=metrics(y, oof),
        models=models,
        columns=use,
        label=label,
        histories=histories,
    )


def stack_rows(
    sources: Sequence[Dataset],
    columns: list[str],
    mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, object | None]:
    """Training rows drawn from one or several read depths of the same sites.

    One `Dataset` in gives exactly what indexing it would have given, so the
    ordinary path pays nothing. Several stack one copy of each row per depth.

    **The labels are simply repeated.** They come from m6ACE-Seq and not from
    the nanopore reads, so dropping reads changes how much evidence a row
    carries and not what is true about the site (docs/decisions/0003). Nothing
    here invents a site: the same 97,524 training sites appear five times at
    five depths, which is why the stacked rows are correlated in a way ordinary
    extra data would not be - see docs/decisions/0022.
    """
    if not sources:
        raise ValueError("stack_rows needs at least one Dataset to draw rows from.")
    first = sources[0]
    keep = np.ones(len(first.y), dtype=bool) if mask is None else mask
    reads = _stack_reads(sources, keep)
    if len(sources) == 1:
        return first.X.loc[keep, columns], first.y[keep], reads
    return (
        pd.concat([source.X.loc[keep, columns] for source in sources], axis=0),
        np.tile(first.y[keep], len(sources)),
        reads,
    )


def _stack_reads(sources: Sequence[Dataset], keep: np.ndarray):
    """The read blocks for the same rows, stacked in the same order as the frame.

    Returns None when no source carries reads, which is every run that is not
    read-level. A source set where only *some* carry reads is a bug rather than
    a configuration: the stacked rows would silently lose their reads for some
    depths and keep them for others.
    """
    from m6a.data import ReadBlocks

    present = [source.reads is not None for source in sources]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(
            "Some training depths carry read-level data and some do not. Build "
            "every depth with build_datasets(with_reads=True) or none of them."
        )
    blocks = [source.reads.take(keep) for source in sources]
    if len(blocks) == 1:
        return blocks[0]
    counts = np.concatenate([block.counts for block in blocks])
    return ReadBlocks(
        np.concatenate([block.values for block in blocks]),
        np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
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
        model = models[position]
        # The sweep scores the *subsampled* dataset, so a read-level model gets
        # that dataset's thinned reads - which is the whole point: fewer reads
        # is less evidence, and a MIL model should feel that directly rather
        # than through a summary statistic computed over fewer of them.
        extra = {}
        if getattr(model, "CONSUMES_READS", False):
            if dataset.reads is None:
                raise RuntimeError(
                    "A read-level model cannot score a dataset built without "
                    "reads. Pass with_reads=True when building the depth sweep's "
                    "datasets too, or the sweep silently measures nothing."
                )
            extra["reads"] = dataset.reads.take(holdout)
        scores[holdout] = model.predict_proba(dataset.X.loc[holdout, use], **extra)
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
    train_on: Sequence[Dataset] = (),
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
            # Same for the training curve, and here it saves time rather than
            # memory: an iterative model scores its held-out fold at every
            # iteration, which is a real slowdown paid once instead of ten
            # times. Only repetition 0's curve is ever logged - consistent with
            # fold/* and the depth sweep being repetition-0 quantities.
            record_history=repetition == 0,
            # The extra depths are the same rows in the same order, so the
            # re-split fold mask applies to them unchanged - only `split` needs
            # refolding. cross_validate checks the index alignment rather than
            # assuming it.
            train_on=train_on,
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
