"""Cross-source evaluation: is a change better under BOTH labellings?

docs/decisions/0029. Gene-grouped CV on dataset0 asks one question - does it
predict dataset0's labels on unseen genes - and the project's largest gain
(`everything` over `quantiles`, +0.065) turned out to be worth nothing on data1,
a second sequencing run of largely the same sites with a different labelling.

This evaluates a config against a baseline config across both files:

**One split over both files.** A gene and both sequencing copies of a shared site
are always in the same fold. dataset0's genes keep their canonical seed-4262
fold, so nothing recorded moves; genes only data1 has are split with the same
seed among themselves. data1 transcripts with no known gene are left out - they
could belong to any fold.

**Arms** - per fold, the config is trained five ways and scores the held-out
genes of BOTH files:

  dataset0     trained on dataset0 only (what ships today)
  data1        trained on data1 only
  pooled       both files; each site with its own file's label. A site
               sequenced in both counts once (weight 1/2 per copy)
  pooled_both  both files; a shared site's two run-copies each carry BOTH
               labels (weight 1/4 each), so a disagreement is a 0.5 target and
               the model cannot learn "this run, so this labelling"
  ensemble     mean of the dataset0 and data1 models' scores

The baseline config goes through exactly the same arms and folds, so every gain
is paired.

**Readouts**, all as the config's gain over the baseline, with 95% intervals
from resampling whole genes (paired: same resamples for both):

- per arm and file: PR AUC, the baseline's, the gain;
- the selection rule (0032 amends 0029): rank on the worse of the two files'
  gains; report the mean too. The worse one also
  vetoes a model below `VETO` (-0.005, fixed in 0029 before any result);
- the crossed test on shared sites, dataset0 arm: dataset0 vs data1
  measurements x dataset0 vs data1 labels - where a gain survives and where it
  does not;
- data1 genes absent from dataset0, dataset0 arm - the gene-population check.

Not imported by predict.py.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from m6a import external
from m6a.data import SUBSAMPLE_SEED, assign_folds, load_labels
from m6a.evaluation import metrics

SOURCES = ("dataset0", "data1")
FITTED_ARMS = ("dataset0", "data1", "pooled", "pooled_both")
ARMS = FITTED_ARMS + ("ensemble",)
VETO = -0.005
# Logged on every cross-source run so the W&B decision view can show only runs
# evaluated this way (1 = CV only, 2 = + held-out data1/data2 per 0028,
# 3 = cross-source per 0029).
EVAL_SCHEMA = 3
N_RESAMPLES = 500
SEED = 4262
KEYS = ["transcript_id", "transcript_position"]
# The crossed test's four cells: (measurements from, labels from).
CELLS = {"A": ("dataset0", "dataset0"), "B": ("dataset0", "data1"),
         "C": ("data1", "dataset0"), "D": ("data1", "data1")}


@dataclass
class Source:
    """One file's labelled sites, in one row order, at every training depth."""

    name: str
    index: pd.MultiIndex
    X: dict            # depth (None = full) -> DataFrame, rows in `index` order
    y: np.ndarray
    genes: np.ndarray
    folds: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


# ----------------------------------------------------------------- loading

def union_folds(genes0, genes1, seed: int = SEED, n_folds: int = 5) -> dict:
    """gene -> fold for every gene in either file.

    dataset0's genes get exactly the assignment crossval gives them (the same
    function over the same gene set), so the canonical split is unchanged.
    Genes only data1 has are assigned among themselves with the same seed.
    """
    g0 = pd.DataFrame({"gene_id": pd.unique(np.asarray(genes0))})
    out = dict(zip(g0["gene_id"], assign_folds(g0, seed, n_folds, "gene_id")))
    only1 = sorted(set(np.asarray(genes1)) - set(out))
    if only1:
        g1 = pd.DataFrame({"gene_id": only1})
        out.update(zip(g1["gene_id"], assign_folds(g1, seed, n_folds, "gene_id")))
    return out


def load(features: str, depths, data_dir: str | Path, *, subsample_seed: int = SUBSAMPLE_SEED,
         use_cache: bool = True, seed: int = SEED, n_folds: int = 5, log=print) -> dict:
    """Both files as Sources, extracted at `depths`, on the union gene split."""
    from m6a import feature_cache

    data_dir = Path(data_dir)
    depths = list(dict.fromkeys(list(depths) + [None]))
    labels0 = load_labels(data_dir / "data.info.labelled").set_index(KEYS)
    info1 = external.load_info(data_dir, "data1")
    gene_map = external.load_genes(data_dir)
    raw = {}
    for name, json_path, labels, gene_of in (
        ("dataset0", data_dir / "dataset0.json.gz", labels0["label"],
         labels0["gene_id"]),
        ("data1", external.paths(data_dir, "data1")[0], info1["label"],
         pd.Series(gene_map.reindex(info1.index.get_level_values(0)).fillna("").to_numpy(),
                   index=info1.index)),
    ):
        ext = feature_cache.extract(json_path, features, depths, seed=subsample_seed,
                                    use_cache=use_cache, log=log)
        full = ext[None].features
        keep = full.index.isin(labels.index)
        index = full.index[keep]
        genes = gene_of.reindex(index).to_numpy()
        known = genes != ""          # data1 transcripts no source could map
        index, genes = index[known], genes[known]
        raw[name] = (index, {d: ext[d].features.loc[index] for d in depths},
                     labels.reindex(index).to_numpy().astype(int), genes)

    fold_of = union_folds(raw["dataset0"][3], raw["data1"][3], seed, n_folds)
    columns = list(raw["dataset0"][1][None].columns)
    return {name: Source(name, index, {d: X[columns] for d, X in Xs.items()}, y, genes,
                         np.array([fold_of[g] for g in genes]))
            for name, (index, Xs, y, genes) in raw.items()}


# --------------------------------------------------------------- training rows

def training_rows(sources: dict, arm: str, fold: int, depths) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(X, y, weight) for one arm's training rows in one fold, stacked over
    `depths` (each depth copy carries the same labels and weights, as
    docs/decisions/0022 does for depth augmentation)."""
    s0, s1 = sources["dataset0"], sources["data1"]
    blocks = []

    def add(src, rows, labels, weights):
        for d in depths:
            blocks.append((src.X[d].iloc[rows], labels, weights))

    if arm in ("dataset0", "data1"):
        src = sources[arm]
        rows = np.flatnonzero(src.folds != fold)
        add(src, rows, src.y[rows], np.ones(len(rows)))
    else:
        train0 = np.flatnonzero(s0.folds != fold)
        train1 = np.flatnonzero(s1.folds != fold)
        # Labels of the same site in the other file, where it is shared.
        other_of_0 = pd.Series(s1.y, index=s1.index).reindex(s0.index[train0])
        other_of_1 = pd.Series(s0.y, index=s0.index).reindex(s1.index[train1])
        shared0 = other_of_0.notna().to_numpy()
        shared1 = other_of_1.notna().to_numpy()
        if arm == "pooled":
            add(s0, train0, s0.y[train0], np.where(shared0, 0.5, 1.0))
            add(s1, train1, s1.y[train1], np.where(shared1, 0.5, 1.0))
        elif arm == "pooled_both":
            # Unique sites: their own label, weight 1. Shared sites: each
            # run-copy appears twice, once per labelling, weight 1/4 each -
            # four rows summing to the one site.
            add(s0, train0, s0.y[train0], np.where(shared0, 0.25, 1.0))
            add(s1, train1, s1.y[train1], np.where(shared1, 0.25, 1.0))
            add(s0, train0[shared0], other_of_0.to_numpy()[shared0].astype(int),
                np.full(shared0.sum(), 0.25))
            add(s1, train1[shared1], other_of_1.to_numpy()[shared1].astype(int),
                np.full(shared1.sum(), 0.25))
        else:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {FITTED_ARMS}.")
    X = pd.concat([b[0] for b in blocks], axis=0)
    return X, np.concatenate([b[1] for b in blocks]), np.concatenate([b[2] for b in blocks])


def _fit(model_class, params, X, y, w):
    model = model_class(**params)
    accepts = "sample_weight" in inspect.signature(model.fit).parameters
    if not np.allclose(w, 1.0) and not accepts:
        raise SystemExit(
            f"{model_class.__name__}.fit takes no sample_weight, which the pooled "
            "arms need. Use --xsrc-arms dataset0,data1,ensemble, or add sample_weight "
            "to the model (see m6a/models/lightgbm.py)."
        )
    model.fit(X, y, **({"sample_weight": w} if accepts else {}))
    return model


def out_of_fold(sources: dict, model_class, params: dict, train_depths, arms=ARMS,
                log=print) -> dict:
    """arm -> file -> out-of-fold scores (one per site, in the Source's row order).
    Every score comes from a model whose training rows hold no site of that gene,
    in either file."""
    fitted = [a for a in FITTED_ARMS if a in arms or (a in ("dataset0", "data1") and "ensemble" in arms)]
    oof = {a: {s: np.full(len(sources[s]), np.nan) for s in SOURCES} for a in fitted}
    for fold in sorted(np.unique(sources["dataset0"].folds)):
        for arm in fitted:
            X, y, w = training_rows(sources, arm, fold, train_depths)
            model = _fit(model_class, params, X, y, w)
            for s in SOURCES:
                held = sources[s].folds == fold
                oof[arm][s][held] = model.predict_proba(sources[s].X[None][held])
            log(f"    fold {fold} {arm:<12s} trained on {len(X):,} rows")
    if "ensemble" in arms:
        oof["ensemble"] = {s: (oof["dataset0"][s] + oof["data1"][s]) / 2 for s in SOURCES}
    return {a: oof[a] for a in arms}


# ----------------------------------------------------------------- readouts

def _ap(y, s) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def paired_gain(y, cand, base, clusters, n: int = N_RESAMPLES, seed: int = SEED) -> dict:
    """Config minus baseline PR AUC, with a paired gene-resampled interval."""
    diffs = np.array([_ap(y[i], cand[i]) - _ap(y[i], base[i])
                      for i in external.resample_indices(clusters, n, seed)])
    low, high = np.nanpercentile(diffs, [2.5, 97.5])
    m = metrics(y, cand)
    return {"n": int(len(y)), "n_positive": int(y.sum()), "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"], "baseline_pr_auc": _ap(y, base),
            "gain": _ap(y, cand) - _ap(y, base), "ci_low": float(low), "ci_high": float(high),
            "win_rate": float(np.nanmean(diffs > 0))}


def summarise(sources: dict, cand: dict, base: dict, n: int = N_RESAMPLES,
              headline_arm: str = "dataset0") -> dict:
    """The report block: matrix, selection rule per arm, crossed test, new genes.
    `headline_arm` is how the shipped model is trained; its row is the one the
    selection rule is read from."""
    out = {"arms": {}, "veto": VETO, "headline_arm": headline_arm}
    for arm in cand:
        cells = {s: paired_gain(sources[s].y, cand[arm][s], base[arm][s], sources[s].genes, n)
                 for s in SOURCES}
        gains = [cells[s]["gain"] for s in SOURCES]
        out["arms"][arm] = {**cells, "gain_mean": float(np.mean(gains)),
                            "gain_worst": float(np.min(gains)),
                            "eligible": bool(np.min(gains) >= VETO)}

    s0, s1 = sources["dataset0"], sources["data1"]
    shared = s0.index.intersection(s1.index)
    if "dataset0" in cand and len(shared):
        pos = {"dataset0": s0.index.get_indexer(shared), "data1": s1.index.get_indexer(shared)}
        clusters = s0.genes[pos["dataset0"]]
        labels = {"dataset0": s0.y[pos["dataset0"]], "data1": s1.y[pos["data1"]]}
        crossed = {}
        for cell, (meas, lab) in CELLS.items():
            crossed[cell] = {"measurements": meas, "labels": lab, **paired_gain(
                labels[lab], cand["dataset0"][meas][pos[meas]],
                base["dataset0"][meas][pos[meas]], clusters, n)}
        out["crossed"] = crossed
        new_genes = ~pd.Series(s1.genes).isin(set(s0.genes)).to_numpy()
        out["data1_new_genes"] = paired_gain(
            s1.y[new_genes], cand["dataset0"]["data1"][new_genes],
            base["dataset0"]["data1"][new_genes], s1.genes[new_genes], n)
    return out
