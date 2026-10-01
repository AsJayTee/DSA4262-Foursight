"""Guardrails for cross-source evaluation (src/m6a/crosssource.py, decision 0029).

Every failure pinned here is silent: a split that moves dataset0's canonical
folds (every recorded number stops being comparable), a held-out gene leaking
into training through the OTHER file, or a shared site counted twice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import crosssource as xs  # noqa: E402
from m6a.data import assign_folds  # noqa: E402


def make_sources(n_genes: int = 30, seed: int = 0):
    """Two small files: dataset0 has genes G0..G{n-1}; data1 shares the first
    two thirds of them (same sites) and adds genes of its own."""
    rng = np.random.default_rng(seed)
    rows0 = [(f"T{g}", p, f"G{g}") for g in range(n_genes) for p in range(4)]
    rows1 = ([(f"T{g}", p, f"G{g}") for g in range(2 * n_genes // 3) for p in range(4)]
             + [(f"N{g}", p, f"H{g}") for g in range(10) for p in range(4)])
    sources = {}
    for name, rows in (("dataset0", rows0), ("data1", rows1)):
        index = pd.MultiIndex.from_tuples([(t, p) for t, p, _ in rows], names=xs.KEYS)
        genes = np.array([g for _, _, g in rows], dtype=object)
        X = pd.DataFrame({"f": rng.normal(size=len(rows)),
                          "site": [hash(k) % 10_000 for k in index]}, index=index)
        sources[name] = xs.Source(name, index, {None: X, 1: X}, rng.integers(0, 2, len(rows)),
                                  genes, None)
    fold_of = xs.union_folds(sources["dataset0"].genes, sources["data1"].genes, n_folds=3)
    for s in sources.values():
        s.folds = np.array([fold_of[g] for g in s.genes])
    return sources


def test_dataset0_keeps_its_canonical_folds():
    genes0 = np.array([f"G{i}" for i in range(50)])
    canonical = assign_folds(pd.DataFrame({"gene_id": genes0}), 4262, 5, "gene_id")
    union = xs.union_folds(genes0, np.array(["G1", "G2", "NEW1", "NEW2"]))
    assert [union[g] for g in genes0] == canonical.tolist()
    assert "NEW1" in union and "NEW2" in union


def test_a_gene_is_in_one_fold_in_both_files():
    s = make_sources()
    fold_of = {}
    for src in s.values():
        for g, f in zip(src.genes, src.folds):
            assert fold_of.setdefault(g, f) == f


def test_pooled_weights_count_a_shared_site_once():
    s = make_sources()
    for arm in ("pooled", "pooled_both"):
        X, y, w = xs.training_rows(s, arm, fold=0, depths=[None])
        per_site = pd.Series(w, index=X.index).groupby(level=[0, 1]).sum()
        assert np.allclose(per_site.to_numpy(), 1.0), arm


def test_pooled_both_gives_each_shared_run_copy_both_labels():
    s = make_sources()
    X, y, w = xs.training_rows(s, "pooled_both", fold=0, depths=[None])
    key = s["dataset0"].index[s["dataset0"].folds != 0][0]   # a shared site (T*)
    labels = sorted(y[X.index.get_indexer_for([key])])
    expected = sorted([s["dataset0"].y[s["dataset0"].index.get_loc(key)],
                       s["data1"].y[s["data1"].index.get_loc(key)]] * 2)
    assert labels == expected


class _LeakDetector:
    """Scores 1 for any site it was trained on, 0 otherwise."""

    def __init__(self, **_):
        self.seen = set()

    def fit(self, X, y, sample_weight=None):
        self.seen = set(X.index)

    def predict_proba(self, X):
        return np.array([float(k in self.seen) for k in X.index])


def test_no_held_out_site_is_trained_on_in_any_arm():
    s = make_sources()
    oof = xs.out_of_fold(s, _LeakDetector, {}, train_depths=[None, 1], log=lambda *a: None)
    for arm, by_file in oof.items():
        for name, scores in by_file.items():
            assert np.nansum(scores) == 0, f"{arm} saw held-out sites of {name}"


def test_identical_models_have_zero_gain_and_pass_the_rule():
    s = make_sources()
    scores = {a: {n: np.random.default_rng(1).random(len(s[n])) for n in xs.SOURCES}
              for a in ("dataset0",)}
    block = xs.summarise(s, scores, scores, n=20)
    row = block["arms"]["dataset0"]
    assert row["gain_mean"] == 0 and row["eligible"]
    assert all(cell["gain"] == 0 for cell in block["crossed"].values())
