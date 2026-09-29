"""Guardrails for the held-out data1/data2 evaluation (src/m6a/external.py).

The failures these pin are all silent ones: a slice that quietly includes sites
the model trained on, a bootstrap that splits a transcript across resamples, a
paired comparison whose two arms were not scored on the same sites. Each would
produce a plausible number and no warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import external  # noqa: E402
from m6a.tracking import flat_metrics  # noqa: E402


def index(pairs):
    return pd.MultiIndex.from_tuples(pairs, names=["transcript_id", "transcript_position"])


def test_slices_exclude_what_they_claim_to():
    train = index([("T1", 10), ("T1", 20), ("T2", 5)])
    train_genes = {"G1", "G2"}
    data1 = index([("T1", 10), ("T1", 30), ("T2", 5), ("T3", 1), ("T3", 2),
                   ("T4", 7), ("T5", 9)])
    # T3 is a new transcript of training gene G1 - the leak new_genes exists to
    # exclude. T4 is a genuinely new gene. T5 could not be mapped to any gene.
    genes = ["G1", "G1", "G2", "G1", "G1", "G9", ""]
    masks = external.slice_masks(data1, train, genes, train_genes)
    # new_sites: not a training site. T1:30 is new even though T1 was trained on.
    assert masks["new_sites"].tolist() == [False, True, False, True, True, True, True]
    # new_transcripts: no site of the transcript was trained on.
    assert masks["new_transcripts"].tolist() == [False, False, False, True, True, True, True]
    # new_genes: no transcript of the GENE was trained on, and the gene is known.
    assert masks["new_genes"].tolist() == [False, False, False, False, False, True, False]


def test_resamples_are_whole_transcripts_and_deterministic():
    clusters = np.array(["a", "a", "a", "b", "c", "c"])
    first = list(external.resample_indices(clusters, n=20, seed=1))
    again = list(external.resample_indices(clusters, n=20, seed=1))
    assert all(np.array_equal(x, y) for x, y in zip(first, again))
    sizes = {"a": 3, "b": 1, "c": 2}
    for rows in first:
        drawn = pd.Series(clusters[rows]).value_counts()
        # A transcript is drawn whole: its row count is a multiple of its size.
        assert all(count % sizes[c] == 0 for c, count in drawn.items())
        assert sum(count // sizes[c] for c, count in drawn.items()) == 3


def _block(scores, labels, idx, train_index):
    info = pd.DataFrame({"label": labels, "n_reads": 30}, index=idx)
    genes = np.array([f"G{t}" for t in idx.get_level_values(0)], dtype=object)
    return {"_scores": {"data1": pd.Series(scores, index=idx), "info1": info,
                        "train_index": train_index, "genes": genes,
                        "train_genes": {"Gtrain"}}}


def test_identical_arms_differ_by_exactly_zero():
    rng = np.random.default_rng(0)
    idx = index([(f"T{i // 5}", i) for i in range(400)])
    labels = (rng.random(400) < 0.2).astype(int)
    scores = rng.random(400) + labels * 0.3
    arm = _block(scores, labels, idx, index([("X", 0)]))
    result = external.compare(arm, arm, n=50)
    for row in result.values():
        assert row["mean_difference"] == 0
        assert row["ci_low"] == 0 and row["ci_high"] == 0


def test_arms_scored_on_different_sites_are_refused():
    idx_a = index([("T1", 1), ("T2", 2)])
    idx_b = index([("T1", 1), ("T3", 3)])
    a = _block([0.1, 0.9], [0, 1], idx_a, index([("X", 0)]))
    b = _block([0.1, 0.9], [0, 1], idx_b, index([("X", 0)]))
    with pytest.raises(RuntimeError):
        external.compare(a, b, n=5)


def test_data2_block_reads_a_perfectly_ordered_score_as_rank_one():
    fractions = np.repeat([0.0, 0.25, 0.5, 1.0], 10)
    idx = index([(f"tx{i // 10}", i) for i in range(40)])
    info = pd.DataFrame({"label": fractions, "n_reads": 100}, index=idx)
    block = external.data2_block(pd.Series(fractions + 0.01, index=idx), info)
    assert block["spearman"] == pytest.approx(1.0)
    assert block["roc_auc_100_vs_0"] == pytest.approx(1.0)
    assert block["mean_score_by_fraction"][0.5] == pytest.approx(0.51)


def test_flat_keys_for_the_held_out_block():
    row = {"n": 100, "n_positive": 8, "pr_auc": 0.36, "roc_auc": 0.83, "pr_auc_lift": 4.5,
           "ci_low": 0.30, "ci_high": 0.42}
    report = {
        "external": {"data1": {"new_transcripts": row, "new_sites": row},
                     "data2": {"spearman": 0.2, "roc_auc_100_vs_0": 0.7,
                               "mean_score_by_fraction": {0.0: 0.1, 0.25: 0.3, 1.0: 0.4}}},
        "external_comparisons": {"features": {"new_transcripts": {
            "mean_difference": 0.01, "ci_low": -0.01, "ci_high": 0.03, "win_rate": 0.8}}},
    }
    flat = flat_metrics(report)
    assert flat["ext/data1/new_transcripts/pr_auc"] == 0.36
    assert flat["ext/data2/mean_score/25pct"] == 0.3
    assert flat["ext/data2/mean_score/100pct"] == 0.4
    assert flat["ext_compare/features/new_transcripts/win_rate"] == 0.8


def test_the_bootstrap_resamples_a_gene_together():
    idx = index([("T1", 1), ("T2", 2), ("T3", 3)])
    # T1 and T2 are one gene; T3 has no known gene and stands alone.
    clusters = external.clusters_for(idx, np.array(["G1", "G1", ""], dtype=object))
    assert clusters.tolist() == ["G1", "G1", "T3"]


def test_missing_data_fails_with_the_command_that_fixes_it(tmp_path):
    with pytest.raises(SystemExit, match="download_data.py --set data1"):
        external.require(tmp_path)
