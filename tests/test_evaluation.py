"""Guardrails for the evaluation harness.

These protect three things that are easy to break and hard to notice:

1. Read subsampling is reproducible and never invents data. Every depth number
   in GAPS.md rests on that.
2. The feature cache returns exactly what recomputing would have returned. A
   cache that is subtly wrong produces plausible numbers, which is the worst
   kind of wrong.
3. The paired comparison actually beats the unpaired one. That is the whole
   reason m6a.compare exists, and it is worth a test that fails if someone
   "simplifies" it back.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_smoke import make_fake_dataset  # noqa: E402

from m6a import crossval, feature_cache, registry  # noqa: E402
from m6a.compare import paired_comparison  # noqa: E402
from m6a.data import Site, iter_sites, subsample_reads  # noqa: E402
from m6a.evaluation import (  # noqa: E402
    calibration_summary,
    calibration_table,
    depth_bands,
    expected_calibration_error,
    metrics,
    metrics_by,
)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A small dataset plus a cache directory of its own."""
    monkeypatch.setenv("M6A_CACHE_DIR", str(tmp_path / "cache"))
    json_path, labels_path = make_fake_dataset(tmp_path, n_sites=360, seed=7)
    return json_path, labels_path


# --------------------------------------------------------------------------
# read subsampling
# --------------------------------------------------------------------------

def a_site(n_reads: int = 40) -> Site:
    reads = np.arange(n_reads * 9, dtype=np.float32).reshape(n_reads, 9)
    return Site("ENST00000000233", 244, "AAGACCA", reads)


def test_subsampling_keeps_real_reads_and_never_invents_any():
    site = a_site(40)
    kept = subsample_reads(site, 3)
    assert kept.n_reads == 3
    original = {tuple(row) for row in site.reads}
    assert all(tuple(row) in original for row in kept.reads)
    assert len({tuple(row) for row in kept.reads}) == 3, "drawn without replacement"


def test_subsampling_is_a_no_op_when_the_site_is_already_shallow():
    site = a_site(3)
    assert subsample_reads(site, 10) is site
    assert subsample_reads(site, 3) is site


def test_subsampling_does_not_depend_on_iteration_order_or_the_depth_list():
    """The reason the draw is keyed rather than taken from one RNG stream.

    A shared stream gives reproducible results only as long as nobody edits the
    depth list or the order sites are visited in. Neither is a property worth
    resting the depth sweep on.
    """
    sites = [Site(f"ENST{i:011d}", 100 + i, "AAGACCA", a_site(30).reads) for i in range(5)]

    forwards = {s.key: subsample_reads(s, 3).reads.tobytes() for s in sites}
    backwards = {s.key: subsample_reads(s, 3).reads.tobytes() for s in reversed(sites)}
    assert forwards == backwards

    # ...and asking for more depths does not shift the ones already asked for.
    interleaved = {}
    for s in sites:
        for depth in (1, 3, 5, 10):
            if depth == 3:
                interleaved[s.key] = subsample_reads(s, depth).reads.tobytes()
    assert forwards == interleaved


def test_a_different_subsample_seed_draws_different_reads():
    site = a_site(40)
    assert not np.array_equal(
        subsample_reads(site, 5, seed=4262).reads,
        subsample_reads(site, 5, seed=1).reads,
    )


def test_subsampling_rejects_a_depth_below_one():
    with pytest.raises(ValueError):
        subsample_reads(a_site(), 0)


# --------------------------------------------------------------------------
# the feature cache
# --------------------------------------------------------------------------

def test_cache_returns_exactly_what_recomputing_would_have(fake):
    json_path, _ = fake

    cold = feature_cache.extract(json_path, "quantiles_v1", [None, 3], log=lambda *_: None)
    warm = feature_cache.extract(json_path, "quantiles_v1", [None, 3], log=lambda *_: None)
    assert not cold[3].from_cache and warm[3].from_cache

    for depth in (None, 3):
        assert np.array_equal(cold[depth].features.to_numpy(), warm[depth].features.to_numpy())
        assert cold[depth].columns == warm[depth].columns
        assert cold[depth].features.index.equals(warm[depth].features.index)

    # ...and what predict.py's own path would have produced, which is the one
    # that actually has to agree.
    direct = registry.get("features", "quantiles_v1")().transform(iter_sites(json_path))
    assert list(direct.columns) == cold[None].columns
    assert direct.index.equals(cold[None].features.index)
    assert np.allclose(direct.to_numpy(), cold[None].features.to_numpy())


def test_cache_entries_are_keyed_separately_per_depth_and_feature_set(fake):
    json_path, _ = fake
    feature_cache.extract(json_path, "quantiles_v1", [None, 1, 3], log=lambda *_: None)
    feature_cache.extract(json_path, "pooled_v1", [None], log=lambda *_: None)
    assert len(list(feature_cache.cache_dir().glob("*.npz"))) == 4


def test_changing_the_input_file_invalidates_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("M6A_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, _ = make_fake_dataset(tmp_path / "a", seed=1)
    second, _ = make_fake_dataset(tmp_path / "b", seed=2)

    a = feature_cache.extract(first, "pooled_v1", log=lambda *_: None)[None]
    b = feature_cache.extract(second, "pooled_v1", log=lambda *_: None)[None]
    assert not b.from_cache
    assert not np.allclose(a.features.to_numpy(), b.features.to_numpy())


def test_subsampled_features_differ_from_full_depth_ones(fake):
    json_path, _ = fake
    out = feature_cache.extract(json_path, "quantiles_v1", [None, 1], log=lambda *_: None)
    assert not np.allclose(out[None].features.to_numpy(), out[1].features.to_numpy())
    # At depth 1 there is no spread between reads to measure.
    assert (out[1].features["mean_0_iqr"] == 0).all()
    # Site metadata reports true depth even in a subsampled entry.
    assert (out[1].sites["n_reads"] == out[None].sites["n_reads"]).all()
    assert (out[1].sites["n_reads"] >= 20).all()


# --------------------------------------------------------------------------
# stratified metrics and calibration
# --------------------------------------------------------------------------

def test_depth_bands_are_ordered_and_cover_the_sgnex_range():
    bands = depth_bands([1, 2, 3, 4, 5, 19, 20, 46, 500])
    assert list(bands) == ["1", "2", "3-4", "3-4", "5-9", "10-19", "20-31", "32-46", "304+"]
    assert bands.ordered


def test_thin_strata_get_a_reason_instead_of_a_number():
    y = np.array([0] * 100 + [1] * 3 + [0] * 100 + [1] * 40)
    scores = np.linspace(0, 1, len(y))
    groups = np.array(["thin"] * 103 + ["thick"] * 140)

    table = metrics_by(y, scores, groups, min_positive=10)
    assert np.isnan(table.loc["thin", "pr_auc"])
    assert "3 positive" in table.loc["thin", "note"]
    assert np.isfinite(table.loc["thick", "pr_auc"])
    assert table.loc["thick", "note"] == ""


def test_calibration_never_silently_drops_sites():
    """A degenerate score distribution must not produce an empty table.

    qcut drops every bin boundary when all scores are identical, which used to
    leave the table empty and report a calibration error of exactly zero for a
    model that is as wrong as it is possible to be.
    """
    y = (np.arange(1000) < 45).astype(int)          # 4.5% positive
    table = calibration_table(y, np.full(1000, 0.5))
    assert len(table) == 1
    assert int(table["n"].sum()) == 1000
    assert expected_calibration_error(y, np.full(1000, 0.5)) == pytest.approx(0.455, abs=1e-3)

    # ...and the ordinary case still bins into deciles covering every site.
    scores = np.linspace(0, 1, 1000)
    assert int(calibration_table(y, scores)["n"].sum()) == 1000


def test_ece_is_blind_to_whether_the_model_is_any_good():
    """The reason ECE must never be used to rank models.

    Predicting the base rate for every site is perfectly calibrated and
    completely useless. PR AUC is the metric that can tell the difference.
    """
    y = (np.arange(1000) < 45).astype(int)
    useless = np.full(1000, y.mean())
    assert expected_calibration_error(y, useless) == pytest.approx(0.0, abs=1e-9)
    assert metrics(y, useless)["pr_auc"] == pytest.approx(y.mean(), abs=0.01)


def test_calibration_summary_measures_the_overcount():
    y = np.array([1] * 100 + [0] * 900)      # 10% positive
    scores = np.full(1000, 0.2)              # ...predicted at 20%
    summary = calibration_summary(y, scores)
    assert summary["count_ratio"] == pytest.approx(2.0)
    assert summary["mean_predicted"] == pytest.approx(0.2)
    assert summary["actual_rate"] == pytest.approx(0.1)
    assert summary["ece"] == pytest.approx(0.1)
    # worse than predicting the base rate everywhere
    assert summary["brier"] > summary["brier_baseline"]


# --------------------------------------------------------------------------
# paired vs unpaired - the point of m6a.compare
# --------------------------------------------------------------------------

def per_fold(values: list[float]) -> list[dict]:
    return [{"fold": i, "pr_auc": v} for i, v in enumerate(values)]


def test_pairing_finds_a_consistent_gain_that_the_unpaired_test_misses():
    """This is the failure GAPS.md recorded, reproduced in miniature.

    Fold difficulty varies far more than the gap between the two models, so the
    unpaired view calls a real and perfectly consistent improvement noise. The
    paired view sees that the candidate wins on every fold.
    """
    baseline = [0.420, 0.510, 0.450, 0.470, 0.440]
    gains = [0.010, 0.014, 0.011, 0.013, 0.012]   # small, and positive every time
    candidate = [b + g for b, g in zip(baseline, gains)]

    result = paired_comparison(per_fold(baseline), per_fold(candidate), "a", "b")
    assert result["wins"] == 5
    assert result["mean_difference"] == pytest.approx(0.012)
    assert result["p_value"] < 0.001, "consistent on every fold"
    assert result["unpaired_p_value"] > 0.5, "swamped by fold-to-fold variation"
    assert result["ci_low"] > 0, "the whole interval is on the better side of zero"


def test_pairing_does_not_bless_an_inconsistent_difference():
    baseline = [0.42, 0.51, 0.45, 0.47, 0.44]
    candidate = [0.46, 0.47, 0.49, 0.43, 0.46]
    result = paired_comparison(per_fold(baseline), per_fold(candidate), "a", "b")
    assert result["p_value"] > 0.05
    assert result["wins"] < 5


def test_an_identical_difference_on_every_fold_is_flagged_rather_than_tested():
    """Degenerate, and worth naming: zero variance is not infinite evidence."""
    baseline = [0.42, 0.51, 0.45, 0.47, 0.44]
    result = paired_comparison(
        per_fold(baseline), per_fold([b + 0.01 for b in baseline]), "a", "b"
    )
    assert result["sd_difference"] == 0
    assert result["p_value"] == 0.0 and np.isinf(result["t_statistic"])


def test_comparing_a_run_with_itself_is_not_significant():
    values = [0.42, 0.51, 0.45, 0.47, 0.44]
    result = paired_comparison(per_fold(values), per_fold(values), "a", "a")
    assert result["mean_difference"] == 0
    assert np.isnan(result["p_value"]), "a zero-variance difference has no t-test"


def test_runs_on_different_folds_cannot_be_paired():
    ok = paired_comparison(
        per_fold([0.42, 0.51, 0.45, 0.47, 0.44]),
        per_fold([0.44, 0.52, 0.46, 0.49, 0.45]),
        "a", "b",
    )
    assert ok["n_folds"] == 5
    with pytest.raises(ValueError, match="not paired"):
        paired_comparison(
            [{"fold": 0, "pr_auc": 0.4}], [{"fold": 3, "pr_auc": 0.5}], "a", "b"
        )


def test_the_predict_path_never_imports_the_comparison_module():
    """scipy is present on most machines only because scikit-learn pulls it in.

    m6a.evaluation is imported by predict.py, which evaluators run from a clean
    clone with the base dependencies and nothing else. Keeping the paired test
    in m6a.compare is what stops the graded path from depending on that
    accident - so check the import graph, not just the intention.
    """
    probe = (
        "import sys; sys.path.insert(0, 'src');"
        "import m6a.data, m6a.evaluation;"
        "from m6a import registry;"
        "registry.available('features'); registry.available('models');"
        "print('m6a.compare' in sys.modules or 'm6a.crossval' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "predict.py's import graph reached the evaluation-only modules"
    )


# --------------------------------------------------------------------------
# cross-validation and the out-of-fold table
# --------------------------------------------------------------------------

def test_the_oof_table_reproduces_the_metrics_it_was_built_from(fake):
    """The table is no longer stored, but every number still comes out of it.

    docs/decisions/0009 dropped `oof.csv`; it did not drop the table. Every
    metric, stratum, calibration figure and plot is computed from the in-memory
    vector before it is discarded, so it still has to agree with the per-fold
    numbers `cross_validate` reports alongside it.
    """
    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )
    result = crossval.cross_validate(
        dataset, registry.get("models", "lightgbm"), {"n_estimators": 30},
        log=lambda *_: None,
    )

    table = result.oof
    pooled = metrics(table["label"].to_numpy(), table["score"].to_numpy())
    assert pooled["pr_auc"] == pytest.approx(result.pooled["pr_auc"], abs=1e-6)
    for fold in result.per_fold:
        part = table[table["fold"] == fold["fold"]]
        assert metrics(part["label"].to_numpy(), part["score"].to_numpy())["pr_auc"] == (
            pytest.approx(fold["pr_auc"], abs=1e-6)
        )

    assert not hasattr(crossval, "save_oof"), (
        "the per-site table is not persisted any more - see docs/decisions/0009"
    )


def test_a_gene_is_never_scored_by_a_model_that_trained_on_it(fake):
    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )
    table = crossval.oof_table(dataset, np.zeros(len(dataset)))
    assert (table.groupby("gene_id")["fold"].nunique() == 1).all()


def test_pairing_refuses_two_runs_that_were_split_differently(fake):
    json_path, labels_path = fake
    a = crossval.build_dataset(json_path, labels_path, "pooled_v1", log=lambda *_: None)
    b = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", seed=1, log=lambda *_: None
    )
    table_a = crossval.oof_table(a, np.zeros(len(a)))
    table_b = crossval.oof_table(b, np.zeros(len(b)))

    crossval.assert_same_folds(table_a, table_a, "a", "a")
    with pytest.raises(SystemExit, match="fold assignment"):
        crossval.assert_same_folds(table_a, table_b, "seed 4262", "seed 1")


def test_column_subsets_split_motif_from_signal():
    columns = ["mean_0_mean", "n_reads", "motif_GGACT", "motif_AAACA"]
    assert crossval.column_subset(columns, "all") == columns
    assert crossval.column_subset(columns, "motif") == ["motif_GGACT", "motif_AAACA"]
    assert crossval.column_subset(columns, "signal") == ["mean_0_mean", "n_reads"]
    with pytest.raises(ValueError):
        crossval.column_subset(["mean_0_mean"], "motif")


def test_datasets_at_different_depths_line_up_row_for_row(fake):
    json_path, labels_path = fake
    built = crossval.build_datasets(
        json_path, labels_path, "pooled_v1", [None, 1, 5], log=lambda *_: None
    )
    reference = built[None]
    for depth, dataset in built.items():
        assert dataset.X.index.equals(reference.X.index)
        assert (dataset.folds == reference.folds).all()
        assert (dataset.y == reference.y).all(), "subsampling must not touch labels"


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_evaluate_runs_the_whole_report(tmp_path, monkeypatch):
    json_path, labels_path = make_fake_dataset(tmp_path, n_sites=360, seed=3)
    out = tmp_path / "reports"

    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py",
         "--config", "configs/lightgbm.yaml",
         "--json", str(json_path), "--labels", str(labels_path),
         "--depth-sweep", "--depths", "1,5,full",
         "--ablate", "--compare-features", "quantiles_v1",
         "--out", str(out), "--min-positive", "5"],
        cwd=ROOT, capture_output=True, text=True,
        env={**dict(__import__("os").environ), "M6A_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((out / "lightgbm_pooled.json").read_text())
    assert len(report["per_fold"]) == 5
    assert {row["depth"] for row in report["depth_sweep"]["rows"]} == {"1", "5", "full"}
    assert report["depth_sweep"]["subsample_seed"] == 4262
    assert set(report["ablations"]) == {"signal", "motif"}
    assert "features" in report["comparisons"]
    assert report["calibration"]["summary"]["count_ratio"] > 0
    assert report["by_depth"] and report["by_motif"]


def test_training_evaluates_inline_and_leaves_no_per_site_table(tmp_path):
    """What a training run leaves behind, after 0007 and 0009.

    Training now runs the evaluation itself, because an instance that is
    terminated before anyone evaluates the model has lost the run. So the report
    has to come out of `train.py` with the depth sweep in it - and the per-site
    score table has to *not* be written, which is the half of 0001 that 0009
    reversed.
    """
    json_path, labels_path = make_fake_dataset(tmp_path, n_sites=360, seed=5)
    model_dir = tmp_path / "model"
    reports = tmp_path / "reports"
    environment = {
        **dict(__import__("os").environ),
        "M6A_CACHE_DIR": str(tmp_path / "cache"),
        "M6A_REPORT_DIR": str(reports),
    }

    train = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "configs/lightgbm.yaml",
         "--json", str(json_path), "--labels", str(labels_path),
         "--out", str(model_dir), "--no-wandb", "--no-cache"],
        cwd=ROOT, capture_output=True, text=True, env=environment,
    )
    assert train.returncode == 0, train.stderr

    meta = json.loads((model_dir / "meta.json").read_text())
    assert len(meta["metrics_per_fold"]) == 5
    assert "oof" not in meta, "the per-site table is gone - see docs/decisions/0009"
    assert not list(model_dir.glob("*.csv")), "no per-site scores may be written to disk"
    # predict.py reads these four and must keep finding them.
    assert all(key in meta for key in ("name", "features", "columns", "model"))

    report = json.loads((reports / "lightgbm_pooled.json").read_text())
    stored = {fold["fold"]: fold["pr_auc"] for fold in meta["metrics_per_fold"]}
    for fold in report["per_fold"]:
        assert fold["pr_auc"] == pytest.approx(stored[fold["fold"]], abs=1e-6)
    # standard is the default profile, so a depth sweep happens without asking.
    assert report["profile"] == "standard"
    assert report["depth_sweep"]["rows"]
