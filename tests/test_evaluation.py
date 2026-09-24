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
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_smoke import make_fake_dataset  # noqa: E402

from m6a import crossval, feature_cache, registry  # noqa: E402
from m6a.compare import paired_comparison  # noqa: E402
from m6a.data import Site, iter_sites, subsample_reads  # noqa: E402
from m6a.models.base import BaseModel  # noqa: E402
from m6a.evaluation import (  # noqa: E402
    calibration_summary,
    calibration_table,
    count_swing,
    depth_bands,
    expected_calibration_error,
    metrics,
    metrics_by,
    operating_points,
    threshold_table,
)


def _have_torch() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


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
    bands = depth_bands([1, 2, 3, 4, 5, 19, 20, 46, 500, 700])
    assert list(bands) == [
        "1", "2", "3-4", "3-4", "5-9", "10-19", "20-31", "32-46", "304-599", "600+",
    ]
    assert bands.ordered
    # The top band was split at 600 (docs/decisions/0017) because every model
    # drops sharply above 304 and one open-ended bucket could not say whether
    # that is a cliff or a slide. True depth tops out at 991 on this data.


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
    # M6A_REPORT_DIR is set session-wide in conftest.py so no test can write
    # where a real run writes; this one needs to know where that is.
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


# --------------------------------------------------------------------------
# thresholds (docs/decisions/0019)
# --------------------------------------------------------------------------

def test_the_threshold_table_agrees_with_counting_by_hand():
    """Every row must match a direct `score >= t` count over the raw arrays.

    The implementation is vectorised - one sort, then a searchsorted per
    threshold - because `evaluation.py` may not import sklearn's curve helpers
    (AGENTS.md section 4). That is exactly the kind of code that is subtly wrong
    at the boundaries: an off-by-one in the cumulative sum, or `side="right"`
    instead of `"left"`, produces a table that looks entirely plausible and is
    shifted by one site.

    Ties are included on purpose. Real scores tie constantly - a gradient-boosted
    tree puts thousands of sites on identical leaf values - and a site exactly on
    the threshold is the case the two sides of `>=` disagree about.
    """
    rng = np.random.default_rng(4262)
    y = (rng.random(3000) < 0.045).astype(int)
    score = np.round(rng.beta(2, 20, 3000) + 0.3 * y, 2)  # rounded: many ties

    table = threshold_table(y, score)
    for row in table.itertuples():
        called = score >= row.threshold
        assert int(called.sum()) == row.predicted_positives
        assert int((called & (y == 1)).sum()) == row.true_positives
        if row.predicted_positives:
            assert row.precision == pytest.approx(y[called].mean())
        else:
            # Not 1.0. A model that calls nothing has no precision, and
            # sklearn's convention would put a perfect score in the run table.
            assert np.isnan(row.precision)
        assert row.recall == pytest.approx(int((called & (y == 1)).sum()) / y.sum())


def test_count_matched_picks_the_threshold_that_counts_right():
    """The operating point a Task 2 site count needs, and the swing around it."""
    rng = np.random.default_rng(7)
    y = (rng.random(5000) < 0.05).astype(int)
    score = np.clip(rng.beta(2, 12, 5000) + 0.35 * y, 0, 1)

    table = threshold_table(y, score)
    points = operating_points(table, int(y.sum()))

    assert set(points) == {"f1_max", "count_matched", "half"}
    assert points["half"]["threshold"] == pytest.approx(0.5)

    # No other threshold on the grid may call a count closer to the truth.
    gap = abs(points["count_matched"]["predicted_positives"] - y.sum())
    assert gap <= (table["predicted_positives"] - y.sum()).abs().min()
    # f1_max is the best F1 on the grid, by construction.
    assert points["f1_max"]["f1"] == pytest.approx(table["f1"].max())

    swing = count_swing(table)
    assert swing["low_count"] >= swing["high_count"]  # count falls as the cut rises
    assert swing["ratio"] >= 1.0


def test_thresholds_never_log_a_non_finite_point():
    """Nothing registered as a series may be NaN or infinite.

    `Tracker.log` drops non-finite scalars, but `log_curve_series` calls
    `run.log` directly for every point of a curve and does not - so a NaN here
    reaches W&B and draws as a real measurement at zero. Precision, F1 and
    log10(count) all go undefined at the top of the range once nothing is being
    called, so the series are truncated there rather than padded
    (docs/decisions/0019 section 4).
    """
    from m6a import report as reporting

    rng = np.random.default_rng(11)
    n = 2000
    y = (rng.random(n) < 0.05).astype(int)
    oof = pd.DataFrame({
        "label": y,
        # Deliberately capped below 1.0, so the top of the grid calls nothing.
        "score": np.clip(rng.beta(2, 12, n) + 0.3 * y, 0, 0.9),
    })

    report = reporting.new("quick", subsample_seed=4262, log=lambda *_: None)
    reporting.thresholds(report, oof)

    series = report.data["curves"]["series"]
    assert "curve/threshold/threshold" in series
    for key, values in series.items():
        if not key.startswith("curve/threshold/"):
            continue
        assert values, f"{key} is empty"
        assert all(np.isfinite(v) for v in values), f"{key} logs a non-finite point"

    # The truncated ones stop short of the full grid; the count itself does not.
    grid = len(series["curve/threshold/threshold"])
    assert len(series["curve/threshold/predicted_positives"]) == grid
    assert len(series["curve/threshold/precision"]) < grid


# --------------------------------------------------------------------------
# comparing many runs at once (docs/decisions/0020)
# --------------------------------------------------------------------------

def _compare_runs_module():
    """Load scripts/compare_runs.py by path - `scripts/` is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compare_runs", ROOT / "scripts" / "compare_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_runs_sharing_a_name_do_not_collapse_into_one():
    """W&B run names are not unique, and the figure keys its rows by name.

    Re-running a config gives a second run with the same display name - the
    normal case, not an edge one. Without disambiguation the two land on the
    same dict key and one vanishes from the figure with nothing said, which is
    the silent kind of wrong this harness exists to avoid.
    """
    module = _compare_runs_module()
    runs = [
        {"name": "lightgbm_quantiles", "id": "aaa111"},
        {"name": "lightgbm_quantiles", "id": "bbb222"},
        {"name": "baseline_logistic", "id": "ccc333"},
    ]
    module.label_runs(runs)

    assert len({run["name"] for run in runs}) == 3, "names must be unique after labelling"
    # The shared name gets its id; the unique one is left alone.
    assert runs[0]["name"] == "lightgbm_quantiles (aaa111)"
    assert runs[1]["name"] == "lightgbm_quantiles (bbb222)"
    assert runs[2]["name"] == "baseline_logistic"
    assert [run["wandb_name"] for run in runs] == [
        "lightgbm_quantiles", "lightgbm_quantiles", "baseline_logistic",
    ]


def test_the_diagonal_is_only_drawn_between_comparable_axes():
    """y = x means something only when both axes are the same quantity.

    `oof/pr_auc` against `depth/3/pr_auc` is the depth collapse and the diagonal
    is the claim. Against `calib/calibrated` it would be a line a reader could
    only misread, so it is not drawn.
    """
    module = _compare_runs_module()
    assert module.same_quantity("oof/pr_auc", "depth/3/pr_auc")
    assert module.same_quantity("fold/0/pr_auc", "rep/pr_auc_mean") is False
    assert not module.same_quantity("oof/pr_auc", "calib/calibrated")
    assert not module.same_quantity("oof/pr_auc", "calib/ece")


# --------------------------------------------------------------------------
# the training curve (docs/decisions/0021)
# --------------------------------------------------------------------------

class _PlainModel(BaseModel):
    """A model that does not train iteratively - and cannot accept a validation set.

    `fit` deliberately has no `validation` parameter. That is the normal shape
    for a model in this repo, and it is what makes the guard below meaningful:
    if cross-validation ever hands an eval set to a model that did not ask for
    one, this raises a TypeError instead of ignoring it.
    """

    def fit(self, X, y, groups=None):
        self.n_rows = len(X)

    def predict_proba(self, X):
        return np.linspace(0.1, 0.9, len(X))


class _IterativeModel(BaseModel):
    REPORTS_TRAINING_CURVE = True

    def fit(self, X, y, groups=None, validation=None):
        self.n_rows = len(X)
        self.validation_rows = None if validation is None else len(validation[0])
        self.history = [
            {"iteration": float(i + 1),
             "train_logloss": 1.0 - 0.1 * i,
             "valid_logloss": 1.2 - 0.1 * i,
             "train_pr_auc": 0.1 * i,
             "valid_pr_auc": 0.05 * i + len(X) / 1e6}
            for i in range(4)
        ]

    def predict_proba(self, X):
        return np.linspace(0.1, 0.9, len(X))


def test_a_model_that_does_not_train_iteratively_is_never_handed_a_validation_set(fake):
    """The capability flag is the opt-in, not the `fit` signature.

    Most models cannot say how their fit progressed, and forcing every one of
    them to absorb a parameter so they can ignore it is how an eval set ends up
    silently discarded. `_PlainModel.fit` would raise on an unexpected keyword,
    so this passing is the assertion.
    """
    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )

    plain = crossval.cross_validate(dataset, _PlainModel, log=lambda *_: None)
    assert plain.histories == [[]] * 5, "a plain model contributes no history"

    iterative = crossval.cross_validate(dataset, _IterativeModel, log=lambda *_: None)
    assert all(len(history) == 4 for history in iterative.histories)
    # The held-out rows, and only those: a model handed its own training rows as
    # a validation set would report a curve that means nothing.
    for fold, model in zip(sorted(np.unique(dataset.folds)), iterative.models):
        assert model.validation_rows == int((dataset.folds == fold).sum())


def test_the_booster_reports_no_training_curve(fake):
    """LightGBM must stay out of the training-curve panel (docs/decisions/0024).

    It *does* train iteratively - one boosting round per tree - so the temptation
    to set the flag is real and the comment in lightgbm.py says not to. A
    boosting round is not an epoch, and 600 of one beside 40 of the other on a
    shared x axis is a panel nobody can read.

    This also pins the fit signature: no `validation` parameter at all, so
    handing it one raises instead of being silently ignored.
    """
    import inspect

    model_class = registry.get("models", "lightgbm")
    assert model_class.REPORTS_TRAINING_CURVE is False
    assert "validation" not in inspect.signature(model_class.fit).parameters

    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )
    result = crossval.cross_validate(
        dataset, model_class, {"n_estimators": 20}, log=lambda *_: None
    )
    assert result.histories == [[]] * 5

    from m6a import report as reporting
    from m6a.tracking import flat_metrics

    report = reporting.new("quick", subsample_seed=4262, log=lambda *_: None)
    reporting.training_curve(report, result)
    assert not any(key.startswith("fit/") for key in flat_metrics(report.data))


@pytest.mark.skipif(
    not _have_torch(), reason="torch not installed - pip install -e '.[mil]'"
)
def test_recording_the_training_curve_does_not_change_the_model(fake):
    """Scoring the held-out fold each epoch must cost time and nothing else.

    The curve is computed with the network in eval mode under `no_grad`, so it
    draws no randomness and updates no weights - but that is a property of how
    it is written, not a guarantee of the framework. If it ever perturbed the
    fit, every number a torch model reports would move by a little, which is the
    hardest kind of regression to notice.
    """
    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )
    model_class = registry.get("models", "mlp")
    params = {"epochs": 3, "hidden": [8], "batch_size": 64}

    without = crossval.cross_validate(
        dataset, model_class, params, record_history=False, log=lambda *_: None
    )
    with_curve = crossval.cross_validate(
        dataset, model_class, params, record_history=True, log=lambda *_: None
    )

    assert without.histories == [[]] * 5
    assert all(len(history) == 3 for history in with_curve.histories)
    np.testing.assert_allclose(
        without.oof["score"].to_numpy(), with_curve.oof["score"].to_numpy(), rtol=0, atol=0
    )


def test_the_training_curve_is_the_mean_over_the_canonical_splits_folds():
    """One curve per run, meaned over folds - and it is repetition 0's.

    Five curves on one panel would be unreadable and would not overlay against
    another run, which is the whole reason it is logged as a series at all
    (docs/decisions/0016).
    """
    from m6a import report as reporting

    histories = [
        [{"iteration": 1.0, "train_logloss": 0.5, "valid_logloss": 0.6,
          "train_pr_auc": 0.30, "valid_pr_auc": 0.20},
         {"iteration": 2.0, "train_logloss": 0.4, "valid_logloss": 0.5,
          "train_pr_auc": 0.40, "valid_pr_auc": 0.40}],
        [{"iteration": 1.0, "train_logloss": 0.7, "valid_logloss": 0.8,
          "train_pr_auc": 0.50, "valid_pr_auc": 0.30},
         {"iteration": 2.0, "train_logloss": 0.6, "valid_logloss": 0.9,
          "train_pr_auc": 0.60, "valid_pr_auc": 0.20}],
    ]
    result = crossval.CVResult(
        oof=pd.DataFrame(), per_fold=[], pooled={}, histories=histories
    )
    report = reporting.new("quick", subsample_seed=4262, log=lambda *_: None)
    reporting.training_curve(report, result, name="probe")

    series = report.data["curves"]["series"]
    assert series["curve/train/iteration"] == [1.0, 2.0]
    assert series["curve/train/valid_pr_auc"] == pytest.approx([0.25, 0.30])
    assert series["curve/train/train_pr_auc"] == pytest.approx([0.40, 0.50])
    assert series["curve/train/valid_pr_auc_sd"] == pytest.approx(
        [np.std([0.2, 0.3], ddof=1), np.std([0.4, 0.2], ddof=1)]
    )

    summary = report.data["training_curve"]
    # The two objectives disagree, which is the point of logging both: held-out
    # PR AUC is best at iteration 2 and held-out logloss at iteration 1.
    assert summary["best_iteration"] == 2
    assert summary["logloss_best_iteration"] == 1
    assert summary["n_folds"] == 2

    from m6a.tracking import flat_metrics

    flat = flat_metrics(report.data)
    assert flat["fit/best_iteration"] == 2
    assert flat["fit/n_iterations"] == 2


# --------------------------------------------------------------------------
# read-level data (docs/decisions/0023)
# --------------------------------------------------------------------------

def test_read_blocks_hold_exactly_the_reads_the_sites_had():
    """The ragged structure is only useful if it is lossless, and `take` reorders.

    `ReadBlocks` carries no site ids - its whole contract is "row i here is row
    i there" - so the fold split and the label join both go through `take`, and
    a `take` that dropped or reordered reads would be undetectable downstream.
    """
    from m6a.data import read_blocks

    sites = [a_site(n) for n in (20, 47, 3, 91)]
    for i, site in enumerate(sites):
        site.transcript_id = f"ENST{i:011d}"

    blocks = read_blocks(sites)
    assert len(blocks) == 4
    assert blocks.counts.tolist() == [20, 47, 3, 91]
    assert blocks.total_reads == 161
    for i, site in enumerate(sites):
        np.testing.assert_array_equal(blocks.site(i), site.reads)

    picked = blocks.take(np.array([3, 0]))
    assert picked.counts.tolist() == [91, 20]
    np.testing.assert_array_equal(picked.site(0), sites[3].reads)
    np.testing.assert_array_equal(picked.site(1), sites[0].reads)

    masked = blocks.take(np.array([False, True, False, True]))
    assert masked.counts.tolist() == [47, 91]
    with pytest.raises(IndexError):
        blocks.take(np.array([9]))


def test_subsampling_blocks_matches_subsampling_sites():
    """One keyed draw, two call sites, and they must not drift apart.

    docs/decisions/0003 put the draw in `data.py` so there would be exactly one
    implementation. A ragged-array version that re-derived the key would be a
    second one, and the two would disagree the first time either was touched.
    """
    from m6a.data import read_blocks, subsample_blocks

    sites = [
        Site(f"ENST{i:011d}", 100 + i, "AAGACCA",
             np.arange(n * 9, dtype=np.float32).reshape(n, 9))
        for i, n in enumerate((40, 25, 2, 60))
    ]
    blocks = read_blocks(sites)
    keys = [(s.transcript_id, s.position) for s in sites]

    for depth in (1, 3, 10):
        thinned = subsample_blocks(blocks, keys, depth)
        for i, site in enumerate(sites):
            np.testing.assert_array_equal(
                thinned.site(i), subsample_reads(site, depth).reads
            )
    # A site already at or below the depth is untouched, here as there.
    assert subsample_blocks(blocks, keys, 10).counts.tolist() == [10, 10, 2, 10]

    with pytest.raises(ValueError, match="out of step"):
        subsample_blocks(blocks, keys[:2], 3)


def test_reads_follow_the_label_join_and_are_checked_against_the_site_table(fake):
    """The join is inner and can reorder, so the reads are reindexed, not assumed."""
    json_path, labels_path = fake
    built = crossval.build_datasets(
        json_path, labels_path, "pooled_v1", [None, 3],
        with_reads=True, log=lambda *_: None,
    )
    full, thin = built[None], built[3]

    assert full.reads is not None and len(full.reads) == len(full)
    # `sites.n_reads` is the true depth recorded during extraction. If the reads
    # had not followed the join, these would disagree - which is exactly the
    # check build_datasets makes, and this asserts it means something.
    np.testing.assert_array_equal(
        full.reads.counts, full.sites["n_reads"].to_numpy()
    )
    # The subsampled dataset shares the row order and carries thinned reads.
    assert thin.X.index.equals(full.X.index)
    assert (thin.reads.counts <= 3).all()
    np.testing.assert_array_equal(
        thin.reads.counts, np.minimum(full.reads.counts, 3)
    )

    # Without with_reads, nothing is loaded and nothing pays for it.
    plain = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )
    assert plain.reads is None


def test_a_read_level_model_refuses_a_dataset_with_no_reads(fake):
    """Loud, not silent. A MIL model quietly scoring on site features alone
    looks exactly like a MIL model that does not work, and the whole question
    being asked is which of those is true.
    """
    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )

    class _NeedsReads(BaseModel):
        CONSUMES_READS = True

        def fit(self, X, y, groups=None, validation=None, reads=None):
            assert reads is not None

        def predict_proba(self, X, reads=None):
            return np.full(len(X), 0.5)

    with pytest.raises(RuntimeError, match="with_reads=True"):
        crossval.cross_validate(dataset, _NeedsReads, log=lambda *_: None)

    with_reads = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", with_reads=True, log=lambda *_: None
    )
    result = crossval.cross_validate(with_reads, _NeedsReads, log=lambda *_: None)
    assert len(result.per_fold) == 5


# --------------------------------------------------------------------------
# depth-augmented training (docs/decisions/0022)
# --------------------------------------------------------------------------

def test_stacked_training_rows_are_the_same_sites_at_several_depths(fake):
    """Five depths is five copies of every site, not five times the evidence.

    The labels are tiled rather than recomputed, because they come from
    m6ACE-Seq and not from the reads (docs/decisions/0003) - so a site that is
    positive at full depth is positive at depth 1 with less to go on.
    """
    json_path, labels_path = fake
    built = crossval.build_datasets(
        json_path, labels_path, "pooled_v1", [None, 1, 5], log=lambda *_: None
    )
    sources = [built[None], built[1], built[5]]
    columns = built[None].columns

    holdout = built[None].folds == 0
    X, y, reads = crossval.stack_rows(sources, columns, ~holdout)

    assert reads is None, "no source carries reads, so neither does the stack"
    assert len(X) == 3 * int((~holdout).sum())
    assert len(y) == len(X)
    np.testing.assert_array_equal(y, np.tile(built[None].y[~holdout], 3))
    # No held-out site is anywhere in the training rows, at any depth. That is
    # the guarantee the whole harness rests on and stacking is where it would
    # quietly break.
    held_out_sites = set(built[None].X.index[holdout])
    assert not held_out_sites & set(X.index)
    # The first block is the full-depth rows, unchanged.
    pd.testing.assert_frame_equal(
        X.iloc[: int((~holdout).sum())], built[None].X.loc[~holdout, columns]
    )


def test_one_training_set_is_indistinguishable_from_no_stacking(fake):
    """The default path has to be the old path exactly, or every number moves."""
    json_path, labels_path = fake
    dataset = crossval.build_dataset(
        json_path, labels_path, "pooled_v1", log=lambda *_: None
    )
    model_class = registry.get("models", "lightgbm")
    params = {"n_estimators": 30}

    plain = crossval.cross_validate(dataset, model_class, params, log=lambda *_: None)
    explicit = crossval.cross_validate(
        dataset, model_class, params, train_on=[dataset], log=lambda *_: None
    )
    np.testing.assert_array_equal(
        plain.oof["score"].to_numpy(), explicit.oof["score"].to_numpy()
    )


def test_training_rows_in_a_different_order_are_refused(fake):
    """The fold mask is positional, so a misaligned training set is a silent lie.

    It would not raise and it would not look wrong - it would train on the
    wrong sites and report a plausible number, which is the failure mode this
    repo spends most of its guards on.
    """
    json_path, labels_path = fake
    built = crossval.build_datasets(
        json_path, labels_path, "pooled_v1", [None, 1], log=lambda *_: None
    )
    shuffled = crossval.Dataset(
        X=built[1].X.iloc[::-1],
        y=built[1].y[::-1],
        folds=built[1].folds[::-1],
        sites=built[1].sites.iloc[::-1],
        columns=built[1].columns,
        depth=1,
    )
    with pytest.raises(RuntimeError, match="same order"):
        crossval.cross_validate(
            built[None], registry.get("models", "lightgbm"), {"n_estimators": 5},
            train_on=[built[None], shuffled], log=lambda *_: None,
        )


def test_train_depths_are_parsed_and_checked():
    """`null` is full depth, and anything that is not a read count is refused."""
    from m6a.config import describe_train_depths, parse_train_depths

    assert parse_train_depths(None) == [None]
    assert parse_train_depths([1, 3, None]) == [1, 3, None]
    assert parse_train_depths(["full", 5, 5]) == [None, 5], "duplicates collapse"
    assert describe_train_depths([1, 3, None]) == "1,3,full"

    with pytest.raises(ValueError, match="below 1"):
        parse_train_depths([0])
    with pytest.raises(ValueError, match="not a read count"):
        parse_train_depths(["shallow"])
    with pytest.raises(ValueError, match="non-empty list"):
        parse_train_depths([])


def test_a_model_with_no_curve_logs_no_fit_keys():
    """A run that cannot report a curve must leave the keys absent, not zero.

    `fit/best_iteration = 0` in the run table would read as "the fit peaked
    immediately", which is a claim; a blank reads as "this model does not train
    in steps", which is the truth.
    """
    from m6a import report as reporting
    from m6a.tracking import flat_metrics

    result = crossval.CVResult(oof=pd.DataFrame(), per_fold=[], pooled={}, histories=[[]])
    report = reporting.new("quick", subsample_seed=4262, log=lambda *_: None)
    reporting.training_curve(report, result)

    assert "training_curve" not in report.data
    assert not any(key.startswith("fit/") for key in flat_metrics(report.data))
