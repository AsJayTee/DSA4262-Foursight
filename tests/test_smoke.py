"""End-to-end guardrails.

These are deliberately few. The one that matters most is
test_predict_path_has_no_heavy_imports: it protects the 5% documentation grade
by making sure the script other students run never acquires a dependency on
wandb, boto3 or torch.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import registry
from m6a.data import assign_folds, iter_sites, load_labels
from m6a.evaluation import metrics, validate_submission

DRACH_7MERS = ["AAGACCA", "GGGACTT", "TGAACAG", "AGGACAA"]


def make_fake_dataset(directory: Path, n_sites: int = 240, seed: int = 0) -> tuple[Path, Path]:
    """A small synthetic dataset with the same shape as the real one.

    Lets the suite run anywhere - CI, a fresh clone - without the 180 MB file.
    A weak signal is injected so a model can actually learn something.
    """
    rng = np.random.default_rng(seed)
    json_path = directory / "fake.json.gz"
    labels_path = directory / "fake.info.labelled"
    rows = []

    with gzip.open(json_path, "wt") as out:
        for i in range(n_sites):
            transcript = f"ENST{i // 6:011d}"
            gene = f"ENSG{i // 12:011d}"
            position = 100 + (i % 6) * 17
            label = int(rng.random() < 0.25)
            n_reads = int(rng.integers(20, 60))

            reads = np.column_stack([
                rng.normal(0.008, 0.003, n_reads),
                rng.normal(4.3, 1.5, n_reads),
                rng.normal(110 + 6 * label, 9, n_reads),  # the signal
                rng.normal(0.008, 0.003, n_reads),
                rng.normal(5.2, 1.8, n_reads),
                rng.normal(111 + 8 * label, 10, n_reads),
                rng.normal(0.007, 0.002, n_reads),
                rng.normal(3.0, 1.2, n_reads),
                rng.normal(86, 5, n_reads),
            ])
            kmer = DRACH_7MERS[i % len(DRACH_7MERS)]
            record = {transcript: {str(position): {kmer: np.round(reads, 5).tolist()}}}
            out.write(json.dumps(record) + "\n")
            rows.append((gene, transcript, position, label))

    pd.DataFrame(rows, columns=["gene_id", "transcript_id", "transcript_position", "label"]).to_csv(
        labels_path, index=False
    )
    return json_path, labels_path


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_registry_finds_the_builtin_implementations():
    assert "pooled_v1" in registry.available("features")
    assert "quantiles_v1" in registry.available("features")
    assert "lightgbm" in registry.available("models")
    assert "logistic" in registry.available("models")


def test_registry_reports_modules_it_could_not_import():
    # A module that fails to import is skipped rather than raised, so the
    # failure has to be visible somewhere or a broken module goes unnoticed.
    assert isinstance(registry.failures("models"), dict)


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------

def test_folds_are_deterministic_and_never_split_a_gene(tmp_path):
    _, labels_path = make_fake_dataset(tmp_path)
    labels = load_labels(labels_path)

    first = assign_folds(labels, seed=4262, n_folds=5)
    second = assign_folds(labels, seed=4262, n_folds=5)
    assert first.equals(second), "same seed must give the same folds, on every machine"

    per_gene = labels.assign(fold=first).groupby("gene_id")["fold"].nunique()
    assert (per_gene == 1).all(), "a gene must never appear in two folds"


def test_a_different_seed_gives_a_different_split(tmp_path):
    _, labels_path = make_fake_dataset(tmp_path)
    labels = load_labels(labels_path)
    assert not assign_folds(labels, seed=4262).equals(assign_folds(labels, seed=1))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_pr_auc_lift_is_one_for_a_random_classifier():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.045).astype(int)
    scores = metrics(y, rng.random(4000))
    assert scores["pr_auc_lift"] == pytest.approx(1.0, abs=0.35)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config", ["configs/baseline.yaml", "configs/lightgbm.yaml"])
def test_train_then_predict_produces_a_valid_submission(tmp_path, config):
    json_path, labels_path = make_fake_dataset(tmp_path)
    model_dir = tmp_path / "model"
    output = tmp_path / "predictions.csv"

    train = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", config,
         "--json", str(json_path), "--labels", str(labels_path),
         "--out", str(model_dir), "--no-wandb"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert train.returncode == 0, train.stderr

    predict = subprocess.run(
        [sys.executable, "scripts/predict.py", "--model", str(model_dir),
         "--input", str(json_path), "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert predict.returncode == 0, predict.stderr

    problems = validate_submission(output, json_path)
    assert problems == [], "\n".join(problems)


def test_validator_catches_a_broken_submission(tmp_path):
    json_path, _ = make_fake_dataset(tmp_path, n_sites=12)
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {"transcript_id": ["ENST99999999999"], "transcript_position": [1], "score": [1.7]}
    ).to_csv(bad, index=False)

    problems = validate_submission(bad, json_path)
    assert any("outside [0, 1]" in p for p in problems)
    assert any("no prediction" in p for p in problems)
    assert any("not in the JSON" in p for p in problems)


def test_predict_path_has_no_heavy_imports():
    """predict.py must run on a clean machine with only the base dependencies.

    Evaluators install what pyproject lists and nothing more. If discovery ever
    pulls torch, boto3 or wandb into the import graph, this catches it here
    rather than on someone else's machine during grading.
    """
    probe = (
        "import sys; sys.path.insert(0, 'src');"
        "import m6a.data, m6a.evaluation;"
        "from m6a import registry;"
        "registry.available('features'); registry.available('models');"
        "print(','.join(m for m in ('wandb','boto3','torch') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"predict.py's import graph pulled in: {leaked}"


@pytest.mark.skipif(
    not (ROOT / "data" / "sample" / "sample.json.gz").exists(),
    reason="data/sample not built yet - run: make sample",
)
def test_committed_sample_is_readable():
    sites = list(iter_sites(ROOT / "data" / "sample" / "sample.json.gz"))
    assert len(sites) > 0
    assert all(site.reads.shape[1] == 9 for site in sites)
    assert all(len(site.motif) == 5 for site in sites)
