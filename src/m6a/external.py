"""Held-out evaluation on the course's later releases, data1 and data2.

Cross-validation on dataset0 answers "is it better on our data". It cannot say
whether an improvement survives a different labelling - and the first time
anyone checked, most of it did not: `everything` beats `quantiles` by +0.065 in
cross-validation and by +0.008 on data1's unseen sites (GAPS.md). So every
standard run now also scores the model it ships on:

**data1** - a second labelled run (docs/data.md). Only sites the model never
trained on count, in two slices:

  new_transcripts  transcripts absent from dataset0. The headline: no site of
                   these transcripts, and so no neighbour label, was seen.
  new_sites        sites absent from dataset0, many on transcripts it trained on.

Its labels disagree with dataset0's on 5.75% of shared sites, so a score here
measures agreement with a *different* labelling. Using dataset0's own labels as
the prediction scores PR AUC 0.326 on the shared sites - context for how low
these numbers run.

**data2** - one synthetic sequence mixed at 0-100% modified molecules. Reported
as the mean score at each fraction and a rank correlation: does the score rise
with the fraction? It is a diagnostic, never a target.

**Report only.** Nothing here gates a comparison; the decision stays with
whoever reads it (docs/decisions/0028).

Uncertainty: one test set, so intervals come from resampling data1's
*transcripts* with replacement - sites on a transcript are correlated, so
resampling sites would understate the error. A comparison uses the same
resamples for both arms, so it is paired.

Not imported by predict.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from m6a.evaluation import metrics

SLICES = ("new_transcripts", "new_sites")
HEADLINE = "new_transcripts"
N_RESAMPLES = 2000
SEED = 4262
FILES = {"data1": ("data1", "dataset1.json.gz"), "data2": ("data2", "dataset2.json.gz")}


def paths(data_dir: str | Path, name: str) -> tuple[Path, Path]:
    folder, signal = FILES[name]
    return Path(data_dir) / folder / signal, Path(data_dir) / folder / "data.info"


def require(data_dir: str | Path) -> None:
    """Fail at the start of a run, not after an hour of fitting."""
    missing = [str(p) for name in FILES for p in paths(data_dir, name) if not p.exists()]
    if missing:
        raise SystemExit(
            "This profile evaluates on data1 and data2, and they are not in "
            f"{data_dir}:\n  " + "\n  ".join(missing) + "\n"
            "Fetch them with:\n"
            "  python scripts/download_data.py --set data1\n"
            "  python scripts/download_data.py --set data2\n"
            "or pass --no-external to skip (the run will then lack the ext/* keys "
            "and cannot be compared with runs that have them)."
        )


def load_info(data_dir: str | Path, name: str) -> pd.DataFrame:
    info = pd.read_csv(paths(data_dir, name)[1])
    return info.set_index(["transcript_id", "transcript_position"])[["label", "n_reads"]]


def score(model, features: str, columns: list[str], json_path: Path, *,
          use_cache: bool = True, log=print) -> pd.Series:
    """The model's score for every site in an external file, indexed like it.

    Features come through the same cache as training (keyed on file content),
    and a read-level model gets the file's reads, exactly as in cross-validation.
    """
    from m6a import feature_cache

    extraction = feature_cache.extract(json_path, features, [None], use_cache=use_cache,
                                       log=log)[None]
    X = extraction.features[columns]
    kwargs = {}
    if getattr(model, "CONSUMES_READS", False):
        # extract_reads returns sites in file order, which is the order extract
        # built its index in - so row i of the blocks is row i of X.
        kwargs["reads"] = feature_cache.extract_reads(json_path, use_cache=use_cache, log=log)
    return pd.Series(model.predict_proba(X, **kwargs), index=X.index, name="score")


def slice_masks(index: pd.MultiIndex, train_index: pd.MultiIndex) -> dict[str, np.ndarray]:
    seen_sites = index.isin(train_index)
    seen_transcripts = index.get_level_values(0).isin(set(train_index.get_level_values(0)))
    return {"new_transcripts": ~seen_transcripts, "new_sites": ~seen_sites}


def resample_indices(clusters: np.ndarray, n: int = N_RESAMPLES, seed: int = SEED):
    """Row indices for `n` cluster-bootstrap resamples: whole transcripts drawn
    with replacement. Deterministic in `seed`, so two arms scored separately
    still get identical resamples."""
    codes, uniques = pd.factorize(clusters)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniques))
    starts = np.concatenate([[0], np.cumsum(counts)])
    rng = np.random.default_rng(seed)
    for _ in range(n):
        picked = rng.integers(0, len(uniques), len(uniques))
        # Concatenate the picked transcripts' row ranges without a Python loop:
        # position j of the output is start[picked cluster] + (j - where that
        # cluster's block begins in the output).
        lengths = counts[picked]
        begin = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        yield order[np.arange(lengths.sum()) + np.repeat(starts[picked] - begin, lengths)]


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def data1_block(scores: pd.Series, info: pd.DataFrame, train_index: pd.MultiIndex,
                n: int = N_RESAMPLES) -> dict:
    scores = scores.reindex(info.index)
    out = {}
    for name, mask in slice_masks(info.index, train_index).items():
        y = info["label"].to_numpy()[mask].astype(int)
        s = scores.to_numpy()[mask]
        boots = [_ap(y[i], s[i]) for i in resample_indices(
            info.index.get_level_values(0)[mask], n)]
        low, high = np.nanpercentile(boots, [2.5, 97.5])
        out[name] = {"n": int(mask.sum()), "n_positive": int(y.sum()),
                     **metrics(y, s), "ci_low": float(low), "ci_high": float(high)}
    return out


def data2_block(scores: pd.Series, info: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr

    scores = scores.reindex(info.index)
    frame = pd.DataFrame({"fraction": info["label"].to_numpy(), "score": scores.to_numpy()})
    extremes = frame[frame["fraction"].isin([0.0, 1.0])]
    return {
        "n": int(len(frame)),
        "mean_score_by_fraction": {float(k): float(v) for k, v in
                                   frame.groupby("fraction")["score"].mean().items()},
        "spearman": float(spearmanr(frame["fraction"], frame["score"]).correlation),
        "roc_auc_100_vs_0": float(metrics((extremes["fraction"] == 1.0).to_numpy().astype(int),
                                          extremes["score"].to_numpy())["roc_auc"]),
    }


def evaluate(model, features: str, columns: list[str], train_index: pd.MultiIndex,
             data_dir: str | Path, *, use_cache: bool = True, log=print) -> dict:
    """Score data1 and data2. Returns the report block, plus the per-site scores
    under `_scores` for a paired comparison - that key never reaches disk."""
    s1 = score(model, features, columns, paths(data_dir, "data1")[0], use_cache=use_cache, log=log)
    s2 = score(model, features, columns, paths(data_dir, "data2")[0], use_cache=use_cache, log=log)
    info1, info2 = load_info(data_dir, "data1"), load_info(data_dir, "data2")
    return {
        "data1": data1_block(s1, info1, train_index),
        "data2": data2_block(s2, info2),
        "_scores": {"data1": s1.reindex(info1.index), "info1": info1, "train_index": train_index},
    }


def compare(baseline: dict, candidate: dict, n: int = N_RESAMPLES) -> dict:
    """Paired difference on data1, candidate minus baseline, per slice.

    Both arms must have been scored on the same data1 sites with the same
    training sites held out, or the difference is not paired - checked, not
    assumed.
    """
    a, b = baseline["_scores"], candidate["_scores"]
    if not a["info1"].index.equals(b["info1"].index) or not a["train_index"].equals(b["train_index"]):
        raise RuntimeError("The two arms were not scored on the same data1 sites.")
    info = a["info1"]
    out = {}
    for name, mask in slice_masks(info.index, a["train_index"]).items():
        y = info["label"].to_numpy()[mask].astype(int)
        sa, sb = a["data1"].to_numpy()[mask], b["data1"].to_numpy()[mask]
        diffs = np.array([_ap(y[i], sb[i]) - _ap(y[i], sa[i]) for i in resample_indices(
            info.index.get_level_values(0)[mask], n)])
        low, high = np.nanpercentile(diffs, [2.5, 97.5])
        out[name] = {
            "baseline_pr_auc": _ap(y, sa), "candidate_pr_auc": _ap(y, sb),
            "mean_difference": _ap(y, sb) - _ap(y, sa),
            "ci_low": float(low), "ci_high": float(high),
            # Share of resamples on which the candidate was better - the
            # bootstrap's analogue of the win count.
            "win_rate": float(np.nanmean(diffs > 0)),
        }
    return out


def strip(block: dict) -> dict:
    """The block without its in-memory per-site scores, for JSON and W&B."""
    return {k: v for k, v in block.items() if not k.startswith("_")}
