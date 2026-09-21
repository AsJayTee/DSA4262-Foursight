"""Comparing two runs against each other.

Separate from `m6a.evaluation` on purpose. This module imports scipy, and
predict.py imports `m6a.evaluation`. scipy is installed on most machines only
because scikit-learn happens to pull it in, and the script other students run
from a clean clone must not depend on that accident - see AGENTS.md section 4.
Nothing in the graded prediction path imports this file.

**Why paired.** Both runs are scored on the same five folds, so fold difficulty
is common to both and cancels in the difference. Comparing two pooled numbers
against the fold-to-fold standard deviation instead throws that away: it asks
whether the two runs' *levels* are distinguishable given how much folds vary,
which is a much weaker question and the wrong one. On this repo's data it was
also wrong in practice - it talked the team out of a real +0.0145 improvement
that wins on 5 folds out of 5.

**The honest caveat**: five folds is five paired observations, and they are not
independent - each fold's model trains on the other four, so 73.4% of fold 0's
training rows are also in fold 1's. The naive t-test treats them as independent
and returns a p-value smaller than the evidence supports.

So every comparison here reports **three** numbers, and the third is the one to
quote:

    unpaired  (wrong)       ignores that both runs saw the same folds
    paired    (optimistic)  the naive paired t-test
    corrected (quote this)  Nadeau & Bengio, variance inflated for the overlap

The corrected value is always the largest. On `quantiles_v1` vs `pooled_v1` the
naive test gives p = 0.0465 over five folds and the corrected test gives
0.1305 - the effect wins 5 folds out of 5 and is still not established at that
sample size. `repeated_cross_validate` is the way out: ten repetitions gives 50
paired observations, and the same comparison comes out at +0.0164, 50/50 wins,
corrected p = 0.000095.

Read the win count alongside whichever p-value you quote. A difference that is
positive on every observation does not depend on the scatter estimate at all,
and one that flips sign is worth much less than its p-value suggests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "m6a.compare needs scipy for the paired t-test. Install the training "
        "extras:  pip install -e '.[train]'\n"
        "It is kept out of m6a.evaluation deliberately, so that predict.py never "
        "depends on scipy being present."
    ) from exc


def _keyed(observations: list[dict], metric: str) -> dict[tuple[int, int], float]:
    """Metric values keyed by (repetition, fold).

    A plain 5-fold run has no `repetition` field and is repetition 0, so the two
    shapes pair against each other without any special case at the call site.
    """
    out: dict[tuple[int, int], float] = {}
    for row in observations:
        key = (int(row.get("repetition", 0)), int(row["fold"]))
        if key in out:
            raise ValueError(
                f"Repetition {key[0]} fold {key[1]} appears twice in one run's "
                "metrics; observations must be unique."
            )
        out[key] = float(row[metric])
    return out


def _align(
    baseline: list[dict], candidate: list[dict], metric: str
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """The observations the two runs have in common, in a fixed order.

    Repetitions are intersected: a `--repeats 5` run and a `--repeats 10` run
    share their first five repetitions exactly (the seed chain is stable in its
    length - docs/decisions/0012), so pairing them on those five is valid and
    refusing would throw away a comparison we can actually make.

    Folds are **not** intersected. Two runs that disagree about the folds inside
    a repetition did not see the same split, and pairing them on the overlap
    would produce a plausible number for a comparison nobody made.
    """
    left, right = _keyed(baseline, metric), _keyed(candidate, metric)
    reps_a = {rep for rep, _ in left}
    reps_b = {rep for rep, _ in right}
    shared = sorted(reps_a & reps_b)
    if not shared:
        raise ValueError(
            f"Fold ids differ ({sorted(left)} vs {sorted(right)}); "
            "these runs are not paired."
        )

    keys: list[tuple[int, int]] = []
    for rep in shared:
        folds_a = sorted(fold for r, fold in left if r == rep)
        folds_b = sorted(fold for r, fold in right if r == rep)
        if folds_a != folds_b:
            raise ValueError(
                f"Repetition {rep} has folds {folds_a} in one run and {folds_b} "
                "in the other; these runs are not paired. Both have to use the "
                "same split - seed 4262, grouped by gene_id (AGENTS.md section 3)."
            )
        keys.extend((rep, fold) for fold in folds_a)

    return (
        keys,
        np.array([left[k] for k in keys], dtype=float),
        np.array([right[k] for k in keys], dtype=float),
    )


def corrected_variance(diff: np.ndarray, n_folds: int) -> float:
    """Nadeau & Bengio's variance for the mean of correlated CV differences.

    The naive paired t-test divides the sample variance by `n`, which assumes the
    `n` differences are `n` independent observations. In k-fold cross-validation
    they are not: each fold's model trains on the other k-1, so any two training
    sets overlap heavily - measured on this data, 73.4% of fold 0's training rows
    are also in fold 1's. The models are near-copies, they agree with each other
    more than independent experiments would, the observed scatter understates the
    true uncertainty, and the p-value comes out too small.

    The correction inflates the variance by the ratio of test-set size to
    training-set size, which for k-fold CV is 1/(k-1):

        var(mean) = var(d) * (1/n + 1/(k-1))

    rather than `var(d)/n`. Note what that means with many observations: at
    k = 5 the correction floors the variance at `var(d)/4` no matter how many
    repetitions you run, because repeating a split does not make the training
    sets stop overlapping. Fifty splits of one dataset are still one dataset.

    Nadeau & Bengio (2003), *Inference for the Generalization Error*; the
    underlying problem is Dietterich (1998).
    """
    if n_folds < 2:
        raise ValueError(
            f"The corrected t-test needs at least 2 folds to know the "
            f"train/test ratio, got {n_folds}."
        )
    n = len(diff)
    return float(diff.var(ddof=1) * (1.0 / n + 1.0 / (n_folds - 1)))


def paired_comparison(
    baseline: list[dict],
    candidate: list[dict],
    name_baseline: str = "baseline",
    name_candidate: str = "candidate",
    metric: str = "pr_auc",
    n_folds: int | None = None,
) -> dict:
    """Compare two runs' metrics, paired by (repetition, fold).

    `baseline` and `candidate` are metric dicts as `cross_validate` or
    `repeated_cross_validate` produce them. The returned dict carries three
    framings: the **corrected** paired test, which is the one to quote; the
    uncorrected paired test, kept so numbers recorded before the correction
    existed stay traceable; and the unpaired one, so the gap between paired and
    unpaired is visible rather than asserted.

    `n_folds` is the k of the cross-validation, which is what sets the
    train/test overlap the correction accounts for. It defaults to the number of
    folds in one repetition, which is right unless you are passing something
    other than CV results.
    """
    keys, a, b = _align(baseline, candidate, metric)
    folds_a = [fold for _, fold in keys]
    repetitions = sorted({rep for rep, _ in keys})
    if n_folds is None:
        n_folds = len([fold for rep, fold in keys if rep == repetitions[0]])

    diff = b - a
    n = len(diff)

    result = {
        "metric": metric,
        # Kept named n_folds for the recorded numbers that already quote it. It
        # is the observation count, which equals the fold count only when there
        # is one repetition; `cv_folds` is the k the correction needs.
        "n_folds": n,
        "cv_folds": int(n_folds),
        "n_repeats": len(repetitions),
        "repetitions": repetitions,
        "baseline": name_baseline,
        "candidate": name_candidate,
        "folds": folds_a,
        "observations": [list(k) for k in keys],
        "baseline_per_fold": a.tolist(),
        "candidate_per_fold": b.tolist(),
        "difference_per_fold": diff.tolist(),
        "baseline_mean": float(a.mean()),
        "candidate_mean": float(b.mean()),
        "baseline_sd": float(a.std(ddof=1)) if n > 1 else float("nan"),
        "candidate_sd": float(b.std(ddof=1)) if n > 1 else float("nan"),
        "mean_difference": float(diff.mean()),
        "wins": int((diff > 0).sum()),
    }

    if n < 2 or np.all(diff == 0):
        # The same run twice, usually. There is no difference to test, which is
        # not the same as a difference that failed a test.
        result.update(
            sd_difference=0.0, t_statistic=float("nan"), p_value=float("nan"),
            ci_low=float(diff.mean()), ci_high=float(diff.mean()),
            unpaired_p_value=float("nan"),
            t_statistic_corrected=float("nan"), p_value_corrected=float("nan"),
            ci_low_corrected=float(diff.mean()), ci_high_corrected=float(diff.mean()),
        )
        return result

    sd = float(diff.std(ddof=1))
    if sd == 0:
        # A non-zero difference that is identical on every fold. scipy returns
        # t = inf with a warning; say so directly instead. Real per-fold
        # differences never land here, so this is a guard, not a result to
        # quote: p = 0 reflects the degeneracy, not overwhelming evidence.
        result.update(
            sd_difference=0.0,
            t_statistic=float("inf") if diff.mean() > 0 else float("-inf"),
            p_value=0.0,
            ci_low=float(diff.mean()), ci_high=float(diff.mean()),
            unpaired_p_value=float(stats.ttest_ind(b, a, equal_var=False).pvalue),
            # Zero variance survives the correction as zero variance, so the
            # corrected test is just as degenerate. Say so rather than inventing
            # a finite p-value that looks like a result.
            t_statistic_corrected=float("inf") if diff.mean() > 0 else float("-inf"),
            p_value_corrected=0.0,
            ci_low_corrected=float(diff.mean()), ci_high_corrected=float(diff.mean()),
        )
        return result

    t_stat, p_value = stats.ttest_rel(b, a)
    half_width = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)

    # The corrected test is the one to quote. Same differences, same degrees of
    # freedom, a variance that admits the training sets overlap - so its p-value
    # is always the larger of the two, and a comparison that only clears 0.05
    # uncorrected has not cleared it.
    corrected_se = np.sqrt(corrected_variance(diff, n_folds))
    t_corrected = float(diff.mean() / corrected_se)
    p_corrected = float(2 * stats.t.sf(abs(t_corrected), n - 1))
    half_width_corrected = stats.t.ppf(0.975, n - 1) * corrected_se

    result.update(
        sd_difference=sd,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        ci_low=float(diff.mean() - half_width),
        ci_high=float(diff.mean() + half_width),
        # The test the pipeline used to invite by reporting only pooled numbers:
        # it ignores that both runs saw the same folds, and is shown here only so
        # the gap between the two framings is visible.
        unpaired_p_value=float(stats.ttest_ind(b, a, equal_var=False).pvalue),
        t_statistic_corrected=t_corrected,
        p_value_corrected=p_corrected,
        ci_low_corrected=float(diff.mean() - half_width_corrected),
        ci_high_corrected=float(diff.mean() + half_width_corrected),
    )
    return result


def stratified_paired_comparison(
    baseline: dict[str, list[dict]],
    candidate: dict[str, list[dict]],
    name_baseline: str = "baseline",
    name_candidate: str = "candidate",
    metric: str = "pr_auc",
    n_folds: int = 5,
    order: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """One paired comparison per stratum. Returns (rows, names that were skipped).

    A stratum is testable only if both arms scored it on the same
    (repetition, fold) observations and there are at least two of them. Anything
    else is returned in the skipped list by name rather than quietly omitted -
    a band missing from a table reads as "no difference" to anyone skimming, and
    that is not what it means.

    Strata are never paired across mismatched observation ids. `paired_comparison`
    refuses that, and it is refused here too rather than caught and silently
    reduced to the overlap: two arms that disagree about which folds could score
    a band did not measure the same thing in it.
    """
    present = set(baseline) | set(candidate)
    # `order` fixes the reading order (depth bands are ordinal, and alphabetical
    # would put "600+" between "5-9" and "84-303"). It is filtered to strata that
    # actually occurred, so bands that are empty by construction - everything
    # below 20 reads, on this training set - are not reported as "not tested".
    # "Not tested" should mean "we could not test it", not "it does not exist".
    names = [s for s in order if s in present] if order else sorted(present)
    rows: list[dict] = []
    skipped: list[str] = []

    for stratum in names:
        left, right = baseline.get(stratum), candidate.get(stratum)
        if not left or not right:
            skipped.append(str(stratum))
            continue
        ids_a = {(o.get("repetition", 0), o["fold"]) for o in left}
        ids_b = {(o.get("repetition", 0), o["fold"]) for o in right}
        shared = ids_a & ids_b
        if len(shared) < 2:
            skipped.append(str(stratum))
            continue

        keep = lambda rows_: [
            o for o in rows_ if (o.get("repetition", 0), o["fold"]) in shared
        ]
        result = paired_comparison(
            keep(left), keep(right), name_baseline, name_candidate,
            metric=metric, n_folds=n_folds,
        )
        rows.append(
            {
                "stratum": str(stratum),
                "n_observations": result["n_folds"],
                "n_sites": int(np.mean([o["n"] for o in keep(right)])),
                "baseline_mean": result["baseline_mean"],
                "candidate_mean": result["candidate_mean"],
                "mean_difference": result["mean_difference"],
                "wins": result["wins"],
                "win_ratio": f"{result['wins']}/{result['n_folds']}",
                "p_value": result["p_value"],
                "p_value_corrected": result["p_value_corrected"],
                "ci_low_corrected": result["ci_low_corrected"],
                "ci_high_corrected": result["ci_high_corrected"],
            }
        )

    return rows, skipped


# --------------------------------------------------------------------------
# bootstrap over sites
# --------------------------------------------------------------------------

# Seed for the bootstrap resampling. A third knob, distinct from the split seed
# (4262, config.py) and the subsample seed (4262, data.py), which happens to hold
# the same number. Changing this one is safe: it only redraws which sites land in
# which resample, and it invalidates no stored fold assignment and no depth
# number. It is reported alongside any interval it produces.
BOOTSTRAP_SEED = 4262


def paired_bootstrap(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    n_resamples: int = 2000,
    seed: int = BOOTSTRAP_SEED,
    metric: str = "pr_auc",
) -> dict:
    """Resample sites with replacement; return an interval per arm and on the gap.

    This answers the one question the per-fold numbers cannot: **would this hold
    on a different sample of sites?** Cross-validation tells you about the split;
    this tells you about the 121,838 sites that happened to be in the dataset.
    `0.4759` is a point estimate and nothing else in the harness puts an error bar
    on it.

    It must run **in-process, while the out-of-fold vector is still in memory** -
    per-site scores are not stored any more (docs/decisions/0009), so there is no
    file to come back to. Only the resulting interval is logged.

    Both arms are scored on the **same** resample each time, so site difficulty
    cancels in the difference for the same reason fold difficulty cancels in the
    paired fold test. Resampling them independently would measure something
    strictly weaker.

    What it does not do: make the estimate independent of the data. Resampling
    121,838 sites two thousand times is still those 121,838 sites, and every
    resample inherits whatever is unrepresentative about them - the depth >= 20
    floor most of all.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    score_fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    y_true = np.asarray(y_true)
    names = list(scores)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    draws: dict[str, list[float]] = {name: [] for name in names}
    differences: list[float] = []
    skipped = 0

    for _ in range(n_resamples):
        index = rng.integers(0, n, size=n)
        resampled = y_true[index]
        # A resample with one class in it has no AUC of any kind. Vanishingly
        # rare at 5,475 positives, but it must not become a nan in the interval.
        if resampled.min() == resampled.max():
            skipped += 1
            continue
        values = [float(score_fn(resampled, np.asarray(scores[name])[index]))
                  for name in names]
        for name, value in zip(names, values):
            draws[name].append(value)
        if len(names) == 2:
            differences.append(values[1] - values[0])

    def interval(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
            # Percentile interval. Not BCa: the bias correction needs a jackknife
            # over 121,838 sites, which is 121,838 more AP computations for a
            # refinement smaller than the width being reported.
            "ci_low": float(np.percentile(array, 2.5)),
            "ci_high": float(np.percentile(array, 97.5)),
        }

    result = {
        "metric": metric,
        "n_resamples": len(next(iter(draws.values()))),
        "n_requested": n_resamples,
        "n_skipped": skipped,
        "n_sites": int(n),
        "seed": int(seed),
        "arms": {name: interval(values) for name, values in draws.items()},
    }
    if differences:
        result["difference"] = {
            **interval(differences),
            "baseline": names[0],
            "candidate": names[1],
            # The share of resamples on the better side of zero. Not a p-value,
            # and not to be quoted as one - it is a description of the interval.
            "fraction_positive": float(np.mean(np.asarray(differences) > 0)),
        }
    return result
def comparison_frame(result: dict) -> pd.DataFrame:
    """The per-observation table: baseline, candidate, and the difference.

    Indexed by fold for a single-repetition run, and by (repetition, fold) when
    there is more than one - so the five-row table people already read keeps its
    shape and does not grow a constant column.
    """
    columns = {
        result["baseline"]: result["baseline_per_fold"],
        result["candidate"]: result["candidate_per_fold"],
        "difference": result["difference_per_fold"],
    }
    if result.get("n_repeats", 1) > 1:
        index = pd.MultiIndex.from_tuples(
            [tuple(k) for k in result["observations"]], names=["repetition", "fold"]
        )
    else:
        index = pd.Index(result["folds"], name="fold")
    return pd.DataFrame(columns, index=index)


def verdict(result: dict) -> str:
    """One sentence a teammate can act on, with the caveat attached.

    Reads the **corrected** p-value, because that is the one the evidence
    actually supports. The uncorrected one is still in the result dict, and still
    printed, so numbers recorded before the correction existed stay traceable.
    """
    n = result["n_folds"]
    mean = result["mean_difference"]
    wins = result["wins"]
    p = result.get("p_value_corrected", result["p_value"])
    naive = result["p_value"]
    direction = "better than" if mean > 0 else "worse than"
    unit = "paired observations" if result.get("n_repeats", 1) > 1 else "folds"

    if not np.isfinite(p):
        return (
            f"{result['candidate']} and {result['baseline']} scored identically on "
            f"all {n} {unit} - nothing to test."
        )
    if p < 0.05 and wins in (0, n):
        return (
            f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']}, "
            f"winning {wins}/{n} {unit}, corrected p = {p:.4f}. Consistent in "
            f"direction and significant even after correcting for the overlap "
            f"between training sets."
        )
    if p < 0.05:
        return (
            f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']} "
            f"(corrected p = {p:.4f}) but wins only {wins}/{n} {unit}, so the "
            "direction is not consistent. Treat with caution."
        )
    if naive < 0.05 <= p:
        return (
            f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']}, "
            f"winning {wins}/{n} {unit}. The uncorrected paired test says p = "
            f"{naive:.4f}; corrected for the overlap between training sets it is "
            f"p = {p:.4f}, which does not clear 0.05. Read the win count: a "
            f"difference that holds on every {unit[:-1]} is worth more than this "
            f"p-value suggests, and more repetitions (--repeats 10) are the way to "
            f"settle it rather than quoting the uncorrected number."
        )
    return (
        f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']}, "
        f"winning {wins}/{n} {unit}, corrected p = {p:.4f}. Not distinguishable "
        f"from noise - which is not the same as being no better."
    )
