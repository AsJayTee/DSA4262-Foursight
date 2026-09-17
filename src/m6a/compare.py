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

**The honest caveat**: five folds is five paired observations. The t-test has
four degrees of freedom, so it detects only large and consistent effects, and a
p-value near 0.05 from it is weak evidence. Read the per-fold differences and
the win count alongside it; a difference that is positive on every fold is worth
more than the p-value alone suggests, and one that flips sign is worth less.
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


def paired_comparison(
    baseline: list[dict],
    candidate: list[dict],
    name_baseline: str = "baseline",
    name_candidate: str = "candidate",
    metric: str = "pr_auc",
) -> dict:
    """Compare two runs' per-fold metrics, paired by fold.

    `baseline` and `candidate` are per-fold metric dicts as `cross_validate`
    produces them. The returned dict carries both framings: the paired test,
    which is the valid one, and the unpaired one, so the difference between them
    is visible rather than asserted.
    """
    folds_a = [int(f["fold"]) for f in baseline]
    folds_b = [int(f["fold"]) for f in candidate]
    if folds_a != folds_b:
        raise ValueError(
            f"Fold ids differ ({folds_a} vs {folds_b}); these runs are not paired."
        )

    a = np.array([f[metric] for f in baseline], dtype=float)
    b = np.array([f[metric] for f in candidate], dtype=float)
    diff = b - a
    n = len(diff)

    result = {
        "metric": metric,
        "n_folds": n,
        "baseline": name_baseline,
        "candidate": name_candidate,
        "folds": folds_a,
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
        )
        return result

    t_stat, p_value = stats.ttest_rel(b, a)
    half_width = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)

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
    )
    return result


def comparison_frame(result: dict) -> pd.DataFrame:
    """The per-fold table: baseline, candidate, and the difference that matters."""
    return pd.DataFrame(
        {
            result["baseline"]: result["baseline_per_fold"],
            result["candidate"]: result["candidate_per_fold"],
            "difference": result["difference_per_fold"],
        },
        index=pd.Index(result["folds"], name="fold"),
    )


def verdict(result: dict) -> str:
    """One sentence a teammate can act on, with the caveat attached."""
    n = result["n_folds"]
    mean = result["mean_difference"]
    wins = result["wins"]
    p = result["p_value"]
    direction = "better than" if mean > 0 else "worse than"

    if not np.isfinite(p):
        return (
            f"{result['candidate']} and {result['baseline']} scored identically on "
            f"all {n} folds - nothing to test."
        )
    if p < 0.05 and wins in (0, n):
        return (
            f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']}, "
            f"winning {wins}/{n} folds, paired p = {p:.4f}. Consistent in direction "
            f"and significant, though on only {n} paired folds."
        )
    if p < 0.05:
        return (
            f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']} "
            f"(paired p = {p:.4f}) but wins only {wins}/{n} folds, so the direction "
            "is not consistent. Treat with caution."
        )
    return (
        f"{result['candidate']} is {mean:+.4f} {direction} {result['baseline']}, "
        f"winning {wins}/{n} folds, paired p = {p:.4f}. Not distinguishable from "
        f"noise on {n} folds - which is not the same as being no better."
    )
