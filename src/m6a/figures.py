"""Plots for an evaluation run. Every one of them ends up in W&B.

A result that exists only as a table of numbers on a terminated instance is a
result nobody will read. These figures are the half of an evaluation a teammate
can take in without rerunning anything (docs/decisions/0007 section 5).

**The primary figure for any run is a distribution, not a table.** Five folds -
fifty, with repeated cross-validation - are a spread, and the spread is the
thing that says whether a gap between two runs means anything. Individual points
are always drawn, because with this few of them you should be able to count them
(docs/decisions/0009 section 3).

**matplotlib is imported inside each function, never at module level.** It is a
`train`-extra dependency and predict.py runs on an evaluator's machine with only
the base list installed. Nothing on the prediction path imports this module, and
this keeps that true even if something one day does. See AGENTS.md section 4 and
docs/decisions/0004.

The backend is forced to Agg on import of pyplot here: a Ronin instance has no
display, and the default backend would fail there rather than on the laptop
where this was written.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

# A validated categorical palette. Fixed order, never cycled - slot 1 is always
# the subject of the plot and slot 2 always the thing it is being compared
# against, so two figures side by side mean the same thing.
#
# **BLUE and ORANGE are the only two that may share a figure as categories.**
# Measured, not assumed: BLUE/ORANGE separate by dE 24.7 under protanopia and
# 33.6 at normal vision, comfortably clear. BLUE/ORANGE/**RED** together does
# not pass - RED against ORANGE is dE 10.8 at *normal* vision, below the 15
# floor, so full-colour readers cannot reliably tell them apart either. An
# earlier version of this comment claimed the first three slots held up when
# every pair was on screen; that claim was wrong and no figure here relies on
# it. RED is a diverging pole against BLUE and never a third category.
#
# A third series in one figure goes to MUTED as context, the way depth_sweep
# draws ROC AUC and threshold_sweep draws F1. A figure that genuinely needs more
# than two categories needs a wider validated ramp first - see
# docs/decisions/0020 and `scripts/compare_runs.py`.
BLUE = "#2a78d6"     # slot 1: the run this report is about
ORANGE = "#eb6834"   # slot 2: the run it is compared against
RED = "#d03b3b"      # the losing side of a difference; diverging pole against BLUE

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#75746f"    # context series: present, not competing for attention
GRID = "#e3e2de"

FIGSIZE = (7.2, 4.3)
DPI = 150


def _plt():
    """pyplot, on a backend that works without a display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _canvas(title: str, xlabel: str, ylabel: str, figsize=FIGSIZE):
    """One figure, themed. Hairline axes and grid, so the data is the loudest thing."""
    plt = _plt()
    figure, axes = plt.subplots(figsize=figsize, dpi=DPI)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    axes.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    axes.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    axes.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=MUTED, labelsize=8.5, length=0)
    return figure, axes


def _caption(figure, text: str) -> None:
    """A line under the plot saying what the reader is looking at.

    Several of these figures answer questions that are easy to confuse with each
    other - scored-at-reduced-depth against measured-at-true-depth, most of all -
    so the distinction is written on the figure rather than left to whoever
    remembers it.
    """
    figure.text(0.012, 0.012, text, color=MUTED, fontsize=7.6, ha="left", va="bottom")
    figure.subplots_adjust(bottom=0.24)


# --------------------------------------------------------------------------
# the primary figure: a metric as a distribution
# --------------------------------------------------------------------------

def metric_distribution(
    vectors: dict[str, Sequence[float]],
    metric: str = "PR AUC",
    title: str | None = None,
    caption: str = "",
    colours: Sequence[str] | None = None,
) -> Any:
    """One row per arm, every fold (and repetition) drawn as its own point.

    This is the figure to read first. A pooled number hides that the same model
    scores 0.4548 on one fold and 0.5081 on another; two arms overlaid on one
    axis make it obvious whether a gap between them is larger than that spread or
    lost inside it.

    With enough observations a violin is drawn behind the points to show the
    shape. The points stay on top either way - a summary of five numbers is a
    worse object than the five numbers.
    """
    names = list(vectors)
    # `colours` lets a caller with more than two arms say which is which -
    # scripts/compare_runs.py paints the reference run one way and every
    # candidate the other. Left alone, the house default holds: slot 1 is the
    # subject, slot 2 is what it is compared against.
    if colours is not None:
        colours = list(colours)
    else:
        colours = [BLUE, ORANGE][: len(names)] or [BLUE]
    while len(colours) < len(names):
        colours.append(MUTED)

    height = max(2.6, 1.05 * len(names) + 1.9)
    figure, axes = _canvas(
        title or f"{metric} across folds",
        metric,
        "",
        figsize=(FIGSIZE[0], height),
    )
    rng = np.random.default_rng(0)  # jitter only; nothing here depends on the draw

    # Drawn top-down, so the first arm - the run the report is about - is the one
    # the eye lands on first.
    for position, (name, colour) in enumerate(zip(names, colours)):
        row = len(names) - 1 - position
        values = np.asarray(list(vectors[name]), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        if values.size >= 20:
            body = axes.violinplot(
                values, positions=[row], vert=False, widths=0.72,
                showmeans=False, showextrema=False, showmedians=False,
            )
            for part in body["bodies"]:
                part.set_facecolor(colour)
                part.set_alpha(0.16)
                part.set_edgecolor("none")

        jitter = rng.uniform(-0.13, 0.13, values.size) if values.size > 1 else np.zeros(1)
        axes.scatter(
            values, np.full(values.size, row) + jitter,
            s=46, color=colour, alpha=0.85, zorder=3,
            # A 2px surface ring rather than a dark border, so overlapping points
            # stay countable without the marks reading as outlined blocks.
            edgecolors=SURFACE, linewidths=1.6,
        )
        mean = float(values.mean())
        axes.plot(
            [mean, mean], [row - 0.3, row + 0.3],
            color=colour, linewidth=2.4, zorder=4, solid_capstyle="round",
        )
        spread = f"sd {values.std(ddof=1):.4f}" if values.size > 1 else "single point"
        axes.annotate(
            f"mean {mean:.4f}   {spread}   n={values.size}",
            xy=(mean, row + 0.34), color=INK_SOFT, fontsize=8, ha="center", va="bottom",
        )

    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names[::-1], color=INK, fontsize=9.5)
    axes.set_ylim(-0.6, len(names) - 0.3)
    axes.grid(axis="y", visible=False)
    # Arm names sit outside the axes, so the left margin has to be wide enough for
    # the longest one or it is silently clipped at the figure edge.
    figure.subplots_adjust(left=min(0.42, 0.11 + 0.011 * max(len(n) for n in names)))
    _caption(
        figure,
        caption
        or "Each point is one fold. Folds differ in size and positive rate, so this\n"
           "spread is fold difficulty, not the uncertainty on a comparison.",
    )
    return figure


def comparison_distribution(result: dict, metric: str = "PR AUC") -> Any:
    """Both arms of a paired comparison overlaid on one axis.

    The figure that answers "is this better" at a glance, and that makes the
    fold-difficulty problem visible instead of something a reader has to be told.
    When the two overlap too much to call, the corrected paired t-test printed in
    the caption is the tie-breaker (docs/decisions/0009 section 4).
    """
    p_value = result.get("p_value_corrected")
    which = "corrected" if p_value is not None else "uncorrected"
    if p_value is None:
        p_value = result.get("p_value", float("nan"))

    caption = (
        f"Paired difference {result['mean_difference']:+.4f}, "
        f"wins {result['wins']}/{result['n_folds']}, "
        f"{which} p = {p_value:.4f}.\n"
        "Both arms were scored on the same folds, so read the paired difference, "
        "not the gap between the two means."
    )
    return metric_distribution(
        {result["candidate"]: result["candidate_per_fold"],
         result["baseline"]: result["baseline_per_fold"]},
        metric=metric,
        title=f"{metric}: {result['candidate']} against {result['baseline']}",
        caption=caption,
    )


def difference_strip(result: dict, metric: str = "PR AUC") -> Any:
    """The per-fold differences against zero - the quantity the test is run on.

    Diverging by construction: a point's side of zero is its whole meaning, so
    the two sides get the two poles and zero gets the neutral rule.
    """
    differences = np.asarray(result["difference_per_fold"], dtype=float)
    folds = result["folds"]
    mean = float(differences.mean())

    figure, axes = _canvas(
        f"Per-fold {metric} difference: {result['candidate']} minus {result['baseline']}",
        f"difference in {metric}",
        "fold",
        figsize=(FIGSIZE[0], max(3.0, 0.42 * len(folds) + 2.4)),
    )
    axes.axvline(0, color=INK_SOFT, linewidth=1.2, zorder=2)

    # Folds top-down, and the summary interval on its own row underneath them
    # rather than as a band behind everything: the CI here spans most of the
    # plot, and drawn as a shaded region it reads as the background rather than
    # as the estimate it is.
    for position, (fold, difference) in enumerate(zip(folds, differences)):
        row = len(folds) - position
        colour = BLUE if difference > 0 else RED
        axes.plot([0, difference], [row, row], color=colour, linewidth=2, zorder=3)
        axes.scatter(
            [difference], [row], s=58, color=colour, zorder=4,
            edgecolors=SURFACE, linewidths=1.6,
        )

    if np.isfinite(result.get("ci_low", float("nan"))):
        axes.plot(
            [result["ci_low"], result["ci_high"]], [0, 0],
            color=INK_SOFT, linewidth=2, zorder=3, solid_capstyle="butt",
        )
        for edge in (result["ci_low"], result["ci_high"]):
            axes.plot([edge, edge], [-0.16, 0.16], color=INK_SOFT, linewidth=1.4, zorder=3)
    axes.scatter([mean], [0], s=66, color=INK_SOFT, zorder=4,
                 edgecolors=SURFACE, linewidths=1.6)
    axes.annotate(
        f"mean {mean:+.4f}", xy=(mean, 0.3), color=INK_SOFT,
        fontsize=8.5, ha="center", va="bottom",
    )

    axes.set_yticks(range(len(folds) + 1))
    axes.set_yticklabels(["mean, 95% CI"] + [str(f) for f in folds][::-1],
                         color=INK, fontsize=9)
    axes.set_ylim(-0.75, len(folds) + 0.75)
    axes.grid(axis="y", visible=False)
    figure.subplots_adjust(left=0.22)
    _caption(
        figure,
        "Every point on one side of zero matters more than the p-value: a consistent "
        "sign\ndoes not depend on the scatter estimate, and with five folds the "
        "scatter estimate is thin.",
    )
    return figure


# --------------------------------------------------------------------------
# curves over the pooled out-of-fold scores
# --------------------------------------------------------------------------

def pr_curve(arms: dict[str, tuple[np.ndarray, np.ndarray]], base_rate: float) -> Any:
    """Precision against recall, with the random-classifier line drawn in.

    A PR curve without that line is unreadable: at a 4.49% positive rate the
    floor is 0.0449, not 0.5, and a curve that looks low may be ten times random.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    figure, axes = _canvas("Precision-recall, out of fold", "recall", "precision")
    for (name, (y_true, y_score)), colour in zip(arms.items(), (BLUE, ORANGE)):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        score = average_precision_score(y_true, y_score)
        axes.plot(recall, precision, color=colour, linewidth=2,
                  label=f"{name}  (PR AUC {score:.4f})")

    axes.axhline(base_rate, color=MUTED, linewidth=1.2, zorder=2)
    axes.annotate(
        f"random classifier  {base_rate:.4f}",
        xy=(0.98, base_rate), xytext=(0.98, base_rate + 0.03),
        color=MUTED, fontsize=8, ha="right",
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    if len(arms) > 1:
        axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT, loc="upper right")
    else:
        axes.set_title(
            f"Precision-recall, out of fold: {next(iter(arms))}",
            color=INK, fontsize=11, loc="left", pad=12,
        )
    _caption(
        figure,
        "Every site is scored by a model that saw no transcript of its gene.",
    )
    return figure


def roc_curve(arms: dict[str, tuple[np.ndarray, np.ndarray]]) -> Any:
    """True positive rate against false positive rate, with the chance diagonal.

    Reported because the course evaluates on it. Rank your own experiments on the
    PR curve: at 4.49% positives every model here sits above 0.90 on this one.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve as sk_roc_curve

    figure, axes = _canvas("ROC, out of fold", "false positive rate", "true positive rate")
    for (name, (y_true, y_score)), colour in zip(arms.items(), (BLUE, ORANGE)):
        fpr, tpr, _ = sk_roc_curve(y_true, y_score)
        axes.plot(fpr, tpr, color=colour, linewidth=2,
                  label=f"{name}  (ROC AUC {roc_auc_score(y_true, y_score):.4f})")

    axes.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, zorder=2)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT, loc="lower right")
    _caption(
        figure,
        "ROC AUC flatters an imbalanced problem. Read the precision-recall figure first.",
    )
    return figure


def reliability(table: pd.DataFrame, summary: dict) -> Any:
    """Predicted probability against observed rate, by score decile.

    Rank-based metrics cannot see this at all: a model can order sites perfectly
    and still be wrong about magnitude by a factor of two, which is exactly what
    breaks a Task 2 claim of the form "cell line X has N modified sites".
    """
    figure, axes = _canvas(
        "Reliability: predicted against observed",
        "mean predicted probability",
        "observed positive rate",
    )
    predicted = table["mean_predicted"].to_numpy(dtype=float)
    observed = table["observed_rate"].to_numpy(dtype=float)
    top = float(max(predicted.max(), observed.max())) * 1.08 or 1.0

    axes.plot([0, top], [0, top], color=MUTED, linewidth=1.2, zorder=2)
    axes.annotate("perfectly calibrated", xy=(top * 0.97, top * 0.97), xytext=(-2, 7),
                  textcoords="offset points", color=MUTED, fontsize=8,
                  ha="right", va="bottom")
    axes.plot(predicted, observed, color=BLUE, linewidth=2, zorder=3)
    axes.scatter(predicted, observed, s=52, color=BLUE, zorder=4,
                 edgecolors=SURFACE, linewidths=1.6)
    axes.set_xlim(0, top)
    axes.set_ylim(0, top)
    _caption(
        figure,
        f"ECE {summary['ece']:.4f}. Summing these scores counts "
        f"{summary['count_ratio']:.2f}x too many positives\n"
        f"({summary['expected_positives']:,.0f} expected against "
        f"{summary['actual_positives']:,.0f} actual).",
    )
    return figure


# --------------------------------------------------------------------------
# read depth - two different questions, two figures
# --------------------------------------------------------------------------

def depth_sweep(rows: list[dict], subsample_seed: int) -> Any:
    """PR AUC when the same sites are scored with reads thrown away.

    Not the same question as `depth_bands` below, and the two are easy to
    confuse: this one holds the sites fixed and removes evidence, which is the
    Task 2 situation - trained on depth >= 20, predicting on SG-NEx at median
    depth 3.
    """
    ordered = [row for row in rows if row["depth"] != "full"]
    ordered.sort(key=lambda row: int(row["depth"]))
    full = next((row for row in rows if row["depth"] == "full"), None)

    x = list(range(len(ordered) + (1 if full else 0)))
    labels = [row["depth"] for row in ordered] + (["full"] if full else [])
    series = ordered + ([full] if full else [])

    figure, axes = _canvas(
        "Scored at reduced read depth",
        "reads kept per site",
        "area under the curve",
    )
    # Emphasis, not two equal series: PR AUC is the number that discriminates and
    # ROC AUC is context, so ROC recedes to the de-emphasis grey rather than
    # taking a second categorical hue it would compete for attention with.
    axes.plot(x, [row["roc_auc"] for row in series], color=MUTED, linewidth=2, zorder=3)
    axes.scatter(x, [row["roc_auc"] for row in series], s=40, color=MUTED, zorder=4,
                 edgecolors=SURFACE, linewidths=1.6)
    axes.annotate("ROC AUC", xy=(x[-1], series[-1]["roc_auc"]), xytext=(-6, 9),
                  textcoords="offset points", color=MUTED, fontsize=8.5, ha="right")

    axes.plot(x, [row["pr_auc"] for row in series], color=BLUE, linewidth=2, zorder=3)
    axes.scatter(x, [row["pr_auc"] for row in series], s=48, color=BLUE, zorder=4,
                 edgecolors=SURFACE, linewidths=1.6)
    axes.annotate("PR AUC", xy=(x[-1], series[-1]["pr_auc"]), xytext=(-6, -16),
                  textcoords="offset points", color=BLUE, fontsize=8.5, ha="right")

    axes.set_xticks(x)
    axes.set_xticklabels(labels)
    axes.set_ylim(0, 1)
    _caption(
        figure,
        "Depth is the number of distinct RNA molecules measured at a site, not repeated "
        "readings\nof one (docs/data.md#read-depth). Models trained at full depth; "
        f"subsample seed {subsample_seed}.",
    )
    return figure


def depth_bands(rows: list[dict], spans: dict[str, tuple[int, int]]) -> Any:
    """PR AUC lift on sites grouped by the depth they really had.

    The other depth question: no reads are removed here, the sites simply differ
    in how much coverage they came with. Lift rather than PR AUC because PR AUC
    is bounded below by each band's own positive rate.

    **Drawn as a step, not a line.** Each band's metric is one number that holds
    across a *range* of read counts, and the ranges are wildly uneven - 20-31 is
    12 reads wide, 84-303 is 220. A dot per band joined by a line invites the
    reader to see a trend between two dots with nothing measured in between, and
    hides how much of the axis each measurement is responsible for. A segment
    spanning the band says both things at once: constant here, and *this* wide.

    The x axis is logarithmic because the bands are roughly geometric; on a
    linear axis the first five would pile up against the origin.
    """
    scored = [row for row in rows
              if row.get("pr_auc_lift") is not None and row["group"] in spans]
    if not scored:
        return None
    scored.sort(key=lambda row: spans[row["group"]][0])

    figure, axes = _canvas(
        "Measured at true read depth",
        "reads at the site (log scale; bar width is the band's range)",
        "PR AUC lift over random",
    )
    lift = [float(row["pr_auc_lift"]) for row in scored]
    for row, value in zip(scored, lift):
        first, last = spans[row["group"]]
        axes.plot([first, last], [value, value], color=BLUE, linewidth=3.5,
                  solid_capstyle="butt", zorder=3)
        axes.annotate(
            f"{value:.1f}x", xy=((first * last) ** 0.5, value), xytext=(0, 9),
            textcoords="offset points", color=INK_SOFT, fontsize=8, ha="center",
        )
        axes.annotate(
            str(row["group"]), xy=((first * last) ** 0.5, value), xytext=(0, -14),
            textcoords="offset points", color=MUTED, fontsize=7.5, ha="center",
        )

    axes.set_xscale("log")
    span = max(lift) - min(lift)
    axes.set_ylim(min(lift) - max(span, 0.5) * 0.55, max(lift) + max(span, 0.5) * 0.55)
    _caption(
        figure,
        "No reads were removed: these are sites that genuinely differ in coverage. Not the\n"
        "same question as the sweep, which holds the sites fixed and takes evidence away.\n"
        "Each bar is one measurement covering that whole range - there is no trend to read\n"
        "between two bars. Lift, not PR AUC, because each band has its own positive rate.",
    )
    return figure


def stratum_differences(rows: list[dict], stratum_label: str, title: str) -> Any:
    """The paired difference inside each stratum, with its corrected interval.

    The figure for "is it better *where we are currently weak*". A model that
    trades a little pooled PR AUC for a real gain at low read depth is what
    Task 2 needs, and a single pooled number cannot show it.

    Intervals are the corrected ones, which are wide - that is the honest width.
    Bands whose interval crosses zero are drawn in the neutral ink rather than a
    side's colour, because colouring them by the sign of a difference the data
    cannot resolve is how a reader talks themselves into a finding.
    """
    if not rows:
        return None

    figure, axes = _canvas(
        title, "difference in PR AUC", stratum_label,
        figsize=(FIGSIZE[0], max(3.0, 0.44 * len(rows) + 2.3)),
    )
    axes.axvline(0, color=INK_SOFT, linewidth=1.2, zorder=2)

    for position, row in enumerate(rows):
        y = len(rows) - 1 - position
        low, high = float(row["ci_low_corrected"]), float(row["ci_high_corrected"])
        difference = float(row["mean_difference"])
        resolved = np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
        colour = (BLUE if difference > 0 else RED) if resolved else MUTED

        if np.isfinite(low) and np.isfinite(high):
            axes.plot([low, high], [y, y], color=colour, linewidth=2, zorder=3)
            for edge in (low, high):
                axes.plot([edge, edge], [y - 0.14, y + 0.14],
                          color=colour, linewidth=1.3, zorder=3)
        axes.scatter([difference], [y], s=58, color=colour, zorder=4,
                     edgecolors=SURFACE, linewidths=1.6)
        axes.annotate(
            row["win_ratio"], xy=(difference, y + 0.22), color=MUTED,
            fontsize=7.5, ha="center", va="bottom",
        )

    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([row["stratum"] for row in rows][::-1], color=INK, fontsize=9)
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.grid(axis="y", visible=False)
    figure.subplots_adjust(left=0.2)
    _caption(
        figure,
        "95% intervals, corrected for the overlap between training sets. Grey means the\n"
        "interval crosses zero. These are the weakest numbers the harness produces:\n"
        "one test per band, strongly correlated, uncorrected for multiplicity by design.\n"
        "Descriptive - where a difference concentrates - not confirmatory.",
    )
    return figure


# --------------------------------------------------------------------------
# thresholds - the half a rank metric cannot answer
# --------------------------------------------------------------------------

def threshold_sweep(table: pd.DataFrame, points: dict[str, dict]) -> Any:
    """Precision and recall against the threshold, with F1 behind them.

    Precision and recall are the two quantities in tension, so they take the two
    categorical slots. F1 is a *summary of those two* rather than a third
    independent series, so it recedes to the context grey - the same treatment
    `depth_sweep` gives ROC AUC. Its peak is marked because that is the only
    thing anyone reads off an F1 curve.
    """
    thresholds = table["threshold"].to_numpy(dtype=float)
    figure, axes = _canvas(
        "Precision, recall and F1 against the decision threshold",
        "threshold (a site is called positive at or above this score)",
        "",
    )

    axes.plot(thresholds, table["f1"].to_numpy(dtype=float), color=MUTED,
              linewidth=2, zorder=3, label="F1")
    axes.plot(thresholds, table["precision"].to_numpy(dtype=float), color=BLUE,
              linewidth=2, zorder=4, label="precision")
    axes.plot(thresholds, table["recall"].to_numpy(dtype=float), color=ORANGE,
              linewidth=2, zorder=4, label="recall")

    best = points.get("f1_max")
    if best is not None:
        at = float(best["threshold"])
        axes.axvline(at, color=INK_SOFT, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        # Annotated at the foot of the rule, not the top: the top right is the
        # only region all three curves leave clear, and the legend needs it.
        axes.annotate(
            f"best F1 {float(best['f1']):.3f} at {at:.2f}",
            xy=(at, 0.0), xytext=(5, 6), textcoords="offset points",
            color=INK_SOFT, fontsize=8, ha="left", va="bottom",
        )

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    # Upper right is the one corner no curve reaches: precision tops out below
    # 0.8, and recall and F1 are both falling by then. Anywhere else the legend
    # sits on top of a line.
    axes.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT, loc="upper right")
    _caption(
        figure,
        "Every other metric in this report integrates over this axis. PR AUC and ROC AUC\n"
        "rank sites; they never pick a threshold, and counting modified sites needs one.",
    )
    return figure


def threshold_count(table: pd.DataFrame, points: dict[str, dict], n_positive: int) -> Any:
    """How many sites get called positive, against the threshold. Log scale.

    **The figure behind any Task 2 site count.** The count spans orders of
    magnitude across thresholds a reasonable person might pick, so a claim that a
    cell line has N modified sites is partly a claim about N and partly about a
    choice that usually goes unrecorded. One series, so no legend - the title
    names it.

    The y axis is logarithmic because the count is: on a linear axis every
    operating point worth choosing is flattened against zero.
    """
    counts = table["predicted_positives"].to_numpy(dtype=float)
    thresholds = table["threshold"].to_numpy(dtype=float)
    # log scale cannot draw a zero, and the count reaches zero once nothing is
    # called. Stop the line where the calls stop rather than clipping, which
    # would draw a floor that is not there.
    drawn = counts > 0
    figure, axes = _canvas(
        "Sites called positive, against the decision threshold",
        "threshold (a site is called positive at or above this score)",
        "sites called positive (log scale)",
    )

    axes.plot(thresholds[drawn], counts[drawn], color=BLUE, linewidth=2, zorder=4)
    axes.axhline(n_positive, color=MUTED, linewidth=1.2, zorder=2)
    axes.annotate(
        f"{n_positive:,} sites really are positive",
        xy=(0.02, n_positive), xytext=(0, 6), textcoords="offset points",
        color=MUTED, fontsize=8, ha="left", va="bottom",
    )

    # Alternating label heights, alternating **in threshold order** rather than
    # in the order the operating points happen to be named. The three can land
    # within a few hundredths of each other, and alternating by name put two
    # neighbouring labels at the same height and printed one over the other.
    ordered = sorted(points.items(), key=lambda kv: float(kv[1]["threshold"]))
    for index, (name, row) in enumerate(ordered):
        at = float(row["threshold"])
        called = int(row["predicted_positives"])
        if called <= 0:
            continue
        axes.axvline(at, color=INK_SOFT, linewidth=1.1, linestyle=(0, (4, 3)), zorder=3)
        axes.scatter([at], [called], s=52, color=BLUE, zorder=5,
                     edgecolors=SURFACE, linewidths=1.6)
        axes.annotate(
            f"{name}\n{at:.2f} -> {called:,}",
            xy=(at, called), xytext=(6, 12 if index % 2 == 0 else -28),
            textcoords="offset points", color=INK_SOFT, fontsize=7.8, ha="left",
            # The neighbouring rule often passes behind these two lines of text.
            # A surface-coloured pad knocks it out, the same idea as the surface
            # ring the scatter marks carry - but it only works above the rules,
            # and the next point's axvline is drawn after this annotation.
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.4),
            zorder=6,
        )

    axes.set_yscale("log")
    axes.set_xlim(0, 1)
    _caption(
        figure,
        "The sensitivity analysis any site count needs. Two defensible thresholds give\n"
        "counts an order of magnitude apart, and the scores are miscalibrated on top of\n"
        "that - so a count quoted without a threshold is an arbitrary cut reported as a\n"
        "measurement.",
    )
    return figure


# --------------------------------------------------------------------------
# many runs at once - the report's answer to the live W&B panels
# --------------------------------------------------------------------------

def run_scatter(
    points: list[dict],
    x_label: str,
    y_label: str,
    title: str,
    diagonal: bool = True,
    reference: str | None = None,
) -> Any:
    """One point per run, labelled, with the y = x diagonal drawn in.

    The two things a W&B scatter panel cannot do are draw a reference line and
    put a name next to a dot, and both are the whole point when the question is
    "how far does each of our models fall short of *this* line".

    The diagonal is only drawn when the two axes are the same quantity - full
    depth against depth 3, say, where the distance below the line is the
    collapse. Against two different quantities it would be a line with no
    meaning, so the caller passes `diagonal=False` and the caption says why.

    Identity is carried by the label beside each point, never by colour alone:
    beyond two categories this palette has no validated hues to give
    (docs/decisions/0020).
    """
    if not points:
        return None

    xs = np.array([float(p["x"]) for p in points])
    ys = np.array([float(p["y"]) for p in points])
    figure, axes = _canvas(title, x_label, y_label, figsize=(FIGSIZE[0], 5.0))

    if diagonal:
        low = float(min(xs.min(), ys.min()))
        high = float(max(xs.max(), ys.max()))
        pad = max((high - low) * 0.25, 0.02)
        line = [low - pad, high + pad]
        axes.plot(line, line, color=MUTED, linewidth=1.2, zorder=2)
        # Annotated at the *lower* end. The upper end is where the label column
        # below puts its first entry, and the two printed over each other.
        axes.annotate(
            "y = x", xy=(line[0], line[0]), xytext=(4, -3), textcoords="offset points",
            color=MUTED, fontsize=8, ha="left", va="top",
        )

    for point in points:
        is_reference = reference is not None and point["name"] == reference
        axes.scatter(
            [point["x"]], [point["y"]], s=74,
            color=ORANGE if is_reference else BLUE, zorder=4,
            edgecolors=SURFACE, linewidths=1.8,
        )

    # Labels run to the right of their point, so the right margin has to hold the
    # longest one or it is clipped at the figure edge.
    axes.margins(x=0.30, y=0.18)

    # **Labels are stacked in a column, not hung off each dot.** Two models that
    # score similarly are two dots a few pixels apart, and a label beside each
    # one then prints over its neighbour - which happened to every pair tried
    # here, and alternating the labels above and below only moved which pair
    # collided. A column spaced evenly down the axis cannot collide by
    # construction, whatever the points do, and a leader line keeps each label
    # attached to its dot.
    ordered = sorted(points, key=lambda p: float(p["y"]))
    x_low, x_high = axes.get_xlim()
    y_low, y_high = axes.get_ylim()
    label_x = max(float(p["x"]) for p in points) + (x_high - x_low) * 0.05
    if len(ordered) == 1:
        heights = [float(ordered[0]["y"])]
    else:
        first, last = y_low + (y_high - y_low) * 0.12, y_low + (y_high - y_low) * 0.88
        heights = [
            first + (last - first) * i / (len(ordered) - 1)
            for i in range(len(ordered))
        ]

    for point, height in zip(ordered, heights):
        is_reference = reference is not None and point["name"] == reference
        axes.plot(
            [float(point["x"]), label_x], [float(point["y"]), height],
            color=GRID, linewidth=1.0, zorder=3, solid_capstyle="round",
        )
        axes.annotate(
            point["name"] + ("  (reference)" if is_reference else ""),
            xy=(label_x, height), xytext=(4, 0), textcoords="offset points",
            color=INK_SOFT, fontsize=8, ha="left", va="center", zorder=5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.4),
        )
    _caption(
        figure,
        "One point per W&B run, labelled - which is what a W&B scatter panel cannot do,\n"
        "along with the reference line. Each point is a whole run, so nothing here is a\n"
        "significance test; pair the runs for that."
        + ("\nDistance below the diagonal is the drop between the two axes."
           if diagonal else ""),
    )
    return figure


def feature_importance(importances: dict[str, float], top_n: int = 20) -> Any:
    """The columns the shipped model actually splits on."""
    ranked = sorted(importances.items(), key=lambda kv: -kv[1])[:top_n]
    if not ranked:
        return None
    names = [name for name, _ in ranked][::-1]
    gains = [float(gain) for _, gain in ranked][::-1]

    figure, axes = _canvas(
        f"Feature importance, top {len(ranked)}", "gain", "",
        figsize=(FIGSIZE[0], max(3.2, 0.26 * len(ranked) + 1.6)),
    )
    for row, gain in enumerate(gains):
        axes.plot([0, gain], [row, row], color=BLUE, linewidth=7, zorder=3,
                  solid_capstyle="round")
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names, color=INK, fontsize=8)
    axes.grid(axis="y", visible=False)
    figure.subplots_adjust(left=0.32, bottom=0.1)
    return figure
