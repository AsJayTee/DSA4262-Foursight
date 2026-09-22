#!/usr/bin/env python
"""Put N finished W&B runs on one figure. The report's version of the live panels.

    # three runs, overlaid distributions and the depth-collapse scatter
    python scripts/compare_runs.py lightgbm_quantiles lightgbm_pooled baseline_logistic

    # by run id, and pair everything against the first one named
    python scripts/compare_runs.py y2lk7ilf 3bw2tvmm --pair

    # any two flat metric keys on the scatter
    python scripts/compare_runs.py a b c --x oof/pr_auc --y calib/calibrated

**Why this is a script and not a W&B panel.** W&B draws distributions one run
per image, so comparing three models means opening three pages and flicking
between three pictures; and its scatter panel cannot draw a reference line or
put a run's name beside its dot. Those three gaps are the whole reason this
exists. See docs/decisions/0020.

It reads from W&B and writes PNGs to `report/figures/`. Nothing is fitted and
nothing is logged back - the inputs are already durable, so unlike an
evaluation this can be re-run from any machine at any time, including after
every instance that produced the runs has been terminated.

`--pair` adds the Nadeau & Bengio corrected paired test of every run against the
first one named. It refuses to pair runs whose dataset fingerprints disagree,
for the reason docs/decisions/0015 exists: two runs on different data produce
two plausible numbers whose difference means nothing.

Needs wandb and matplotlib, both in the `train` extra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m6a import tracking
from m6a.env import load_env

DEFAULT_OUT = "report/figures"
# Full depth against the depth SG-NEx actually has. The diagonal is "no
# collapse", every model sits far below it, and that gap is the largest known
# risk in the project (GAPS.md) - so it is what the scatter shows unless asked
# otherwise. Both axes are PR AUC, which is what makes y = x mean anything.
DEFAULT_X = "oof/pr_auc"
DEFAULT_Y = "depth/3/pr_auc"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "runs", nargs="+",
        help="W&B run ids or names. The first is the reference: it is drawn "
             "differently and is what --pair pairs everything against.",
    )
    ap.add_argument(
        "--metric", default="pr_auc",
        help="The per-(repetition, fold) metric to draw as a distribution "
             "(default: %(default)s)",
    )
    ap.add_argument("--x", default=DEFAULT_X, help=f"Scatter x key (default: {DEFAULT_X})")
    ap.add_argument("--y", default=DEFAULT_Y, help=f"Scatter y key (default: {DEFAULT_Y})")
    ap.add_argument(
        "--pair", action="store_true",
        help="Also run the corrected paired test of each run against the first. "
             "Refuses on a dataset-fingerprint mismatch.",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Where the PNGs go (default: {DEFAULT_OUT})")
    ap.add_argument("--prefix", default="runs", help="Filename prefix for the figures")
    args = ap.parse_args()
    if len(args.runs) < 2:
        ap.error(
            "Name at least two runs - one run on its own is its own run page.\n"
            "  python scripts/compare_runs.py lightgbm_quantiles lightgbm_pooled"
        )
    return args


def same_quantity(x_key: str, y_key: str) -> bool:
    """Do two flat keys measure the same thing, so that y = x means something?

    `oof/pr_auc` against `depth/3/pr_auc` - yes, both PR AUC, and the diagonal is
    "no collapse". `oof/pr_auc` against `calib/ece` - no, and a diagonal there
    would be a line a reader could only misread.
    """
    return x_key.rsplit("/", 1)[-1] == y_key.rsplit("/", 1)[-1]


def label_runs(runs: list[dict]) -> None:
    """Make every run's `name` unique within this comparison, in place.

    W&B run names are **not** unique - re-running the same config produces
    another run with the same display name, which is the normal case here rather
    than an edge one. Everything downstream keys on `name`: the distribution
    figure uses it for the row, the scatter for the point label, and
    `paired_comparison` for the arm. Two runs called `lightgbm_quantiles` would
    collide in the figure's dict and one would silently vanish.

    So where a name is shared, the run id is appended and `name` becomes the
    display label. The original stays as `wandb_name` for anyone who needs to
    look the run back up.
    """
    seen: dict[str, int] = {}
    for run in runs:
        seen[run["name"]] = seen.get(run["name"], 0) + 1
    for run in runs:
        run["wandb_name"] = run["name"]
        if seen[run["name"]] > 1:
            run["name"] = f"{run['name']} ({run['id']})"


def fetch(references: list[str], metric: str, log=print) -> list[dict]:
    """Pull every named run, reporting what each one turned out to be missing."""
    runs = [tracking.fetch_run(reference, metric) for reference in references]
    label_runs(runs)
    for record in runs:
        log(
            f"  {record['name']:<34} {record['id']}  "
            f"{len(record['observations'])} observations  {record['url']}"
        )
    return runs


def draw_distributions(runs: list[dict], metric: str, out: Path, prefix: str, log=print):
    """Every run's metric vector on one axis, one row each.

    The figure W&B cannot produce: its distribution panels are one image per run,
    so three models is three pictures and no shared axis.
    """
    from m6a import figures

    vectors = {
        run["name"]: [float(o[metric]) for o in run["observations"]]
        for run in runs
        if run["observations"]
    }
    missing = [run["name"] for run in runs if not run["observations"]]
    if missing:
        log(
            f"\n  no per-fold {metric} vector on: {', '.join(missing)} - left off the\n"
            "  distribution figure. A run logged before the harness existed has "
            "only\n  its pooled number."
        )
    if not vectors:
        log("  nothing to draw: no run carries a per-fold vector.")
        return None

    counts = {len(v) for v in vectors.values()}
    caption = (
        "Each point is one fold of one repetition, pulled from W&B. Runs with "
        "different\nobservation counts are not equally precise - read the n beside "
        "each row."
    )
    if len(counts) > 1:
        log(
            f"\n  NOTE: observation counts differ {sorted(counts)}. A 5-point run and a\n"
            "  50-point run are not equally precise and their spreads are not "
            "comparable\n  (docs/decisions/0013). Both are drawn; the n is on the figure."
        )

    # The reference gets slot 2 - the thing being compared against - and every
    # candidate slot 1. Two hues only: beyond BLUE/ORANGE this palette has no
    # pair that survives the colour-vision check, and it does not need one here
    # because every row is labelled (docs/decisions/0020).
    reference = runs[0]["name"]
    colours = [
        figures.ORANGE if name == reference else figures.BLUE for name in vectors
    ]
    pretty = metric.replace("_", " ").upper()
    figure = figures.metric_distribution(
        vectors,
        metric=pretty,
        title=f"{pretty} across folds and repetitions, {len(vectors)} runs",
        caption=caption,
        colours=colours,
    )
    path = out / f"{prefix}_distribution.png"
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    log(f"\n  {path}")
    return figure


def draw_scatter(runs: list[dict], x_key: str, y_key: str, out: Path, prefix: str, log=print):
    """One labelled point per run, with the y = x diagonal where it means something."""
    from m6a import figures

    points, missing = [], []
    for run in runs:
        x, y = run["summary"].get(x_key), run["summary"].get(y_key)
        if x is None or y is None:
            missing.append(run["name"])
            continue
        points.append({"name": run["name"], "x": float(x), "y": float(y)})

    if missing:
        log(
            f"\n  {', '.join(missing)} has no {x_key} or {y_key}, so it is not on the\n"
            "  scatter. A run evaluated at --profile quick logs no depth keys."
        )
    if not points:
        log(f"  nothing to draw: no run carries both {x_key} and {y_key}.")
        return None

    diagonal = same_quantity(x_key, y_key)
    if not diagonal:
        log(
            f"\n  {x_key} and {y_key} are not the same quantity, so no y = x line is\n"
            "  drawn - a diagonal between two different measures means nothing."
        )
    figure = figures.run_scatter(
        points, x_key, y_key,
        f"{y_key} against {x_key}, one point per run",
        diagonal=diagonal,
        reference=runs[0]["name"],
    )
    path = out / f"{prefix}_scatter.png"
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    log(f"  {path}")
    return figure


def pair_against_reference(runs: list[dict], metric: str, log=print) -> None:
    """The corrected paired test of every other run against the first one named.

    A picture says which run is higher; it does not say whether that survives the
    overlap between training folds. This is the number that does
    (docs/decisions/0006), and it refuses to run across mismatched data
    (docs/decisions/0015).
    """
    from m6a.compare import paired_comparison, verdict

    reference = runs[0]
    if not reference["observations"]:
        log(
            f"\n  {reference['name']} has no per-fold vector, so there is nothing to "
            "pair\n  against. Name a run that does as the first argument."
        )
        return

    missing = [k for k, v in reference["fingerprint"].items() if v is None]
    if missing:
        log(
            f"\n  {reference['name']} does not record {', '.join(missing)}, so no other "
            "run\n  can be checked against it. Re-run it; every run logged after\n"
            "  docs/decisions/0015 carries the fingerprint."
        )
        return

    for candidate in runs[1:]:
        if not candidate["observations"]:
            continue
        log("")
        try:
            tracking.require_same_dataset(
                reference["fingerprint"], {**candidate["fingerprint"],
                                           "run_id": candidate["id"]},
                candidate["name"],
            )
        except SystemExit as exc:
            # One mismatched pair must not kill the others: the figures are
            # already written and the remaining runs may pair perfectly well.
            log(f"  not paired with {reference['name']}:\n{exc}")
            continue

        result = paired_comparison(
            reference["observations"], candidate["observations"],
            reference["name"], candidate["name"],
            n_folds=int(reference["fingerprint"]["split_n_folds"]),
        )
        log(
            f"  {candidate['name']} vs {reference['name']}: "
            f"{result['mean_difference']:+.4f}  "
            f"wins {result['wins']}/{result['n_folds']}  "
            f"corrected p = {result['p_value_corrected']:.4g}"
        )
        log(f"    {verdict(result)}")


def main() -> int:
    args = parse_args()
    load_env()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Pulling {len(args.runs)} runs from W&B")
    runs = fetch(args.runs, args.metric)

    figures_drawn = [
        draw_distributions(runs, args.metric, out, args.prefix),
        draw_scatter(runs, args.x, args.y, out, args.prefix),
    ]

    if args.pair:
        pair_against_reference(runs, args.metric)

    from m6a.tracking import close_figures

    close_figures({str(i): f for i, f in enumerate(figures_drawn) if f is not None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
