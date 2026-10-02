#!/usr/bin/env python
"""Build the project's W&B dashboard from code.

    python scripts/build_dashboard.py            # create or update both views
    python scripts/build_dashboard.py --dry-run  # print what would be built

Two saved views of the shared workspace (docs/decisions/0030):

  Decisions          runs scored on held-out data (eval_schema >= 2), sorted by
                     xsrc/gain_worst - the selection rule of docs/decisions/0032
  History (CV only)  the earlier runs (eval_schema 1) and the panels that read
                     them: depth collapse, calibration, thresholds, curves

**The views are owned by this file.** Change a panel here and re-run; an edit
made by hand in the W&B UI is overwritten on the next build. The view URLs are
kept in docs/wandb-views.json so every teammate updates the same views rather
than creating copies.

Needs the train extra plus wandb-workspaces (pip install -e '.[train]').
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a.env import load_env  # noqa: E402

VIEWS_FILE = ROOT / "docs" / "wandb-views.json"
from m6a.report import REFERENCE_RANGE  # noqa: E402

# Labels for the integer x axes of the ladder curves (m6a.report.cross_source_series).
# A W&B line panel cannot name its ticks, so the axis title carries the key.
TRANSFER_X = "0 = dataset0 labels  |  1 = data1 labels  |  2 = data1 genes not in dataset0"
CROSSED_X = "0 = A  |  1 = B labels swapped  |  2 = C run swapped  |  3 = D both swapped"
ARM_X = "trained on: 0 = dataset0  |  1 = pooled  |  2 = pooled_both  |  3 = data1  |  4 = ensemble"
DIAG = "curve/xsrc/diag/"
# How each series of the diagonal panel is drawn. W&B keys marks by run, so the
# build styles the runs that exist; run_xsrc_batch.py rebuilds after a batch.
DIAG_MARKS = {DIAG + "data1_gain": "points", DIAG + "y_equals_x": "dashed",
              DIAG + "veto_data1": "dotted", DIAG + "veto_dataset0": "dotted"}
DIAG_TITLES = {DIAG + "data1_gain": "gain", DIAG + "y_equals_x": "y = x (transfers)",
               DIAG + "veto_data1": "veto", DIAG + "veto_dataset0": "veto"}


def note(ws, wr, text: str):
    return wr.MarkdownPanel(markdown=text)


def per_run(run_ids: list[str], styles: dict[str, str]) -> dict[str, str]:
    return {f"{rid}:{key}": value for rid in run_ids for key, value in styles.items()}


def decisions_view(ws, wr, entity: str, project: str, run_ids: list[str]):
    read_me = (
        "**How to read this view.** Every run is a model evaluated against a baseline "
        "(usually `quantiles`) on held-out genes of BOTH files - dataset0 and data1, "
        "which are labelled differently ([decision 0029](https://github.com/AsJayTee/"
        "DSA4262-Foursight/blob/main/docs/decisions/0029-models-are-selected-on-cross-source-gain.md)). "
        "Every number is a **gain**: the model's PR AUC minus the baseline's, on identical sites.\n\n"
        "**The rule** ([0032](https://github.com/AsJayTee/DSA4262-Foursight/blob/main/docs/decisions/0032-rank-on-the-worse-gain.md)): rank on `xsrc/gain_worst`, the worse of the two gains (the table is sorted by it) - we do not know which labelling the test set uses. Gaps under ~0.005 are ties; break them on ""`xsrc/data1_new_genes/gain`. Reject if "
        "`xsrc/eligible` is 0 (the worse gain is below -0.005). `oof/pr_auc` is dataset0 "
        "cross-validation only and over-ranked `everything` by ~3x - context, not the decision.\n\n"
        "**Reading the lines.** Most panels have an integer x axis: each point is one "
        "condition, named in the axis title. A flat line means the gain does not depend on "
        "the condition; a line that falls means the gain is lost there. Hover for exact "
        "values; filter the run table to compare two runs.\n\n"
        "Make a run with `python scripts/evaluate.py --config <cfg> --cross-source`. "
        "Panel guide: `docs/wandb-panels.md`."
    )
    diag = (
        "**The diagonal.** Each dot is one model, as it would ship (trained on dataset0). "
        "x = its gain under dataset0's labels, y = under data1's. **On the dashed line** the "
        "gain transfers fully. **Below it** part of the gain belongs to dataset0's labelling - "
        "`everything` sits far below. **Left of or below a dotted line** the model is vetoed."
    )
    crossed = (
        "**Crossed test** on the 67,320 sites in both files, trained on dataset0. "
        "A = dataset0's measurements + dataset0's labels. B = data1's labels. "
        "C = data1's measurements (another sequencing run). D = both. "
        "A drop from A to B: the gain is tied to dataset0's labelling. A to C: to its run."
    )
    data2 = (
        "**data2** is one synthetic sequence at 0-100% modified molecules. A model that "
        "measures modification should rise steadily. Caveat: the samples were sequenced to "
        "different depths (median 550-1,205 reads), so read-count features confound this "
        "curve (`analysis/newdata/data2_audit.py`). Drawn only by standard runs."
    )
    sections = [
        ws.Section(name="Read me", is_open=True, panels=[note(ws, wr, read_me)]),
        ws.Section(name="Should we ship this?", is_open=True, panels=[
            wr.LinePlot(title="Gain on dataset0 vs data1 - on the dashed line = transfers",
                        x=DIAG + "dataset0_gain", y=list(DIAG_MARKS),
                        title_x="gain under dataset0's labels",
                        title_y="gain under data1's labels",
                        range_x=REFERENCE_RANGE, range_y=REFERENCE_RANGE,
                        line_marks=per_run(run_ids, DIAG_MARKS),
                        line_titles=per_run(run_ids, DIAG_TITLES)),
            wr.LinePlot(title="Gain as the test moves away from dataset0 (flat = transfers)",
                        x="curve/xsrc/transfer/step", y=["curve/xsrc/transfer/gain"],
                        title_x=TRANSFER_X, title_y="gain over baseline (PR AUC)"),
            note(ws, wr, diag),
        ]),
        ws.Section(name="Where is the gain lost?", is_open=True, panels=[
            wr.LinePlot(title="Crossed test: gain in cells A-D",
                        x="curve/xsrc/crossed/cell", y=["curve/xsrc/crossed/gain"],
                        title_x=CROSSED_X, title_y="gain over baseline (PR AUC)"),
            note(ws, wr, crossed),
            wr.LinePlot(title="Same, with 95% gene-bootstrap interval (filter to one run)",
                        x="curve/xsrc/transfer/step",
                        y=["curve/xsrc/transfer/gain", "curve/xsrc/transfer/ci_low",
                           "curve/xsrc/transfer/ci_high"],
                        title_x=TRANSFER_X, title_y="gain over baseline (PR AUC)"),
        ]),
        ws.Section(name="What should it train on?", is_open=True, panels=[
            wr.LinePlot(title="Mean gain by training data (context; the rule uses the lower line of the next panel)",
                        x="curve/xsrc/arm/index", y=["curve/xsrc/arm/gain_mean"],
                        title_x=ARM_X, title_y="mean gain over baseline"),
            wr.LinePlot(title="Gain by training data, under each labelling",
                        x="curve/xsrc/arm/index",
                        y=["curve/xsrc/arm/dataset0_gain", "curve/xsrc/arm/data1_gain"],
                        title_x=ARM_X, title_y="gain over baseline"),
            wr.LinePlot(title="Raw PR AUC trade-off: each vertex is a training arm "
                              "(dataset0 -> pooled -> pooled_both -> data1)",
                        x="curve/xsrc/path/dataset0_pr_auc", y=["curve/xsrc/path/data1_pr_auc"],
                        title_x="PR AUC on dataset0 held-out genes (4.5% positive)",
                        title_y="PR AUC on data1 held-out genes (7.3% positive)"),
        ]),
        ws.Section(name="Does the score track modification? (data2)", is_open=False, panels=[
            wr.LinePlot(title="Mean score vs fraction of molecules modified",
                        x="curve/data2/fraction", y=["curve/data2/mean_score"],
                        title_x="fraction of molecules modified", title_y="mean score"),
            note(ws, wr, data2),
        ]),
        ws.Section(name="Held out on data1 (standard runs)", is_open=False, panels=[
            wr.LinePlot(title="data1 PR AUC by how far from training (only 0 is leak-free)",
                        x="curve/ext/data1/slice",
                        y=["curve/ext/data1/pr_auc", "curve/ext/data1/ci_low",
                           "curve/ext/data1/ci_high"],
                        title_x="0 = new genes  |  1 = new transcripts  |  2 = new sites",
                        title_y="PR AUC"),
        ]),
    ]
    runset = ws.RunsetSettings(
        filters=[ws.Config("eval_schema") >= 2],
        order=[ws.Ordering(ws.Summary("xsrc/gain_worst"), ascending=False)],
        pinned_columns=["config:features", "config:baseline", "summary:xsrc/gain_worst",
                        "summary:xsrc/data1_new_genes/gain", "summary:xsrc/gain_mean", "summary:xsrc/eligible",
                        "summary:xsrc/dataset0/dataset0/gain", "summary:xsrc/dataset0/data1/gain",
                        "summary:xsrc/pooled_both/gain_mean", "summary:oof/pr_auc",
                        "summary:ext/data1/new_genes/pr_auc"],
    )
    return ws.Workspace(entity=entity, project=project, name="Decisions",
                        sections=sections, runset_settings=runset)


def history_view(ws, wr, entity: str, project: str):
    read_me = (
        "**Runs evaluated before 2026-09-29** (`eval_schema` 1): dataset0 cross-validation "
        "only. Their numbers are correct but answer a narrower question - see the Decisions "
        "view. Panel guide: `docs/wandb-panels.md`."
    )
    sections = [
        ws.Section(name="Read me", is_open=True, panels=[note(ws, wr, read_me)]),
        ws.Section(name="Headline (dataset0 CV)", is_open=True, panels=[
            wr.ScatterPlot(title="Accuracy at full depth (x) vs at SG-NEx depth 3 (y)",
                           x=wr.SummaryMetric("oof/pr_auc"), y=wr.SummaryMetric("depth/3/pr_auc")),
        ]),
        ws.Section(name="Depth collapse", is_open=True, panels=[
            wr.LinePlot(title="PR AUC vs log10 reads per site (SG-NEx median depth 3 is x = 0.48)",
                        x="curve/depth/log10_reads", y=["curve/depth/pr_auc"]),
            wr.LinePlot(title="Lift by true read-depth band",
                        x="curve/band/log10_reads", y=["curve/band/pr_auc_lift"]),
        ]),
        ws.Section(name="Calibration and counts", is_open=False, panels=[
            wr.LinePlot(title="Reliability (y = x is calibrated)",
                        x="curve/reliability/predicted", y=["curve/reliability/observed"]),
            wr.LinePlot(title="Threshold -> sites called (log10)",
                        x="curve/threshold/threshold", y=["curve/threshold/log10_predicted_positives"]),
            wr.ScatterPlot(title="Accuracy vs calibration", x=wr.SummaryMetric("oof/pr_auc"),
                           y=wr.SummaryMetric("calib/calibrated")),
        ]),
        ws.Section(name="Curves and fit progress", is_open=False, panels=[
            wr.LinePlot(title="Precision-recall", x="curve/pr/recall", y=["curve/pr/precision"]),
            wr.LinePlot(title="ROC", x="curve/roc/fpr", y=["curve/roc/tpr"]),
            wr.LinePlot(title="Fit progress - average precision (gradient-descent models)",
                        x="curve/train/iteration",
                        y=["curve/train/valid_pr_auc", "curve/train/train_pr_auc"]),
            wr.LinePlot(title="Fit progress - logloss", x="curve/train/iteration",
                        y=["curve/train/valid_logloss", "curve/train/train_logloss"]),
        ]),
    ]
    runset = ws.RunsetSettings(
        filters=[ws.Config("eval_schema") == 1],
        order=[ws.Ordering(ws.Summary("oof/pr_auc"), ascending=False)],
        pinned_columns=["config:features", "config:model", "config:train_depths",
                        "summary:oof/pr_auc", "summary:rep/pr_auc_mean", "summary:depth/1/pr_auc",
                        "summary:depth/3/pr_auc", "summary:calib/count_ratio"],
    )
    return ws.Workspace(entity=entity, project=project, name="History (CV only)",
                        sections=sections, runset_settings=runset)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Build the views, save nothing")
    args = ap.parse_args()
    load_env(ROOT / ".env")
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError:
        raise SystemExit("wandb-workspaces is not installed: pip install -e '.[train]'")
    entity = os.environ.get("WANDB_ENTITY") or "dsa4262-team"
    project = os.environ.get("WANDB_PROJECT", "dsa4262-project")

    recorded = json.loads(VIEWS_FILE.read_text()) if VIEWS_FILE.exists() else {}
    # wandb-workspaces joins the URL path with os.path on Windows, giving
    # "team\project" - unusable in a browser and unparseable by from_url.
    url = lambda v: v.replace("\\", "/")  # noqa: E731
    import wandb
    run_ids = [r.id for r in wandb.Api().runs(f"{entity}/{project}",
                                               filters={"config.eval_schema": {"$gte": 2}})]
    builds = (lambda *a: decisions_view(*a, run_ids=run_ids), history_view)
    for build in builds:
        view = build(ws, wr, entity, project)
        n_panels = sum(len(s.panels) for s in view.sections)
        if args.dry_run:
            print(f"[dry run] {view.name}: {len(view.sections)} sections, {n_panels} panels")
            continue
        if view.name in recorded:
            # Update the recorded view in place, rather than making a copy.
            existing = ws.Workspace.from_url(url(recorded[view.name]))
            existing.sections, existing.runset_settings = view.sections, view.runset_settings
            existing.save()
            view = existing
        else:
            view.save()
            recorded[view.name] = url(view.url)
        print(f"{view.name}: {n_panels} panels -> {url(view.url)}")
    if not args.dry_run:
        VIEWS_FILE.write_text(json.dumps({k: url(v) for k, v in recorded.items()},
                                         indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
