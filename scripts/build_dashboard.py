#!/usr/bin/env python
"""Build the project's W&B dashboard from code.

    python scripts/build_dashboard.py            # create or update both views
    python scripts/build_dashboard.py --dry-run  # print what would be built

Two saved views of the shared workspace (docs/decisions/0030):

  Decisions          runs scored on held-out data (eval_schema >= 2), sorted by
                     xsrc/gain_mean - the selection rule of docs/decisions/0029
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
ARMS = ("dataset0", "data1", "pooled", "pooled_both", "ensemble")


def note(ws, wr, text: str):
    return wr.MarkdownPanel(markdown=text)


def decisions_view(ws, wr, entity: str, project: str):
    read_me = (
        "**How to read this view.** Every row is a model evaluated against a baseline "
        "(usually `quantiles`) on held-out genes of BOTH files - dataset0 and data1, "
        "which are labelled differently ([decision 0029](https://github.com/AsJayTee/"
        "DSA4262-Foursight/blob/main/docs/decisions/0029-models-are-selected-on-cross-source-gain.md)).\n\n"
        "**The rule:** choose on `xsrc/gain_mean` (mean gain across the two labellings); "
        "reject if `xsrc/eligible` is 0 (the worse gain is below -0.005). "
        "`oof/pr_auc` is dataset0 cross-validation only and over-ranked `everything` by ~3x - "
        "it is context, not the decision.\n\n"
        "Make a row with `python scripts/evaluate.py --config <cfg> --cross-source`."
    )
    transfer = (
        "**Crossed test** (shared sites, trained on dataset0), gain over the baseline: "
        "A = dataset0 run + dataset0 labels; B = labels swapped to data1's; "
        "C = run swapped to data1's; D = both. A gain that drops A->B is tied to "
        "dataset0's labelling; A->C, to its sequencing run."
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
            wr.BarPlot(title="Selection rule: mean gain and worst gain over the baseline",
                       metrics=[wr.SummaryMetric("xsrc/gain_mean"),
                                wr.SummaryMetric("xsrc/gain_worst")]),
            wr.BarPlot(title="Gain under each labelling (trained on dataset0, as shipped)",
                       metrics=[wr.SummaryMetric("xsrc/dataset0/dataset0/gain"),
                                wr.SummaryMetric("xsrc/dataset0/data1/gain")]),
            wr.ScalarChart(title="Eligible (1 = passes the -0.005 veto)",
                           metric=wr.SummaryMetric("xsrc/eligible")),
        ]),
        ws.Section(name="Does it transfer?", is_open=True, panels=[
            wr.ScatterPlot(title="Gain on dataset0 (x) vs gain on data1 (y) - on the diagonal = transfers",
                           x=wr.SummaryMetric("xsrc/dataset0/dataset0/gain"),
                           y=wr.SummaryMetric("xsrc/dataset0/data1/gain")),
            wr.BarPlot(title="Crossed test: where a gain is lost",
                       metrics=[wr.SummaryMetric(f"xsrc/crossed/{c}/gain") for c in "ABCD"]),
            wr.BarPlot(title="data1 genes absent from dataset0 (gene population)",
                       metrics=[wr.SummaryMetric("xsrc/data1_new_genes/gain")]),
            note(ws, wr, transfer),
        ]),
        ws.Section(name="What should it train on?", is_open=True, panels=[
            wr.BarPlot(title="Mean gain by training arm",
                       metrics=[wr.SummaryMetric(f"xsrc/{a}/gain_mean") for a in ARMS]),
            wr.BarPlot(title="Raw data1 PR AUC by training arm",
                       metrics=[wr.SummaryMetric(f"xsrc/{a}/data1/pr_auc") for a in ARMS]),
            wr.BarPlot(title="Raw dataset0 PR AUC by training arm",
                       metrics=[wr.SummaryMetric(f"xsrc/{a}/dataset0/pr_auc") for a in ARMS]),
        ]),
        ws.Section(name="Does the score track modification? (data2)", is_open=False, panels=[
            wr.LinePlot(title="Mean score vs fraction of molecules modified",
                        x="curve/data2/fraction", y=["curve/data2/mean_score"]),
            note(ws, wr, data2),
        ]),
        ws.Section(name="Held out on data1 (standard runs)", is_open=False, panels=[
            wr.BarPlot(title="data1 PR AUC by slice (new genes is leak-free)",
                       metrics=[wr.SummaryMetric(f"ext/data1/{s}/pr_auc")
                                for s in ("new_genes", "new_transcripts", "new_sites")]),
        ]),
    ]
    runset = ws.RunsetSettings(
        filters=[ws.Config("eval_schema") >= 2],
        order=[ws.Ordering(ws.Summary("xsrc/gain_mean"), ascending=False)],
        pinned_columns=["config:features", "config:baseline", "summary:xsrc/gain_mean",
                        "summary:xsrc/gain_worst", "summary:xsrc/eligible",
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
            wr.BarPlot(title="Pooled OOF PR AUC and the 50-observation mean",
                       metrics=[wr.SummaryMetric("oof/pr_auc"), wr.SummaryMetric("rep/pr_auc_mean")]),
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
    for build in (decisions_view, history_view):
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
