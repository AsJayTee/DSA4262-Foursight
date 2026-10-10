"""The report's mechanism and biology figures, from result files already in the repo (10 Oct).

    python analysis/representation/report_figures.py

No torch, no data: every figure reads a CSV/JSON that a named script wrote.
One colour code throughout: blue = cell line 1 (dataset0), orange = cell line 2
(data1). Writes report/figures/fig_*.png:

  fig_corroboration_curve   the shipped scalar_drop: site score vs one neighbour's (models/final)
  fig_gps_attention         where the graph transformer attends; cutting it     (probe_day.py)
  fig_radius                gain vs neighbour radius, per cell line              (xsrc_nets.py)
  fig_comodification        co-modification by distance, local vs transcript     (distance_bands.py)
  fig_gene_structure        positive rate around exon junctions and stop codons  (annotation_context.py)
  fig_ceiling               label oracle and the premise test                    (label_oracle.py, data1_arm_eval.py)
  fig_depth                 PR AUC vs reads per site                             (depth rescoring)
  fig_neighbour_count       graph gain by number of neighbours                   (gain_decomposition.py)
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import common
from m6a import figures as F

FIGURES = common.ROOT / "report" / "figures"
NEWDATA = common.ROOT / "analysis" / "newdata"
R = common.RESULTS
CL1, CL2 = "#2a78d6", "#eb6834"          # validated pair (dataviz palette slots 1-2)
LINES = {"cell line 1": CL1, "cell line 2": CL2}


def figure(ncols: int = 1, width: float = 7.2, height: float = 4.0):
    plt = F._plt()
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=200, squeeze=False)
    fig.patch.set_facecolor(F.SURFACE)
    for ax in axes[0]:
        ax.set_facecolor(F.SURFACE)
        ax.grid(True, color=F.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(F.GRID)
        ax.tick_params(colors=F.MUTED, labelsize=8, length=0)
    return fig, list(axes[0])


def finish(fig, title: str, caption: str, name: str) -> None:
    import textwrap
    width = fig.get_size_inches()[0]
    # ~13 title characters per inch at 10.5 pt: wrap rather than run off the canvas.
    title = "\n".join(textwrap.wrap(title, int(width * 13)))
    fig.suptitle(title, x=0.012, ha="left", color=F.INK, fontsize=10.5)
    # ~19 caption characters per inch at 7 pt.
    caption = "\n".join(line for part in caption.split("\n") for line in textwrap.wrap(part, int(width * 19)))
    fig.text(0.012, 0.01, caption, color=F.MUTED, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.08 + 0.025 * caption.count("\n"), 1, 0.93))
    fig.savefig(FIGURES / f"{name}.png")
    print(f"  {name}.png")


def labels(ax, x: str, y: str, title: str | None = None) -> None:
    ax.set_xlabel(x, color=F.INK_SOFT, fontsize=8.5)
    ax.set_ylabel(y, color=F.INK_SOFT, fontsize=8.5)
    if title:
        ax.set_title(title, color=F.INK, fontsize=9, loc="left")


def corroboration_curve() -> None:
    """What the SHIPPED scalar_drop networks (models/final, numpy) do for one concrete case: a site
    whose own reads give probability p, with ONE neighbour (within 75 nt) whose own reads give q.
    Both have 30 reads. Averaged over the two shipped seeds."""
    def logit(p):
        return np.log(p / (1 - p))

    nets = [np.load(common.ROOT / "models" / "final" / f"scalar_drop_s{s}.npz") for s in (0, 1)]
    q = np.linspace(0.01, 0.95, 120)
    logn = np.log1p(30.0)
    fig, axes = figure(1, 7.4, 4.6)
    ax = axes[0]
    for own, shade in ((0.05, F.MUTED), (0.20, F.INK_SOFT), (0.50, F.INK)):
        out = []
        for w in nets:
            # One neighbour: the weighted mean and the best neighbour are both its score (scalar messages).
            feats = np.stack([np.full_like(q, logit(own)), np.full_like(q, logn), logit(q), logit(q),
                              np.full_like(q, logn), np.ones_like(q)], -1).astype(np.float32)
            h = np.maximum(feats @ w["c_mlp.0.weight"].T + w["c_mlp.0.bias"], 0)
            out.append(logit(own) + (h @ w["c_mlp.2.weight"].T + w["c_mlp.2.bias"])[:, 0])
        final = 1 / (1 + np.exp(-np.mean(out, 0)))
        ax.plot(100 * q, 100 * final, color=shade, linewidth=2.0)
        ax.axhline(100 * own, color=shade, linewidth=0.8, linestyle=":")
        ax.text(96, 100 * final[-1], f"own reads say {own:.0%}", fontsize=7.5, color=shade, va="center")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    labels(ax, "neighbour's own-reads score (%)", "site's final score (%)")
    finish(fig, "Corroboration: a neighbour that looks modified raises a site's score; one that looks unmodified lowers it",
           "The shipped scalar_drop networks (both seeds, numpy), for a site with one neighbour within 75 nt, both with "
           "30 reads. Dotted: the site's own-reads score,\ni.e. no neighbour effect. Scores are rankings from networks "
           "trained with positives up-weighted, not calibrated probabilities.", "fig_corroboration_curve")


def gps_attention() -> None:
    att = pd.read_csv(R / "probe_gps_attention.csv")
    cut = pd.read_csv(R / "probe_gps.csv")
    fig, axes = figure(2, 9, 3.9)
    a = att[att["trained on"] == "dataset0"].groupby("distance band", sort=False)["attention share"].mean()
    ax = axes[0]
    ax.bar(range(len(a)), 100 * a.to_numpy(), color=F.MUTED, width=0.6)
    ax.set_xticks(range(len(a)))
    ax.set_xticklabels(a.index, fontsize=7.5)
    for i, v in enumerate(a):
        ax.text(i, 100 * v + 0.6, f"{100 * v:.0f}%", ha="center", fontsize=7.2, color=F.INK)
    labels(ax, "distance from the scored site", "share of attention (%)", "where its attention goes")
    ax = axes[1]
    g = cut[cut["trained on"] == "dataset0"].set_index("scoring")
    full = g.loc["as trained (whole transcript)"]
    steps = ["as trained (whole transcript)", "attention cut to 150 nt", "attention cut to 50 nt"]
    for line, colour in LINES.items():
        ax.plot(range(3), [g.loc[s, line] - full[line] for s in steps], "o-", color=colour, linewidth=1.8,
                label=line + (" (training line)" if line == "cell line 1" else " (unseen line)"))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["whole transcript\n(as trained)", "cut to 150 nt", "cut to 50 nt"], fontsize=7.5)
    ax.axhline(0, color=F.MUTED, linewidth=0.8)
    labels(ax, "attention range at scoring time", "change in PR AUC", "cutting the long range")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=F.INK_SOFT)
    finish(fig, "The graph transformer reads transcript-wide context, and that context is cell-line-specific",
           "gps trained on cell line 1 (seed 0, 5 folds). Half its attention lands beyond 150 nt; cutting it costs "
           "the training line 2.6-3x more than the unseen line.\nCut attention is an input the network never saw - "
           "read the asymmetry, not the size. Source: results/probe_gps*.csv.", "fig_gps_attention")


def radius() -> None:
    rows = []
    for nt, model in ((50, "h2gcn_aux"), (100, "h2gcn_aux_r100"), (200, "h2gcn_aux_r200"), (400, "h2gcn_aux_r400")):
        x = json.load(open(R / f"{model}_nn__xsrc.json"))["cross_source"]["arms"]
        rows.append({"radius": nt, "cell line 2": x["dataset0"]["data1"]["gain"],
                     "cell line 1": x["pooled_both"]["dataset0"]["gain"]})
    t = pd.DataFrame(rows)
    fig, axes = figure(1, 6.4, 4.0)
    ax = axes[0]
    names = {"cell line 1": "cell line 1 (training line, held-out genes)", "cell line 2": "cell line 2 (unseen line)"}
    for line, colour in LINES.items():
        ax.plot(t.radius, t[line], "o-", color=colour, linewidth=1.8, label=names[line])
    ax.set_xscale("log", base=2)
    ax.set_xticks(t.radius)
    ax.set_xticklabels([f"{r} nt" for r in t.radius])
    ax.axhline(0, color=F.MUTED, linewidth=0.8)
    labels(ax, "neighbour radius", "PR AUC gain over LightGBM")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=F.INK_SOFT)
    finish(fig, "Wider context helps the training cell line; ~100 nt is what transfers",
           "h2gcn_aux with only the radius changed, seed 0. Cell line 2: trained on cell line 1 only. Cell line 1: "
           "trained on both, held-out genes.\nSource: results/h2gcn_aux*_nn__xsrc.json.", "fig_radius")


def comodification() -> None:
    b = pd.read_csv(NEWDATA / "distance_bands" / "bands.csv")
    fig, axes = figure(1, 6.8, 4.0)
    ax = axes[0]
    for line, key, colour in (("cell line 1", "dataset0 (cell line 1), all sites", CL1),
                              ("cell line 2", "data1 (cell line 2), all sites", CL2)):
        g = b[b.data == key]
        x = np.arange(len(g))
        ax.plot(x, g.local_part, "o-", color=colour, linewidth=1.8, label=f"{line}: local clustering")
        ax.fill_between(x, g.local_lo, g.local_hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, g.transcript_part, ":", color=colour, linewidth=1.4, label=f"{line}: m6A-rich transcripts")
    ax.set_xticks(x)
    ax.set_xticklabels(g.band, fontsize=7.5)
    ax.axhline(1, color=F.MUTED, linewidth=0.8)
    labels(ax, "distance between two sites on one transcript", "enrichment of modified pairs (x)")
    ax.legend(frameon=False, fontsize=7, labelcolor=F.INK_SOFT, ncol=2)
    finish(fig, "Modified sites cluster within ~100-200 nt, the same in both cell lines",
           "Solid: pairs of modified sites beyond what each transcript's own positive rate predicts\n(95% interval, "
           "resampled transcripts). Dotted: the part explained by m6A-rich transcripts, which differs\nby cell line "
           "(3.3x vs 2.0x). Source: analysis/newdata/distance_bands/bands.csv.", "fig_comodification")


def _profile(path):
    t = pd.read_csv(path, header=[0, 1], index_col=0).dropna(how="all")
    lift = t["lift vs file rate"]
    mids = [np.mean([float(v) for v in re.findall(r"-?\d+", s)[:2]]) for s in lift.index]
    return np.array(mids), lift


def gene_structure() -> None:
    fig, axes = figure(2, 9.6, 3.9)
    for ax, (fname, xlab, title) in zip(axes, (
            ("distance_from_the_modified_a_to_the_near.csv", "distance to the nearest exon junction (nt)",
             "exon junctions"),
            ("distance_to_the_stop_codon_nt_negative_c.csv", "position relative to the stop codon (nt)",
             "stop codon"))):
        x, lift = _profile(NEWDATA / "annotation_context" / fname)
        if "stop" not in fname:
            # Bins of unequal width and an open last one ([400, inf)): evenly spaced, labelled by bin.
            x = np.arange(len(lift))
            ax.set_xticks(x)
            bins = [[int(float(v)) for v in re.findall(r"-?\d+", b)[:2]] for b in lift.index]
            ax.set_xticklabels([f"{lo}+" if hi > 10**6 else f"{lo}-{hi}" for lo, hi in bins], fontsize=7.5)
        for line, col, colour in (("cell line 1", "dataset0", CL1), ("cell line 2", "data1", CL2)):
            ax.plot(x, lift[col], "o-", color=colour, linewidth=1.6, markersize=3, label=line)
        ax.axhline(1, color=F.MUTED, linewidth=0.8)
        if "stop" in fname:
            ax.axvline(0, color=F.INK_SOFT, linewidth=0.8, linestyle="--")
        labels(ax, xlab, "positive rate relative to the file's rate (x)", title)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=F.INK_SOFT)
    finish(fig, "Gene structure shapes where m6A sits, in both cell lines",
           "Ensembl 91 annotation, used for analysis only (never a model input). Exclusion near junctions and "
           "enrichment after the stop codon match the exon-junction-complex\nmodel (Uzonyi 2023, He 2023, Luo 2023). "
           "Source: analysis/newdata/annotation_context/.", "fig_gene_structure")


def ceiling() -> None:
    o = pd.read_csv(R / "label_oracle.csv")
    o = o[o["scored against"] == "cell line 2 labels"]
    p = pd.read_csv(R / "data1_arm.csv")
    fig, axes = figure(2, 9.6, 3.9)
    ax = axes[0]
    names = {"cell line 1's label alone": "cell line 1's true labels",
             "cell line 1's label, ties broken by the model": "true labels +\nmodel to break ties",
             "model trained on cell line 1 (h2gcn_aux)": "our model,\ntrained on cell line 1"}
    vals = [o.set_index("score").loc[k, "pr_auc"] for k in names]
    ax.bar(range(3), vals, color=[F.MUTED, F.MUTED, CL1], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color=F.INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names.values(), fontsize=7.5)
    labels(ax, "", "PR AUC against cell line 2's labels", "a perfect cell line 1 model, on cell line 2")
    ax = axes[1]
    order = ["cell line 1 only", "cell line 2 only", "both"]
    for model, colour, mk in (("h2gcn_aux", F.INK, "o"), ("gps", F.MUTED, "s")):
        v = [p.set_index("comparison").loc[f"{model}, trained on {o_}: PR AUC on cell line 2", "value"] for o_ in order]
        ax.plot(range(3), v, mk + "-", color=colour, linewidth=1.6, label="constrained H2GCN" if model == "h2gcn_aux"
                else "graph transformer")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["cell line 1", "cell line 2\n(its own labels)", "both"], fontsize=7.5)
    labels(ax, "trained on", "PR AUC on cell line 2", "does training on cell line 2 itself help?")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=F.INK_SOFT)
    finish(fig, "Part of the cross-cell-line gap cannot be closed from these labels",
           "Left: 67,320 sites in both files; knowing cell line 1's labels perfectly scores below our reads-based model. "
           "Right: seed 0, cell line 2's held-out genes;\nits own labels add 0.007 (constrained) or 0.019 (transformer). "
           "Sources: results/label_oracle.csv, data1_arm.csv.", "fig_ceiling")


def depth() -> None:
    rows = []
    for f in ("depth_rescore_final_batch.json", "depth_rescore_twohead_aux.json"):
        rows += json.load(open(R / f))
    d = pd.DataFrame(rows)
    d = d[d.arm == "pooled_both"]
    full = pd.read_csv(R / "report_models_pooled.csv")
    full = full[full.version == "both"]
    fig, axes = figure(2, 9.6, 3.9)
    for ax, (src, line) in zip(axes, (("dataset0", "cell line 1"), ("data1", "cell line 2"))):
        # Lines are models here, so neither cell-line colour: ink, solid vs dash-dot.
        for model, ls, name in (("h2gcn_twohead_aux", "o-", "two-head H2GCN"),
                                ("h2gcn_aux", "s-.", "H2GCN + own-reads head")):
            g = d[(d.model == model) & (d.source == src)].groupby("depth").pr_auc.mean()
            top = full[(full.model == model) & (full["scored on"] == line)].pr_auc.iloc[0]
            ax.plot(list(g.index) + [40], list(g.values) + [top], ls, color=F.INK, linewidth=1.6, label=name)
        lg = d[(d.model == "h2gcn_aux") & (d.source == src)].groupby("depth").lgbm_pr_auc.mean()
        lt = full[(full.model == "lightgbm") & (full["scored on"] == line)].pr_auc.iloc[0]
        ax.plot(list(lg.index) + [40], list(lg.values) + [lt], "o--", color=F.MUTED, linewidth=1.4, label="LightGBM")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 40])
        ax.set_xticklabels(["1", "3", "10", "all (>=20)"])
        labels(ax, "reads per site at scoring", "PR AUC", line)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=F.INK_SOFT)
    finish(fig, "Every model loses most of its signal at the read depths SG-NEx actually has",
           "Trained on both cell lines; scored with each held-out site's reads subsampled. SG-NEx median depth is ~3 "
           "reads. Seeds averaged where available.\nSources: results/depth_rescore*.json, report_models_pooled.csv.",
           "fig_depth")


def neighbour_count() -> None:
    g = pd.read_csv(R / "gain_decomposition.csv")
    g = g[(g.stratum == "neighbours within 50 nt") & (g["trained on"] == "dataset0")]
    order = ["0", "1", "2-3", "4+"]
    fig, axes = figure(1, 6.4, 4.0)
    ax = axes[0]
    for (src, line), off in zip((("dataset0", "cell line 1"), ("data1", "cell line 2")), (-0.08, 0.08)):
        t = g[g["scored on"] == src].set_index(g[g["scored on"] == src].level.astype(str)).loc[order]
        x = np.arange(4) + off
        ax.errorbar(x, t.gain, yerr=[t.gain - t.ci_low, t.ci_high - t.gain], fmt="o-", color=LINES[line],
                    linewidth=1.6, capsize=0, elinewidth=1.0,
                    label=line + (" (unseen line)" if line == "cell line 2" else " (held-out genes)"))
    ax.set_xticks(range(4))
    ax.set_xticklabels(order)
    ax.axhline(0, color=F.MUTED, linewidth=0.8)
    labels(ax, "other candidate sites within 50 nt", "PR AUC gain of the graph over reads alone")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=F.INK_SOFT)
    finish(fig, "The graph's gain grows with the number of neighbours on the training line, less on the unseen one",
           "h2gcn_aux minus DeepSet (same read encoder, no graph), trained on cell line 1; 95% gene-resampled "
           "intervals.\nSource: results/gain_decomposition.csv.", "fig_neighbour_count")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for make in (corroboration_curve, gps_attention, radius, comodification, gene_structure, ceiling, depth,
                 neighbour_count):
        make()


if __name__ == "__main__":
    main()
