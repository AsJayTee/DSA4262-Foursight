"""Does the shipped model's score rise with the share of modified molecules? (dataset2, 10 Oct)

    python analysis/newdata/data2_scores.py

dataset2 (the course's in-vitro series): one synthetic RNA, sequenced as seven
samples made with a KNOWN share of modified molecules - 0, 25, 50, 70, 75, 95 and
100% - at the same 189 positions (data.info: one "transcript" per sample, label =
the fraction). It was never used for training.

Scored with the shipped model (models/final, numpy), two scores per site:
  final      the ensemble's raw output, averaged over its six networks (uses neighbours)
  own reads  the constrained designs' read-only score (no neighbours)
Every neighbour here shares the site's fraction, so corroboration should sharpen
the trend - the mechanism in the clearest possible setting.

Per-site scores (with the fraction labels) go to .cache/representation/ (course
data, not committed); the per-fraction summary to results/data2_score_summary.csv
and the figure to report/figures/fig_data2_fractions.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "sgnex"))

from m6a.data import iter_sites  # noqa: E402
from m6a.models.site_graph import SiteGraphEnsemble  # noqa: E402
from score_sgnex import score  # noqa: E402

DATA = ROOT / "data0" / "data2"
CACHE = ROOT / ".cache" / "representation" / "data2_scores.csv"
RESULTS = ROOT / "analysis" / "representation" / "results" / "data2_score_summary.csv"
FIGURE = ROOT / "report" / "figures" / "fig_data2_fractions.png"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main() -> None:
    ids, pos, kmers, counts, chunks = [], [], [], [], []
    for s in iter_sites(DATA / "dataset2.json.gz"):
        ids.append(s.transcript_id)
        pos.append(s.position)
        kmers.append(s.kmer)
        counts.append(len(s.reads))
        chunks.append(np.asarray(s.reads, dtype=np.float32))
    frame = pd.DataFrame({"transcript_id": ids, "transcript_position": pos, "kmer": kmers})
    out = score(SiteGraphEnsemble.load(ROOT / "models" / "final"), frame, np.concatenate(chunks), np.asarray(counts))
    info = pd.read_csv(DATA / "data.info")[["transcript_id", "transcript_position", "label"]]
    out = out.merge(info, on=["transcript_id", "transcript_position"]).rename(columns={"label": "fraction"})
    own_cols = [c for c in out.columns if c.startswith("own_")]
    out["own"] = out[own_cols].mean(axis=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)

    summary = (out.assign(final_pct=100 * sigmoid(out.score_raw), own_pct=100 * sigmoid(out.own))
               .groupby("fraction")
               .agg(sites=("score_raw", "size"), median_reads=("n_reads", "median"),
                    final_median=("final_pct", "median"), final_q25=("final_pct", lambda v: v.quantile(.25)),
                    final_q75=("final_pct", lambda v: v.quantile(.75)),
                    own_median=("own_pct", "median"), own_q25=("own_pct", lambda v: v.quantile(.25)),
                    own_q75=("own_pct", lambda v: v.quantile(.75))).reset_index())
    rho_final = spearmanr(out.fraction, out.score_raw).statistic
    rho_own = spearmanr(out.fraction, out.own).statistic
    summary.to_csv(RESULTS, index=False)
    print(summary.round(2).to_string(index=False))
    print(f"Spearman (site score vs fraction): final {rho_final:.3f}, own reads {rho_own:.3f}")
    draw(out, rho_final, rho_own)


def draw(out: pd.DataFrame, rho_final: float, rho_own: float) -> None:
    from m6a import figures as F
    plt = F._plt()
    fractions = sorted(out.fraction.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200, sharey=True)
    fig.patch.set_facecolor(F.SURFACE)
    rng = np.random.default_rng(4262)
    for ax, (col, title, rho) in zip(axes, (("score_raw", "final score (site + neighbours)", rho_final),
                                            ("own", "own-reads score (no neighbours)", rho_own))):
        ax.set_facecolor(F.SURFACE)
        ax.grid(True, axis="y", color=F.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(F.GRID)
        ax.tick_params(colors=F.MUTED, labelsize=8, length=0)
        vals = [100 * sigmoid(out.loc[out.fraction == f, col].to_numpy()) for f in fractions]
        body = ax.violinplot(vals, positions=range(len(fractions)), widths=0.8, showextrema=False)
        for b in body["bodies"]:
            b.set_facecolor("#2a78d6")
            b.set_edgecolor("none")
            b.set_alpha(0.18)
        for i, v in enumerate(vals):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=5, color="#2a78d6", alpha=0.55, linewidths=0)
        ax.plot(range(len(fractions)), [np.median(v) for v in vals], "o-", color=F.INK, linewidth=1.6,
                markersize=4, zorder=4, label="median")
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f"{100 * f:.0f}%" for f in fractions], fontsize=8)
        ax.set_xlabel("share of modified molecules in the sample", color=F.INK_SOFT, fontsize=8.5)
        ax.set_title(f"{title}\nSpearman with the fraction: {rho:.2f}", color=F.INK, fontsize=9, loc="left")
    axes[0].set_ylabel("model score (%)", color=F.INK_SOFT, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("dataset2: scores separate unmodified from modified RNA, but plateau above ~50% modified", x=0.012, ha="left",
                 color=F.INK, fontsize=10.5)
    fig.text(0.012, 0.01, "One synthetic RNA sequenced at 7 known modification levels; each point is one of its 189 "
             "positions. Never used for training. Shipped model (models/final). Scores are uncalibrated\n(training "
             "up-weighted modified sites); x positions are categories, not to scale. Every neighbour shares the site's "
             "level here, so neighbours reinforce the trend. Source: results/data2_score_summary.csv.",
             color=F.MUTED, fontsize=6.8, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(FIGURE)
    print(f"-> {FIGURE}")


if __name__ == "__main__":
    main()
