"""Task 2, deliverable 2: preliminary analyses A-E across the SG-NEx cell lines (10 Oct).

    python analysis/sgnex/explore_lines.py

Reads score_sgnex.py's output (data/sgnex_scores/, copied off Ronin). No torch. Uses
score_raw (the mean of the six networks' raw outputs): one scale for every sample,
so lines can be compared - unlike score_rank, which is a position within one file.
Raw scores still depend on read count (fewer reads pull them toward the middle), so
every comparison is made within a read-count band or on the same sites.

  A  trust checks   reads per site per line; do two samples of one line agree
                    (vs two different lines, on the same sites); does the motif
                    known to be strongest (GGACU) score highest in every line
  B  along genes    mean score by position around the stop codon and by distance
                    to the nearest exon junction (Ensembl 91; analysis only)
  C  how much m6A   mean score per line by read band; and the share of sites over
                    a cut-off borrowed from the labelled cell lines - set where
                    precision is 30/50/70% on cell line 1, then CHECKED on cell
                    line 2, where the answer is known, before being applied here
  D  genes          gene-level scores on sites well covered in every line: genes
                    high everywhere vs high in one line only
  E  similarity     correlation between lines (site and gene level), clustered

Writes analysis/sgnex/explore/*.csv, *.png and summary.md.
"""

from __future__ import annotations

import gzip
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "newdata"))
sys.path.insert(0, str(ROOT / "analysis" / "representation"))

from annotation_context import read_gtf, transcript_layout  # noqa: E402

SCORES = ROOT / "data" / "sgnex_scores"
GTF = ROOT / "data" / "annotation" / "gtf_91.gtf.gz"
OUT = ROOT / "analysis" / "sgnex" / "explore"
LINES = ["A549", "H9", "HEYA8", "Hct116", "HepG2", "K562", "MCF7"]
BANDS = [(1, 2), (3, 9), (10, 19), (20, 10**9)]       # reads per site
BAND_NAMES = ["1-2 reads", "3-9 reads", "10-19 reads", "20+ reads"]
WELL = 10                                              # "well covered" for D and E
COLS = ["transcript_id", "transcript_position", "kmer", "n_reads", "score_raw", "score_rank"]


def text_table(df: pd.DataFrame, index: bool = True) -> str:
    """A plain fixed-width table in a code block (no extra dependency)."""
    fence = "`" * 3
    return f"{fence}\n{df.to_string(index=index)}\n{fence}"


def band(n: np.ndarray) -> np.ndarray:
    out = np.full(len(n), "", dtype=object)
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        out[(n >= lo) & (n <= hi)] = name
    return out


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=lambda c: c in COLS)
    # Per-sample files of the first run kept versioned IDs and spike-ins (score_sgnex.py).
    d = d[d.transcript_id.str.startswith("ENST")]
    d["transcript_id"] = d.transcript_id.str.split(".").str[0]
    return d.drop_duplicates(["transcript_id", "transcript_position"])


def gene_names(wanted: set[str]) -> pd.DataFrame:
    rx = {k: re.compile(rf'{k} "([^"]+)"') for k in ("gene_id", "gene_name", "transcript_id", "transcript_biotype")}
    rows = []
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) > 8 and f[2] == "transcript":
                t = rx["transcript_id"].search(f[8]).group(1)
                if t in wanted:
                    rows.append({k: (m.group(1) if (m := r.search(f[8])) else "") for k, r in rx.items()})
    return pd.DataFrame(rows).set_index("transcript_id")


def annotate(sites: pd.DataFrame) -> pd.DataFrame:
    """Region, position relative to the stop codon, distance to the nearest junction."""
    wanted = set(sites.transcript_id)
    layout = transcript_layout(read_gtf(GTF, wanted))
    region = np.full(len(sites), "unannotated", dtype=object)
    to_stop = np.full(len(sites), np.nan)
    to_junction = np.full(len(sites), np.nan)
    for t, rows in sites.groupby("transcript_id").indices.items():
        lay = layout.get(t)
        if lay is None:
            continue
        p = sites.transcript_position.to_numpy()[rows]
        if lay["junctions"].size:
            j = np.sort(lay["junctions"])
            k = np.clip(np.searchsorted(j, p), 1, len(j)) - 1
            to_junction[rows] = np.minimum(np.abs(p - j[k]), np.abs(p - j[np.minimum(k + 1, len(j) - 1)]))
        if lay["stop"] is not None and lay["start"] is not None:
            to_stop[rows] = p - lay["stop"]
            region[rows] = np.where(p < lay["start"], "5'UTR", np.where(p <= lay["stop"] + 2, "CDS", "3'UTR"))
        else:
            region[rows] = "non-coding"
    return sites.assign(region=region, to_stop=to_stop, to_junction=to_junction)


# ---------------------------------------------------------------- C: borrowed cut-offs
def labelled_cutoffs() -> pd.DataFrame:
    """Raw-score cut-offs from the labelled cell lines, per read band, checked across lines.

    The cached out-of-fold raw outputs (trained on both lines, held-out genes) of the
    shipped designs - two-head, res_gate, scalar_drop, every seed - averaged as the
    shipped ensemble averages its networks. Full depth from the cross-validation
    cache; 1, 3 and 10 reads from depth_rescore.py's thinned scores."""
    import common  # noqa: F401  (analysis/representation; sets paths)
    import day_significance as D
    from m6a import crosssource as xs
    from m6a.config import Config
    cfg = Config.load(D.BASELINE)
    src = xs.load(cfg.features, [None], common.DATA_DIR, seed=cfg.split.seed, n_folds=cfg.split.n_folds,
                  log=D.quiet)
    n0 = len(src["dataset0"])
    y = {"cell line 1": src["dataset0"].y, "cell line 2": src["data1"].y}
    models = ["h2gcn_twohead_aux", "res_gate", "scalar_drop"]
    raw = {}
    full = [D.preds(m, "pooled_both", s) for m in models for s in (0, 1, 2)]
    raw["20+ reads"] = np.mean([p for p in full if p is not None], 0)
    for d, name in ((1, "1-2 reads"), (3, "3-9 reads"), (10, "10-19 reads")):
        got = [np.load(p)[f"d{d}"] for m in models for p in (D.NETS / "depth_scores").glob(f"{m}_pooled_both_s*.npz")]
        raw[name] = np.mean(got, 0)
    rows = []
    for name, score in raw.items():
        s1, s2 = score[:n0], score[n0:]
        prec, rec, thr = precision_recall_curve(y["cell line 1"], s1)
        for target in (0.3, 0.5, 0.7):
            ok = np.flatnonzero(prec[:-1] >= target)
            if not len(ok):
                continue
            t = thr[ok[0]]
            call1, call2 = s1 >= t, s2 >= t
            rows.append({"read band": name, "target precision (cell line 1)": target, "cut-off (raw)": t,
                         "cell line 1: called %": 100 * call1.mean(),
                         "cell line 1: precision": y["cell line 1"][call1].mean(),
                         "cell line 1: true rate %": 100 * y["cell line 1"].mean(),
                         "cell line 2: called %": 100 * call2.mean(),
                         "cell line 2: precision": y["cell line 2"][call2].mean() if call2.any() else np.nan,
                         "cell line 2: true rate %": 100 * y["cell line 2"].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Task 2 preliminary analyses (A-E)", "",
          "Generated by `analysis/sgnex/explore_lines.py` from the pooled SG-NEx scores "
          "(all of a cell line's samples combined). Score = `score_raw` (comparable across lines).", ""]
    pooled = {l: load(SCORES / f"{l}_pooled.csv.gz") for l in LINES}

    # A1. reads per site
    a1 = pd.DataFrame([{"line": l, "sites": len(d), "median reads": d.n_reads.median(),
                        **{f"{b} %": 100 * (band(d.n_reads.to_numpy()) == b).mean() for b in BAND_NAMES}}
                       for l, d in pooled.items()])
    a1.to_csv(OUT / "A1_reads_per_line.csv", index=False)
    md += ["## A1. Reads per site (samples combined)", "", a1.round(1).pipe(text_table, index=False), ""]

    # A2. do samples of one line agree, compared with two different lines, on the same sites?
    rows = []
    samples = {l: sorted(SCORES.glob(f"SGNex_{l}_directRNA_*.csv.gz")) for l in LINES}
    per = {p.name: load(p) for ps in samples.values() for p in ps}
    first = {l: per[ps[0].name] for l, ps in samples.items()}
    pairs = [("same line", l, a.name, b.name) for l, ps in samples.items() for a, b in itertools.combinations(ps, 2)]
    pairs += [("different lines", f"{l1} vs {l2}", samples[l1][0].name, samples[l2][0].name)
              for l1, l2 in itertools.combinations(LINES, 2)]
    for kind, label, a, b in pairs:
        m = per[a].merge(per[b], on=["transcript_id", "transcript_position"], suffixes=("_a", "_b"))
        low = np.minimum(m.n_reads_a, m.n_reads_b).to_numpy()
        for name in BAND_NAMES:
            sel = band(low) == name
            if sel.sum() > 1000:
                rows.append({"kind": kind, "pair": label, "read band (fewer of the two)": name, "sites": int(sel.sum()),
                             "spearman": m.score_raw_a[sel].corr(m.score_raw_b[sel], method="spearman")})
    a2 = pd.DataFrame(rows)
    a2.to_csv(OUT / "A2_sample_agreement.csv", index=False)
    a2s = a2.groupby(["kind", "read band (fewer of the two)"]).spearman.agg(["mean", "min", "max", "count"]).reset_index()
    md += ["## A2. Do samples agree? (Spearman correlation of raw score at shared sites)", "",
           "Same line = two samples of one cell line; different lines = one sample from each of two lines. "
           "If 'same line' is not clearly above 'different lines', single-site differences between lines "
           "cannot be told from sample noise.", "", a2s.round(3).pipe(text_table, index=False), ""]

    # A3. motif ranking per line (sites with >= WELL reads)
    a3 = pd.concat([d[d.n_reads >= WELL].assign(motif=lambda f: f.kmer.str[1:6]).groupby("motif").score_raw.mean()
                    .rename(l) for l, d in pooled.items()], axis=1)
    a3["mean"] = a3.mean(1)
    a3 = a3.sort_values("mean", ascending=False)
    a3.to_csv(OUT / "A3_motif_scores.csv")
    md += [f"## A3. Mean score by central 5-mer (sites with >= {WELL} reads), top 6", "",
           a3.head(6).round(2).pipe(text_table), "",
           "Known from the labelled data: GGACT is the most-modified motif. "
           f"Top motif here: {a3.index[0]}.", ""]

    # B. along genes
    union = pd.concat([d[["transcript_id", "transcript_position"]] for d in pooled.values()]).drop_duplicates()
    ann = annotate(union.reset_index(drop=True))
    brow = []
    for l, d in pooled.items():
        x = d[d.n_reads >= WELL].merge(ann, on=["transcript_id", "transcript_position"])
        stop_bin = (np.floor(x.to_stop / 100) * 100)
        for b, g in x[(x.to_stop >= -1000) & (x.to_stop < 1000)].groupby(stop_bin):
            brow.append({"line": l, "profile": "around stop codon", "bin start (nt)": b, "sites": len(g),
                         "mean score": g.score_raw.mean()})
        jb = pd.cut(x.to_junction, [0, 25, 50, 100, 200, 400, 10**9], right=False)
        for b, g in x.groupby(jb, observed=True):
            brow.append({"line": l, "profile": "distance to junction", "bin start (nt)": b.left, "sites": len(g),
                         "mean score": g.score_raw.mean()})
        for r, g in x.groupby("region"):
            brow.append({"line": l, "profile": "region", "bin start (nt)": r, "sites": len(g),
                         "mean score": g.score_raw.mean()})
    bt = pd.DataFrame(brow)
    bt.to_csv(OUT / "B_along_genes.csv", index=False)
    reg = bt[bt.profile == "region"].pivot(index="line", columns="bin start (nt)", values="mean score")
    stop = bt[bt.profile == "around stop codon"].pivot(index="bin start (nt)", columns="line", values="mean score")
    jun = bt[bt.profile == "distance to junction"].pivot(index="bin start (nt)", columns="line", values="mean score")
    md += [f"## B. Along genes (mean raw score, sites with >= {WELL} reads)", "", "By region:", "",
           reg.round(2).pipe(text_table), "", "Distance to nearest exon junction (bin start, nt):", "",
           jun.round(2).pipe(text_table), "",
           f"Peak bin around the stop codon, per line: "
           + ", ".join(f"{l} {int(stop[l].idxmax()):+d}" for l in LINES) + " nt", ""]

    # C. how much m6A
    c1 = pd.DataFrame([{"line": l, **{b: d.score_raw[band(d.n_reads.to_numpy()) == b].mean() for b in BAND_NAMES}}
                       for l, d in pooled.items()])
    c1.to_csv(OUT / "C1_mean_score_by_band.csv", index=False)
    cut = labelled_cutoffs()
    cut.to_csv(OUT / "C2_cutoffs_checked_on_cell_line_2.csv", index=False)
    crow = []
    for l, d in pooled.items():
        bd = band(d.n_reads.to_numpy())
        for _, r in cut.iterrows():
            sel = bd == r["read band"]
            crow.append({"line": l, "read band": r["read band"], "target precision": r["target precision (cell line 1)"],
                         "called %": 100 * (d.score_raw[sel] >= r["cut-off (raw)"]).mean()})
    c3 = pd.DataFrame(crow)
    c3.to_csv(OUT / "C3_called_share_by_line.csv", index=False)
    md += ["## C. How much m6A", "", "C1. Mean raw score by read band (compare lines within a column):", "",
           c1.round(2).pipe(text_table, index=False), "",
           "C2. Cut-offs set on cell line 1's labels, checked on cell line 2 (where the answer is known):", "",
           cut.round(3).pipe(text_table, index=False), "",
           "C3. Share of sites called modified, per line, at each borrowed cut-off (20+ reads):", "",
           c3[c3["read band"] == "20+ reads"].pivot(index="line", columns="target precision",
                                                     values="called %").round(1).pipe(text_table), ""]

    # D and E on sites well covered in EVERY line, so coverage differences cancel.
    common = None
    for l, d in pooled.items():
        k = d[d.n_reads >= WELL].set_index(["transcript_id", "transcript_position"]).score_raw.rename(l)
        common = k.to_frame() if common is None else common.join(k, how="inner")
    genes = gene_names(set(common.index.get_level_values(0)))
    common = common.reset_index()
    common = common.join(genes[["gene_id", "gene_name"]], on="transcript_id")
    gene = common.groupby(["gene_id", "gene_name"])[LINES].max()
    gene.to_csv(OUT / "D_gene_scores.csv")
    top = gene.rank(pct=True) >= 0.9
    everywhere = top.all(1)
    only = {l: (top[l] & (gene.drop(columns=l).rank(pct=True) < 0.5).all(1)) for l in LINES}
    d_rows = [{"line": l, "genes top 10% here and bottom half in every other line": int(v.sum()),
               "examples": ", ".join(gene.index[v].get_level_values(1)[:8])} for l, v in only.items()]
    pd.DataFrame(d_rows).to_csv(OUT / "D_line_specific_genes.csv", index=False)
    md += [f"## D. Genes (max raw score over sites with >= {WELL} reads in every line)", "",
           f"{len(common):,} sites on {len(gene):,} genes are well covered in all 7 lines.", "",
           f"Genes in the top 10% in every line: {int(everywhere.sum())} "
           f"(e.g. {', '.join(gene.index[everywhere].get_level_values(1)[:10])}).", "",
           pd.DataFrame(d_rows).pipe(text_table, index=False), ""]

    site_corr = common[LINES].corr(method="spearman")
    gene_corr = gene.corr(method="spearman")
    site_corr.to_csv(OUT / "E_site_correlation.csv")
    gene_corr.to_csv(OUT / "E_gene_correlation.csv")
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    order = dendrogram(linkage(squareform(1 - gene_corr.to_numpy(), checks=False), "average"),
                       labels=LINES, no_plot=True)["ivl"]
    md += ["## E. Which lines look alike (Spearman, same sites / same genes in every line)", "",
           "Site level:", "", site_corr.round(3).pipe(text_table), "", "Gene level:", "",
           gene_corr.round(3).pipe(text_table), "", f"Clustering order (gene level): {' - '.join(order)}", ""]
    _figures(a2, stop, jun, c1, gene_corr.loc[order, order])
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def _figures(a2, stop, jun, c1, corr) -> None:
    sys.path.insert(0, str(ROOT / "analysis" / "representation"))
    from m6a import figures as F
    plt = F._plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    fig.patch.set_facecolor(F.SURFACE)
    ax = axes[0]
    for kind, c in (("same line", "#2a78d6"), ("different lines", "#eb6834")):
        g = a2[a2.kind == kind].groupby("read band (fewer of the two)", sort=False).spearman.mean().reindex(BAND_NAMES)
        ax.plot(range(4), g.to_numpy(), "o-", color=c, label=kind)
    ax.set_xticks(range(4)); ax.set_xticklabels(BAND_NAMES, fontsize=8)
    ax.set_title("A2. Agreement between samples", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Spearman correlation of raw score")
    ax = axes[1]
    for l in stop.columns:
        ax.plot(stop.index + 50, stop[l], linewidth=1.4, label=l)
    ax.axvline(0, color=F.INK_SOFT, linestyle="--", linewidth=0.8)
    ax.set_title("B. Around the stop codon (sites >= 10 reads)", loc="left", fontsize=10)
    ax.set_xlabel("position relative to stop codon (nt)"); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axes[2]
    im = ax.imshow(corr.to_numpy(), cmap="Blues", vmin=corr.to_numpy().min(), vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if corr.iat[i, j] > corr.to_numpy().mean() else F.INK)
    ax.set_title("E. Gene-level similarity between lines", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png")


if __name__ == "__main__":
    main()
