"""Data for the dashboard's Model stats page.

    python analysis/site_types/build_table.py && python analysis/site_types/by_type.py \
        && python analysis/site_types/shortlist.py          # held-out results (first)
    python analysis/sgnex/export_model_stats.py

Aggregates only; no per-site labels or reads leave the cache. Sections:

  performance   PR AUC / ROC AUC of each method on held-out genes (README.md, Results;
                m6Anet from analysis/m6anet/README.md)
  depth         PR AUC and pairwise accuracy of the shipped ensemble at 1, 3, 10 and 20+
                reads (analysis/site_types/results/by_type.csv)
  depth_vs_m6anet  the model and m6Anet at 1, 3, 10 and 20+ reads on the same sites and
                the same read draw: PR AUC, pairwise accuracy, top-3 gene recovery
                (m6Anet below 20 reads: analysis/m6anet/infer_low_depth.py)
  topk          share of held-out genes whose top k sites include an m6ACE-seq-positive
                site, k = 1..5: model, m6Anet, sequence-motif rule, random (shortlist.csv)
  gene_best     share of genes with a positive site, by the percentile of the gene's
                best-ranked site (gene_best.csv)
  reliability   pairwise accuracy by DRACH motif x depth, worse of the two cell lines
  sgnex_depth   reads per site in each SG-NEx line (samples pooled, transcript copies of
                one genome position merged, as Explore shows them): sites, median, and
                the share in each band of the reliability table (export_site_shortlist.py)
  landmarks     enrichment of positive calls by distance to the stop codon and to the
                nearest exon junction:
                  labelled   held-out sites of both labelled lines: m6ACE-seq labels, the
                             model and m6Anet, each "positive" at the file's own base rate q
                             (labels: modified; methods: ranked in the file's top q);
                  sgnex      the seven SG-NEx lines (pooled samples), sites with >= 10 reads,
                             "positive" = top 5% of the file.
                Enrichment = share positive in the bin / share positive overall.

Writes dashboard/public/data/model_stats.json.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analysis" / "site_types" / "results"
SITES = ROOT / ".cache" / "site_types" / "sites.csv.gz"
M6ANET = ROOT / "data" / "m6anet"
SCORES = ROOT / "data" / "sgnex_scores"
SITE_DATA = ROOT / "data" / "dashboard" / "shards"
GTF = ROOT / "data" / "annotation" / "gtf_91.gtf.gz"
OUT = ROOT / "dashboard" / "public" / "data" / "model_stats.json"
LINES = ["A549", "H9", "HEYA8", "Hct116", "HepG2", "K562", "MCF7"]
STOP_BINS = np.arange(-1000, 1001, 50)     # signed distance to the stop codon (nt)
JUNCTION_BINS = np.arange(0, 1001, 25)     # distance to the nearest exon-exon junction (nt)
SGNEX_MIN_READS = 10
SGNEX_TOP = 0.05
CELL = {"dataset0": "HCT116", "data1": "HEK293T"}

# Held-out PR AUC / ROC AUC, both labelled lines, models trained on both (README.md Results).
PERFORMANCE = [
    {"method": "Random", "pr": {"HCT116": 0.045, "HEK293T": 0.072}, "roc": {"HCT116": 0.5, "HEK293T": 0.5}},
    {"method": "Logistic regression", "pr": {"HCT116": 0.415, "HEK293T": 0.331}, "roc": {"HCT116": 0.899, "HEK293T": 0.813}},
    {"method": "Quantiles + LightGBM", "pr": {"HCT116": 0.472, "HEK293T": 0.392}, "roc": {"HCT116": 0.918, "HEK293T": 0.829}},
    {"method": "m6Anet (pretrained)", "pr": {"HCT116": 0.503, "HEK293T": 0.356}, "roc": {"HCT116": 0.932, "HEK293T": 0.819},
     "note": "trained on HCT116: not held out there"},
    {"method": "DeepSet (reads only)", "pr": {"HCT116": 0.541, "HEK293T": 0.429}, "roc": {"HCT116": 0.936, "HEK293T": 0.840}},
    {"method": "This model", "pr": {"HCT116": 0.607, "HEK293T": 0.460}, "roc": {"HCT116": 0.958, "HEK293T": 0.865}},
]


def transcript_landmarks():
    """transcript -> (stop-codon start, internal junction positions), 0-based transcript coords."""
    exons, cds = {}, {}
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if p[2] not in ("exon", "CDS"):
                continue
            t = re.search(r'transcript_id "([^"]+)"', p[8]).group(1)
            (exons if p[2] == "exon" else cds).setdefault(t, []).append((int(p[3]), int(p[4]), p[6]))
    out = {}
    for t, ex in exons.items():
        strand = ex[0][2]
        ex = sorted(ex, key=lambda e: e[0], reverse=strand == "-")
        cum = np.concatenate([[0], np.cumsum([e[1] - e[0] + 1 for e in ex])])
        stop = None
        if t in cds:
            # Last coding base in transcript coordinates; the stop codon starts after it.
            g = max(c[1] for c in cds[t]) if strand == "+" else min(c[0] for c in cds[t])
            for i, (s, e, _) in enumerate(ex):
                if s <= g <= e:
                    stop = int(cum[i] + (g - s if strand == "+" else e - g)) + 1
                    break
        out[t] = (stop, cum[1:-1].astype(np.int64))
    return out


def distances(tx: pd.Series, pos: pd.Series, marks: dict):
    stop = np.full(len(tx), np.nan)
    junction = np.full(len(tx), np.nan)
    t_arr, p_arr = tx.str.split(".").str[0].to_numpy(), pos.to_numpy()
    for t, idx in pd.Series(np.arange(len(t_arr))).groupby(t_arr).indices.items():
        m = marks.get(t)
        if m is None:
            continue
        s, j = m
        p = p_arr[idx]
        if s is not None:
            stop[idx] = p - s
        if len(j):
            junction[idx] = np.abs(p[:, None] - j[None, :]).min(1)
    return stop, junction


def profile(dist: np.ndarray, positive: dict, edges: np.ndarray) -> dict:
    """Per bin: sites, and for each series, share positive / share positive overall."""
    inside = ~np.isnan(dist) & (dist >= edges[0]) & (dist < edges[-1])
    b = np.digitize(dist[inside], edges) - 1
    out = {"centre": ((edges[:-1] + edges[1:]) / 2).tolist(), "sites": np.bincount(b, minlength=len(edges) - 1).tolist()}
    for name, pos in positive.items():
        p = pos[inside].astype(float)
        overall = pos[~np.isnan(dist)].mean()
        share = np.bincount(b, weights=p, minlength=len(edges) - 1) / np.maximum(out["sites"], 1)
        out[name] = [round(float(v / overall), 3) if n >= 30 else None for v, n in zip(share, out["sites"])]
    return out


def labelled_landmarks(marks: dict) -> dict:
    d = pd.read_csv(SITES, usecols=["file", "transcript_id", "transcript_position", "label", "rank_full"])
    parts = []
    for file in ("dataset0", "data1"):
        m = pd.read_csv(M6ANET / f"{file}_pretrained_out" / "data.site_proba.csv",
                        usecols=["transcript_id", "transcript_position", "probability_modified"])
        parts.append(m.assign(file=file))
    m = pd.concat(parts)
    m["tx"] = m.transcript_id.str.split(".").str[0]
    d["tx"] = d.transcript_id.str.split(".").str[0]
    d = d.merge(m[["file", "tx", "transcript_position", "probability_modified"]], how="left",
                on=["file", "tx", "transcript_position"])
    out = {}
    for file, f in d.groupby("file"):
        q = f.label.mean()
        model = f.rank_full >= f.rank_full.quantile(1 - q)
        m6a = f.probability_modified.rank(pct=True) >= 1 - q
        stop, junction = distances(f.transcript_id, f.transcript_position, marks)
        series = {"labels": f.label.to_numpy() == 1, "model": model.to_numpy(), "m6anet": m6a.to_numpy()}
        out[CELL[file]] = {"base_rate": round(float(q), 4), "stop": profile(stop, series, STOP_BINS),
                           "junction": profile(junction, series, JUNCTION_BINS)}
        print(f"labelled {CELL[file]}: {len(f):,} sites", flush=True)
    return out


def sgnex_landmarks(marks: dict) -> dict:
    out = {}
    for line in LINES:
        d = pd.read_csv(SCORES / f"{line}_pooled.csv.gz", usecols=["transcript_id", "transcript_position", "n_reads", "score_rank"])
        d = d[d.n_reads >= SGNEX_MIN_READS]
        top = (d.score_rank >= d.score_rank.quantile(1 - SGNEX_TOP)).to_numpy()
        stop, junction = distances(d.transcript_id, d.transcript_position, marks)
        out[line] = {"stop": profile(stop, {"model": top}, STOP_BINS)["model"],
                     "junction": profile(junction, {"model": top}, JUNCTION_BINS)["model"], "sites": len(d)}
        print(f"SG-NEx {line}: {len(d):,} sites with >= {SGNEX_MIN_READS} reads", flush=True)
    return {"stop_centre": ((STOP_BINS[:-1] + STOP_BINS[1:]) / 2).tolist(),
            "junction_centre": ((JUNCTION_BINS[:-1] + JUNCTION_BINS[1:]) / 2).tolist(),
            "min_reads": SGNEX_MIN_READS, "top": SGNEX_TOP, "lines": out}


def sgnex_depth() -> list[dict]:
    """Reads per site in each SG-NEx line, from the Explore site data."""
    reads = {line: [] for line in LINES}
    shards = sorted(SITE_DATA.glob("*.json.gz"))
    if not shards:
        sys.exit(f"No site data in {SITE_DATA}. Run: python analysis/sgnex/export_site_shortlist.py")
    for f in shards:
        with gzip.open(f, "rt") as fh:
            for gene in json.load(fh).values():
                for line, rec in gene["lines"].items():
                    reads[line].extend(rec["n"])
    out = []
    for line, n in reads.items():
        n = np.asarray(n)
        out.append({"line": line, "sites": int(n.size), "median": float(np.median(n)),
                    "bands": {"1-2": round(float((n <= 2).mean()), 4), "3-9": round(float(((n >= 3) & (n < 10)).mean()), 4),
                              "10-19": round(float(((n >= 10) & (n < 20)).mean()), 4), "20+": round(float((n >= 20).mean()), 4)}})
    return out


def depth_vs_m6anet() -> dict | None:
    """Model vs m6Anet at 1, 3, 10 and >= 20 reads, same sites and same read draw.

    m6Anet at reduced depth: analysis/m6anet/infer_low_depth.py on the inputs written by
    make_thinned_inputs.py (the default draw, seed 4262), so the model's column for that
    draw is used here rather than the five-draw average. Returns None until the m6Anet
    outputs exist."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    sys.path.insert(0, str(ROOT / "analysis" / "site_types"))
    from shortlist import MIN_SITES, merge_copies

    thinned = M6ANET / "thinned_out"
    if not thinned.exists():
        return None
    d = pd.read_csv(SITES)
    d["tx"] = d.transcript_id.str.split(".").str[0]
    cols = {"full": ("rank_full", None), "10 reads": ("rank_d10_draw4262", 10),
            "3 reads": ("rank_d3_draw4262", 3), "1 reads": ("rank_d1_draw4262", 1)}
    out = {}
    for file in ("dataset0", "data1"):
        f = d[d.file == file].copy()
        res = {}
        for depth, (model_col, k) in cols.items():
            src = (M6ANET / f"{file}_pretrained_out" / "data.site_proba.csv") if k is None \
                else (thinned / f"{file}_d{k}" / "data.site_proba.csv")
            m = pd.read_csv(src, usecols=["transcript_id", "transcript_position", "probability_modified"])
            m["tx"] = m.transcript_id.str.split(".").str[0]
            g = f.merge(m[["tx", "transcript_position", "probability_modified"]], how="left", on=["tx", "transcript_position"])
            # m6Anet leaves the last 4 HEK293T sites (one transcript, ENST00000641784) unscored
            # at every depth; both methods are scored on the sites m6Anet covers, so the
            # comparison stays paired. Anything beyond a handful means a broken input.
            missing = int(g.probability_modified.isna().sum())
            if missing > 10:
                raise SystemExit(f"m6Anet {file} {depth}: {missing} sites without a prediction")
            g = g[g.probability_modified.notna()].copy()
            y = g.label.to_numpy()
            row = {}
            for name, score in (("model", g[model_col]), ("m6anet", g.probability_modified)):
                g["_s"] = score.to_numpy()
                # Top-3 recovery, sites merged as on the dashboard (shortlist.py).
                s = merge_copies(g, "_s", "n_reads" if k is None else None)
                sizes = s.groupby(["gene"]).agg(n=("label", "size"), p=("label", "sum"))
                keep = sizes[(sizes.n >= MIN_SITES) & (sizes.p > 0)].index
                top3 = s[s.gene.isin(keep)].sort_values("score", ascending=False).groupby("gene").head(3)
                row[name] = {"pr_auc": round(float(average_precision_score(y, score)), 3),
                             "pair_acc": round(float(roc_auc_score(y, score)), 3),
                             "top3": round(float((top3.groupby("gene").label.max() > 0).mean()), 3)}
            res[depth] = row
        out[CELL[file]] = res
        print(f"depth vs m6Anet {CELL[file]}: " + ", ".join(f"{k}: {v['model']['pr_auc']} vs {v['m6anet']['pr_auc']}" for k, v in res.items()), flush=True)
    return out


def main() -> None:
    by_type = pd.read_csv(RESULTS / "by_type.csv")
    short = pd.read_csv(RESULTS / "shortlist.csv")
    best = pd.read_csv(RESULTS / "gene_best.csv")
    cl = lambda s: s.split(" ")[0]  # noqa: E731

    depth = {cl(c): {r.depth: {"pr_auc": round(r.pr_auc, 3), "pair_acc": round(r.pair_acc, 3)} for r in g.itertuples()}
             for c, g in by_type[by_type.family == "all"].groupby("cell_line")}
    full = short[short["evaluated on"] == "full reads"]
    topk = {m: {cl(c): [round(v, 3) for v in g.sort_values("k").hit_rate] for c, g in x.groupby("cell_line")}
            for m, x in full.groupby("method")}
    gene_best = {}
    for where, key in [("full reads", "full"), ("1 reads (thinned, 5 draws)", "one_read")]:
        x = best[best["evaluated on"] == where]
        gene_best[key] = [{"band": b, **{cl(r.cell_line): round(r._5, 3) for r in g.itertuples()}}
                          for b, g in x.groupby("gene's best site", sort=False)]
    motif = by_type[(by_type.family == "motif") & (by_type.note.isna() | (by_type.note == ""))]
    reliability = {}
    for (m, dep), g in motif.groupby(["type", "depth"]):
        if len(g) == 2:
            reliability.setdefault(m, {})[dep] = round(float(g.pair_acc.min()), 3)

    marks = transcript_landmarks()
    stats = {
        "performance": PERFORMANCE, "depth": depth, "topk": topk, "gene_best": gene_best,
        "reliability": reliability, "depths": ["full", "10 reads", "3 reads", "1 reads"],
        "landmarks": {"labelled": labelled_landmarks(marks), "sgnex": sgnex_landmarks(marks)},
        "depth_vs_m6anet": depth_vs_m6anet(),
        "sgnex_depth": sgnex_depth(),
        "sources": {"held_out": "analysis/site_types/results/", "baselines": "README.md (Results)",
                    "script": "analysis/sgnex/export_model_stats.py"},
    }
    OUT.write_text(json.dumps(stats, separators=(",", ":")))
    print(f"-> {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
