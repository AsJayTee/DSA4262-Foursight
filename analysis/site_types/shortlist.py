"""Does a gene's top-k shortlist find real m6A? (report/task2_plan.md, 4.1)

    python analysis/site_types/build_table.py    # first
    python analysis/site_types/shortlist.py

Replays the dashboard's use: a researcher takes a gene, sorts its sites by the
model's rank, and tests the top k. On held-out genes of both labelled cell lines,
for k = 1..5:

  hit rate   share of genes where at least one of the top k is truly modified
             (among genes that have at least one modified site - otherwise
             there is nothing to find);
  precision  share of the top k that are truly modified.

Against two ways to choose without the dashboard:
  motif rule  sites ordered by how often their DRACH motif is modified, learned
              from the other folds' genes of the same cell line (never the gene
              itself); ties broken at random, averaged over TIE_DRAWS orders;
  random      the exact expectation of a blind pick.

Sites are merged as on the dashboard (plan 1.3): copies of one genome position
on several transcripts of a gene become one site (rank = read-weighted mean,
modified if any copy is). Genes need at least MIN_SITES sites, so a top 3 is a
choice. Repeated at 10, 3 and 1 reads (thinned, averaged over random draws) and
on two other real HCT116 runs. Intervals: 95%, resampling genes; the model -
motif difference is paired on the same genes.

Second table: does a gene's best site, ranked against the whole cell line, say
whether the gene has any modified site at all? (The "rank in cell line" column.)

Writes analysis/site_types/results/shortlist.csv and gene_best.csv (aggregates).
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "site_types"
RESULTS = Path(__file__).resolve().parent / "results"
KS = [1, 2, 3, 4, 5]
MIN_SITES = 5
TIE_DRAWS = 20
N_BOOT = 1000
CELL = {"dataset0": "HCT116 (cell line 1)", "data1": "HEK293T (cell line 2)"}
M6ANET = ROOT / "data" / "m6anet"
OTHER = "m6Anet (pretrained)"


def merge_copies(d: pd.DataFrame, score: str, weight: str | None) -> pd.DataFrame:
    """One row per physical site per gene: read-weighted mean rank, modified if any copy is."""
    key = np.where(d.genome_pos >= 0, d.chrom.astype(str) + ":" + d.genome_pos.astype(str),
                   d.transcript_id + "@" + d.transcript_position.astype(str))
    w = d[weight].to_numpy(float) if weight else np.ones(len(d))
    t = pd.DataFrame({"file": d.file.to_numpy(), "gene": d.gene_id.to_numpy(), "key": key,
                      "fold": d.fold.to_numpy(), "motif": d.motif.to_numpy(),
                      "label": d.label.to_numpy(), "ws": w * d[score].to_numpy(), "w": w})
    g = t.groupby(["file", "gene", "key"], sort=False)
    out = g.agg(fold=("fold", "first"), motif=("motif", "first"), label=("label", "max"),
                ws=("ws", "sum"), w=("w", "sum")).reset_index()
    out["score"] = out.ws / out.w
    return out.drop(columns=["ws", "w"])


def motif_scores(s: pd.DataFrame) -> np.ndarray:
    """Each site's motif modification rate, learned from the other folds of the same file."""
    out = np.empty(len(s))
    for (file, fold), idx in s.groupby(["file", "fold"]).indices.items():
        train = s[(s.file == file) & (s.fold != fold)]
        rate = train.groupby("motif").label.mean()
        out[idx] = s.motif.iloc[idx].map(rate).fillna(train.label.mean()).to_numpy()
    return out


def per_gene(s: pd.DataFrame, col: str, rng=None) -> pd.DataFrame:
    """For each gene: modified count in the top k, by `col` (random tie-break if rng given)."""
    key = s[col].to_numpy() + (rng.random(len(s)) * 1e-9 if rng is not None else 0)
    s = s.assign(_k=key).sort_values(["file", "gene", "_k"], ascending=[True, True, False])
    s["_pos"] = s.groupby(["file", "gene"]).cumcount()
    out = {f"top{k}": s[s._pos < k].groupby(["file", "gene"]).label.sum() for k in KS}
    return pd.DataFrame(out)


def random_expectation(n: int, p: int, k: int) -> tuple[float, float]:
    hit = 1 - comb(n - p, k) / comb(n, k) if k <= n else 1.0
    return hit, p / n


def summarise(s: pd.DataFrame, label: str, rng) -> list[dict]:
    sizes = s.groupby(["file", "gene"]).agg(n=("label", "size"), p=("label", "sum"))
    sizes = sizes[sizes.n >= MIN_SITES]
    model = per_gene(s, "score").loc[sizes.index]
    motif = [per_gene(s, "motif_rate", np.random.default_rng(i)).loc[sizes.index] for i in range(TIE_DRAWS)]
    other = per_gene(s, "m6anet").loc[sizes.index] if "m6anet" in s else None
    rows = []
    for file in sizes.index.get_level_values(0).unique():
        z = sizes.loc[file]
        has = z.p.to_numpy() > 0
        genes_with = int(has.sum())
        for k in KS:
            m_top = model.loc[file][f"top{k}"].to_numpy()
            mo_top = np.mean([mt.loc[file][f"top{k}"].to_numpy() for mt in motif], 0)
            mo_hit = np.mean([(mt.loc[file][f"top{k}"].to_numpy() > 0) for mt in motif], 0)
            r = np.array([random_expectation(int(n), int(p), k) for n, p in zip(z.n, z.p)])
            hit = {"model": (m_top > 0)[has], "motif rule": mo_hit[has], "random": r[has, 0]}
            prec = {"model": m_top / np.minimum(k, z.n), "motif rule": mo_top / np.minimum(k, z.n),
                    "random": r[:, 1]}
            if other is not None:
                o_top = other.loc[file][f"top{k}"].to_numpy()
                hit[OTHER], prec[OTHER] = (o_top > 0)[has], o_top / np.minimum(k, z.n)
            # Paired gene resampling for the model - motif difference.
            idx = rng.integers(0, genes_with, (N_BOOT, genes_with))
            diff = (hit["model"].astype(float) - hit["motif rule"])[idx].mean(1)
            vs = (hit["model"].astype(float) - hit[OTHER])[idx].mean(1) if OTHER in hit else None
            for method in hit:
                h = hit[method].astype(float)
                boot = h[idx].mean(1)
                rows.append({"evaluated on": label, "cell_line": CELL[file], "k": k, "method": method,
                             "genes": len(z), "genes_with_modified": genes_with,
                             "hit_rate": h.mean(), "hit_lo": np.percentile(boot, 2.5),
                             "hit_hi": np.percentile(boot, 97.5), "precision": float(np.mean(prec[method])),
                             "model_minus_motif": diff.mean() if method == "model" else np.nan,
                             "diff_lo": np.percentile(diff, 2.5) if method == "model" else np.nan,
                             "diff_hi": np.percentile(diff, 97.5) if method == "model" else np.nan,
                             "model_minus_m6anet": vs.mean() if method == "model" and vs is not None else np.nan,
                             "vs_m6anet_lo": np.percentile(vs, 2.5) if method == "model" and vs is not None else np.nan,
                             "vs_m6anet_hi": np.percentile(vs, 97.5) if method == "model" and vs is not None else np.nan})
    return rows


def add_m6anet(d: pd.DataFrame) -> pd.DataFrame:
    """m6Anet's pretrained site probabilities on the same sites, where its output exists
    (analysis/m6anet/run_ronin.sh). Not held out on HCT116: m6Anet was trained on SG-NEx
    HCT116 with HCT116 m6ACE-seq labels (analysis/m6anet/README.md)."""
    parts = []
    for file in ("dataset0", "data1"):
        p = M6ANET / f"{file}_pretrained_out" / "data.site_proba.csv"
        if not p.exists():
            return d
        m = pd.read_csv(p, usecols=["transcript_id", "transcript_position", "probability_modified"])
        parts.append(m.assign(file=file))
    m = pd.concat(parts)
    m["tx"] = m.transcript_id.str.split(".").str[0]
    key = d.assign(tx=d.transcript_id.str.split(".").str[0])[["file", "tx", "transcript_position"]]
    got = key.merge(m[["file", "tx", "transcript_position", "probability_modified"]], how="left",
                    on=["file", "tx", "transcript_position"])
    print(f"m6Anet scores found for {got.probability_modified.notna().mean():.1%} of sites", flush=True)
    return d.assign(m6anet=got.probability_modified.to_numpy())


def gene_best(s: pd.DataFrame, label: str) -> pd.DataFrame:
    g = s.groupby(["file", "gene"]).agg(n=("label", "size"), p=("label", "sum"), best=("score", "max"))
    g = g[g.n >= MIN_SITES].reset_index()
    bins = [(0.99, 1.01, "top 1%"), (0.95, 0.99, "1-5%"), (0.90, 0.95, "5-10%"), (0.75, 0.90, "10-25%"),
            (0.0, 0.75, "below top 25%")]
    rows = []
    for file, f in g.groupby("file"):
        for lo, hi, name in bins:
            b = f[(f.best >= lo) & (f.best < hi)]
            if len(b):
                rows.append({"evaluated on": label, "cell_line": CELL[file], "gene's best site": name,
                             "genes": len(b), "share with a modified site": (b.p > 0).mean(),
                             "mean modified sites": b.p.mean(), "mean sites": b.n.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    d = pd.read_csv(CACHE / "sites.csv.gz")
    rng = np.random.default_rng(4262)
    draws = sorted({int(c.split("draw")[1]) for c in d if c.startswith("rank_d1_draw")})
    settings = [("full reads", ["rank_full"], "n_reads")] + [
        (f"{k} reads (thinned, {len(draws)} draws)", [f"rank_d{k}_draw{r}" for r in draws], None) for k in (10, 3, 1)]
    rows, best = [], []
    d = add_m6anet(d)
    for label, cols, weight in settings:
        parts = []
        for c in cols:
            s = merge_copies(d, c, weight)
            s["motif_rate"] = motif_scores(s)
            if c == "rank_full" and "m6anet" in d:
                # Same sites, merged the same way; full depth only (m6Anet needs 20+ reads).
                s["m6anet"] = merge_copies(d, "m6anet", weight)["score"].to_numpy()
            parts.append(pd.DataFrame(summarise(s, label, rng)))
            best.append(gene_best(s, label))
        # Average over random draws of which reads were kept.
        t = pd.concat(parts)
        keys = ["evaluated on", "cell_line", "k", "method", "genes", "genes_with_modified"]
        rows.append(t.groupby(keys, sort=False).mean(numeric_only=True).reset_index())
        print(f"{label}: done", flush=True)

    runs = CACHE / "hct116_runs.csv.gz"
    if runs.exists():
        r = pd.read_csv(runs)
        for run, x in r.groupby("run"):
            sub = d.iloc[x.row.to_numpy()].copy()
            sub["rank_run"], sub["n_reads_run"] = x.rank_run.to_numpy(), x.n_reads_run.to_numpy()
            label = f"other HCT116 run: {run.replace('SGNex_Hct116_directRNA_', '')}"
            s = merge_copies(sub, "rank_run", "n_reads_run")
            s["motif_rate"] = motif_scores(s)
            rows.append(pd.DataFrame(summarise(s, label, rng)))
            best.append(gene_best(s, label))
            print(f"{label}: done", flush=True)

    pd.concat(rows).to_csv(RESULTS / "shortlist.csv", index=False, float_format="%.4f")
    pd.concat(best).groupby(["evaluated on", "cell_line", "gene's best site"], sort=False) \
        .mean(numeric_only=True).reset_index().to_csv(RESULTS / "gene_best.csv", index=False, float_format="%.4f")
    print(f"-> {RESULTS}/shortlist.csv, gene_best.csv")


if __name__ == "__main__":
    main()
