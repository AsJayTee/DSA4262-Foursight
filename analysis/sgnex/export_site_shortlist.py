"""Data for the Task 2 site shortlist dashboard (report/task2_plan.md, sections 1.3, 3 and 5).

    python analysis/site_types/by_type.py           # the trust notes come from here
    python analysis/sgnex/export_site_shortlist.py

Reads the SG-NEx scores in data/sgnex_scores/ (score_sgnex.py: each line's samples
combined, each sample alone, and each multi-run batch combined with --batch) and
the Ensembl 91 annotation. For every cell line:

  - each site is placed on the genome; copies of one genome position on several
    transcripts of a gene are merged into one site: rank and own-reads rank are
    read-weighted means of the copies, reads are summed (plan 1.3);
  - the ranking comes from the line's combined file; each batch of cells is
    ranked from its own file, for the "holds up?" column (plan 3.3);
  - sites on transcripts missing from Ensembl 91 cannot be placed and are dropped.

Ranks are percentiles within the file (score_rank, what predict.py outputs), stored
as integers 0-9999; higher = higher priority. No probability is stored or implied.

Writes data/dashboard/ (gitignored; plan 5.3 on why it is not committed):
  genes.json.gz            search index: id, name, location, per line site count and best rank
  shards/NNN.json.gz       per gene: its sites and each line's numbers, 256 files
  trust.json               pairwise accuracy by motif x read band (worse of the two
                           labelled cell lines) and each motif's modification rate,
                           from analysis/site_types/results/by_type.csv
  meta.json                cell lines, batches, column meanings
Only model outputs on public SG-NEx data and aggregate label statistics: never a
course label or read (AGENTS.md section 5).
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data" / "sgnex_scores"
GTF = ROOT / "data" / "annotation" / "gtf_91.gtf.gz"
BY_TYPE = ROOT / "analysis" / "site_types" / "results" / "by_type.csv"
OUT = ROOT / "data" / "dashboard"
N_SHARDS = 256
OWN = ["own_res_gate_s0", "own_res_gate_s1", "own_scalar_drop_s0", "own_scalar_drop_s1"]
REGION_ORDER = ["3' UTR", "coding", "5' UTR", "non-coding"]
# Batches of cells (SG-NEx "replicates"). A batch sequenced more than once is scored
# from its runs combined (score_sgnex.py --batch); a single-run batch is its sample file.
BATCHES = {
    "A549": {"replicate5": "SGNex_A549_directRNA_replicate5_run1", "replicate6": "SGNex_A549_directRNA_replicate6_run1"},
    "H9": {"replicate2": "H9_replicate2_pooled", "replicate3": "H9_replicate3_pooled", "replicate4": "H9_replicate4_pooled"},
    "HEYA8": {"replicate1": "HEYA8_replicate1_pooled", "replicate2": "HEYA8_replicate2_pooled",
              "replicate3": "SGNex_HEYA8_directRNA_replicate3_run1"},
    "Hct116": {"replicate3": "Hct116_replicate3_pooled", "replicate4": "SGNex_Hct116_directRNA_replicate4_run3"},
    "HepG2": {"replicate5": "SGNex_HepG2_directRNA_replicate5_run2", "replicate6": "SGNex_HepG2_directRNA_replicate6_run1"},
    "K562": {"replicate4": "SGNex_K562_directRNA_replicate4_run1", "replicate5": "SGNex_K562_directRNA_replicate5_run1",
             "replicate6": "SGNex_K562_directRNA_replicate6_run1"},
    "MCF7": {"replicate3": "SGNex_MCF7_directRNA_replicate3_run1", "replicate4": "SGNex_MCF7_directRNA_replicate4_run1"},
}
READ_BANDS = [(1, 2, "1 reads"), (3, 9, "3 reads"), (10, 19, "10 reads"), (20, 10**9, "full")]


def log(*a) -> None:
    print(*a, flush=True)


def annotation():
    exons, cds, tx_gene, gene_name = {}, {}, {}, {}
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if p[2] not in ("exon", "CDS"):
                continue
            t = re.search(r'transcript_id "([^"]+)"', p[8]).group(1)
            g = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            m = re.search(r'gene_name "([^"]+)"', p[8])
            tx_gene[t], gene_name[g] = g, m.group(1) if m else g
            (exons if p[2] == "exon" else cds).setdefault(t, []).append((p[0], int(p[3]), int(p[4]), p[6]))
    layout = {}
    for t, ex in exons.items():
        strand = ex[0][3]
        ex = sorted(ex, key=lambda e: e[1], reverse=strand == "-")
        cum = np.concatenate([[0], np.cumsum([e[2] - e[1] + 1 for e in ex])])
        st, en = np.array([e[1] for e in ex]), np.array([e[2] for e in ex])
        span = None
        if t in cds:
            lo, hi = min(e[1] for e in cds[t]), max(e[2] for e in cds[t])
            ends = []
            for g in (lo, hi):
                i = np.flatnonzero((st <= g) & (g <= en))
                ends.append(None if not len(i) else cum[i[0]] + (g - st[i[0]] if strand == "+" else en[i[0]] - g))
            span = None if None in ends else (min(ends), max(ends))
        layout[t] = (ex[0][0], strand, st, en, cum, span)
    return layout, tx_gene, gene_name


def place(df: pd.DataFrame, layout, tx_gene) -> pd.DataFrame:
    """Add gene, chrom, strand, genome position, region; drop what cannot be placed."""
    df = df[df.transcript_id.isin(layout.keys())].reset_index(drop=True)
    n = len(df)
    gpos, region = np.empty(n, np.int64), np.empty(n, dtype=object)
    chrom, strand = np.empty(n, dtype=object), np.empty(n, dtype=object)
    pos_all = df.transcript_position.to_numpy()
    for t, idx in df.groupby("transcript_id", sort=False).indices.items():
        c, s, st, en, cum, span = layout[t]
        pos = pos_all[idx]
        i = np.clip(np.searchsorted(cum, pos, side="right") - 1, 0, len(st) - 1)
        off = pos - cum[i]
        gpos[idx] = np.where(pos < cum[-1], np.where(s == "+", st[i] + off, en[i] - off), -1)
        chrom[idx], strand[idx] = c, s
        region[idx] = "non-coding" if span is None else \
            np.where(pos < span[0], "5' UTR", np.where(pos > span[1], "3' UTR", "coding"))
    df = df.assign(gene=df.transcript_id.map(tx_gene), chrom=chrom, strand=strand, gpos=gpos, region=region)
    return df[df.gpos >= 0]


def read_scores(name: str, cols: list[str]) -> pd.DataFrame:
    d = pd.read_csv(SCORES / f"{name}.csv.gz", usecols=cols)
    # Per-sample files of the 10 Oct run predate ID normalisation (score_sgnex.py docstring).
    d = d[d.transcript_id.str.startswith("ENST")]
    return d.assign(transcript_id=d.transcript_id.str.split(".").str[0])


def merge(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """One row per (gene, genome position): read-weighted means of value_cols, reads summed."""
    w = df.n_reads.to_numpy(float)
    t = df[["gene", "gpos", "n_reads"]].copy()
    for c in value_cols:
        t[c] = df[c].to_numpy() * w
    g = t.groupby(["gene", "gpos"], sort=False).sum()
    for c in value_cols:
        g[c] = g[c] / g.n_reads
    return g


def trust_table() -> dict:
    """Pairwise accuracy by motif x read band, and each motif's modification rate.

    The rate (mean over the two labelled lines, full depth) is what the dashboard's
    "rank by DNA letters alone" orders by: the rule of thumb the model is compared with."""
    t = pd.read_csv(BY_TYPE)
    t = t[t.family == "motif"]
    acc = {}
    for (motif, depth), x in t.groupby(["type", "depth"]):
        judged = x[x.note.isna() | (x.note == "")]
        # Worse of the two labelled cell lines, the same rule models are chosen by (decision 0032).
        acc.setdefault(motif, {})[depth] = None if len(judged) < 2 else round(float(judged.pair_acc.min()), 3)
    rate = t[t.depth == "full"].groupby("type").base_rate.mean().round(4).to_dict()
    return {"read_bands": READ_BANDS, "pair_accuracy": acc, "motif_rate": rate}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shards").mkdir(exist_ok=True)
    layout, tx_gene, gene_name = annotation()
    log(f"annotation: {len(layout):,} transcripts")
    static = []      # per line: one row per copy (site x transcript) with region and kmer
    per_line = {}
    for line, batches in BATCHES.items():
        d = read_scores(f"{line}_pooled", ["transcript_id", "transcript_position", "kmer", "n_reads",
                                            "score_rank", *OWN])
        d["own_rank"] = d[OWN].mean(1).rank(pct=True)
        dropped = len(d)
        d = place(d, layout, tx_gene)
        log(f"{line}: {len(d):,} copies placed, {dropped - len(d):,} dropped (transcript not in Ensembl 91)")
        static.append(d[["gene", "gpos", "chrom", "strand", "kmer", "region", "transcript_id", "n_reads"]])
        m = merge(d, ["score_rank", "own_rank"])
        for b, name in batches.items():
            x = place(read_scores(name, ["transcript_id", "transcript_position", "n_reads", "score_rank"]),
                      layout, tx_gene)
            xb = merge(x, ["score_rank"])
            m[f"rank_{b}"] = xb.score_rank.reindex(m.index)
            m[f"reads_{b}"] = xb.n_reads.reindex(m.index)
        per_line[line] = m
        log(f"{line}: {len(m):,} sites, batches {list(batches)}")

    # Fixed site details, once per (gene, position): letters, region, transcripts.
    s = pd.concat(static, ignore_index=True)
    reg_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    s["reg_i"] = s.region.map(reg_rank)
    site_static = s.sort_values(["gene", "gpos", "reg_i"]).groupby(["gene", "gpos"], sort=True).agg(
        chrom=("chrom", "first"), strand=("strand", "first"), kmer=("kmer", "first"),
        region=("region", lambda r: r.value_counts().index[0]), tx=("transcript_id", lambda t: sorted(set(t))))
    log(f"{len(site_static):,} physical sites over {site_static.index.get_level_values(0).nunique():,} genes")

    lines = list(BATCHES)
    genes_index, shards = [], {i: {} for i in range(N_SHARDS)}
    as_int = lambda v: [int(round(x * 9999)) if x == x else -1 for x in v]  # noqa: E731
    by_gene_line = {line: {g: f for g, f in m.groupby(level=0, sort=False)} for line, m in per_line.items()}
    for gene, st in site_static.groupby(level=0, sort=False):
        pos = st.index.get_level_values(1).to_numpy()
        txs = sorted({t for ts in st.tx for t in ts})
        tx_i = {t: i for i, t in enumerate(txs)}
        entry = {"id": gene, "name": gene_name.get(gene, gene), "chrom": st.chrom.iloc[0], "strand": st.strand.iloc[0],
                 "tx": txs, "sites": {"pos": pos.tolist(), "kmer": st.kmer.tolist(),
                                      "region": [REGION_ORDER.index(r) for r in st.region],
                                      "tx": [[tx_i[t] for t in ts] for ts in st.tx]}, "lines": {}}
        summary = {"id": gene, "name": entry["name"], "chrom": entry["chrom"], "n": {}, "best": {}}
        where = {p: i for i, p in enumerate(pos)}
        for line in lines:
            f = by_gene_line[line].get(gene)
            if f is None:
                continue
            gp = f.index.get_level_values(1)
            rec = {"i": [where[p] for p in gp], "r": as_int(f.score_rank), "n": f.n_reads.astype(int).tolist(),
                   "o": as_int(f.own_rank),
                   "b": [as_int(f[f"rank_{b}"]) for b in BATCHES[line]],
                   "bn": [[int(x) if x == x else 0 for x in f[f"reads_{b}"]] for b in BATCHES[line]]}
            entry["lines"][line] = rec
            summary["n"][line], summary["best"][line] = len(f), max(rec["r"])
        shard = int(gene[4:]) % N_SHARDS
        summary["shard"] = shard
        shards[shard][gene] = entry
        genes_index.append(summary)

    for i, content in shards.items():
        with gzip.open(OUT / "shards" / f"{i:03d}.json.gz", "wt", compresslevel=9) as fh:
            json.dump(content, fh, separators=(",", ":"))
    with gzip.open(OUT / "genes.json.gz", "wt", compresslevel=9) as fh:
        json.dump(genes_index, fh, separators=(",", ":"))
    (OUT / "trust.json").write_text(json.dumps(trust_table(), indent=1))
    (OUT / "meta.json").write_text(json.dumps({
        "model": "models/final (decision 0034): h2gcn_twohead_aux + res_gate + scalar_drop, 2 seeds each",
        "annotation": "Ensembl 91 (GRCh38)", "lines": lines, "batches": {k: list(v) for k, v in BATCHES.items()},
        "regions": REGION_ORDER, "rank_scale": "percentile within the file x 9999; higher = higher priority; -1 = not scored",
        "columns": {"r": "rank, line's samples combined", "n": "reads, combined", "o": "rank from the site's own reads only",
                    "b": "rank in each batch of cells", "bn": "reads in each batch"},
        "warnings": ["Hct116: one sample is the model's training data; its ranks are flattered."]}, indent=1))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    log(f"-> {OUT}: {len(genes_index):,} genes, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
