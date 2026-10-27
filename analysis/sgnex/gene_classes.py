"""Which kinds of genes carry more m6A? Depth-matched, and checked against GLORI.

The raw score cannot answer this. It rises with read depth, and read depth is
transcript abundance, so any class of abundant genes (ribosomal proteins are
the extreme) looks "highly modified" on raw scores whatever its biology.
Depth is the number of distinct molecules measured at a site:
docs/data.md#read-depth.

So each site is ranked only against sites of the same depth band in the same
cell line (m6a.evaluation.depth_bands), and called "high" if it is in the top
5% of its band. A gene's m6A share is the fraction of its sites called high;
a class's effect is its genes' mean share minus that of all other
protein-coding genes. Genes, not sites, are the unit, because the sites of one
gene are not independent; the interval resamples genes.

A class effect is reported as a finding only if it has the same sign in all
seven lines at matched depth AND GLORI agrees where it is available. GLORI's
side uses GLORI's own calls (glori_check.py's labelled sites, >= 10 reads):
each site's call minus the GLORI rate of its depth band, averaged per gene.
GLORI's sensitivity also follows abundance, which is why its side is
depth-matched too.

Classes: ribosomal-protein genes, mitochondrially encoded genes, transcription
factors (Lambert et al., Cell 2018: TF_names_v_1.01.txt), long non-coding
RNAs, and the 50 MSigDB hallmark sets (h.all.v2023.2.Hs.symbols.gmt). Both
lists go in data/external/glori/ (URLs in LISTS below).

    python analysis/sgnex/gene_classes.py     # run glori_check.py first

Writes analysis/sgnex/gene_classes/{summary.md, effects.csv}.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "sgnex"))
sys.path.insert(0, str(ROOT / "analysis" / "newdata"))

from annotation_context import read_cdna  # noqa: E402
from glori_check import GLORI, GLORI_FILES, find_a_offset, read_exons, to_genome  # noqa: E402
from m6a.evaluation import depth_bands  # noqa: E402

SCORES = ROOT / "data" / "sgnex_scores"
ANN = ROOT / "data" / "annotation"
OUT = ROOT / "analysis" / "sgnex" / "gene_classes"
LINES = ["A549", "H9", "HEYA8", "Hct116", "HepG2", "K562", "MCF7"]
LISTS = {
    "TF_names_v_1.01.txt": "https://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt",
    "h.all.symbols.gmt": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt",
}
TOP = 0.05
MIN_SITES = 3
LNC = {"lincRNA", "antisense", "sense_intronic", "sense_overlapping", "processed_transcript",
       "bidirectional_promoter_lncRNA", "3prime_overlapping_ncRNA", "macro_lncRNA"}
RNG = np.random.default_rng(4262)


def gene_table(path: Path) -> pd.DataFrame:
    rows = []
    pat = {k: re.compile(f'{k} "([^"]+)"') for k in ("gene_id", "gene_name", "gene_biotype")}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] == "gene":
                rows.append((*(pat[k].search(f[8]).group(1) for k in pat), f[0]))
    return pd.DataFrame(rows, columns=["gene_id", "gene_name", "gene_biotype", "chrom"]).set_index("gene_id")


def classes(genes: pd.DataFrame) -> dict[str, set[str]]:
    pc = genes[genes.gene_biotype == "protein_coding"]
    out = {
        "ribosomal proteins": set(pc.index[pc.gene_name.str.match(r"^RP[LS]\d+[A-Z]?\d*$")]),
        "mitochondrially encoded": set(pc.index[pc.chrom == "MT"]),
        "long non-coding RNAs": set(genes.index[genes.gene_biotype.isin(LNC)]),
    }
    by_name = pd.Series(pc.index, index=pc.gene_name)
    by_name = by_name[~by_name.index.duplicated()]
    tf = {x.strip() for x in (GLORI / "TF_names_v_1.01.txt").read_text().split()}
    out["transcription factors"] = set(by_name.reindex(list(tf)).dropna())
    for line in (GLORI / "h.all.symbols.gmt").read_text().splitlines():
        name, _, *members = line.split("\t")
        out[name.replace("HALLMARK_", "hallmark: ").lower()] = set(by_name.reindex(members).dropna())
    return out


def effect(share: pd.Series, members: set[str], background: set[str], n_boot: int = 500) -> tuple[float, float, float, int]:
    """Mean gene share in the class minus the rest of the background, with a gene bootstrap 95% interval."""
    a = share[share.index.isin(members)].to_numpy()
    b = share[share.index.isin(background - members)].to_numpy()
    if len(a) < 10:
        return np.nan, np.nan, np.nan, len(a)
    d = a.mean() - b.mean()
    boot = (RNG.choice(a, (n_boot, len(a))).mean(1) - RNG.choice(b, (n_boot, len(b))).mean(1))
    return d, *np.quantile(boot, [0.025, 0.975]), len(a)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [f for f in LISTS if not (GLORI / f).exists()] + \
              [f"labelled_{line}.csv.gz" for line in GLORI_FILES if not (GLORI / f"labelled_{line}.csv.gz").exists()]
    if missing:
        raise SystemExit(f"Missing in {GLORI}: {missing}. Run analysis/sgnex/glori_check.py, and fetch the lists:\n  "
                         + "\n  ".join(f"curl -L -o {f} {u}" for f, u in LISTS.items()))
    genes = gene_table(ANN / "gtf_91.gtf.gz")
    cls = classes(genes)
    coding = set(genes.index[genes.gene_biotype == "protein_coding"])

    scores = {line: pd.read_csv(SCORES / f"{line}_pooled.csv.gz",
                                usecols=["transcript_id", "transcript_position", "kmer", "n_reads", "score_raw"])
              for line in LINES}
    for d in scores.values():
        d["transcript_id"] = d.transcript_id.str.split(".").str[0]
    exons, tx_gene = read_exons(ANN / "gtf_91.gtf.gz", set().union(*(set(d.transcript_id) for d in scores.values())))
    a_offset = find_a_offset(scores["A549"], read_cdna(ANN / "cdna_91.fa.gz", set(scores["A549"].transcript_id)))
    train = pd.read_csv(ROOT / "data0" / "data.info.labelled")
    train_keys = set(zip(train.transcript_id, train.transcript_position))

    share, share20, raw = {}, {}, {}
    for line, s in scores.items():
        if line == "Hct116":
            # Training sites score higher for having been trained on (explore summary C).
            s = s[[k not in train_keys for k in zip(s.transcript_id, s.transcript_position)]]
        s = to_genome(s, exons, a_offset)
        s["gene_id"] = s.transcript_id.map(tx_gene)
        s = s.sort_values("n_reads", ascending=False).drop_duplicates(["chrom", "gpos", "strand"])
        s["band"] = depth_bands(s.n_reads)
        s["high"] = s.groupby("band", observed=True).score_raw.rank(pct=True) > 1 - TOP
        g = s.groupby("gene_id")
        keep = g.size() >= MIN_SITES
        share[line] = g.high.mean()[keep]
        raw[line] = g.score_raw.mean()[keep]
        deep = s[s.n_reads >= 20].groupby("gene_id")
        share20[line] = deep.high.mean()[deep.size() >= MIN_SITES]
        print(f"{line}: {keep.sum():,} genes with >= {MIN_SITES} sites", flush=True)

    glori = {}
    for line in GLORI_FILES:
        s = pd.read_csv(GLORI / f"labelled_{line}.csv.gz", usecols=["gene_id", "n_reads", "label"])
        s = s[s.n_reads >= 10]
        s["band"] = depth_bands(s.n_reads)
        s["excess"] = s.label - s.groupby("band", observed=True).label.transform("mean")
        g = s.groupby("gene_id")
        glori[line] = g.excess.mean()[g.size() >= MIN_SITES]

    rows = []
    for name, members in cls.items():
        bg = coding | members if name == "long non-coding RNAs" else coding
        row = {"class": name}
        for line in LINES:
            d, lo, hi, n = effect(share[line], members, bg)
            row |= {f"{line}": d, f"{line}_lo": lo, f"{line}_hi": hi, f"{line}_genes": n,
                    f"{line}_raw": effect(raw[line], members, bg, n_boot=1)[0],
                    f"{line}_20plus": effect(share20[line], members, bg, n_boot=1)[0]}
        for line in GLORI_FILES:
            d, lo, hi, n = effect(glori[line], members, bg)
            row |= {f"glori_{line}": d, f"glori_{line}_lo": lo, f"glori_{line}_hi": hi}
        rows.append(row)
    res = pd.DataFrame(rows)
    sign = np.sign(res[LINES])
    res["lines_same_sign"] = np.maximum((sign > 0).sum(axis=1), (sign < 0).sum(axis=1))
    res["lines_ci_excludes_0"] = sum(((res[f"{l}_lo"] > 0) | (res[f"{l}_hi"] < 0)).astype(int) for l in LINES)
    gcols = [f"glori_{l}" for l in GLORI_FILES]
    res["glori_agrees"] = (np.sign(res[gcols]).eq(np.sign(res[LINES].median(axis=1)), axis=0)).sum(axis=1)
    res["glori_ci_excludes_0"] = sum(((res[f"glori_{l}_lo"] > 0) | (res[f"glori_{l}_hi"] < 0)).astype(int)
                                     for l in GLORI_FILES)
    res["median_effect"] = res[LINES].median(axis=1)
    res["finding"] = (res.lines_same_sign == 7) & (res.glori_agrees == len(GLORI_FILES)) & \
                     (res.lines_ci_excludes_0 >= 4) & (res.glori_ci_excludes_0 >= 2)
    res = res.sort_values("median_effect", ascending=False)
    res.to_csv(OUT / "effects.csv", index=False)

    pct = lambda x: f"{100 * x:+.1f}" if pd.notna(x) else "  -"  # noqa: E731
    show = pd.DataFrame({
        "class": res["class"],
        "genes (A549)": res["A549_genes"],
        "raw A549": res["A549_raw"].round(2),
        **{line: res[line].map(pct) for line in LINES},
        "same sign": res.lines_same_sign,
        "CI excl. 0": res.lines_ci_excludes_0,
        **{f"GLORI {l}": res[f"glori_{l}"].map(pct) for l in GLORI_FILES},
        "finding": res.finding.map({True: "yes", False: ""}),
    })
    with open(OUT / "summary.md", "w", encoding="utf8") as fh:
        fh.write("# Which kinds of genes carry more m6A?\n\nGenerated by `analysis/sgnex/gene_classes.py`. "
                 f"Per line: a class's genes' mean share of sites in the top {TOP:.0%} of their depth band, minus "
                 "other protein-coding genes', in percentage points (the expected share is "
                 f"{TOP:.0%}). `raw A549` is the same difference on raw score, not depth-matched. GLORI columns: "
                 "the class's excess GLORI call rate over its depth band, percentage points (>= 10 reads). "
                 "`finding` = same sign in all 7 lines, CI excludes 0 in >= 4, GLORI agrees in sign in all 4 "
                 "and its CI excludes 0 in >= 2. Descriptive: 54 classes, no multiplicity correction.\n\n")
        fh.write("```\n" + show.to_string(index=False) + "\n```\n")
    print((OUT / "summary.md").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
