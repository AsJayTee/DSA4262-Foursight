"""Score the shipped model on SG-NEx cell lines against GLORI labels.

GLORI is a chemical (not antibody) single-base m6A assay. Public GLORI site
lists exist for four of the seven SG-NEx lines, so for those lines the scores
stop being "predictions on unlabelled data" and become a held-out test - in
cell lines the model never trained on, and at SG-NEx's real read depths
(median 3) rather than simulated thinning.

| SG-NEx line | GLORI source |
|---|---|
| A549 | GSE290408 (`A549_GLORI3_hg38_SNuc_KDremoved.bed.gz`) |
| H9 | GSE290408 (`H9_GLORI3_hg38_SNuc_KDremoved.bed.gz`) |
| HepG2 | GSE301419, sgCTRL (GSM9083038) |
| HCT116 | GSE301419, sgCTRL (GSM9083040) |

Note GSE210563, the GLORI paper's own series, has no SG-NEx line at all
(HEK293T, HeLa, mouse only).

How a label is made, and what it costs:

- GLORI is genomic (hg38); our sites are transcript positions (Ensembl 91,
  GRCh38). Each site's A is mapped to the genome through the GTF exons. The
  convention is checked, not assumed: the 7-mer must match the cDNA at the
  site, and GLORI sites must land on our candidate As (an off-by-one gives a
  match rate near zero, because our candidates are all DRACH centres).
- One genomic A appears once per isoform that covers it. Each copy is scored on
  its own molecules, so the copy with the most reads stands for the site.
- The files list only sites GLORI called. Negatives are our candidates in genes
  where GLORI called at least one site, i.e. genes it demonstrably covered. A
  modified site GLORI missed is therefore counted as a negative, which makes
  every precision here a lower bound - worst for HepG2 and HCT116, whose
  GLORI runs called only ~7k sites.
- HCT116 is our training line: its 121,838 training sites are dropped before
  scoring. Those sites are used instead to measure how far GLORI and the
  course's labels agree, which bounds what any model could score.

The motif-only row ranks each site by its 5-mer's modification rate in the
training labels (data/dashboard/trust.json). At 1 read the model is expected
to approach it; the gap above it is what the reads add.

    python analysis/sgnex/glori_check.py
    # needs data/sgnex_scores/ (score_sgnex.py), data/annotation/ (annotation_context.py)
    # and the GLORI files in data/external/glori/ (URLs in GLORI_FILES below)

Writes analysis/sgnex/glori/{summary.md, metrics.csv, mapping.csv}.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "newdata"))

from annotation_context import read_cdna  # noqa: E402
from m6a.evaluation import metrics  # noqa: E402

SCORES = ROOT / "data" / "sgnex_scores"
ANN = ROOT / "data" / "annotation"
GLORI = ROOT / "data" / "external" / "glori"
OUT = ROOT / "analysis" / "sgnex" / "glori"

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
GLORI_FILES = {
    "A549": f"{GEO}/series/GSE290nnn/GSE290408/suppl/GSE290408_A549_GLORI3_hg38_SNuc_KDremoved.bed.gz",
    "H9": f"{GEO}/series/GSE290nnn/GSE290408/suppl/GSE290408_H9_GLORI3_hg38_SNuc_KDremoved.bed.gz",
    "HepG2": f"{GEO}/samples/GSM9083nnn/GSM9083038/suppl/GSM9083038_HepG2_sgCTRL.totalm6A.FDR.csv.gz",
    "Hct116": f"{GEO}/samples/GSM9083nnn/GSM9083040/suppl/GSM9083040_HCT116_sgCTRL.totalm6A.FDR.csv.gz",
}
BANDS = [(1, 2, "1-2 reads"), (3, 9, "3-9 reads"), (10, 19, "10-19 reads"), (20, 10**9, "20+ reads")]


def read_glori(path: Path) -> pd.DataFrame:
    """chrom (Ensembl style, no 'chr'), 1-based position as written, strand, ratio."""
    if path.name.endswith(".bed.gz"):
        g = pd.read_csv(path, sep="\t", header=None, usecols=[0, 2, 4, 5], names=["chrom", "pos", "ratio", "strand"])
    else:
        g = pd.read_csv(path).rename(columns={"Chr": "chrom", "Sites": "pos", "Strand": "strand", "Ratio": "ratio"})
    g["chrom"] = g.chrom.str.replace("^chr", "", regex=True).replace({"M": "MT"})
    return g[["chrom", "pos", "strand", "ratio"]]


def read_exons(path: Path, wanted: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, gene = [], {}
    tid, gid = re.compile(r'transcript_id "([^"]+)"'), re.compile(r'gene_id "([^"]+)"')
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "exon":
                continue
            t = tid.search(f[8]).group(1)
            if t in wanted:
                rows.append((t, f[0], int(f[3]), int(f[4]), f[6]))
                gene[t] = gid.search(f[8]).group(1)
    return pd.DataFrame(rows, columns=["transcript_id", "chrom", "start", "end", "strand"]), gene


def to_genome(sites: pd.DataFrame, exons: pd.DataFrame, a_offset: int) -> pd.DataFrame:
    """Add chrom, gpos (1-based) and strand for each site's A (transcript pos + a_offset)."""
    sites = sites.sort_values("transcript_id").reset_index(drop=True)
    gpos = np.full(len(sites), -1, dtype=np.int64)
    chrom = np.empty(len(sites), dtype=object)
    strand = np.empty(len(sites), dtype=object)
    bounds = sites.groupby("transcript_id").indices
    for t, ex in exons.groupby("transcript_id"):
        idx = bounds.get(t)
        if idx is None:
            continue
        plus = ex.strand.iloc[0] == "+"
        ex = ex.sort_values("start", ascending=plus)
        s, e = ex.start.to_numpy(), ex.end.to_numpy()
        tx_end = np.cumsum(e - s + 1)
        tx_start = tx_end - (e - s + 1)
        a = sites.transcript_position.to_numpy()[idx] + a_offset
        i = np.searchsorted(tx_end, a, side="right")
        ok = i < len(tx_end)
        i = np.minimum(i, len(tx_end) - 1)
        within = a - tx_start[i]
        g = np.where(plus, s[i] + within, e[i] - within)
        gpos[idx] = np.where(ok, g, -1)
        chrom[idx] = ex.chrom.iloc[0]
        strand[idx] = "+" if plus else "-"
    sites["chrom"], sites["gpos"], sites["strand"] = chrom, gpos, strand
    return sites[sites.gpos > 0]


def find_a_offset(sites: pd.DataFrame, cdna: dict[str, str]) -> int:
    """Where the A sits relative to transcript_position, from the 7-mers."""
    sample = sites.sample(min(20000, len(sites)), random_state=0)
    best = None
    for off in range(-4, 2):
        ok = sum(cdna.get(t, "")[p + off:p + off + 7] == k
                 for t, p, k in zip(sample.transcript_id, sample.transcript_position, sample.kmer) if p + off >= 0)
        best = max(best or (0, off), (ok, off))
    ok, off = best
    rate = ok / len(sample)
    print(f"7-mer matches cDNA {rate:.1%} with the window starting {-off} nt before transcript_position", flush=True)
    if rate < 0.95:
        raise SystemExit("7-mers do not match Ensembl 91 cDNA: run analysis/newdata/annotation_context.py to fetch it.")
    return off + 3


def band(n: np.ndarray) -> np.ndarray:
    out = np.empty(len(n), dtype=object)
    for lo, hi, name in BANDS:
        out[(n >= lo) & (n <= hi)] = name
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [line for line, url in GLORI_FILES.items() if not (GLORI / url.rsplit("/", 1)[1]).exists()]
    if missing:
        urls = "\n  ".join(GLORI_FILES[m] for m in missing)
        raise SystemExit(f"GLORI files missing for {missing}. Download into {GLORI}:\n  curl -O {urls}")
    motif_rate = json.loads((ROOT / "data" / "dashboard" / "trust.json").read_text())["motif_rate"]

    scores = {line: pd.read_csv(SCORES / f"{line}_pooled.csv.gz",
                                usecols=["transcript_id", "transcript_position", "kmer", "n_reads", "score_raw"])
              for line in GLORI_FILES}
    for d in scores.values():
        d["transcript_id"] = d.transcript_id.str.split(".").str[0]
    wanted = set().union(*(set(d.transcript_id) for d in scores.values()))
    exons, gene = read_exons(ANN / "gtf_91.gtf.gz", wanted)
    cdna = read_cdna(ANN / "cdna_91.fa.gz", set(scores["A549"].transcript_id))
    a_offset = find_a_offset(scores["A549"], cdna)

    train = pd.read_csv(ROOT / "data0" / "data.info.labelled")
    train_keys = set(zip(train.transcript_id, train.transcript_position))

    rows, mapping, ratio_rows, notes = [], [], [], []
    # HCT116 comes first so the course labels' recall against GLORI is known
    # before any line is scored at that recall.
    lines = ["Hct116"] + [line for line in GLORI_FILES if line != "Hct116"]
    course_recall = None
    for line in lines:
        url = GLORI_FILES[line]
        s = to_genome(scores[line], exons, a_offset)
        s["gene_id"] = s.transcript_id.map(gene)
        if line == "Hct116":
            s["train"] = [k in train_keys for k in zip(s.transcript_id, s.transcript_position)]
            lab = train.set_index(["transcript_id", "transcript_position"]).label
            s["course_label"] = [lab.get(k, np.nan) for k in zip(s.transcript_id, s.transcript_position)]
        # One row per genomic A: the isoform copy with the most molecules.
        s = s.sort_values("n_reads", ascending=False).drop_duplicates(["chrom", "gpos", "strand"])

        g = read_glori(GLORI / url.rsplit("/", 1)[1])
        cand = set(zip(s.chrom, s.gpos, s.strand))
        # Off-by-one check: a GLORI site must land on one of our candidate As.
        on = {k: np.array([(c, p + k, st) in cand for c, p, st in zip(g.chrom, g.pos, g.strand)]) for k in (-1, 0, 1)}
        shift = max(on, key=lambda k: on[k].sum())
        hit = on[shift]
        mapping.append({"line": line, "glori_sites": len(g), "position_shift": shift,
                        "glori_sites_on_our_candidates": int(hit.sum()), "share": round(hit.mean(), 4),
                        **{f"share_at_shift_{k:+d}": round(v.mean(), 4) for k, v in on.items() if k != shift}})
        print(f"{line}: {len(g):,} GLORI sites, {hit.mean():.1%} on our candidates (shift {shift:+d})", flush=True)

        g = g[hit].assign(gpos=g.pos[hit] + shift)
        s = s.merge(g[["chrom", "gpos", "strand", "ratio"]], on=["chrom", "gpos", "strand"], how="left")
        s["label"] = s.ratio.notna().astype(int)
        s = s[s.gene_id.isin(set(s.gene_id[s.label == 1]))]
        s["motif_only"] = s.kmer.str[1:6].map(motif_rate).fillna(0.0)
        # Reused by gene_classes.py. Under data/, so never committed: it carries
        # the course's training flags for HCT116.
        s.drop(columns=["course_label"], errors="ignore").to_csv(GLORI / f"labelled_{line}.csv.gz", index=False)

        if line == "Hct116":
            # Agreement between the two labellings on the training sites: an upper
            # bound for any model trained on one and judged on the other.
            t = s[s.train & s.course_label.notna()]
            course_recall = float(t.course_label[t.label == 1].mean())
            notes.append(f"HCT116 training sites in GLORI-covered genes: {len(t):,}. Course labels as a "
                         f"predictor of GLORI: precision {t.label[t.course_label == 1].mean():.3f}, recall "
                         f"{t.course_label[t.label == 1].mean():.3f}; GLORI positive rate {t.label.mean():.3f}.")
            print(notes[-1], flush=True)
            s = s[~s.train]

        s["band"] = band(s.n_reads.to_numpy())
        for name, sub in [("all", s)] + [(b, s[s.band == b]) for _, _, b in BANDS]:
            if sub.label.sum() < 20 or sub.label.sum() == len(sub):
                continue
            for scorer in ("score_raw", "motif_only"):
                m = metrics(sub.label, sub[scorer])
                k = max(1, int(round(0.01 * len(sub))))
                top = sub.nlargest(k, scorer)
                # Precision once the ranking has recovered as many GLORI sites as
                # the course labels do: comparable with the labels' own precision.
                ranked = sub.label.to_numpy()[np.argsort(-sub[scorer].to_numpy(), kind="stable")]
                cut = int(np.searchsorted(np.cumsum(ranked), course_recall * ranked.sum()))
                rows.append({"line": line, "band": name, "scorer": scorer, **m,
                             "precision_top1pct": float(top.label.mean()),
                             "precision_at_course_recall": float(ranked[:cut + 1].mean())})
        pos = s[s.label == 1]
        rho = spearmanr(pos.score_raw, pos.ratio).statistic
        rho20 = spearmanr(pos.score_raw[pos.n_reads >= 20], pos.ratio[pos.n_reads >= 20]).statistic
        # GGACT sites are both more often and more heavily modified, so motif alone
        # could produce a correlation; within one motif it cannot.
        gg = pos[(pos.n_reads >= 20) & (pos.kmer.str[1:6] == "GGACT")]
        ratio_rows.append({"line": line, "positives": len(pos), "spearman_score_vs_glori_ratio": round(rho, 3),
                           "same_at_20plus_reads": round(rho20, 3),
                           "same_20plus_GGACT_only": round(spearmanr(gg.score_raw, gg.ratio).statistic, 3),
                           "GGACT_20plus_positives": len(gg)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(mapping).to_csv(OUT / "mapping.csv", index=False)
    wide = res.pivot_table(index=["line", "band"], columns="scorer",
                           values=["pr_auc", "pr_auc_lift", "precision_top1pct",
                                   "precision_at_course_recall"]).round(3)
    counts = res[res.scorer == "score_raw"].set_index(["line", "band"])[["n", "n_positive", "positive_rate"]]
    with open(OUT / "summary.md", "w", encoding="utf8") as fh:
        fh.write("# The shipped model against GLORI, on SG-NEx\n\nGenerated by `analysis/sgnex/glori_check.py`. "
                 "Score = `score_raw` of the pooled line. Negatives = candidates in genes GLORI called a site in, "
                 "so precision is a lower bound. HCT116 excludes its training sites.\n\n")
        fh.write("## Mapping\n\n```\n" + pd.DataFrame(mapping).to_string(index=False) + "\n```\n\n")
        fh.write("## Sites and positives\n\n```\n" + counts.round(4).to_string() + "\n```\n\n")
        fh.write("## Model vs motif-only\n\n```\n" + wide.to_string() + "\n```\n\n")
        fh.write("## Does the score track GLORI's m6A fraction? (positives only)\n\n```\n"
                 + pd.DataFrame(ratio_rows).to_string(index=False) + "\n```\n\n")
        fh.write("## Label agreement\n\n" + "\n".join(notes) + "\n")
    print((OUT / "summary.md").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
