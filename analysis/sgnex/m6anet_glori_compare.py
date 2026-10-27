"""Our model vs m6Anet on SG-NEx, both graded by GLORI, on identical sites.

Inputs: glori_check.py's labelled sites (our score, GLORI label, read count)
and m6Anet's `data.site_proba.csv` from run_m6anet_glori.sh, copied back to
data/m6anet_glori/<line>_out/. m6Anet is the pretrained HCT116_RNA002 model,
trained on HCT116; ours was trained on HCT116 and HEK293T (dataset0 + data1).
Neither saw A549, H9 or HepG2, so all three are held out for both.

Two m6Anet rows:

- **as published:** m6Anet drops every site with fewer than 20 reads, so it
  says nothing about ~70-85% of SG-NEx sites. `coverage` is the share of
  sites it scores.
- **all depths:** the same model with that minimum lowered to 1
  (analysis/m6anet/infer_low_depth.py), compared band by band.

Each band reports both models on exactly the same sites. The difference in PR
AUC has a 95% interval from resampling genes (sites in one gene are not
independent), and `gene_wins` is the share of resamples ours wins.

    python analysis/sgnex/m6anet_glori_compare.py --lines A549
    python analysis/sgnex/m6anet_glori_compare.py            # every line with m6Anet output

Writes analysis/sgnex/m6anet_glori/{summary.md, metrics.csv}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "sgnex"))

from glori_check import BANDS, GLORI, band  # noqa: E402
from m6a.evaluation import metrics  # noqa: E402

M6ANET = ROOT / "data" / "m6anet_glori"
OUT = ROOT / "analysis" / "sgnex" / "m6anet_glori"
N_BOOT = 200


def outputs(line: str) -> list[Path]:
    """Every m6Anet site file for a line: `<line>_out/` and any `<line>_out_shard*/`."""
    return sorted(M6ANET.glob(f"{line}_out*/data.site_proba.csv"))


def gene_bootstrap(sub: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    """95% interval of PR AUC (ours - m6Anet) over gene resamples, and the share ours wins."""
    genes = sub.gene_id.to_numpy()
    codes, uniq = pd.factorize(genes)
    idx_by_gene = pd.Series(np.arange(len(sub))).groupby(codes).indices
    y, a, b = sub.label.to_numpy(), sub.score_raw.to_numpy(), sub.m6anet.to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx_by_gene[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if y[pick].sum() == 0:
            continue
        diffs.append(average_precision_score(y[pick], a[pick]) - average_precision_score(y[pick], b[pick]))
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi), float((diffs > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=["A549", "H9", "HepG2"])
    args = ap.parse_args()
    lines = [l for l in args.lines if outputs(l)]
    if not lines:
        raise SystemExit(f"No m6Anet output under {M6ANET}/<line>_out*/. Run analysis/sgnex/run_m6anet_glori.sh "
                         "on Ronin and copy data/m6anet_glori/<line>_out*/data.site_proba.csv back.")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4262)
    rows, ratio_rows, cover_rows = [], [], []
    for line in lines:
        s = pd.read_csv(GLORI / f"labelled_{line}.csv.gz",
                        usecols=["transcript_id", "transcript_position", "gene_id", "n_reads", "score_raw",
                                 "motif_only", "label", "ratio"])
        # Shards, plus an unsharded run that may have been stopped mid-write: a
        # cut-off last row is skipped, and a site scored twice is kept once.
        m = pd.concat([pd.read_csv(p, usecols=["transcript_id", "transcript_position", "probability_modified"],
                                   on_bad_lines="skip") for p in outputs(line)])
        m = m.dropna().drop_duplicates(["transcript_id", "transcript_position"])
        m["transcript_position"] = m.transcript_position.astype(int)
        s = s.merge(m.rename(columns={"probability_modified": "m6anet"}),
                    on=["transcript_id", "transcript_position"], how="left")
        scored = s.m6anet.notna()
        print(f"{line}: m6Anet scored {scored.mean():.1%} of {len(s):,} graded sites", flush=True)
        deep = s.n_reads >= 20
        cover_rows.append({"line": line, "graded sites": len(s),
                           "m6Anet as published scores": f"{deep.mean():.1%}",
                           "GLORI sites it can see": f"{s.label[deep].sum() / s.label.sum():.1%}",
                           "all-depth run scored": f"{scored.mean():.1%}"})
        s = s[scored]
        s["band"] = band(s.n_reads.to_numpy())
        for name, sub in [("all depths", s)] + [(b, s[s.band == b]) for _, _, b in BANDS]:
            if sub.label.sum() < 20:
                continue
            ours, theirs, motif = (metrics(sub.label, sub[c]) for c in ("score_raw", "m6anet", "motif_only"))
            k = max(1, int(round(0.01 * len(sub))))
            lo, hi, wins = gene_bootstrap(sub, rng)
            rows.append({"line": line, "band": name, "sites": len(sub), "positive_rate": ours["positive_rate"],
                         "pr_auc_ours": ours["pr_auc"], "pr_auc_m6anet": theirs["pr_auc"],
                         "pr_auc_motif": motif["pr_auc"],
                         "diff": ours["pr_auc"] - theirs["pr_auc"], "diff_lo": lo, "diff_hi": hi,
                         "gene_wins": wins,
                         "top1pct_ours": sub.nlargest(k, "score_raw").label.mean(),
                         "top1pct_m6anet": sub.nlargest(k, "m6anet").label.mean(),
                         "roc_auc_ours": ours["roc_auc"], "roc_auc_m6anet": theirs["roc_auc"]})
            print(f"  {name:12s} ours {ours['pr_auc']:.3f}  m6Anet {theirs['pr_auc']:.3f}  "
                  f"diff {ours['pr_auc'] - theirs['pr_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        pos = s[(s.label == 1) & (s.n_reads >= 20)]
        ratio_rows.append({"line": line, "GLORI sites, 20+ reads": len(pos),
                           "ours": round(spearmanr(pos.score_raw, pos.ratio).statistic, 3),
                           "m6Anet": round(spearmanr(pos.m6anet, pos.ratio).statistic, 3)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    show = res.assign(**{c: res[c].round(3) for c in res.columns if res[c].dtype == float})
    show["positive_rate"] = (res.positive_rate * 100).round(1)
    with open(OUT / "summary.md", "w", encoding="utf8") as fh:
        fh.write("# Our model vs m6Anet on SG-NEx, graded by GLORI\n\nGenerated by "
                 "`analysis/sgnex/m6anet_glori_compare.py`. m6Anet = pretrained HCT116_RNA002 (trained on "
                 "HCT116); ours trained on HCT116 + HEK293T. Neither saw A549, H9 or HepG2. Identical sites per row. `diff` = PR AUC ours - m6Anet, interval "
                 f"from {N_BOOT} gene resamples; `gene_wins` = share of resamples ours wins. Precision is a "
                 "lower bound for both (GLORI lists only what it detected).\n\n")
        fh.write("## What m6Anet as published can score\n\n```\n" + pd.DataFrame(cover_rows).to_string(index=False)
                 + "\n```\n\n")
        fh.write("## Head to head\n\n```\n" + show[["line", "band", "sites", "positive_rate", "pr_auc_ours",
                                                    "pr_auc_m6anet", "pr_auc_motif", "diff", "diff_lo", "diff_hi",
                                                    "gene_wins", "top1pct_ours", "top1pct_m6anet"]]
                 .to_string(index=False) + "\n```\n\n")
        fh.write("## Does the score track GLORI's m6A fraction? (Spearman, GLORI sites with 20+ reads)\n\n```\n"
                 + pd.DataFrame(ratio_rows).to_string(index=False) + "\n```\n")
    print((OUT / "summary.md").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
