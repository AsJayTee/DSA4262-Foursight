"""How well the shipped model ranks each type of site (report/task2_plan.md, 4.2).

    python analysis/site_types/build_table.py    # first
    python analysis/site_types/by_type.py

For each labelled cell line, read depth, and site type (DRACH motif, nearby
sites within 150 nt, region of the transcript, and motif x depth), on held-out
genes:

  pairwise accuracy  ROC AUC: pick one modified and one unmodified site of this
                     type; how often the model ranks the modified one higher.
                     0.5 = chance. The headline, because it does not depend on
                     how often this type is modified.
  PR AUC, base rate  reported beside it. PR AUC / base rate ("lift") is NOT used
                     to compare types: it cannot exceed 1 / base rate, so a
                     commonly modified motif (GGACT, 23% modified) is capped at
                     4.4x while a rare one can reach 100x+. That cap is why
                     GAPS.md found GGACT's lift lowest.

Intervals: 95%, from resampling whole genes (sites of one gene are not
independent). At 1, 3 and 10 reads every network sees reads thinned at random;
each metric is averaged over the available random draws, and draw_sd shows how
much one draw moves it. A stratum with fewer than MIN_CLASS modified or
unmodified sites is reported as too few to judge.

Writes analysis/site_types/results/by_type.csv (aggregates only, no labels).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / ".cache" / "site_types" / "sites.csv.gz"
RESULTS = Path(__file__).resolve().parent / "results"
N_BOOT = 200
MIN_CLASS = 20
NBR_BINS = [(0, 0, "none"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10**6, "6+")]


def ranked(score: np.ndarray, y: np.ndarray):
    order = np.argsort(-score, kind="stable")
    return y[order].astype(float), order


def ap_auc(ys: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted PR AUC (average precision) and ROC AUC, for sites already sorted by score."""
    wp, wn = w * ys, w * (1 - ys)
    tp, seen = np.cumsum(wp), np.cumsum(w)
    P, N = wp.sum(), wn.sum()
    if P == 0 or N == 0:
        return np.nan, np.nan
    ap = float((wp * tp / np.maximum(seen, 1e-12)).sum() / P)
    # ROC AUC: for every positive, the weight of negatives ranked below it.
    neg_below = N - np.cumsum(wn)
    auc = float((wp * neg_below).sum() / (P * N))
    return ap, auc


def metrics(score, y, genes, rng) -> dict:
    ys, order = ranked(score, y)
    g = pd.factorize(genes[order])[0]
    ap, auc = ap_auc(ys, np.ones(len(ys)))
    boots = []
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, g.max() + 1, g.max() + 1), minlength=g.max() + 1)
        boots.append(ap_auc(ys, counts[g].astype(float)))
    b = np.array(boots)
    return {"pr_auc": ap, "pair_acc": auc,
            "pr_auc_lo": np.nanpercentile(b[:, 0], 2.5), "pr_auc_hi": np.nanpercentile(b[:, 0], 97.5),
            "pair_acc_lo": np.nanpercentile(b[:, 1], 2.5), "pair_acc_hi": np.nanpercentile(b[:, 1], 97.5)}


def nbr_bin(n: np.ndarray) -> np.ndarray:
    out = np.empty(len(n), dtype=object)
    for lo, hi, name in NBR_BINS:
        out[(n >= lo) & (n <= hi)] = name
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    d = pd.read_csv(TABLE)
    d["nearby_sites"] = nbr_bin(d.neighbours.to_numpy())
    d["all"] = "all sites"
    draws = sorted({int(c.split("draw")[1]) for c in d if c.startswith("rank_d1_draw")})
    depth_cols = {"full": ["rank_full"]} | {f"{k} reads": [f"rank_d{k}_draw{r}" for r in draws] for k in (10, 3, 1)}
    families = ["all", "motif", "nearby_sites", "region"]
    rng = np.random.default_rng(4262)
    rows = []
    for file, f in d.groupby("file"):
        cell = {"dataset0": "HCT116 (cell line 1)", "data1": "HEK293T (cell line 2)"}[file]
        for depth, cols in depth_cols.items():
            for fam in families:
                for value, s in f.groupby(fam):
                    y = s.label.to_numpy()
                    base = {"cell_line": cell, "depth": depth, "family": fam, "type": value,
                            "sites": len(s), "modified": int(y.sum()), "base_rate": y.mean(),
                            "genes": s.gene_id.nunique()}
                    if y.sum() < MIN_CLASS or (1 - y).sum() < MIN_CLASS:
                        rows.append(base | {"note": "too few to judge"})
                        continue
                    per_draw = [metrics(s[c].to_numpy(), y, s.gene_id.to_numpy(), rng) for c in cols]
                    m = {k: float(np.mean([p[k] for p in per_draw])) for k in per_draw[0]}
                    m["draw_sd_pair_acc"] = float(np.std([p["pair_acc"] for p in per_draw])) if len(cols) > 1 else 0.0
                    rows.append(base | m | {"draws": len(cols), "note": ""})
            print(f"{cell} {depth}: done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "by_type.csv", index=False, float_format="%.4f")
    print(f"-> {RESULTS / 'by_type.csv'} ({len(out)} rows, draws at depth: {draws})")


if __name__ == "__main__":
    main()
