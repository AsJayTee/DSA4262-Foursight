"""At what distance do modified sites cluster, and is it the same in both cell lines?

    python analysis/newdata/distance_bands.py

For the radius test (2026-10-06): a neighbour radius of ~100 nt transferred best
between cell lines, while wider radii helped only the training cell line. Two
label-only analyses, independent of any model:

A. Co-modification by distance band. For every pair of candidate sites on the
   same transcript, binned by distance, the number of pairs where BOTH are
   positive, against two expectations:
     global      every site positive at the file's overall rate
     transcript  positives placed at random WITHIN each transcript - the exact
                 expectation of a within-transcript shuffle, k(k-1)/(n(n-1))
                 per pair for a transcript with n sites and k positives
   so  total = observed / global  splits into
       transcript part = transcript / global   (positive-rich transcripts)
       local part      = observed / transcript (clustering beyond that)
   with 95% intervals from resampling transcripts.
   dataset0 and data1 are DIFFERENT CELL LINES (course staff, 2026-10-05); the
   shared-site rows hold positions fixed and change only the cell line.

B. How many neighbours a site has within each radius.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a import external  # noqa: E402
from m6a.data import load_labels, resolve_data_dir  # noqa: E402

BANDS = [(1, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600)]
RADII = (25, 50, 100, 200, 400)
RESAMPLES = 200
RNG = np.random.default_rng(4262)
OUT = ROOT / "analysis" / "newdata" / "distance_bands"


def per_transcript(frame: pd.DataFrame) -> list[dict]:
    """Pair counts per band for each transcript: all pairs, positive-positive pairs,
    and the within-transcript expectation of positive-positive pairs."""
    rows = []
    for t, g in frame.groupby("transcript_id"):
        p = g["transcript_position"].to_numpy()
        y = g["label"].to_numpy().astype(int)
        n, k = len(p), int(y.sum())
        iu = np.triu_indices(n, 1)
        d = np.abs(p[:, None] - p[None, :])[iu]
        both = (y[:, None] * y[None, :])[iu]
        frac = k * (k - 1) / (n * (n - 1)) if n > 1 else 0.0
        r = {"transcript_id": t, "n": n, "k": k}
        for lo, hi in BANDS:
            sel = (d >= lo) & (d < hi)
            r[f"pairs_{lo}"] = int(sel.sum())
            r[f"both_{lo}"] = int(both[sel].sum())
            r[f"tx_{lo}"] = float(sel.sum() * frac)
        rows.append(r)
    return rows


def decompose(t: pd.DataFrame, rate: float) -> dict:
    out = {}
    for lo, hi in BANDS:
        pairs, both, tx = t[f"pairs_{lo}"].sum(), t[f"both_{lo}"].sum(), t[f"tx_{lo}"].sum()
        glob = pairs * rate ** 2
        out[(lo, hi)] = (both / glob, tx / glob, both / tx if tx else np.nan)
    return out


def report(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rate = frame["label"].mean()
    t = pd.DataFrame(per_transcript(frame))
    point = decompose(t, rate)
    boots = [decompose(t.iloc[RNG.integers(0, len(t), len(t))], rate) for _ in range(RESAMPLES)]
    rows = []
    for band in point:
        local = np.array([b[band][2] for b in boots])
        rows.append({"data": name, "band": f"{band[0]}-{band[1]} nt", "pairs": int(t[f"pairs_{band[0]}"].sum()),
                     "total": point[band][0], "transcript_part": point[band][1], "local_part": point[band][2],
                     "local_lo": np.nanpercentile(local, 2.5), "local_hi": np.nanpercentile(local, 97.5)})
    out = pd.DataFrame(rows)
    print(f"\n{name}: {len(frame):,} sites, {rate:.2%} positive, {len(t):,} transcripts")
    print(out.drop(columns="data").to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return out


def neighbours(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in RADII:
        counts = []
        for _, g in frame.groupby("transcript_id"):
            p = g["transcript_position"].to_numpy()
            counts.append((np.abs(p[:, None] - p[None, :]) <= r).sum(1) - 1)
        c = np.concatenate(counts)
        rows.append({"data": name, "radius": r, "no neighbours": (c == 0).mean(), "1 neighbour": (c == 1).mean(),
                     "2+ neighbours": (c >= 2).mean(), "median": np.median(c), "mean": c.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir()
    keys = ["transcript_id", "transcript_position", "label"]
    d0 = load_labels(data_dir / "data.info.labelled")[keys]
    d1 = external.load_info(data_dir, "data1").reset_index()[keys]
    shared = d0.merge(d1, on=keys[:2], suffixes=("_0", "_1"))
    bands = pd.concat([
        report("dataset0 (cell line 1), all sites", d0),
        report("data1 (cell line 2), all sites", d1),
        report("shared sites, cell line 1 labels", shared.rename(columns={"label_0": "label"})),
        report("shared sites, cell line 2 labels", shared.rename(columns={"label_1": "label"})),
    ])
    bands.to_csv(OUT / "bands.csv", index=False)
    nb = pd.concat([neighbours("dataset0", d0), neighbours("data1", d1)])
    print("\nB. Neighbours within each radius (other candidate sites on the same transcript):")
    print(nb.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    nb.to_csv(OUT / "neighbours.csv", index=False)


if __name__ == "__main__":
    main()
