"""Is co-modification really shorter-ranged in cell line 2, or is that an artefact?

    python analysis/newdata/decay_scale_check.py

The 2026-10-08 batch fitted the decay scale lambda of exp(-d / lambda) to the
training labels (graph.comod_decay_scale): ~101-122 nt on cell line 1, and, for
reference, ~59 nt on cell line 2. Before the report says "co-modification fades
twice as fast in cell line 2", three checks:

  interval     95% interval on lambda from resampling transcripts
  same sites   lambda on the sites present in BOTH files, under each cell
               line's labels - positions held fixed, only the labels change
  same rate    cell line 2 has more positives (7.3% vs 4.5%); thin its
               positives at random to cell line 1's rate and refit (random
               thinning leaves the expected ratio unchanged, so a shift here
               would mean the fit is sensitive to label density)

Writes analysis/newdata/decay_scale/decay_scale.csv.
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

# Copied from analysis/representation/graph.py (importing it needs torch).
FIT_BANDS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 400))

RESAMPLES = 1000
THINNING_REPEATS = 50
RNG = np.random.default_rng(4262)
OUT = ROOT / "analysis" / "newdata" / "decay_scale"


def per_transcript(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(transcripts, bands) arrays of positive-positive pairs and their
    within-transcript expectation - the same counts as graph._band_pair_counts,
    kept per transcript so they can be resampled."""
    both, expected = [], []
    for _, g in frame.groupby("transcript_id"):
        p, y = g.transcript_position.to_numpy(), g.label.to_numpy().astype(int)
        n, k = len(p), int(y.sum())
        if n < 2:
            continue
        iu = np.triu_indices(n, 1)
        d = np.abs(p[:, None] - p[None, :])[iu]
        bb = (y[:, None] * y[None, :])[iu]
        frac = k * (k - 1) / (n * (n - 1))
        sel = [(d > lo) & (d <= hi) for lo, hi in FIT_BANDS]
        both.append([bb[s].sum() for s in sel])
        expected.append([s.sum() * frac for s in sel])
    return np.array(both, float), np.array(expected, float)


def fit(both: np.ndarray, expected: np.ndarray) -> float:
    """graph.comod_decay_scale on summed counts, without its [20, 400] clip so
    the interval is not truncated."""
    excess = both / np.maximum(expected, 1e-9) - 1
    mid = np.array([(lo + hi) / 2 for lo, hi in FIT_BANDS], float)
    ok = (excess > 0) & (expected > 0)
    if ok.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(mid[ok], np.log(excess[ok]), 1, w=np.sqrt(expected[ok]))
    return -1 / slope if slope < 0 else np.inf


def summarise(name: str, frame: pd.DataFrame) -> dict:
    both, expected = per_transcript(frame)
    boots = [fit(both[i].sum(0), expected[i].sum(0))
             for i in (RNG.integers(0, len(both), len(both)) for _ in range(RESAMPLES))]
    boots = np.array(boots)
    finite = boots[np.isfinite(boots)]
    return {"labels": name, "sites": len(frame), "positive %": 100 * frame.label.mean(),
            "lambda_nt": fit(both.sum(0), expected.sum(0)),
            "ci_low": np.percentile(finite, 2.5), "ci_high": np.percentile(finite, 97.5),
            "non-decaying resamples %": 100 * (1 - len(finite) / len(boots))}


def thinned(frame: pd.DataFrame, rate: float) -> pd.DataFrame:
    pos = np.flatnonzero(frame.label.to_numpy() == 1)
    keep = RNG.choice(pos, int(round(rate * len(frame))), replace=False)
    out = frame.copy()
    out["label"] = 0
    out.iloc[keep, out.columns.get_loc("label")] = 1
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir()
    keys = ["transcript_id", "transcript_position", "label"]
    d0 = load_labels(data_dir / "data.info.labelled")[keys]
    d1 = external.load_info(data_dir, "data1").reset_index()[keys]
    shared = d0.merge(d1, on=keys[:2], suffixes=("_0", "_1"))
    rows = [
        summarise("cell line 1, all sites", d0),
        summarise("cell line 2, all sites", d1),
        summarise("shared sites, cell line 1 labels", shared.rename(columns={"label_0": "label"})),
        summarise("shared sites, cell line 2 labels", shared.rename(columns={"label_1": "label"})),
    ]
    table = pd.DataFrame(rows)
    lam = [fit(*(c.sum(0) for c in per_transcript(thinned(d1, d0.label.mean()))))
           for _ in range(THINNING_REPEATS)]
    table = pd.concat([table, pd.DataFrame([{
        "labels": f"cell line 2 thinned to {100 * d0.label.mean():.2f}% positives ({THINNING_REPEATS} draws)",
        "sites": len(d1), "positive %": 100 * d0.label.mean(), "lambda_nt": float(np.median(lam)),
        "ci_low": float(np.percentile(lam, 2.5)), "ci_high": float(np.percentile(lam, 97.5))}])])
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    table.to_csv(OUT / "decay_scale.csv", index=False)


if __name__ == "__main__":
    main()
