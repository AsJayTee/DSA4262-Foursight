"""Is label clustering along transcripts real, or an artefact of site density?

    python analysis/newdata/homophily_shuffle.py

The claim it tests (2026-10-02): a positive site is far more likely to have a
positive neighbour within 50 nt than chance, and dataset0 clusters about twice
as strongly as data1 - the reason neighbour features learned on dataset0 did
not transfer. Two confounds could fake that:

1. Positives concentrate on some transcripts (expression, length, density of
   candidate sites). Then "a positive neighbour" partly means "a positive-rich
   transcript", not local clustering. Control: shuffle labels WITHIN each
   transcript (same positives per transcript, positions randomised), 20 times,
   and compare the lift.
2. The two files cover different sites. Control: repeat on the 67,320 sites in
   both files only, where positions and density are identical, so any
   remaining difference between the labellings is the labelling itself.

lift = P(positive | a positive within w nt) / P(positive | none within w).
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

WINDOWS = (10, 50, 200)
SHUFFLES = 20
RNG = np.random.default_rng(4262)


def prepare(frame: pd.DataFrame) -> list[tuple[np.ndarray, dict]]:
    """Per transcript: labels, and for each window the 'within w, not itself' matrix."""
    out = []
    for _, t in frame.sort_values(["transcript_id", "transcript_position"]).groupby("transcript_id"):
        p = t["transcript_position"].to_numpy()
        d = np.abs(p[:, None] - p[None, :])
        np.fill_diagonal(d, 10**9)
        out.append((t["label"].to_numpy().astype(int), {w: (d <= w) for w in WINDOWS}))
    return out


def lifts(groups, labels_of) -> dict:
    hit, n = {w: [0, 0] for w in WINDOWS}, {w: [0, 0] for w in WINDOWS}   # [with pos nbr, without]
    for i, (y0, near) in enumerate(groups):
        y = labels_of(i, y0)
        for w in WINDOWS:
            has = (near[w].astype(np.int32) @ y) > 0
            hit[w][0] += int(y[has].sum()); n[w][0] += int(has.sum())
            hit[w][1] += int(y[~has].sum()); n[w][1] += int((~has).sum())
    return {w: (hit[w][0] / max(n[w][0], 1)) / max(hit[w][1] / max(n[w][1], 1), 1e-9) for w in WINDOWS}


def report(name: str, frame: pd.DataFrame) -> None:
    groups = prepare(frame)
    observed = lifts(groups, lambda i, y: y)
    shuffled = [lifts(groups, lambda i, y: RNG.permutation(y)) for _ in range(SHUFFLES)]
    print(f"\n{name}: {len(frame):,} sites, {frame['label'].mean():.2%} positive, {len(groups):,} transcripts")
    print(f"  {'window':>8s} {'observed lift':>14s} {'shuffled within transcript':>28s} {'local part (obs / shuffled)':>28s}")
    for w in WINDOWS:
        s = np.array([x[w] for x in shuffled])
        print(f"  {w:>5d} nt {observed[w]:>14.2f} {s.mean():>15.2f} [{np.percentile(s, 2.5):.2f}, {np.percentile(s, 97.5):.2f}]"
              f" {observed[w] / s.mean():>20.2f}x")


def main() -> None:
    data_dir = resolve_data_dir()
    d0 = load_labels(data_dir / "data.info.labelled")[["transcript_id", "transcript_position", "label"]]
    d1 = external.load_info(data_dir, "data1").reset_index()[["transcript_id", "transcript_position", "label"]]
    report("dataset0, all sites", d0)
    report("data1, all sites", d1)
    shared = d0.merge(d1, on=["transcript_id", "transcript_position"], suffixes=("_0", "_1"))
    report("shared sites, dataset0's labels", shared.rename(columns={"label_0": "label"}))
    report("shared sites, data1's labels", shared.rename(columns={"label_1": "label"}))


if __name__ == "__main__":
    main()
