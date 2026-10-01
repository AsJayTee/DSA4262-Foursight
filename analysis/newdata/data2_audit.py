"""Which features peak at a partial mix of modified molecules? (decision 0029, step 2)

    python analysis/newdata/data2_audit.py [feature_set]     # default quantiles_all_v1

data2 is one synthetic sequence at 0, 25, 50, 70, 75, 95 and 100% modified
molecules. Published tools' scores rise with the fraction; ours rise to ~70%
and then fall. The suspicion: features that measure how much a site's reads
DISAGREE (spread, tails, cross-position correlation) are largest at a 50/50
mix and small again at 100%, so a model leaning on them reads a fully modified
site as less modified.

For each feature column, the mean over each sample's 189 sites, expressed in
units of the 0% sample's site-to-site spread. Reported per column:

  monotonic  Spearman of the 7 means against the fraction (+1 = rises steadily)
  peak       fraction where the mean is furthest from the 0% sample
  hump       how far the largest shift exceeds the 100% sample's shift (> 0
             means the 100% sample has fallen back towards unmodified)

Writes analysis/newdata/data2_audit.csv and prints the worst offenders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a import external  # noqa: E402
from m6a.data import resolve_data_dir  # noqa: E402
from m6a.feature_cache import extract  # noqa: E402


def main() -> None:
    features = sys.argv[1] if len(sys.argv) > 1 else "quantiles_all_v1"
    data_dir = resolve_data_dir()
    X = extract(external.paths(data_dir, "data2")[0], features, [None])[None].features
    info = external.load_info(data_dir, "data2").reindex(X.index)
    fraction = info["label"].to_numpy()
    means = X.groupby(fraction).mean()
    spread = X[fraction == 0.0].std().replace(0, np.nan)
    shift = (means - means.loc[0.0]) / spread            # in units of the 0% spread
    fractions = means.index.to_numpy()
    rows = []
    for column in X.columns:
        s = shift[column]
        if s.isna().all():
            continue
        absolute = s.abs()
        rows.append({"feature": column,
                     "monotonic": spearmanr(fractions, s).correlation,
                     "peak": float(absolute.idxmax()),
                     "shift_at_peak": float(s[absolute.idxmax()]),
                     "shift_at_100": float(s.loc[1.0]),
                     "hump": float(absolute.max() - absolute.loc[1.0])})
    table = pd.DataFrame(rows).sort_values("hump", ascending=False)
    out = Path(__file__).resolve().parent / "data2_audit.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 160)
    print(f"{features}: {len(table)} columns. Largest humps (peak at a partial mix, "
          "then back towards unmodified at 100%):")
    print(table.head(25).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    strong = table[table["shift_at_peak"].abs() >= 1.0]
    humped = strong[strong["hump"] >= 0.5 * strong["shift_at_peak"].abs()]
    print(f"\nColumns that move at least one 0%-spread at their peak: {len(strong)}; of these, "
          f"{len(humped)} lose half or more of that shift by 100%.")
    families = humped["feature"].str.extract(r"^(coupling|tx|nbr|left|right|motif|[a-z]+_[a-z0-9]+)")[0]
    print("By family:", families.value_counts().to_dict())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
