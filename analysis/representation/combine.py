"""Do the winners stack, in the harness's setting (every training label used)?

    python analysis/representation/combine.py

Per outer fold, LightGBM (the `everything` parameters) is trained on ALL
training rows with the hand features plus some of:

  kmer_norm   reads standardised against their 7-mer's average read, summarised
              per site. Unsupervised; statistics fitted on the training rows.
  deepset     the cross-fitted network score from crossfit.py
  read_ae_pred  likewise

and every arm is paired against hand alone. Nadeau & Bengio corrected with the
plain 5-fold train/test ratio (~0.25), because here the tree does see every
training row. Still one split and no depth augmentation, so still a screen.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import common
import learn
from probe import LGBM, scores
from m6a import registry

ARMS = {
    "hand": [],
    "hand + kmer_norm": ["kmer_norm"],
    "hand + deepset": ["deepset"],
    "hand + read_ae_pred": ["read_ae_pred"],
    "hand + kmer_norm + deepset": ["kmer_norm", "deepset"],
    "hand + all three": ["kmer_norm", "deepset", "read_ae_pred"],
}


def main() -> None:
    bundle = common.load()
    stacked = {}
    for name in ("deepset", "read_ae_pred"):
        path = common.OUT / f"crossfit_scores_{name}.npz"
        if path.exists():
            stacked[name] = dict(np.load(path))
    arms = {a: parts for a, parts in ARMS.items()
            if all(p == "kmer_norm" or p in stacked for p in parts)}
    records = []
    everyone = np.arange(len(bundle.y))
    for fold in sorted(np.unique(bundle.folds)):
        roles = common.split_roles(bundle, fold)
        held = np.flatnonzero(roles["heldout"])
        train = np.flatnonzero(~roles["heldout"])
        # unsupervised, so fit its per-7-mer statistics on every training read
        kn, _ = learn.kmer_normalised(bundle, {"encoder": ~roles["heldout"]}, held, everyone)
        blocks = {"kmer_norm": pd.DataFrame(kn, index=bundle.X.index,
                                            columns=[f"kn{i}" for i in range(kn.shape[1])])}
        for name, per_fold in stacked.items():
            blocks[name] = pd.DataFrame({f"score_{name}": per_fold[f"fold{fold}"]}, index=bundle.X.index)
        out = {"fold": int(fold), "repetition": 0}
        for arm, parts in arms.items():
            X = pd.concat([bundle.X, *[blocks[p] for p in parts]], axis=1)
            model = registry.get("models", "lightgbm")(**LGBM)
            model.fit(X.iloc[train], bundle.y[train])
            out[arm] = scores(bundle.y[held], model.predict_proba(X.iloc[held]))
        records.append(out)
        print(f"fold {fold}: " + "  ".join(f"{a} {out[a]['pr_auc']:.4f}" for a in arms), flush=True)

    (common.RESULTS / "combine.json").write_text(json.dumps(records, indent=1))
    ratio = np.mean([(bundle.folds == f).sum() / (bundle.folds != f).sum()
                     for f in np.unique(bundle.folds)])
    lines = ["| arm | PR AUC | vs hand | wins | corrected p |", "|---|---:|---:|---:|---:|"]
    base = np.array([r["hand"]["pr_auc"] for r in records])
    for arm in arms:
        v = np.array([r[arm]["pr_auc"] for r in records])
        d = v - base
        if arm == "hand":
            lines.append(f"| {arm} | {v.mean():.4f} | - | - | - |")
            continue
        t = d.mean() / np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))
        p = 2 * stats.t.sf(abs(t), len(d) - 1)
        lines.append(f"| {arm} | {v.mean():.4f} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | {p:.4f} |")
    text = "\n".join(lines)
    (common.RESULTS / "combine.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
