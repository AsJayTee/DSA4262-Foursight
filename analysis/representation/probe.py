"""Judge each saved representation through several lenses (PLAN.md section 4).

    python analysis/representation/probe.py --models hand deepset read_ae
    python analysis/representation/probe.py --models hand deepset --limit 5000   # smoke

Per fold, every probe trains on the probe-half (B) rows and scores the held-out
fold. The `hand` vectors must exist (learn.py --model hand): the incremental
tests (E3) combine them with each z. Writes results/probe_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import common
from m6a import registry

warnings.filterwarnings("ignore")

LGBM = dict(learning_rate=0.05, num_leaves=63, n_estimators=600,
            min_child_samples=40, feature_fraction=0.7)


def scores(y, s) -> dict:
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}


def lgbm(Xtr, ytr, Xte_list):
    model = registry.get("models", "lightgbm")(**LGBM)
    import pandas as pd
    cols = [f"c{i}" for i in range(Xtr.shape[1])]
    model.fit(pd.DataFrame(Xtr, columns=cols), ytr)
    return [model.predict_proba(pd.DataFrame(X, columns=cols)) for X in Xte_list]


def logistic(Xtr, ytr, Xte_list):
    model = LogisticRegression(C=0.5, max_iter=3000).fit(Xtr, ytr)
    return [model.predict_proba(X)[:, 1] for X in Xte_list]


def standardise(train, *others):
    mean, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(a - mean) / sd for a in (train, *others)]


def linear_cka(a, b) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    return float(np.linalg.norm(b.T @ a) ** 2 /
                 (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def pooled_r2(Xtr, Ytr, Xte, Yte) -> float:
    """Variance-weighted R^2 of predicting every column of Y from X (ridge)."""
    # reshape: a one-column Y comes back flat, and (n, 1) - (n,) broadcasts
    # to (n, n) - which is how the first run reported R^2 = -48,000.
    pred = Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte).reshape(Yte.shape)
    return float(1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())


def probe_fold(name, fold, bundle, hand_cache) -> dict:
    roles = common.split_roles(bundle, fold)
    tr, te = np.flatnonzero(roles["probe"]), np.flatnonzero(roles["heldout"])
    ytr, yte = bundle.y[tr], bundle.y[te]
    z, depth_z, info = common.load_embedding(name, fold)
    bad = ~np.isfinite(z[np.concatenate([tr, te])]).all()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    ztr, zte, *zd = standardise(z[tr], z[te], *[depth_z[d] for d in common.EVAL_DEPTHS])
    out = {"fold": int(fold), "repetition": 0, "dims": int(z.shape[1]), "nonfinite": bool(bad),
           "encoder": {k: v for k, v in info.items() if k != "history"},
           "history": info.get("history", [])}

    # E1 - probes of different shapes on z alone
    rng = np.random.default_rng(4262 + fold)
    rotation, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
    p_log = logistic(ztr, ytr, [zte, *zd])
    p_lgb = lgbm(ztr, ytr, [zte, *zd])
    p_rot = lgbm(ztr @ rotation, ytr, [zte @ rotation])[0]
    knn = KNeighborsClassifier(n_neighbors=100, n_jobs=4).fit(ztr, ytr)
    mlp = MLPClassifier((64,), alpha=1e-3, early_stopping=True, max_iter=300, n_iter_no_change=15,
                        random_state=fold).fit(ztr, ytr)
    out["probes"] = {
        "logistic": scores(yte, p_log[0]),
        "knn": scores(yte, knn.predict_proba(zte)[:, 1]),
        "mlp": scores(yte, mlp.predict_proba(zte)[:, 1]),
        "lgbm": scores(yte, p_lgb[0]),
        "lgbm_rotated": scores(yte, p_rot),
    }
    # E5 - probes trained at full depth, scored on held-out sites embedded from fewer reads
    out["depth"] = {str(d): {"logistic": scores(yte, p_log[i + 1]), "lgbm": scores(yte, p_lgb[i + 1])}
                    for i, d in enumerate(common.EVAL_DEPTHS)}

    # E3 - incremental over the hand features, under a tree and under a linear model
    if name != "hand":
        if fold not in hand_cache:
            h, _, _ = common.load_embedding("hand", fold)
            hand_cache[fold] = standardise(h[tr], h[te])[:2]
        htr, hte = hand_cache[fold]
        both_tr, both_te = np.hstack([htr, ztr]), np.hstack([hte, zte])
        out["with_hand"] = {
            "lgbm": scores(yte, lgbm(both_tr, ytr, [both_te])[0]),
            "logistic": scores(yte, logistic(both_tr, ytr, [both_te])[0]),
        }
        # E4 - how much of z the hand features already explain
        sample = rng.choice(len(te), size=min(8000, len(te)), replace=False)
        out["redundancy"] = {
            "r2_hand_to_z": pooled_r2(htr, ztr, hte, zte),
            "cka_with_hand": linear_cka(zte[sample], hte[sample]),
        }

    # E4 - nuisance and effective dimensionality
    depth_tr, depth_te = np.log1p(bundle.n_reads[tr]), np.log1p(bundle.n_reads[te])
    motif = LogisticRegression(max_iter=2000, C=0.5).fit(ztr, bundle.motifs[tr])
    eig = np.clip(np.linalg.eigvalsh(np.cov(zte.T)), 0, None)
    out["content"] = {
        "r2_depth": pooled_r2(ztr, depth_tr[:, None], zte, depth_te[:, None]),
        "motif_accuracy": float((motif.predict(zte) == bundle.motifs[te]).mean()),
        "participation_ratio": float(eig.sum() ** 2 / (eig ** 2).sum()) if eig.sum() > 0 else 0.0,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    bundle = common.load(args.limit, with_reads=False)
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    folds = sorted(np.unique(bundle.folds).tolist())
    names = args.models
    hand_cache: dict = {}
    for name in names:
        started = time.time()
        records = [probe_fold(name, f, bundle, hand_cache) for f in folds]
        suffix = f"__limit{args.limit}" if args.limit else ""
        (common.RESULTS / f"probe_{name}{suffix}.json").write_text(json.dumps(records, indent=1))
        mean = lambda k: np.mean([r["probes"][k]["pr_auc"] for r in records])  # noqa: E731
        extra = ""
        if name != "hand":
            extra = f"  +hand lgbm {np.mean([r['with_hand']['lgbm']['pr_auc'] for r in records]):.4f}"
        print(f"[{name}] logistic {mean('logistic'):.4f}  knn {mean('knn'):.4f}  "
              f"mlp {mean('mlp'):.4f}  lgbm {mean('lgbm'):.4f}  rotated {mean('lgbm_rotated'):.4f}"
              f"{extra}  ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
