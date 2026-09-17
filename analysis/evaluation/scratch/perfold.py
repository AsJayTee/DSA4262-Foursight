"""Per-fold breakdown the pipeline currently computes and throws away,
plus a no-motif ablation and calibration check."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from m6a import registry
from m6a.data import iter_sites, load_labels, align_to_features, assign_folds
from m6a.evaluation import metrics

ex = registry.get("features","quantiles_v1")()
F = ex.transform(iter_sites("data/raw/dataset0.json.gz"))
lab = load_labels("data/raw/data.info.labelled")
J = align_to_features(F, lab)
cols=list(F.columns); X=J[cols]; y=J["label"].to_numpy()
folds = assign_folds(J.reset_index(), seed=4262, n_folds=5, group_by="gene_id").to_numpy()
M = registry.get("models","lightgbm")
P = dict(learning_rate=0.05,num_leaves=63,n_estimators=600,min_child_samples=40,feature_fraction=0.7)

motif_cols=[c for c in cols if c.startswith("motif_")]
sig_cols=[c for c in cols if not c.startswith("motif_")]

def run(use, tag):
    oof=np.zeros(len(y)); per=[]
    for f in sorted(np.unique(folds)):
        h=folds==f
        m=M(**P); m.fit(X.loc[~h,use], y[~h]); oof[h]=m.predict_proba(X.loc[h,use])
        s=metrics(y[h],oof[h]); per.append((f,s['n'],s['positive_rate'],s['roc_auc'],s['pr_auc']))
    o=metrics(y,oof)
    print(f"\n=== {tag} ({len(use)} cols) ===")
    print(f"{'fold':>4} {'n':>7} {'pos%':>6} {'ROC':>7} {'PR':>7}")
    for f,n,pr,r,p in per: print(f"{f:>4} {n:>7,} {100*pr:>5.2f}% {r:>7.4f} {p:>7.4f}")
    prs=np.array([p for *_,p in per])
    print(f"per-fold PR: mean {prs.mean():.4f} sd {prs.std(ddof=1):.4f} range [{prs.min():.4f},{prs.max():.4f}]")
    print(f"POOLED OOF : ROC {o['roc_auc']:.4f}  PR {o['pr_auc']:.4f}  ({o['pr_auc_lift']:.2f}x)")
    return oof

oof_all = run(cols, "quantiles + motif  [SHIPPED]")
run(sig_cols, "signal only, NO motif")
run(motif_cols, "motif one-hot only")

print("\n=== CALIBRATION of shipped model (is_unbalance=True) ===")
b=pd.DataFrame({"p":oof_all,"y":y})
b["bin"]=pd.qcut(b.p,10,duplicates="drop")
c=b.groupby("bin",observed=True).agg(mean_pred=("p","mean"),obs_rate=("y","mean"),n=("y","size"))
print(c.to_string())
print(f"\nmean predicted {oof_all.mean():.4f} vs actual positive rate {y.mean():.4f} "
      f"-> expected count {oof_all.sum():.0f} vs actual {y.sum():.0f} ({oof_all.sum()/y.sum():.2f}x)")
