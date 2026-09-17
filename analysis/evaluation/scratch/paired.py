"""Paired per-fold comparison of pooled_v1 vs quantiles_v1.

Comparing two POOLED numbers against the fold-to-fold SD (0.0203) is the wrong
test: it ignores that both models are scored on the SAME folds, so fold
difficulty cancels. The right test is the per-fold DIFFERENCE.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from scipy import stats
from m6a import registry
from m6a.data import iter_sites, load_labels, align_to_features, assign_folds
from m6a.evaluation import metrics

SETS=["pooled_v1","quantiles_v1"]
M=registry.get("models","lightgbm")
P=dict(learning_rate=0.05,num_leaves=63,n_estimators=600,min_child_samples=40,feature_fraction=0.7)
lab=load_labels("data/raw/data.info.labelled")

F={}
for s in SETS:
    F[s]=registry.get("features",s)().transform(iter_sites("data/raw/dataset0.json.gz"))
J0=align_to_features(F[SETS[0]],lab); y=J0["label"].to_numpy()
folds=assign_folds(J0.reset_index(),seed=4262,n_folds=5,group_by="gene_id").to_numpy()

per={}
for s in SETS:
    cols=list(F[s].columns); X=align_to_features(F[s],lab)[cols]; v=[]
    for f in sorted(np.unique(folds)):
        h=folds==f; m=M(**P); m.fit(X[~h],y[~h])
        v.append(metrics(y[h],m.predict_proba(X[h]))["pr_auc"])
    per[s]=np.array(v)

a,b=per["pooled_v1"],per["quantiles_v1"]; d=b-a
print(f"{'fold':>4} {'pooled':>8} {'quantiles':>10} {'diff':>9}")
for i,(x,z,w) in enumerate(zip(a,b,d)): print(f"{i:>4} {x:>8.4f} {z:>10.4f} {w:>+9.4f}")
print()
print(f"pooled    mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}")
print(f"quantiles mean {b.mean():.4f}  sd {b.std(ddof=1):.4f}")
print(f"UNPAIRED view : gap {b.mean()-a.mean():+.4f} vs fold sd ~{a.std(ddof=1):.4f}  -> looks like noise")
t,p=stats.ttest_rel(b,a)
print(f"PAIRED view   : mean diff {d.mean():+.4f}  sd of diff {d.std(ddof=1):.4f}  t={t:.2f}  p={p:.4f}")
print(f"                wins {int((d>0).sum())}/5 folds")
