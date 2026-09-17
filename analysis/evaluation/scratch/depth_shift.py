"""Does the shipped model survive SG-NEx's read depths?

Train on full-depth Hct116 (as we do now), then score held-out sites whose
reads have been subsampled to SG-NEx-like depths. If PR AUC collapses, every
Task 2 number computed from these scores is meaningless.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from m6a import registry
from m6a.data import iter_sites, load_labels, align_to_features, assign_folds, Site
from m6a.evaluation import metrics

rng = np.random.default_rng(4262)
ex  = registry.get("features","quantiles_v1")()
M   = registry.get("models","lightgbm")
P   = dict(learning_rate=0.05,num_leaves=63,n_estimators=600,min_child_samples=40,feature_fraction=0.7)

lab = load_labels("data/raw/data.info.labelled")
folds_by_gene = dict(zip(lab.gene_id, assign_folds(lab,4262,5,"gene_id")))
tx2gene = dict(zip(lab.transcript_id, lab.gene_id))

DEPTHS = [1,3,5,10,20,None]          # None = full depth, as today
rows = {d: [] for d in DEPTHS}
idx  = {d: [] for d in DEPTHS}

for s in iter_sites("data/raw/dataset0.json.gz"):
    for d in DEPTHS:
        if d is None or s.n_reads <= d:
            sub = s
        else:
            pick = rng.choice(s.n_reads, size=d, replace=False)
            sub = Site(s.transcript_id, s.position, s.kmer, s.reads[pick])
        rows[d].append(ex.site_features(sub)); idx[d].append(s.key)

print(f"{'scored at depth':>16} {'ROC':>7} {'PR':>7} {'lift':>6}")
base = None
for d in DEPTHS:
    F = pd.DataFrame(rows[d], dtype=np.float32)
    F.index = pd.MultiIndex.from_tuples(idx[d], names=["transcript_id","transcript_position"])
    J = align_to_features(F, lab)
    cols=list(F.columns); X=J[cols]; y=J["label"].to_numpy()
    g = J.reset_index().transcript_id.map(tx2gene)
    fo = g.map(folds_by_gene).to_numpy()

    # train on FULL depth (fold-out), score on subsampled — the Task 2 situation
    Ffull = pd.DataFrame(rows[None], dtype=np.float32)
    Ffull.index = pd.MultiIndex.from_tuples(idx[None], names=["transcript_id","transcript_position"])
    Jf = align_to_features(Ffull, lab); Xf = Jf[cols]; yf = Jf["label"].to_numpy()
    gf = Jf.reset_index().transcript_id.map(tx2gene).map(folds_by_gene).to_numpy()

    oof=np.zeros(len(y))
    for f in sorted(np.unique(fo)):
        h = fo==f; hf = gf==f
        m=M(**P); m.fit(Xf[~hf], yf[~hf]); oof[h]=m.predict_proba(X[h])
    s=metrics(y,oof)
    lbl = "full (today)" if d is None else f"{d} read(s)"
    print(f"{lbl:>16} {s['roc_auc']:>7.4f} {s['pr_auc']:>7.4f} {s['pr_auc_lift']:>5.2f}x")
