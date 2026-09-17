"""How much of our PR AUC is explainable by the motif alone, with no signal?"""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0,"src")
from m6a.data import assign_folds
from m6a.evaluation import metrics

rows=[]
with gzip.open("data/raw/dataset0.json.gz","rt") as fh:
    for line in fh:
        r=json.loads(line); t=next(iter(r)); p=next(iter(r[t])); k=next(iter(r[t][p]))
        rows.append((t,int(p),k,k[1:6],len(r[t][p][k])))
d=pd.DataFrame(rows,columns=["transcript_id","transcript_position","kmer","motif","depth"])
lab=pd.read_csv("data/raw/data.info.labelled")
m=d.merge(lab,on=["transcript_id","transcript_position"])
m["fold"]=assign_folds(m,seed=4262,n_folds=5,group_by="gene_id").to_numpy()
y=m.label.to_numpy()

print("FOLD BALANCE")
fb=m.groupby("fold").label.agg(['size','sum','mean'])
fb['mean']=(100*fb['mean']).round(2)
print(fb.to_string()); print()

def oof(scorer):
    o=np.zeros(len(m))
    for f in sorted(m.fold.unique()):
        h=m.fold.values==f
        o[h]=scorer(m[~h],m[h])
    return o

# motif-only: score = positive rate of that 5-mer in the training folds
def motif_rate(tr,te):
    r=tr.groupby("motif").label.mean()
    return te.motif.map(r).fillna(tr.label.mean()).to_numpy()
# 7-mer: does the flanking base add anything on its own?
def kmer_rate(tr,te):
    r=tr.groupby("kmer").label.mean()
    return te.kmer.map(r).fillna(tr.label.mean()).to_numpy()
def depth_only(tr,te):
    return te.depth.to_numpy().astype(float)

for nm,fn in [("motif 5-mer base rate only",motif_rate),
              ("7-mer base rate only",kmer_rate),
              ("read depth only",depth_only)]:
    s=metrics(y,oof(fn))
    print(f"{nm:32s}  ROC {s['roc_auc']:.4f}  PR {s['pr_auc']:.4f}  ({s['pr_auc_lift']:.2f}x)")
print(f"{'random':32s}  ROC 0.5000  PR {y.mean():.4f}  (1.00x)")
print()
print("shipped model (from meta.json): ROC 0.9169  PR 0.4759  (10.59x)")
