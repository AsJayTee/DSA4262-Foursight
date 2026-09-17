"""Does read-level (MIL-style) modelling beat site-level summary statistics,
especially at low depth?

A torch-free proxy for the proposed architecture:
    read 9D + motif -> shared read-level GBT -> pool read probs -> site score

Pooling is a FIXED function (mean / max / q90), not a learned second stage, so
there is no stacking leakage: the read model is trained fold-out and the site
score is a deterministic function of held-out read probabilities.

Reads inherit their site's label (naive instance labelling) - the standard
cheap MIL baseline. That is exactly the assumption m6Anet's attention layer is
designed to relax, so this is a LOWER bound on what read-level modelling can do.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from scipy import stats
from m6a import registry
from m6a.data import iter_sites, load_labels, assign_folds
from m6a.evaluation import metrics
from m6a.features.base import DRACH_MOTIFS

DEPTHS=[1,3,10,None]
MOTIF_IX={m:i for i,m in enumerate(DRACH_MOTIFS)}
lab=load_labels("data/raw/data.info.labelled")
k2l=dict(zip(zip(lab.transcript_id,lab.transcript_position),lab.label))
k2g=dict(zip(zip(lab.transcript_id,lab.transcript_position),lab.gene_id))
gf=dict(zip(lab.gene_id,assign_folds(lab,4262,5,"gene_id")))

# ---- one pass: flat read table + site boundaries, per depth ----
rng=np.random.default_rng(4262)
reads={d:[] for d in DEPTHS}; owner={d:[] for d in DEPTHS}
sy=[]; sf=[]; smot=[]
for si,site in enumerate(iter_sites("data/raw/dataset0.json.gz")):
    if site.key not in k2l: continue
    k=len(sy); sy.append(k2l[site.key]); sf.append(gf[k2g[site.key]]); smot.append(MOTIF_IX[site.motif])
    for d in DEPTHS:
        r=site.reads if (d is None or site.n_reads<=d) else site.reads[rng.choice(site.n_reads,size=d,replace=False)]
        reads[d].append(r.astype(np.float32)); owner[d].append(np.full(len(r),k,dtype=np.int32))
sy=np.asarray(sy); sf=np.asarray(sf); smot=np.asarray(smot)
R={d:np.vstack(reads[d]) for d in DEPTHS}; O={d:np.concatenate(owner[d]) for d in DEPTHS}
del reads, owner
print(f"sites {len(sy):,} | reads full {len(R[None]):,}",flush=True)

M=registry.get("models","lightgbm")
RP=dict(learning_rate=0.05,num_leaves=63,n_estimators=300,min_child_samples=200,feature_fraction=0.8)
COLS=[f"f{i}" for i in range(9)]+["motif"]

def readframe(d,mask=None):
    idx=np.arange(len(R[d])) if mask is None else np.where(mask)[0]
    X=np.column_stack([R[d][idx], smot[O[d][idx]].astype(np.float32)])
    return pd.DataFrame(X,columns=COLS)

def pool(p,own,n,fn):
    out=np.zeros(n)
    order=np.argsort(own,kind="stable"); own_s=own[order]; p_s=p[order]
    bounds=np.searchsorted(own_s,np.arange(n+1))
    for i in range(n):
        seg=p_s[bounds[i]:bounds[i+1]]
        out[i]=fn(seg) if len(seg) else 0.0
    return out

POOLS={"mean":np.mean,"max":np.max,"q90":lambda s:np.quantile(s,0.90)}
res={}
# read-level model trained on FULL-depth reads (as the site model is), fold-out
fitted=[]
for f in range(5):
    trm=sf[O[None]]!=f
    m=M(**RP); m.fit(readframe(None,trm), sy[O[None][trm]]); fitted.append(m)
    print(f"read model fold {f} trained",flush=True)

for d in DEPTHS:
    probs=np.zeros(len(R[d]))
    for f in range(5):
        te=sf[O[d]]==f
        probs[te]=fitted[f].predict_proba(readframe(d,te))
    for nm,fn in POOLS.items():
        s=pool(probs,O[d],len(sy),fn)
        res[(nm,d)]=metrics(sy,s)["pr_auc"]
    print(f"scored depth {d}",flush=True)

tag=lambda d:"full" if d is None else str(d)
QUANT={1:0.1543,3:0.2393,10:0.3732,None:0.4759}   # measured earlier, same folds/seed
POOLED={1:0.1450,3:0.2420,10:0.3630,None:0.4614}
print("\n=== PR AUC: read-level MIL-lite vs site-level summaries ===")
print(f"{'depth':>6} {'MIL mean':>9} {'MIL max':>8} {'MIL q90':>8} {'quantiles':>10} {'pooled':>8} {'motif-only':>11}")
for d in DEPTHS:
    print(f"{tag(d):>6} {res[('mean',d)]:>9.4f} {res[('max',d)]:>8.4f} {res[('q90',d)]:>8.4f} "
          f"{QUANT[d]:>10.4f} {POOLED[d]:>8.4f} {0.1537:>11.4f}")
