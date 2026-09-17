"""Are the signal distributions actually different between cell lines, or is
depth the only thing that shifts? Compare training (Hct116) vs SG-NEx A549,
both restricted to depth>=20 so depth is held constant."""
import json, gzip, sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from m6a.data import iter_sites, READ_FEATURE_NAMES

if len(sys.argv) < 2:
    raise SystemExit(
        "usage: python analysis/evaluation/scratch/dist_cmp.py <a549_sample.jsonl>\n"
        "Nothing in this repo produces that file yet - see the Task 2 section of GAPS.md."
    )
SAMP = sys.argv[1]

def collect(it, cap=20000):
    """Per-site mean of each read feature, for sites with >=20 reads."""
    out=[]; motifs=[]; depths=[]
    for tid,pos,kmer,reads in it:
        depths.append(len(reads))
        if len(reads)<20: continue
        a=np.asarray(reads,dtype=np.float32)
        out.append(a.mean(axis=0)); motifs.append(kmer[1:6])
        if len(out)>=cap: break
    return np.array(out), motifs, depths

def a549_iter():
    for line in open(SAMP):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except Exception: continue
        t=next(iter(r)); p=next(iter(r[t])); k=next(iter(r[t][p]))
        yield t,int(p),k,r[t][p][k]

def train_iter():
    for s in iter_sites("data/raw/dataset0.json.gz"):
        yield s.transcript_id, s.position, s.kmer, s.reads

A,ma,da = collect(a549_iter())
T,mt,dt = collect(train_iter())
print(f"A549  sites sampled {len(da):,} | depth>=20 used {len(A):,} | median depth {int(np.median(da))}")
print(f"Train sites sampled {len(dt):,} | depth>=20 used {len(T):,} | median depth {int(np.median(dt))}")
print()
print("PER-SITE MEAN OF EACH READ FEATURE, depth>=20 both sides")
print(f"{'feature':>10} {'Hct116 med':>11} {'A549 med':>10} {'diff':>8} {'A549/Hct':>9}")
for i,n in enumerate(READ_FEATURE_NAMES):
    t_,a_=np.median(T[:,i]),np.median(A[:,i])
    print(f"{n:>10} {t_:>11.4f} {a_:>10.4f} {a_-t_:>+8.4f} {a_/t_:>9.3f}")
print()
# distributional distance per feature, standardised
print("standardised median shift (|A549-Hct116| / Hct116 IQR):")
for i,n in enumerate(READ_FEATURE_NAMES):
    iqr=np.subtract(*np.percentile(T[:,i],[75,25]))
    print(f"  {n:>10} {abs(np.median(A[:,i])-np.median(T[:,i]))/iqr:>6.3f}")
print()
print("MOTIF COMPOSITION (top 8 by A549 share)")
ca=pd.Series(ma).value_counts(normalize=True); ct=pd.Series(mt).value_counts(normalize=True)
cmp=pd.DataFrame({"Hct116_%":(100*ct).round(2),"A549_%":(100*ca).round(2)}).fillna(0)
print(cmp.sort_values("A549_%",ascending=False).head(8).to_string())
