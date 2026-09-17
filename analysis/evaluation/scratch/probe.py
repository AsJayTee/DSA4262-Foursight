import gzip, json, sys
import numpy as np, pandas as pd

# 1) first raw line, verbatim structure
with gzip.open("data/raw/dataset0.json.gz","rt") as fh:
    l1 = fh.readline()
print("RAW LINE 1 (first 300 chars):"); print(l1[:300]); print()
rec = json.loads(l1)
t = next(iter(rec)); p = next(iter(rec[t])); k = next(iter(rec[t][p]))
print(f"transcript={t} position={p} kmer={k} n_reads={len(rec[t][p][k])} n_feat={len(rec[t][p][k][0])}")
print("read[0] =", rec[t][p][k][0]); print()

lab = pd.read_csv("data/raw/data.info.labelled")
print("LABELS shape:", lab.shape, "cols:", list(lab.columns))
print(lab.head(3).to_string()); print()
print("positives:", int(lab.label.sum()), f"({100*lab.label.mean():.3f}%)")
print("genes:", lab.gene_id.nunique(), "transcripts:", lab.transcript_id.nunique())
print("dup (tx,pos) keys:", int(lab.duplicated(['transcript_id','transcript_position']).sum()))
print("transcripts in >1 gene:", int((lab.groupby('transcript_id').gene_id.nunique()>1).sum()))
print()
# per-gene positive clustering
g = lab.groupby('gene_id').label.agg(['sum','size'])
print("genes with >=1 positive:", int((g['sum']>0).sum()), "of", len(g))
print("sites per gene: median", g['size'].median(), "max", g['size'].max())
print()
# 2) stream all sites: depth, motif, kmer
depths=[]; motifs=[]; kmers=[]; keys=[]
with gzip.open("data/raw/dataset0.json.gz","rt") as fh:
    for line in fh:
        r = json.loads(line)
        t = next(iter(r)); p = next(iter(r[t])); k = next(iter(r[t][p]))
        depths.append(len(r[t][p][k])); kmers.append(k); motifs.append(k[1:6]); keys.append((t,int(p)))
d = pd.DataFrame({"transcript_id":[a for a,_ in keys],"transcript_position":[b for _,b in keys],
                  "depth":depths,"kmer":kmers,"motif":motifs})
print("SITES in json:", len(d), "| distinct 7mers:", d.kmer.nunique(), "| distinct 5mers:", d.motif.nunique())
print("depth: min",d.depth.min(),"p25",int(d.depth.quantile(.25)),"med",int(d.depth.median()),
      "p75",int(d.depth.quantile(.75)),"p95",int(d.depth.quantile(.95)),"max",d.depth.max(),
      "| total reads", int(d.depth.sum()))
print("json dup keys:", int(d.duplicated(['transcript_id','transcript_position']).sum()))
print()
m = d.merge(lab, on=["transcript_id","transcript_position"], how="outer", indicator=True)
print("join:", m._merge.value_counts().to_dict())
m = m[m._merge=="both"]
print("line-for-line aligned?:", bool((d.transcript_id.values==lab.transcript_id.values).all()
      and (d.transcript_position.values==lab.transcript_position.values).all()))
print()
# depth vs label
print("DEPTH vs LABEL")
print(m.groupby('label').depth.describe()[['count','mean','50%','75%','max']].to_string())
m['depth_bin'] = pd.qcut(m.depth, 5, duplicates='drop')
print(m.groupby('depth_bin', observed=True).label.agg(['mean','size']).to_string())
print()
print("PER-MOTIF BASE RATE (5-mer)")
pm = m.groupby('motif').label.agg(['mean','sum','size']).sort_values('mean', ascending=False)
pm['mean'] = (100*pm['mean']).round(2)
print(pm.to_string())
print()
print("POSITION-IN-TRANSCRIPT vs LABEL")
print(m.groupby('label').transcript_position.describe()[['mean','50%','max']].to_string())
d.to_parquet(sys.argv[1]) if len(sys.argv)>1 else None
