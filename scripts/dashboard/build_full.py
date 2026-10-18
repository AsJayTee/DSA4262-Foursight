#!/usr/bin/env python3
"""Stream seven SG-NEx pooled predictions into gene-specific JSON using 64 disk buckets.
Standard library only. Generated files should remain Git-ignored.
"""
import argparse, csv, gzip, json, re, tempfile, hashlib
from collections import defaultdict, Counter
from pathlib import Path
from contextlib import ExitStack
ATTR = re.compile(r'(\w+)\s+"([^"]*)"')
LINES = ['A549','H9','Hct116','HepG2','HEYA8','K562','MCF7']
MOTIFS = {a+b+'AC'+c for a in 'AGT' for b in 'AG' for c in 'ACT'}

def load_gtf(path):
    tx, genes = {}, {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'): continue
            c = line.rstrip('\n').split('\t')
            if len(c)!=9 or c[2] not in ('transcript','exon'): continue
            a = dict(ATTR.findall(c[8])); tid=a.get('transcript_id','').split('.')[0]
            gid=a.get('gene_id','').split('.')[0]
            if not tid or not gid: continue
            name=a.get('gene_name') or gid
            g=genes.setdefault(gid,dict(gene_id=gid,symbol=name,transcripts=[]))
            t=tx.get(tid)
            if t is None:
                t=dict(transcript_id=tid, transcript_name=a.get('transcript_name',tid),gene_id=gid,gene_name=name,
                       chr=c[0] if c[0].startswith('chr') else 'chr'+c[0],strand=c[6],exons=[])
                tx[tid]=t; g['transcripts'].append(tid)
            if c[2]=='exon': t['exons'].append([int(c[3]),int(c[4])])
    for t in tx.values(): t['exons'].sort(key=lambda x:x[0],reverse=t['strand']=='-')
    return tx,genes

def genomic(t,pos):
    offset=pos
    for start,end in t['exons']:
        n=end-start+1
        if offset<n: return start+offset if t['strand']=='+' else end-offset
        offset-=n
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--buckets',type=int,default=64)
    p.add_argument('--lines',nargs='+',choices=LINES,default=LINES)
    args=p.parse_args()
    if args.buckets<1 or args.buckets>256: p.error('buckets must be between 1 and 256')
    gtf=args.input_dir/'annotation/gtf_91.gtf.gz'
    if not gtf.is_file(): p.error(f'GTF not found: {gtf}')
    tx,genes=load_gtf(gtf)
    print(f'GTF: {len(tx):,} transcripts, {len(genes):,} genes',flush=True)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    # A new output directory is strongly preferred: old generated files aren't deleted.
    stats=Counter(); counts=Counter(); per_line=Counter()
    with tempfile.TemporaryDirectory(prefix='foursight_buckets_') as td:
        with ExitStack() as stack:
            files=[stack.enter_context(open(Path(td)/f'{i:03d}.jsonl','w',encoding='utf-8')) for i in range(args.buckets)]
            for line in args.lines:
                path=args.input_dir/'sgnex_scores'/f'{line}_pooled.csv.gz'
                if not path.is_file(): p.error(f'Prediction file not found: {path}')
                with gzip.open(path,'rt',newline='',encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        stats['rows_seen']+=1
                        tid=row['transcript_id'].split('.')[0]; t=tx.get(tid)
                        if not t or not t['exons']:
                            stats['unannotated']+=1; continue
                        try:
                            tp=int(row['transcript_position']); pos=genomic(t,tp)
                            k=row['kmer'].upper(); motif=k[1:6]
                            if pos is None or tp<0 or len(k)!=7 or motif not in MOTIFS:
                                stats['invalid_or_outside']+=1;continue
                            gid=t['gene_id']; key=f'{line}:{tid}:{tp}'
                            record=dict(id=key,gene=t['gene_name'],chr=t['chr'],position=pos,strand=t['strand'],transcript=tid,
                                        transcriptPosition=tp,kmer=k,motif=motif,reads=int(row['n_reads']),
                                        raw=float(row['score_raw']),rank=float(row['score_rank']),line=line,status='evaluated')
                            bucket=int(hashlib.md5(gid.encode()).hexdigest()[:8],16)%args.buckets
                            files[bucket].write(json.dumps([gid,record],separators=(',',':'))+'\n')
                            counts[gid]+=1; per_line[line]+=1;stats['written']+=1
                        except (KeyError,ValueError,TypeError): stats['invalid_or_outside']+=1
                print(f'{line}: {per_line[line]:,} usable sites',flush=True)
        index=[]; total_bytes=0
        for i in range(args.buckets):
            by_gene=defaultdict(list)
            with open(Path(td)/f'{i:03d}.jsonl',encoding='utf-8') as f:
                for line in f:
                    gid,record=json.loads(line);by_gene[gid].append(record)
            for gid,sites in by_gene.items():
                g=genes[gid]; ts=[tx[tid] for tid in g['transcripts'] if tx[tid]['exons']]
                if not ts: continue
                doc=dict(schema_version=1,provenance=dict(kind='processed-predictions',source='team SG-NEx pooled predictions',
                    annotation='Ensembl 91',coordinate_mapping='0-based transcript positions; GTF exons 1-based inclusive'),
                    gene=g['symbol'],gene_id=gid,transcripts=ts,sites=sites)
                relative=f'genes/{gid}.json'; dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(json.dumps(doc,separators=(',',':')),encoding='utf-8')
                total_bytes+=dest.stat().st_size
                index.append(dict(gene_id=gid,symbol=g['symbol'],transcripts=[t['transcript_id'] for t in ts],
                                  sites=len(sites),file=relative,chr=ts[0]['chr']))
            print(f'Bucket {i+1}/{args.buckets}: {len(by_gene)} genes',flush=True)
        index.sort(key=lambda g:(g['symbol'].upper(),g['gene_id']))
        (out/'gene_index.json').write_text(json.dumps(dict(schema_version=1,genes=index),separators=(',',':')),encoding='utf-8')
        report=dict(schema_version=1,lines=args.lines,counts=dict(stats),per_line=dict(per_line),
                    indexed_genes=len(index),gene_file_bytes=total_bytes,notes=[
                    'Predictions are model scores, not confirmed m6A sites or calibrated probabilities.',
                    'Unmatched GTF transcripts are counted but not included in gene-specific exports.',
                    'These are full records, not read coverage; large files may affect browser performance.',
                    'Files in Vite public/ are deployed even when Git-ignored.'])
        (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
if __name__=='__main__': main()
