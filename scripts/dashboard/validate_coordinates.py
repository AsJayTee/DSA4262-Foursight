#!/usr/bin/env python3
"""Diagnose transcript_position and 7-mer conventions using SG-NEx transcriptome FASTA.
Read-only; standard library. No interpretation of model scores.
"""
import argparse, csv, gzip, json, re
from collections import Counter, defaultdict
from pathlib import Path

LINES = ('A549','H9','Hct116','HepG2','HEYA8','K562','MCF7')
RC = str.maketrans('ACGTUNacgtun','TGCAANtgcaan')

def open_text(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path, 'rt', encoding='utf-8')

def sequences(path, wanted):
    current = None
    parts = []
    with open_text(path) as f:
        for line in f:
            if line.startswith('>'):
                if current is not None:
                    yield current, ''.join(parts).upper().replace('U','T')
                # Ensembl header: >ENST... version, possibly decorated
                token = line[1:].split()[0].split('|')[0].split('.')[0]
                current = token if token in wanted else None
                parts = []
            elif current is not None:
                parts.append(line.strip())
        if current is not None:
            yield current, ''.join(parts).upper().replace('U','T')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', required=True, type=Path)
    p.add_argument('--fasta', required=True, type=Path)
    p.add_argument('--genes', nargs='+', default=['PTBP1','HSPA5','POLDIP2','METTL3'])
    p.add_argument('--lines', nargs='+', choices=LINES, default=list(LINES))
    p.add_argument('--samples-per-line', type=int, default=250)
    p.add_argument('--output', type=Path, default=Path('coordinate_validation_report.json'))
    a=p.parse_args()
    if a.samples_per_line < 1: p.error('--samples-per-line must be positive')
    # Pull the transcript IDs from the pilot's GTF, not hard-coded IDs.
    attrs=re.compile(r'(\w+)\s+"([^"]*)"')
    wanted={g.upper() for g in a.genes}
    tx=set()
    with gzip.open(a.input_dir/'annotation'/'gtf_91.gtf.gz','rt') as f:
        for line in f:
            if line.startswith('#'): continue
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=9 or fields[2]!='transcript': continue
            d=dict(attrs.findall(fields[8]))
            if d.get('gene_name','').upper() in wanted:
                tx.add(d['transcript_id'].split('.')[0])
    if not tx: p.error('No matching GTF transcripts found')
    # First N valid 7-mer rows per line, deterministic, bounded memory.
    samples=[]
    per_line=Counter()
    for line in a.lines:
        path=a.input_dir/'sgnex_scores'/f'{line}_pooled.csv.gz'
        with gzip.open(path,'rt',newline='') as f:
            for row in csv.DictReader(f):
                tid=row['transcript_id'].split('.')[0]
                k=row['kmer'].upper().replace('U','T')
                if tid not in tx or len(k)!=7 or any(x not in 'ACGT' for x in k): continue
                try: pos=int(row['transcript_position'])
                except ValueError: continue
                samples.append((line,tid,pos,k))
                per_line[line]+=1
                if per_line[line]>=a.samples_per_line: break
    wanted_tx={s[1] for s in samples}
    seqs=dict(sequences(a.fasta,wanted_tx))
    # Shift is relative to the 0-based center interpretation.
    # offset=0 => sequence[pos-3:pos+4]; offset=-1 => 1-based center.
    shifts=range(-5,6)
    matched=Counter(); evaluable=Counter(); missing=Counter()
    examples=[]
    for line,tid,pos,k in samples:
        seq=seqs.get(tid)
        if seq is None:
            missing[line]+=1
            continue
        for shift in shifts:
            center=pos+shift
            start=center-3
            if start<0 or start+7>len(seq): continue
            for orient,expected in [('forward',k),('reverse_complement',k.translate(RC)[::-1])]:
                key=f'{orient}:shift_{shift:+d}'
                evaluable[key]+=1
                if seq[start:start+7]==expected: matched[key]+=1
        if len(examples)<12:
            examples.append(dict(line=line,transcript=tid,position=pos,kmer=k,
                                 fasta_found=True,seq_length=len(seq)))
    ranked=sorted((dict(hypothesis=k,matches=matched[k],evaluable=evaluable[k],
                        rate=round(matched[k]/evaluable[k],5) if evaluable[k] else 0)
                   for k in evaluable),key=lambda x:x['matches'],reverse=True)
    report=dict(fasta=str(a.fasta),samples_collected=len(samples),per_line=dict(per_line),
                fasta_sequences_found=len(seqs),sample_rows_missing_fasta=sum(missing.values()),
                missing_fasta_by_line=dict(missing),hypotheses=ranked,examples=examples,
                interpretation='forward shift +0 means zero-based center; forward shift -1 means one-based center. Check dominant match rate before using mapped coordinates.',
                warning='Verify FASTA provenance/version; low matches may indicate reference or kmer definition mismatch.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Sampled rows:',len(samples),'| FASTA sequences found:',len(seqs))
    print('Top hypotheses:')
    for h in ranked[:8]:
        print(f"  {h['hypothesis']:<30} {h['matches']:>5}/{h['evaluable']:<5} ({h['rate']:.1%})")
    print('Report:',a.output)

if __name__=='__main__': main()
