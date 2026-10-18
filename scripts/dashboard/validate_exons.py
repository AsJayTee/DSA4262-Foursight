#!/usr/bin/env python3
"""Check every processed transcript exon against the original Ensembl GTF."""
import argparse, gzip, json, re
from pathlib import Path
from collections import defaultdict
ATTR = re.compile(r'(\w+)\s+"([^"]*)"')
p = argparse.ArgumentParser()
p.add_argument('--gtf', type=Path, required=True)
p.add_argument('--processed', type=Path, required=True)
a = p.parse_args()
expected = {}
for path in sorted(a.processed.glob('*.json')):
    doc = json.loads(path.read_text(encoding='utf8'))
    for tx in doc.get('transcripts', []):
        tid = tx['transcript_id'].split('.')[0]
        if tid in expected and expected[tid]['exons'] != tx['exons']:
            raise SystemExit(f'Conflicting annotations for {tid}')
        expected[tid] = tx
if not expected:
    raise SystemExit('No transcripts found in processed JSON')
exons, meta = defaultdict(list), {}
with gzip.open(a.gtf, 'rt', encoding='utf8') as f:
    for line in f:
        if line.startswith('#'): continue
        c = line.rstrip('\n').split('\t')
        if len(c) != 9 or c[2] != 'exon': continue
        attr = dict(ATTR.findall(c[8]))
        tid = attr.get('transcript_id', '').split('.')[0]
        if tid not in expected: continue
        exons[tid].append([int(c[3]), int(c[4])])
        meta[tid] = (c[0].removeprefix('chr'), c[6], attr.get('gene_id'))
failed = []
for tid, tx in expected.items():
    if sorted(exons.get(tid, [])) != sorted(tx['exons']) or meta.get(tid) != (tx['chr'].removeprefix('chr'), tx['strand'], tx['gene_id']):
        failed.append(tid)
print(f'Transcripts checked: {len(expected)}')
print(f'Exact matches: {len(expected)-len(failed)}')
print(f'Mismatches: {len(failed)}')
if failed:
    print('Examples:', ', '.join(failed[:10]))
    raise SystemExit(1)
print('PASS: processed exons, chromosome, strand and gene IDs match GTF')
