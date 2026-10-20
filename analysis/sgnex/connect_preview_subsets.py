"""Export the seven public SG-NEx preview subsets as dashboard JSON."""
import argparse
import csv
import json
import math
import re
from pathlib import Path


def export(source: Path, target: Path) -> dict:
    annotation = {}
    for row in (source / 'matching_transcripts.gtf').read_text().splitlines():
        if row.startswith('#'):
            continue
        fields = row.split('\t')
        attrs = dict(re.findall(r'(\w+) "([^"]+)"', fields[8]))
        tid = attrs.get('transcript_id')
        if not tid:
            continue
        tx = annotation.setdefault(tid, {'gene': attrs.get('gene_name', attrs.get('gene_id', tid)),
                                       'gene_id': attrs.get('gene_id'), 'chromosome': fields[0],
                                       'strand': fields[6], 'exons': []})
        if fields[2] == 'exon':
            tx['exons'].append([int(fields[3]), int(fields[4])])
    target.mkdir(parents=True, exist_ok=True)
    index = {'selection': 'First 1,000 sites per cell line, not a representative sample',
             'coordinates': 'Original transcript positions; genomic conversion not performed',
             'annotation': 'Ensembl 91 / GRCh38.p10', 'lines': []}
    for file in sorted(source.glob('*_subset.csv')):
        line = file.stem.removesuffix('_subset')
        groups = {}
        with file.open() as handle:
            for row in csv.DictReader(handle):
                tid = row['transcript_id']
                ann = annotation.get(tid, {'gene': tid, 'exons': []})
                tx = groups.setdefault(tid, {'id': tid, **ann, 'sites': []})
                site = {'position': int(row['transcript_position']), 'kmer': row['kmer'],
                        'reads': int(row['n_reads']), 'raw': float(row['score_raw']),
                        'rank': float(row['score_rank'])}
                if not all(math.isfinite(site[k]) for k in ['raw', 'rank']) or not 0 <= site['rank'] <= 1:
                    raise ValueError(f'Invalid prediction: {line}/{tid}')
                tx['sites'].append(site)
        for tx in groups.values():
            tx['sites'].sort(key=lambda s: s['position'])
        payload = {'cell_line': line, 'transcripts': sorted(groups.values(), key=lambda t: (t['gene'], t['id']))}
        (target / f'{line}.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
        index['lines'].append({'name': line, 'path': f'{line}.json', 'sites': sum(len(t['sites']) for t in groups.values()), 'transcripts': len(groups)})
    (target / 'index.json').write_text(json.dumps(index, indent=2))
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
