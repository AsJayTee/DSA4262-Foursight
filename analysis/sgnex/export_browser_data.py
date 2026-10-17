"""Export selected transcripts from public SG-NEx prediction CSV(.gz) for Row.

Requires --cell-line and one or more --transcript identifiers. No reads or
labels are exported. Only use predictions derived from public SG-NEx data.
"""
import argparse
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def export(source, cell_line, transcripts, output):
    wanted = {t.split('.')[0] for t in transcripts}
    groups = {t: [] for t in wanted}
    seen = set()
    opener = gzip.open if str(source).endswith('.gz') else open
    with (nullcontext(source) if hasattr(source, 'read') else opener(source, 'rt', encoding='utf-8-sig', newline='')) as handle:
        reader = csv.DictReader(handle)
        required = {'transcript_id', 'transcript_position', 'kmer', 'n_reads'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {required - set(reader.fieldnames or [])}')
        for row in reader:
            tid = row['transcript_id'].split('.')[0]
            if tid not in wanted:
                continue
            site = {k: int(row[k]) for k in ('transcript_position', 'n_reads')}
            site['kmer'] = row['kmer'].upper()
            if site['transcript_position'] < 0 or site['n_reads'] < 1 or not re.fullmatch(r'[ACGTU][AGTU][AG]AC[ACTU][ACGTU]', site['kmer']):
                raise ValueError(f'Invalid site: {tid}:{site["transcript_position"]}')
            key = (tid, site['transcript_position'])
            if key in seen:
                raise ValueError(f'Duplicate site: {key}; export one sample or pooled line at a time')
            seen.add(key)
            for field in ('score_rank', 'score_raw'):
                value = row.get(field, '')
                site[field] = float(value) if value else None
                if site[field] is not None and not math.isfinite(site[field]):
                    raise ValueError(f'Non-finite {field}')
            if site['score_rank'] is not None and not 0 <= site['score_rank'] <= 1:
                raise ValueError('Rank outside [0,1]')
            groups[tid].append(site)
    missing = [t for t, sites in groups.items() if not sites]
    if missing:
        raise ValueError(f'Transcripts not found: {missing}')
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'transcripts': []}
    index['transcripts'] = [t for t in index['transcripts'] if not (t['cell_line'] == cell_line and t['transcript_id'] in wanted)]
    for tid, sites in sorted(groups.items()):
        filename = hashlib.sha256(f'{cell_line}:{tid}'.encode()).hexdigest()[:24] + '.json'
        payload = {'cell_line': cell_line, 'transcript_id': tid, 'sites': sorted(sites, key=lambda s: s['transcript_position'])}
        (output / filename).write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
        index['transcripts'].append({'cell_line': cell_line, 'transcript_id': tid, 'path': filename})
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')
    return sum(map(len, groups.values()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--cell-line', required=True)
    parser.add_argument('--transcript', action='append', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'dashboard/public/data/browser')
    args = parser.parse_args()
    print(f'Exported {export(args.input, args.cell_line, args.transcript, args.output)} sites')
