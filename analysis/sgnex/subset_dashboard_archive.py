"""Create small local preview subsets from the supplied pooled prediction archive.

Takes the first N sites per line, preserving original scores. This is a UI
preview subset, not a representative sample for biological comparisons.
"""
import argparse
import csv
import gzip
import io
import json
import re
import zipfile
from itertools import islice
from pathlib import Path


def prepare(archive: Path, output: Path, count: int = 1000) -> dict:
    if count < 1:
        raise ValueError('Site count must be positive')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'selection': f'First {count} rows of each pooled file; not a representative sample',
                'scores': 'Unchanged from complete pooled input; ranks are NOT recalculated on subsets',
                'source_archive': archive.name, 'cell_lines': []}
    transcripts = set()
    with zipfile.ZipFile(archive) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('_pooled.csv.gz')):
            line = Path(name).name.removesuffix('_pooled.csv.gz')
            with z.open(name) as compressed, gzip.open(compressed, 'rt') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(islice(reader, count))
            tids = {r['transcript_id'].split('.')[0] for r in rows}
            transcripts.update(tids)
            filename = f'{line}_subset.csv'
            with (output / filename).open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest['cell_lines'].append({'cell_line': line, 'file': filename,
                                          'sites': len(rows), 'transcripts': len(tids)})
        annotation = 'data/annotation/gtf_91.gtf.gz'
        genes = set()
        records = []
        headers = []
        with z.open(annotation) as compressed, gzip.open(compressed, 'rt') as stream:
            for row in stream:
                if row.startswith('#'):
                    headers.append(row)
                    continue
                tid = re.search(r'transcript_id "([^"]+)"', row)
                if tid and tid[1].split('.')[0] in transcripts:
                    records.append(row)
                    gene = re.search(r'gene_id "([^"]+)"', row)
                    if gene:
                        genes.add(gene[1])
        with (output / 'matching_transcripts.gtf').open('w', encoding='utf-8') as f:
            f.writelines(headers)
            f.writelines(records)
    manifest['annotation'] = 'matching_transcripts.gtf'
    manifest['annotation_records'] = len(records)
    manifest['genes'] = len(genes)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'README.txt').write_text(
        'SG-NEx local dashboard preview subsets\n\n'
        f'First {count} sites from each of seven pooled prediction files. '
        'Different lines may contain different sites. Transcript boundaries may be cut.\n'
        'Use for UI development, not representative biological comparisons.\n'
        'All prediction columns and values are preserved. Rank scores refer to the '
        'original full pooled files, not these subsets. No model was rerun.\n'
        'matching_transcripts.gtf contains annotation rows for included transcript IDs '
        'from Ensembl 91 / GRCh38.p10. Coordinates have not been converted.\n'
        'These CSV files are preparation inputs, not the reference dashboard JSON import format.\n',
        encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sites', type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.sites), indent=2))
