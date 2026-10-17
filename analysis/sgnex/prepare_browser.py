"""Run the unchanged saved model on processed SG-NEx RNA JSON and export sites.

Score the complete input BEFORE selecting transcripts: graph neighbours and
within-file ranks depend on the input's composition. This can require substantial
RAM for full SG-NEx files. Reuse existing prediction CSVs with export_browser_data
instead when available. Publish only results from public SG-NEx inputs.
"""
import argparse
from pathlib import Path
from io import StringIO

from export_browser_data import ROOT, export

def prepare(source, cell_line, transcripts, model_path, output):
    # Uses existing preprocessing, network forward passes and score aggregation.
    from score_sgnex import read, score, SiteGraphEnsemble
    model = SiteGraphEnsemble.load(model_path)
    frame, values, counts = read(source, limit=None)
    predictions = score(model, frame, values, counts)
    wanted = {t.split('.')[0] for t in transcripts}
    selected = predictions[predictions.transcript_id.isin(wanted)]
    return export(StringIO(selected.to_csv(index=False)), cell_line, transcripts, output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--cell-line', required=True)
    p.add_argument('--transcript', action='append', required=True)
    p.add_argument('--model', type=Path, default=ROOT / 'models/final')
    p.add_argument('--output', type=Path, default=ROOT / 'dashboard/public/data/browser')
    a = p.parse_args()
    print(f'Exported {prepare(a.input, a.cell_line, a.transcript, a.model, a.output)} sites')
