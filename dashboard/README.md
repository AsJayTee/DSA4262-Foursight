# m6A Explorer (Task 2 dashboard)

## Current interface: supplied reference (20 October 2026)

At Josh's request, the active interface matches the supplied standalone
`index.html` from Telegram Desktop. Its original bundled JavaScript and CSS
are preserved in `src/reference/` and loaded by `src/main.jsx`. This preserves
the four reference tabs, layout, interactions, demo data and source labels.
The earlier Row/Data React components remain available but are not mounted.

This is a visual baseline pending real site data. The supplied interface's
synthetic tracks and placeholders are not real predictions. Its aggregate view
reads the existing `public/data/summary.json`; its local-development data URLs
are not connected to a backend here. Use its synthetic demo for now. Import
formats must be checked against the real files when supplied. No model code or
weights were changed. Google Fonts requires connectivity to match typography.

The original attachment contains compiled code, not editable component sources.
Future substantial changes should recover those sources or split the interface
into maintainable components while preserving the reference appearance.

The Row browser documentation below describes the preserved previous interface.

A static React site (Vite), published to GitHub Pages from this repository
([decision 0035](../docs/decisions/0035-dashboard-is-a-static-react-site-on-github-pages.md)).
The plan for its views is in [report/task2_briefing.md](../report/task2_briefing.md).

## Run it locally

Needs Node.js 18 or newer.

```bash
cd dashboard
npm install
npm run dev        # http://localhost:5173/DSA4262-Foursight/
```

`npm run build` writes the static site to `dashboard/dist/` (gitignored);
`npm run preview` serves that build.

## Publish

Pushing changes under `dashboard/` to `main` runs
`.github/workflows/dashboard.yml`. It builds the site and deploys it to
`https://asjaytee.github.io/DSA4262-Foursight/`.

**One-time setup (repository admin):** Settings → Pages → Build and
deployment → Source: **GitHub Actions**.

## Data

Hosted data comes from `public/data/`, and every file there is
written by a script, never by hand:

| File | Written by | Contents |
|---|---|---|
| `summary.json` | `analysis/sgnex/export_dashboard_data.py` | Reads per site per cell line; mean score around the stop codon |
| `browser/index.json` and transcript JSON files | `analysis/sgnex/export_browser_data.py` | Selected public SG-NEx sites and saved prediction scores |

### Row browser

Row supports cell-line and transcript selection, transcript search, panning,
zooming, selectable DRACH candidate sites, read coverage, and a score inspector.
The collapsible sidebar filters by motif and minimum reads. Drag empty track
space to pan, or hold Ctrl while scrolling to zoom; buttons and arrow keys
provide alternatives. The linked table sorts columns and centres selected
sites. CSV export includes only visible filtered rows and labels synthetic
examples explicitly. Candidate, coverage and rank tracks share one axis.
Coordinates are transcript positions, not genome coordinates; exon annotations
and complete transcript lengths are not available in these exports.

The existing Python model runs offline. `analysis/sgnex/prepare_browser.py`
loads `models/final` through the existing SG-NEx scoring code, scores the complete
processed input, then exports requested transcripts. It does not alter weights,
training, or model implementation. Scoring before filtering preserves the graph
context and within-file ranking. Full files may require substantial RAM.

Both preparation and export accept `--input`, `--cell-line`, repeated
`--transcript` identifiers and optional `--output`. Preparation accepts processed
RNA JSON; export accepts existing prediction CSV or CSV.gz. The default output
is `dashboard/public/data/browser`. Only publish results from public SG-NEx.

A transcript export can also be opened with the Row file picker without being
uploaded. Its fields are `cell_line`, `transcript_id`, and `sites`; each site has
`transcript_position`, seven-base `kmer`, positive `n_reads`, and nullable
`score_rank` and `score_raw`. Rank scores must be in [0, 1]. They are not
calibrated modification probabilities. No raw reads or labels are exported.

Until real per-site exports are supplied, use the explicitly labelled synthetic
example to explore the interface. It is not biological evidence.

To add a view's data, add an export to that script (or a new one) and list
it here.

**Never put on the public site:**
- the course's labels (`data.info.labelled`, data1 or data2 labels);
- the course's read files.

The course data is unpublished (AGENTS.md section 5). Model scores and
summaries from public SG-NEx data are fine.

## Layout

```
dashboard/
  index.html          page shell
  vite.config.js      base path = the repository name (GitHub Pages sub-path)
  src/main.jsx        entry point
  src/App.jsx         Row and Data tabs
  src/TranscriptBrowser.jsx  selectable transcript sites and prediction inspector
  src/styles.css      light/dark design tokens
  public/data/        data files the site fetches (exported by scripts)
```
