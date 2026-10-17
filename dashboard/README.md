# m6A Explorer (Task 2 dashboard)

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
