# Foursight m6A (Task 2 dashboard)

A static React site (Vite), published to GitHub Pages from this repository
([decision 0035](../docs/decisions/0035-dashboard-is-a-static-react-site-on-github-pages.md)).
It serves the use case agreed in [report/task2_plan.md](../report/task2_plan.md): helping a
researcher choose which m6A sites of their gene to test first. Why it is built this way:
[decision 0039](../docs/decisions/0039-dashboard-rebuilt-around-the-site-shortlist.md).

| Page | Route | For | Status |
|---|---|---|---|
| Home | `#/` | a layperson | built |
| How it works | `#/how-it-works` | scientists, data science readers | built |
| Explore sites | `#/explore` | a scientist with a gene in mind | built (needs the site data, below) |
| Model stats | `#/model-stats` | scientists judging how far to trust a rank | built |

## Run it locally

Needs Node.js 18 or newer.

```bash
cd dashboard
npm install
npm run dev        # http://localhost:5173/DSA4262-Foursight/
```

`npm run build` writes the static site to `dashboard/dist/` (gitignored); `npm run preview`
serves that build. Light and dark mode follow the system setting; the button in the top bar
overrides it.

The Explore page reads the full site export. Locally, `npm run dev` and `npm run preview` serve
it straight from `data/dashboard/` (`vite.config.js`); make it first if you have not:

```bash
python analysis/sgnex/export_site_shortlist.py   # sites, ranks, batches (needs data/sgnex_scores/)
python analysis/sgnex/export_gene_structure.py   # exons, for the gene map
```

`SITE_DATA=<folder> npm run dev` reads it from elsewhere.

## Publish

Pushing changes under `dashboard/` to `main` runs `.github/workflows/dashboard.yml`, which
builds the site and deploys it to <https://asjaytee.github.io/DSA4262-Foursight/>.

### Publish the site data

The full site export (~95 MB) is not committed (plan section 5.3). It is attached to a GitHub
Release named `site-data`, which the workflow downloads into `public/data/sites/` before
building. Publishing or replacing it makes the file public, so do it deliberately:

```bash
tar -cf site-data.tar -C data/dashboard genes.json.gz meta.json trust.json shards structure
gh release create site-data site-data.tar --title "Site data for the dashboard" \
   --notes "Model outputs on public SG-NEx data; no course labels or reads."
# replacing it later:  gh release upload site-data site-data.tar --clobber
```

A new release does not trigger a deploy by itself: re-run the workflow from the Actions tab
(`workflow_dispatch`). Without the release the site still builds; Explore says its data is
missing.

## Data

The site reads `public/data/` (committed, small) and `public/data/sites/` (the release, above).
Every file is written by a script:

| File | Written by | Contents |
|---|---|---|
| `model_story.json` | `analysis/sgnex/export_model_story.py` | MYC (MCF7): a stretch of the RNA with its sites' read tables, one network's real activations read by read, each site's rank alone and with neighbours, the six networks, and the held-out performance numbers |
| `model_stats.json` | `analysis/sgnex/export_model_stats.py` | held-out performance against baselines and m6Anet, depth curve, top-k recovery, gene-level calibration, reliability by motif and depth, enrichment by distance to the stop codon and exon junctions (labelled sets and the seven SG-NEx lines) |
| `explore.json` | `analysis/sgnex/export_explore_meta.py` | held-out numbers Explore quotes: how often a gene's top k held a confirmed site, by how high its best site ranked; trust-label cut-offs |
| `sites/genes.json.gz` | `analysis/sgnex/export_site_shortlist.py` | gene search index: name, location, sites per cell line |
| `sites/shards/NNN.json.gz` | same | per gene: its sites (position, letters, region, transcripts) and each cell line's rank, molecules and per-batch ranks |
| `sites/trust.json`, `sites/meta.json` | same | trust values by motif and depth, each motif's modification rate; cell lines and batches |
| `sites/structure/NNN.json.gz` | `analysis/sgnex/export_gene_structure.py` | per gene: merged exons and coding span (Ensembl 91), for the gene map |

**Never put on the public site** the course's labels or read files (AGENTS.md section 5).
Model outputs and reads from public SG-NEx data are fine. Example genes are chosen by
name, never by the labels.

## Layout

```
dashboard/
  index.html               page shell, fonts
  src/App.jsx              top bar (with the light/dark toggle), routes, footer
  src/pages/               Home, HowItWorks, Explore, ModelStats
  src/components/          RnaStrand (three.js), ModelViz (text/tables, network, graph),
                           ExploreParts (gene viewer, site tables, site panel), StatsCharts
                           (Model stats charts), StoryCharts (earlier charts, not mounted)
  src/lib/                 route.jsx (hash routes), data.js (loading, rank formatting),
                           sites.js (site data: unpacking, rows per gene, search, CSV)
  src/styles/              base.css (light/dark tokens), home.css, model.css, explore.css, stats.css
  src/TranscriptBrowser.jsx  Josh's Row browser (not mounted; its track idea became the gene map)
  public/data/             data files the site fetches (exported by scripts)
```
