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

The site only reads files in `public/data/`, and every one of them is
written by a script, never by hand:

| File | Written by | Contents |
|---|---|---|
| `summary.json` | `analysis/sgnex/export_dashboard_data.py` | Reads per site per cell line; mean score around the stop codon |

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
  src/App.jsx         the four views (only Overview is wired to data so far)
  src/styles.css      light/dark design tokens
  public/data/        data files the site fetches (exported by scripts)
```
