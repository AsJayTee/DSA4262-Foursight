# 0039. The dashboard is rebuilt around the site shortlist use case

- **Date:** 2026-10-21
- **Status:** Accepted (pages: Home and How it works built; Explore to follow)
- **Affects:** `dashboard/` (all of `src/`, `public/data/`, `index.html`, `package.json`),
  `analysis/sgnex/export_model_story.py` (new). Supersedes
  [0037](0037-use-supplied-dashboard-reference.md) and
  [0038](0038-connect-sgnex-preview-subsets.md); keeps
  [0035](0035-dashboard-is-a-static-react-site-on-github-pages.md) (static React on GitHub Pages).
- **New dependency:** `three` (dashboard only, loaded only by the Home page). GSAP is
  planned for later, added incrementally.

## Context

The team adopted one use case for Task 2 ([report/task2_plan.md](../../report/task2_plan.md)):
a researcher choosing which few m6A sites of their gene to test. The interface on `main`
was a supplied standalone mock-up mounted as a compiled bundle (0037), with one tab on
the first 1,000 sites of each cell line (0038). It read as generic, was hard to navigate,
could not be edited as source, and could not serve the use case: none of MYC, TP53, KRAS,
GAPDH or ACTB was in its data.

## Decision

1. **Three pages, one per reader**, as editable React components with hash routes
   (`#/`, `#/explore`, `#/how-it-works`):
   - **Home**, for a layperson: what m6A is, what we did, one headline result, the limits.
     Its picture is an illustrative three.js RNA strand, not data.
   - **How it works**, for scientists and data science readers: the model as a machine-learning
     problem on MYC (MCF7) - the RNA as text and each site as a table of reads; the read network
     drawn as circles and lines with its real activations; the site graph passing messages -
     with each interactive state precomputed by
     `analysis/sgnex/export_model_story.py` from one public SG-NEx sample and the shipped
     model ("option A"; a live in-browser model is possible later, the weights are 1 MB).
   - **Explore**, the scientist's site shortlist (plan section 3): next, on the full export.
2. **Example genes are chosen by name, never by the course labels.** The earlier showcase
   reads were selected as "transcripts with a modified site", which would leak labels.
3. **Plain SVG charts and our own CSS tokens**, no UI kit or chart library, so the site
   does not look templated. One black-and-white look with an accent for "modified", in light
   and dark (system setting, overridable in the top bar). A serif "journal" look was tried
   and dropped by the team.
4. **Removed:** the compiled mock-up, its preview data and the scripts that made them,
   `summary.json` and `export_dashboard_data.py` (findings A-E were dropped by the plan).
   Josh's Row browser (`TranscriptBrowser.jsx`) was adapted into Explore's gene map and
   then deleted; `export_browser_data.py` and `prepare_browser.py` are kept.

## Why this and not the alternatives

- **Keep the mock-up and connect real data:** its source was compiled, so every change
  meant patching minified code, and its layout was built for a different product.
- **Live model in the browser now:** a JavaScript port of the numpy forward pass must be
  tested against Python; precomputed states give the same lesson today with no risk.
- **A chart library:** faster for standard charts, but the charts here are few and
  specific, and a library's defaults are much of what reads as generic.

## Cost

- **three.js adds about 120 KB (gzipped)**, loaded only on Home. The strand pauses off
  screen and in hidden tabs, and holds still for reduced-motion readers.
- **How it works needs one 1.6 GB public sample to regenerate** (the script's docstring
  has the download command); its output is 12 KB.
- **Phone layouts are unverified below about 500 px wide**: headless Chrome cannot render
  narrower. Check in a browser's device mode.
