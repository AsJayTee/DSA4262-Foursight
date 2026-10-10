# 0035. The Task 2 dashboard is a static React site on GitHub Pages

- **Date:** 2026-10-10
- **Status:** Accepted (hosting and framework decided by the team; the
  views are still to be agreed)
- **Affects:** `dashboard/` (new: Vite + React), `.github/workflows/dashboard.yml`
  (new), `analysis/sgnex/export_dashboard_data.py` (new), `.gitignore`.
- **New dependency:** Node.js 18+ and npm, **for the dashboard only**. The
  Python package, `predict.py` and the graded path are untouched.

## Context

Task 2 requires "an interactive data visualisation platform for other
researchers to explore the results", hosted anywhere. The team chose GitHub
Pages in this repository, built with React for flexibility.

## Decision

1. **A static site.** Everything is precomputed by Python scripts into
   `dashboard/public/data/` and fetched by the browser. No server, so no
   hosting cost, nothing to keep running, and no AWS spend.
2. **Vite + React**, in `dashboard/`, with its own `package.json`. Kept
   apart from the Python project so neither can break the other.
3. **Deployed by GitHub Actions** on every push to `main` that touches
   `dashboard/`. Pages' source must be set to "GitHub Actions" once, in
   the repository settings.
4. **Data files are always written by scripts**, never edited by hand,
   and never contain the course's labels or reads (AGENTS.md section 5).

## Why this and not the alternatives

- **Streamlit or another server app:** quicker for a Python team, but it
  needs hosting. Free tiers sleep or cap memory, which is a risk for a
  graded link.
- **Plain HTML without a framework:** fewer moving parts, but the planned
  views (gene search, a step-by-step model explainer) need components and
  state.
- **Jupyter/Colab:** least effort, but not a platform for other
  researchers.

## Cost

- **A second toolchain (Node) in the repository.** Mitigated: it lives
  entirely under `dashboard/` and its workflow.
- **Data size.** About 13 million site scores cannot be shipped whole;
  views must load per-gene or pre-filtered files. To be decided with the
  views.
