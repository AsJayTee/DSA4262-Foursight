# 0037 — Use the supplied dashboard as the visual baseline

Date: 2026-10-20

Status: accepted on dashboard-josh; superseded by 0039 (2026-10-21)

Josh requested an exact match to the supplied standalone HTML before providing
data. Extract its embedded module and stylesheet without modifying their content,
and mount them through the existing Vite entry point. Preserve the old Row/Data
components for later reuse. Do not change prediction code, models or weights.

This gives the exact supplied interface rather than an approximate redesign.
Its synthetic and placeholder labels stay visible. Existing public aggregate
summaries remain available. Real site-level data integration is deferred until
the supplied dataset and its schema can be inspected.

Cost: the reference is compiled JavaScript with a bundled React runtime, so it
is less maintainable than source components. Recover component sources or
refactor when changing the interface further. Only the supplied entry runtime
is mounted. The old Row browser export schema is not assumed compatible with
the new reference's importer.

Validation: production Vite build succeeds. Extracted JS and CSS are identical
to their attachment blocks. No visual browser verification was performed.
