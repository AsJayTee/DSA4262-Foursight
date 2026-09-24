# 0026. A claim about one stratum must be tested with the stratum's own vectors

- **Date:** 2026-09-24
- **Status:** Accepted. Amends [0017](0017-split-the-top-depth-band.md)'s
  "What it immediately showed" section, which stands as a measurement and not as
  a conclusion.
- **Affects:** how `band/*` and `motif/*` numbers may be quoted; [GAPS.md](../../GAPS.md); [docs/wandb-panels.md](../wandb-panels.md) panel 8

## Context

[0017](0017-split-the-top-depth-band.md) split the top read-depth band at 600
and read the result off the pooled per-band lifts:

| band | 84-303 | 304-599 | 600+ |
|---|---:|---:|---:|
| `lightgbm_quantiles` lift | 10.82x | **11.07x** | **8.84x** |

and concluded: *"304-599 is not depressed at all... The entire drop lives above
600. So this is a **cliff above 600, not a slide from 304**."* That conclusion
propagated into GAPS.md as an unexplained anomaly needing a biological
explanation, and into `docs/wandb-panels.md` panel 8.

**Each of those lifts is a single pooled point estimate with no uncertainty
attached.** At the time that was the only thing available.

It is not any more.
[0018](0018-per-stratum-vectors-are-logged-for-later.md) has logged one metric
per (repetition, fold) per stratum on every standard run since 2026-09-21 -
roughly 50 observations per band, sitting in `strata_observations` in every
report. Nobody had used them for this.

## The measurement

Two bands scored by the **same model on the same fold** pair exactly, so the
question "is this model worse on 600+ sites than on 84-303 sites" is a paired
comparison like any other. On `lightgbm_quantiles`:

| | mean difference | wins | naive p | corrected p |
|---|---:|---:|---:|---:|
| 600+ vs 84-303, **lift** | -0.97 | 17/49 | 0.0899 | 0.6366 |
| 600+ vs 304-599, **lift** | -1.03 | 20/47 | 0.1489 | 0.6829 |
| 600+ vs 84-303, PR AUC | -0.0942 | 13/49 | 0.0000 | 0.1550 |

**On lift - the metric 0017 and GAPS.md quote - the cliff is not significant,
and it is not significant on the naive test either.** The specific contrast
0017's conclusion rests on, 600+ against 304-599, is corrected p = 0.68.

The reason is scatter. Standard deviation of PR AUC across the 50 observations:

| 20-31 | 32-46 | 47-83 | 84-303 | 304-599 | 600+ |
|---:|---:|---:|---:|---:|---:|
| 0.0305 | 0.0336 | 0.0371 | 0.0447 | **0.1172** | **0.1277** |

The top two bands carry three to four times the scatter of every other band,
and 600+ ranges from 0.1390 to 0.6391 across its observations. **The 0.094 gap
the cliff is built on is smaller than one standard deviation of the band's own
observations.** 600+ is 2,837 sites and 129 positives spread over only 192
transcripts, and positives cluster by transcript at 4.68x the binomial variance,
so its effective sample is far smaller than its site count.

## Decision

**A claim about a stratum is made with that stratum's `strata_observations`
vectors and a paired test, never from the pooled per-band value alone.** The
pooled value is for reading a table; it carries no uncertainty and at the thin
end of the depth range it carries a great deal of noise.

Concretely:

- `band/*` and `motif/*` scalars remain what they are and keep their meaning.
  They are descriptive.
- Any sentence of the form "the model is worse on X than on Y" needs the paired
  test over the ~50 observations, and it inherits the multiplicity caveat
  [0008](0008-comparisons-report-both-arms-and-strata.md) section 3 already
  attaches to per-stratum p-values.
- The two thin top bands, 304-599 and 600+, should carry their scatter whenever
  they are quoted.

## Why this and not the alternatives

**Leave 0017 alone and treat the cliff as open.** What was happening, and it was
actively costing work: GAPS.md described 2,837 sites as needing a biological
explanation and named hypotheses (a distinct expression regime, m6ACE antibody
background) for an effect that has not been shown to exist.

**Merge the top bands back together.** Rejected: more resolution is not the
problem, and 0017's *decision* - the 600 edge - is fine. Merging would also
re-hide whatever is or is not there.

**Treat "present in all three models" as corroboration.** It looked like the
strongest argument for the cliff being real and it is not an argument at all.
All three models are scored on the **same sites in the same folds**, so a
band-level artefact driven by which sites fall in the band appears in all three
by construction. Agreement across models rules out a model quirk; it says
nothing about sampling noise in the band.

**Widen the bands so every one is well-powered.** Tempting, and it throws away
the resolution that made the question askable. The honest fix is to keep the
bands and report their uncertainty.

## Consequences

- **This does not show the effect is absent.** Both point estimates are negative
  and both win counts are below half (17/49, 20/47). Something may be there; 50
  correlated observations on 129 positives cannot resolve it. "Unresolved" is
  the claim, not "disproved".
- One methodological caveat, stated because it is load-bearing:
  `paired_comparison` was built to compare two *models* on the same folds, and
  this uses it across two *strata* of one model. The pairing is sound - same
  model, same fold - but the Nadeau & Bengio correction is sized for train/test
  overlap between models and may be mis-scaled here. The conclusion does not
  depend on it: the naive test also declines to call the cliff on lift.
- Every other per-band statement in GAPS.md inherits this standard. The
  lower-depth bands are far better powered (sd 0.03-0.04), so statements about
  them are much safer than statements about the top two.
- 0017's split and its measurements stand. Only the inference from them is
  withdrawn.

## How to check it still holds

The vectors are in any standard run's report JSON:

```python
import json, pandas as pd
from m6a.compare import paired_comparison
d = pd.DataFrame(json.load(open("analysis/evaluation/reports/lightgbm_quantiles.json"))["strata_observations"])
dep = d[d["kind"] == "depth"]
```

Pair two bands on `(repetition, fold)` and pass them through
`paired_comparison`. `600+` against `84-303` on lift must come back above 0.05
on both the naive and corrected p-values, and the per-band standard deviations
must show the top two bands at three to four times the rest.
