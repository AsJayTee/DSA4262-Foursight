# 0013. Every run produces a distribution, so repeated CV is the default

- **Date:** 2026-09-19
- **Status:** Accepted
- **Affects:** `src/m6a/report.py` (`Profile.repeats`), `scripts/train.py`, `scripts/evaluate.py`. Amends [0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md) section 5.

## Context

[0012](0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md) shipped `--repeats`
and made it **opt-in, default 1**, on the grounds that ten repetitions multiply
fit time by ten and that is the wrong default for a quick check.

Using it for a week showed that reasoning to be backwards for this project, for
two reasons that only appear once results are in W&B:

1. **A five-point run cannot be topped up later.** Ronin instances are created,
   used and terminated. If a run produced five observations and you later want
   fifty, you re-download the data, re-extract the features and refit
   everything — on a new instance. The cheap moment to gather evidence is the
   moment the machine exists, which is the whole argument of
   [0007](0007-evaluating-without-a-baseline.md).
2. **Five-point and fifty-point runs are not comparable.** `rep/pr_auc_sd` over
   5 observations and over 50 are different quantities, and the distribution
   figure — the primary figure for any run
   ([0009](0009-distributions-not-per-site-scores.md) section 3) — cannot be
   read side by side. Profiles exist to make runs comparable *by construction*,
   and an opt-in that changes the shape of the headline figure defeats that.

The measured consequence: `quantiles_v1` vs `pooled_v1` is unresolvable on five
folds (corrected p = 0.1305) and comfortably resolved on fifty (p = 0.000095).
Whether a run can answer its own question was, until now, a flag someone had to
remember.

## Decision

`repeats` becomes a field of `Profile`:

| profile | repeats | observations |
|---|---:|---:|
| `quick` | 1 | 5 |
| `standard` | 10 | 50 |
| `full` | 10 | 50 |

`--repeats N` still overrides either way. `train.py` therefore runs
`repeated_cross_validate` rather than `cross_validate`, and **repetition 0 is
still the canonical seed-4262 split**, so the shipped model's headline metrics
are exactly what a single-split run would have reported.

### Why ten and not thirty

The corrected variance is `var(d) x (1/n + 1/(k-1))`. At k = 5 folds the second
term is 0.25 and never goes away:

| observations | variance factor | SE relative to n=50 |
|---:|---:|---:|
| 5 | 0.450 | 1.29x |
| 50 | 0.270 | 1.00x |
| 100 | 0.260 | 0.98x |

Going from 50 to 100 observations doubles the compute and narrows the standard
error by **1.9%**. Ten repetitions is where the curve flattens; past it you are
buying a prettier violin and no statistics.

The term that would actually help is `1/(k-1)` — ten folds would halve it. That
is not available: `n_folds` is part of the frozen split (AGENTS.md section 3),
and changing it would invalidate every recorded `fold/*` key in W&B.

## Why this and not the alternatives

**Leave it opt-in and always type `--repeats 10`.** What 0012 decided. It
relies on everyone remembering, on every run, forever — and the failure is
silent until someone tries to compare two runs and finds one of them has five
points.

**A fourth profile between `standard` and `full`.** Rejected: profiles are a
ladder of thoroughness, and "standard but with a real distribution" is not a
rung, it is what standard should have meant.

**Raise it to 20 for a smoother figure.** See the table — 2% on the standard
error for double the VM time.

## Consequences

- **`standard` now costs roughly ten times the fits.** Measured on the full
  training set, warm cache: `baseline.yaml` ~9s → ~1.5 min, `lightgbm.yaml`
  ~18s → ~2 min, `quantiles.yaml` ~100s → ~11 min. The whole three-config
  battery is about 15 minutes against about 2. Feature extraction is unaffected:
  the split does not enter it (0012 section 4).
- `--quick` is now the meaningfully cheap path, and is the one to use while
  iterating. It was always documented as "not a recorded result"; that is now
  also true of its distribution.
- The depth sweep still scores with repetition 0's models only, so its cost is
  unchanged.
- Runs made before this shipped have five observations and will read as a
  narrow, sparse distribution beside newer ones. They are not wrong, they are
  thinner — `rep/n_observations` says which is which, and
  `compare/*/n_observations` says what a comparison rested on.

## How to check it still holds

```bash
python scripts/train.py --config configs/lightgbm.yaml --smoke
```

prints `profile standard, 10 repetitions x 5 folds = 50 observations` and logs
`rep/n_observations = 50`. With `--quick` it prints no repetition line and logs
no `rep/*` keys at all.

`fold/{0..4}/pr_auc` must still be the canonical split's values in both cases —
that is 0012's guarantee and this record does not touch it.
