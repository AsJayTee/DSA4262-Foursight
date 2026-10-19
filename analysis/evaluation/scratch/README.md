# Scratch scripts

Rough, single-purpose scripts with hardcoded paths. They are not part of the
harness and nothing imports them. Each is kept because it is still the only
source of a number quoted in [GAPS.md](../../../GAPS.md) or
[docs/test-data-assumptions.md](../../../docs/test-data-assumptions.md), or
because it is the predecessor of one that is.

Run them from the repo root. The training data must be where each script
expects it, and there are two conventions:

- `data/raw/dataset0.json.gz` and `data/raw/data.info.labelled`: the older
  scripts (`probe`, `motif_only`, `dist_cmp`, `mil_lite`, and the three
  superseded ones);
- `data0/dataset0.json.gz` and `data0/data.info.labelled`: the four screens
  (`density_mismatch`, `density_screen`, `second_moment_screen`,
  `sparse_file`).

## Superseded: use the harness instead

| Script | Replaced by |
|---|---|
| `perfold.py` | `evaluate.py --config ... --ablate` (per-fold, ablations, calibration) |
| `paired.py` | `evaluate.py --config ... --compare-features pooled_v1` |
| `depth_shift.py` | `evaluate.py --config ... --depth-sweep` |

The harness reproduces every number these produced (per-fold PR AUC, the
paired test, the motif-only floor, calibration and the depth curve) with a CLI,
a fixed and recorded subsample seed, a feature cache, and tests. These three
can be deleted; they were kept so the old numbers could be cross-checked
against the new ones.

## Still the only source

| Script | What it produces | Where the number appears |
|---|---|---|
| `probe.py` | The dataset profile: shape, depth distribution, per-motif base rates, positive clustering by gene | Most of `docs/data.md`, and "Understanding the data" in GAPS.md |
| `motif_only.py` | 7-mer and depth-only base-rate classifiers | "a depth-only classifier scores PR AUC 0.0457" in GAPS.md |
| `dist_cmp.py` | Hct116 vs SG-NEx A549 signal distributions at matched depth | "Cross-cell-line shift is smaller than this file previously assumed" |
| `mil_lite.py` | Read-level (MIL-style) probe: reads inherit the site label, read probabilities pooled by a fixed rule | The read-level table under "Modelling" |
| `sparse_file.py` | `configs/everything.yaml` scored on held-out folds with cross-site columns recomputed as if the file were sparse, beside the same model without them | GAPS.md, Modelling, the cross-site entry ("they may not ship"; measured 2026-09-26); `docs/test-data-assumptions.md` |
| `density_screen.py` | Whether a thinned evaluation file hurts a cross-site model, thinned at random or by coverage, and whether training on thinned copies protects against it (~7 min) | GAPS.md, the same cross-site entry; `docs/test-data-assumptions.md` |
| `density_mismatch.py` | The first version of that question: a fold model fitted on full-density features, scored on randomly thinned ones | Not quoted directly; superseded by `density_screen.py` but kept as its starting point |
| `second_moment_screen.py` | Whether the 24 per-read correlations and nine variances not already in `quantiles_all_v1` carry signal (univariate AUC) and how redundant they are with existing columns | GAPS.md, Modelling, "the other 24 within-site correlations" (a quick run, marked not quotable) |

The screens print per-fold numbers on the canonical split. They are screens,
not results: a number that matters is re-run through `scripts/evaluate.py`
before it is quoted.

`dist_cmp.py` needs an SG-NEx A549 sample in JSONL form, passed as its first
argument. Nothing in this repo produces that file in that form.
