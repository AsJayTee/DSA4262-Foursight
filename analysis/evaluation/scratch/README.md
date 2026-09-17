# Scratch scripts

Rough, single-purpose scripts with hardcoded paths, written before
`scripts/evaluate.py` existed. They are not part of the harness and nothing
imports them. Kept only where they are still the only source of a number quoted
in [GAPS.md](../../../GAPS.md).

Run them from the repo root, with the training data in `data/raw/`.

## Superseded — use the harness instead

| Script | Replaced by |
|---|---|
| `perfold.py` | `evaluate.py --config ... --ablate` (per-fold, ablations, calibration) |
| `paired.py` | `evaluate.py --config ... --compare-features pooled_v1` |
| `depth_shift.py` | `evaluate.py --config ... --depth-sweep` |

The harness reproduces every number these produced — per-fold PR AUC, the
paired test, the motif-only floor, calibration and the depth curve — with a CLI,
a fixed and recorded subsample seed, a feature cache, and tests. These three can
be deleted; they are kept for one release so the old numbers can be
cross-checked against the new ones.

## Still the only source

| Script | What it produces | Where the number appears |
|---|---|---|
| `probe.py` | The dataset profile: shape, depth distribution, per-motif base rates, positive clustering by gene | Most of `docs/data.md`, and "Understanding the data" in GAPS.md |
| `motif_only.py` | 7-mer and depth-only base-rate classifiers | "a depth-only classifier scores PR AUC 0.0457" in GAPS.md |
| `dist_cmp.py` | Hct116 vs SG-NEx A549 signal distributions at matched depth | "Cross-cell-line shift is smaller than this file previously assumed" |
| `mil_lite.py` | Read-level (MIL-style) probe: reads inherit the site label, read probabilities pooled by a fixed rule | The read-level table under "Modelling" |

`dist_cmp.py` needs an SG-NEx A549 sample in JSONL form, passed as its first
argument. Nothing in this repo produces that file yet — see the Task 2 section
of GAPS.md.
