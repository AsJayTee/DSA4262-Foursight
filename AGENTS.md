# Conventions for coding agents

Read this before changing anything. It applies to Claude Code, Codex, and to
people.

This repo is worked on by four people in parallel, each with their own agent.
The rules below exist so that work merges cleanly and so the graded deliverable
keeps working.

---

## Start here

Before planning any work, read:

| File | What it gives you |
|---|---|
| [docs/project-requirements.md](docs/project-requirements.md) | What the project is graded on — tasks, formats, deadlines, assessment weights. The authoritative brief. |
| [GAPS.md](GAPS.md) | What is currently missing, weak, or untested, with evidence. Deliberately records no solutions — work out your own. |
| [docs/data.md](docs/data.md) | The data dictionary: file formats, feature meanings, measured statistics. |
| [docs/setup.md](docs/setup.md) | Getting a machine working, locally or on Ronin. |
| [docs/running-experiments.md](docs/running-experiments.md) | The loop your user follows. Written for them, not you — but it tells you what they expect from you. |
| [docs/decisions/](docs/decisions/) | Why the shared pipeline is shaped the way it is. Read before changing any of it; write one when you do. |

Most people here work through you rather than by reading the source. When you
are asked to explain or change something, assume the person has not read the
file and does not need to. Explain in terms of what it does and why, and hand
back a command they can run.

`GAPS.md` is the backlog. If you close a gap, delete the entry. If you find a
new one, add it with whatever evidence you have.

---

## 1. Experiments are additive

**A new experiment is a new file plus a new config YAML. Do not edit existing
files to add one.**

To add a feature set:

```python
# src/m6a/features/my_idea.py
from m6a.data import Site
from m6a.features.base import FeatureExtractor, motif_onehot
from m6a.registry import register

@register("features", "my_idea_v1")
class MyIdea(FeatureExtractor):
    def site_features(self, site: Site) -> dict[str, float]:
        return {"something": float(site.reads[:, 5].max()), **motif_onehot(site)}
```

To add a model: same shape, in `src/m6a/models/`, subclassing `BaseModel` and
implementing `fit`, `predict_proba`, `save`, `load`.

Then a config that references it by name:

```yaml
# configs/exp_my_idea.yaml
name: exp_my_idea
features: my_idea_v1
model: lightgbm
model_params: {num_leaves: 63}
split: {seed: 4262, n_folds: 5, group_by: gene_id}
```

The registry finds new modules by scanning the package directory, so nothing
needs registering in a shared table. That is what keeps four parallel agents
from producing merge conflicts.

Editing `data.py`, `registry.py`, `evaluation.py`, `crossval.py`,
`compare.py`, `feature_cache.py`, `report.py`, `tracking.py` or the scripts is
sometimes right — but it is shared infrastructure, so say so explicitly rather
than doing it as a side effect of adding an experiment.

Adding a *figure* to `figures.py` is additive and does not need a record, the
same way a feature set does not.

**And write it down.** Any architectural change to the shared pipeline gets a
decision record in [docs/decisions/](docs/decisions/) — one short file saying
what you decided, why, what the alternatives were, and what it costs. Your
teammates and their agents have to be able to find out *why* the pipeline works
the way it does without reading the source or asking you, and "why" is the thing
that never survives in code comments alone.

This applies to: anything in `src/m6a/*.py` outside `features/` and `models/`,
the scripts, the config schema, any file format, the split, the metrics, the
comparison between runs, and any new dependency. It does **not** apply to adding
an experiment — that is additive and explains itself.

One file per decision, numbered, never one shared log: four agents appending to
the same file is four merge conflicts. Same reason the registry scans a
directory instead of keeping a table. The format is in
[docs/decisions/README.md](docs/decisions/README.md); a record that takes ten
minutes to write is one that actually gets written.

---

## 2. Run `make smoke` before handing anything over

```bash
make smoke CONFIG=configs/exp_my_idea.yaml
```

5,000 sites, about four seconds, no W&B. It exercises the whole pipeline.

**Do not give someone a command you have not run.** The person running your
code may be on a VM that costs money and may not read tracebacks. Fix your own
errors first.

---

## 3. Never change the split seed

`seed: 4262`, grouped by `gene_id`, in every config.

If two people use different seeds their metrics are not comparable — and
nothing will warn them, because both numbers look plausible. Transcripts of one
gene share sequence and positions, so a non-grouped split leaks and inflates
AUC.

If you need a different split for a specific reason, say so loudly in the
config `notes` and in the report.

---

## 4. `predict.py` dependencies are frozen

Other students run `scripts/predict.py` on their own machine from a clean
`git clone` plus `pip install -e .`. They have no `.env`, no network, and only
the base dependencies in `pyproject.toml`.

So:

- **Do not add imports to `predict.py`, `data.py`, `evaluation.py`, `registry.py`
  or `features/`** beyond the base dependency list. That includes matplotlib,
  which is a `train` extra and lives behind `m6a.figures`.
- **A direct import on the prediction path must be a declared base dependency**,
  even if something else already installs it. See
  [docs/decisions/0011](docs/decisions/0011-joblib-is-declared-not-inherited.md)
  — joblib sat in `models/baseline.py` undeclared for exactly that reason.
- **If a model needs a heavy dependency (torch), import it inside the methods
  that use it**, never at module level. The registry imports every module in
  `models/` to discover it; a module-level `import torch` would make torch a
  hard requirement of the graded path.

`tests/test_smoke.py::test_predict_path_has_no_heavy_imports` enforces this.
If you break it, the fix is to move the import, not to change the test.

---

## 5. Never commit data

`.gitignore` blocks `data/`, `*.json.gz`, `.env` and `models/*` (except
`models/final/`). Do not add exceptions.

The raw data is unpublished research data and this repo is public. The only
data files that belong in git are `data/sample/`, which is required by the
handout so evaluators can run the prediction script.

Secrets go in `.env`, which is shared over Telegram and never committed.

---

## 6. Metrics

Report **PR AUC** first. At 4.49% positives, ROC AUC flatters everything.
`metrics()` also returns `pr_auc_lift` — PR AUC divided by the positive rate.
A model at 1.0× has learned nothing, whatever its ROC AUC says.

Always quote out-of-fold numbers from the gene-grouped split, never training
scores.

**Never claim one model beats another from two pooled numbers.** Folds differ
in size and positive rate — 4.10% to 5.15% here — so fold-to-fold variation
(sd 0.0203) is larger than most real improvements. Both runs are scored on the
*same* folds, so pair them and compare the per-fold differences:

```bash
python scripts/evaluate.py --config configs/your_experiment.yaml \
       --compare-features quantiles_v1
```

That framing matters: on `pooled_v1` vs `quantiles_v1` the unpaired view gives
p = 0.31 and the naive paired view p = 0.0465 on the identical numbers, and
pairing is the correct framing. But **five folds is five paired observations,
and they are not independent** — each fold's model trains on the other four, so
73.4% of fold 0's training rows are also in fold 1's. `evaluate.py` therefore
reports a third number, and it is the one to quote:

```
unpaired  (wrong)      Welch p = 0.3095
paired    (optimistic) p = 0.0465   wins 5/5
corrected (quote this) p = 0.1305
```

The corrected row is the Nadeau & Bengio resampled t-test. **It is always the
largest of the three**, and if you ever see it come out smaller than the
uncorrected one, the correction is being applied the wrong way round — say so
rather than quoting it.

Read the corrected p-value *and* the win count. 5/5 does not depend on the
scatter estimate and is the more trustworthy signal when there are only five
observations; the corrected p-value is what stops five correlated numbers
reading as five experiments.

**Every run is already a distribution.** `standard` and `full` run the whole
cross-validation **ten times** over independently seeded splits, so any run
reports 50 observations rather than 5 without anyone remembering to ask:

```bash
python scripts/evaluate.py --config configs/your_experiment.yaml \
       --compare-features quantiles_v1
```

Repetition 0 is always the canonical seed-4262 split, so nothing already
recorded moves — see
[0012](docs/decisions/0012-repeated-cv-is-one-run-keyed-by-rep-and-fold.md) and
[0013](docs/decisions/0013-every-run-is-a-distribution.md). It costs ten times
the model fits and no extra feature extraction. Use `--quick` while iterating;
`--repeats N` overrides either way.

A five-point run cannot be topped up after the instance is terminated, and it is
not comparable with a fifty-point one — which is why this is a default and not a
flag.

This is not a formality. The `quantiles_v1` vs `pooled_v1` comparison is
**unresolvable** on five folds (corrected p = 0.1305) and comfortably resolved
on fifty (+0.0164, 50/50 wins, corrected p = 0.000095). If a result matters,
run the repetitions rather than quoting the underpowered number.

Three more things `scripts/evaluate.py` reports, all of which have changed a
conclusion in this repo at least once:

- **Per-stratum metrics** (`--by depth,motif`) — where the model fails, not
  just how well it does on average. A comparison tests inside each stratum too,
  which is how you answer "is this better *at low depth*" rather than only "is
  this better on average". Those per-stratum p-values are the weakest numbers
  the harness produces — one per band, strongly correlated, deliberately not
  corrected for multiplicity. Descriptive, never a headline.
- **Calibration** — the scores are not probabilities. Mean predicted is 0.0809
  against a 0.0449 actual rate, a 1.80× overcount. Rank-based metrics cannot
  see this; any count of modified sites is wrong by that factor.
- **A depth sweep** (`--depth-sweep`) — every training site has ≥ 20 reads and
  SG-NEx has a median of 3. At one read this model scores what a motif-only
  classifier scores. Read [what depth means](docs/data.md#read-depth) first.

Every run writes a JSON report to `analysis/evaluation/reports/` **and** uploads
it to W&B. If you quote a number in `GAPS.md` or the report, quote one that a
command can regenerate.

**Ask before you prepare a run, not after.** Ronin instances are terminated with
nothing pulled off them, so a comparison nobody asked for at the start is a
comparison that needs the whole instance again. Before handing over a command,
ask whether the run is:

- **standalone** → `python scripts/train.py --config <cfg>`. Trains and
  evaluates at the `standard` profile, everything to one W&B run.
- **against a specific baseline** → add
  `python scripts/evaluate.py --config <cfg> --compare-features <name>` (or
  `--compare-with <other.yaml>`).
- **against a run from an instance that is gone** → `--compare-run <run-id>`.
  Pulls that run's metric vector out of W&B and pairs against it with **no
  refit**. Refuses unless both runs record the same dataset fingerprint and
  split ([0015](docs/decisions/0015-comparing-against-a-run-that-no-longer-exists.md)).

Never hand over `--quick` for a run whose number will be quoted. It skips the
depth sweep, and a run without depth numbers cannot be compared against one that
has them — which is the failure the profiles exist to prevent.

---

## 7. Style

- Match the surrounding code. Type hints on public functions, no decoration.
- Comments explain *why*, not *what*. If a line encodes a decision someone
  might undo by accident, say why it is there.
- Error messages name the command that fixes the problem. A teammate who
  cannot read a traceback should still know what to do next.
- No new dependencies without saying so — every one is something that can fail
  on someone else's machine.
- **Explain a domain term the first time it appears in a document**, and link
  to the fuller definition rather than restating it. Most people here are data
  scientists, not biologists, and a term that is silently assumed is a term
  someone will quietly misread. `depth` is the worst offender — it is a count
  of distinct RNA molecules measured at one site, not repeated readings of one
  molecule, and nearly every conclusion in `GAPS.md` depends on that
  distinction. It is defined once in
  [docs/data.md](docs/data.md#read-depth); point at it.

---

## 8. Where things live

| You are doing | It goes in |
|---|---|
| A new feature set or model | `src/m6a/features/`, `src/m6a/models/` |
| Evaluating or comparing runs | `scripts/evaluate.py` — don't write your own |
| A report section, or a profile | `src/m6a/report.py` (shared — write a record) |
| A plot | `src/m6a/figures.py` (additive; matplotlib imported inside the function) |
| Anything that talks to W&B | `src/m6a/tracking.py` — the flat metric keys are a schema |
| Exploration, plots, one-off analysis | `analysis/notebooks/<initials>_<topic>.ipynb` |
| Task 2 / SG-NEx work | `analysis/sgnex/` |
| m6Anet benchmark | `analysis/m6anet/` (separate conda env — see its README) |
| A figure for the report | `report/figures/`, generated by a script |
| Shared infrastructure | `src/m6a/*.py` — flag it in the PR, **and write a `docs/decisions/` record** |
| Why the pipeline is shaped as it is | `docs/decisions/NNNN-*.md` |
| An evaluation number you want to quote | `analysis/evaluation/reports/*.json` |

Notebooks: one owner per file, named with your initials. They are not shared
editing surfaces.
