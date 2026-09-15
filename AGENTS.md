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
| [docs/setup.md](docs/setup.md) | Running things locally or on a VM. |

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

Editing `data.py`, `registry.py`, `evaluation.py` or the scripts is sometimes
right — but it is shared infrastructure, so say so explicitly rather than doing
it as a side effect of adding an experiment.

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
  or `features/`** beyond the base dependency list.
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

---

## 7. Style

- Match the surrounding code. Type hints on public functions, no decoration.
- Comments explain *why*, not *what*. If a line encodes a decision someone
  might undo by accident, say why it is there.
- Error messages name the command that fixes the problem. A teammate who
  cannot read a traceback should still know what to do next.
- No new dependencies without saying so — every one is something that can fail
  on someone else's machine.

---

## 8. Where things live

| You are doing | It goes in |
|---|---|
| A new feature set or model | `src/m6a/features/`, `src/m6a/models/` |
| Exploration, plots, one-off analysis | `analysis/notebooks/<initials>_<topic>.ipynb` |
| Task 2 / SG-NEx work | `analysis/sgnex/` |
| m6Anet benchmark | `analysis/m6anet/` (separate conda env — see its README) |
| A figure for the report | `report/figures/`, generated by a script |
| Shared infrastructure | `src/m6a/*.py` — flag it in the PR |

Notebooks: one owner per file, named with your initials. They are not shared
editing surfaces.
