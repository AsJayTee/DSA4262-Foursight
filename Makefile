# The only interface anyone needs to memorise.
# Run these on the VM. On Windows locally, call the scripts under scripts/ directly.

PY      ?= .venv/bin/python
CONFIG  ?= configs/lightgbm.yaml
MODEL   ?= models/final
INPUT   ?= data/sample/sample.json.gz
OUTPUT  ?= predictions.csv
PROFILE ?= standard
EVAL    ?=
RUNS    ?=
ARGS    ?=

.PHONY: help doctor smoke train evaluate compare predict test sample clean clean-cache

help:
	@echo "make doctor              - check this machine is set up correctly"
	@echo "make smoke  CONFIG=...   - full pipeline on 5k sites, <30s, no W&B"
	@echo "make train  CONFIG=...   - train AND evaluate, everything to W&B"
	@echo "make evaluate CONFIG=... EVAL='--compare-features pooled_v1'"
	@echo "                         - re-evaluate, or compare against a baseline"
	@echo "                           (train already does the standard profile)"
	@echo "                           add --repeats 10 when it is too close to call"
	@echo "make compare RUNS='run-a run-b run-c'"
	@echo "                         - one figure across N finished W&B runs,"
	@echo "                           written to report/figures/"
	@echo "make predict MODEL=... INPUT=... OUTPUT=..."
	@echo "make test                - pytest, incl. end-to-end smoke test"
	@echo "make sample              - regenerate data/sample/ (committed test data)"
	@echo "make clean-cache         - drop the extracted-feature cache"

doctor:
	@$(PY) scripts/doctor.py

smoke:
	@$(PY) scripts/train.py --config $(CONFIG) --smoke

# Trains, then runs the standard evaluation profile inline and logs both to the
# same W&B run. On a disposable instance that is the only place either survives.
train:
	@$(PY) scripts/train.py --config $(CONFIG) --profile $(PROFILE)

# Fits the folds itself, so it needs no prior training run. Use it for
# comparisons: EVAL='--compare-features pooled_v1' or EVAL='--ablate'.
evaluate:
	@$(PY) scripts/evaluate.py --config $(CONFIG) --profile $(PROFILE) $(EVAL)

# Reads finished runs out of W&B and fits nothing, so it needs no instance and
# no data - it works from a laptop after every machine involved is gone.
compare:
	@$(PY) scripts/compare_runs.py $(RUNS) $(ARGS)

predict:
	@$(PY) scripts/predict.py --model $(MODEL) --input $(INPUT) --output $(OUTPUT)

test:
	@$(PY) -m pytest -q

sample:
	@$(PY) scripts/make_sample.py

clean:
	@rm -rf .pytest_cache **/__pycache__ predictions.csv

clean-cache:
	@$(PY) -c "import sys; sys.path.insert(0, 'src'); from m6a import feature_cache; print(f'{feature_cache.clear()} cached feature table(s) removed from {feature_cache.cache_dir()}')"
