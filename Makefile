# The only interface anyone needs to memorise.
# Run these on the VM. On Windows locally, call the scripts under scripts/ directly.

PY      ?= .venv/bin/python
CONFIG  ?= configs/lightgbm.yaml
MODEL   ?= models/final
INPUT   ?= data/sample/sample.json.gz
OUTPUT  ?= predictions.csv

.PHONY: help doctor smoke train predict test sample clean

help:
	@echo "make doctor              - check this machine is set up correctly"
	@echo "make smoke  CONFIG=...   - full pipeline on 5k sites, <30s, no W&B"
	@echo "make train  CONFIG=...   - full training run, logs to W&B"
	@echo "make predict MODEL=... INPUT=... OUTPUT=..."
	@echo "make test                - pytest, incl. end-to-end smoke test"
	@echo "make sample              - regenerate data/sample/ (committed test data)"

doctor:
	@$(PY) scripts/doctor.py

smoke:
	@$(PY) scripts/train.py --config $(CONFIG) --smoke

train:
	@$(PY) scripts/train.py --config $(CONFIG)

predict:
	@$(PY) scripts/predict.py --model $(MODEL) --input $(INPUT) --output $(OUTPUT)

test:
	@$(PY) -m pytest -q

sample:
	@$(PY) scripts/make_sample.py

clean:
	@rm -rf .pytest_cache **/__pycache__ predictions.csv
