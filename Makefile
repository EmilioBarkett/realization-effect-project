.PHONY: test compile analyze audit

PYTHON ?= ./venv/bin/python
RESULTS ?= results/results.csv

test:
	$(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall -q src scripts tests

analyze:
	$(PYTHON) scripts/analyze_realization_results.py $(RESULTS)

audit:
	$(PYTHON) scripts/reparse_realization_results.py results/results.csv results/blocks
