.PHONY: test compile lint check analyze audit steering-summary figures report report-pipeline

PYTHON ?= ./venv/bin/python
RESULTS ?= results/results.csv
STEERING_EVAL ?= results/test/activation_vectors/steering_runs/gemma_realization_steering_train_only_full_v1/steering_eval.csv
STEERING_REPORT_DIR ?= results/final/report_realization_v1/03_steering_intervention
REPORT_DIR ?= reports/final

test:
	$(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall -q src scripts tests

lint:
	$(PYTHON) -m ruff check src scripts tests

check: lint compile test

analyze:
	$(PYTHON) scripts/analyze_realization_results.py $(RESULTS)

audit:
	$(PYTHON) scripts/reparse_realization_results.py results/results.csv results/blocks

steering-summary:
	$(PYTHON) scripts/summarize_steering_report_tables.py \
		--input $(STEERING_EVAL) \
		--output-dir $(STEERING_REPORT_DIR)

figures:
	$(PYTHON) scripts/build_report_figures.py

report:
	cd $(REPORT_DIR) && tectonic report.tex

report-pipeline: steering-summary figures report
