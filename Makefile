.PHONY: test compile lint check

PYTHON ?= ./venv/bin/python
PYTHONPATH ?= src

test:
	$(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall -q src scripts tests

lint:
	$(PYTHON) -m ruff check src scripts tests

check: lint compile test
