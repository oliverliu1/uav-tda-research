.PHONY: help test test-all lint repro-fast

help:
	@echo "Available targets:"
	@echo "  make test        - run the fast test suite (pytest -m 'not slow')"
	@echo "  make test-all    - run the full test suite, including slow tests"
	@echo "  make lint        - run ruff check ."
	@echo "  make repro-fast  - run the seed-42 probe reproduction (uav-tda probe --seed 42)"

test:
	python3 -m pytest -m "not slow" -q

test-all:
	python3 -m pytest -q

lint:
	ruff check .

repro-fast:
	uav-tda probe --seed 42
