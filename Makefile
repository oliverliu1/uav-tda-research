.PHONY: help test test-all lint repro-fast manuscript bundle

help:
	@echo "Available targets:"
	@echo "  make test        - run the fast test suite (pytest -m 'not slow')"
	@echo "  make test-all    - run the full test suite, including slow tests"
	@echo "  make lint        - run ruff check ."
	@echo "  make repro-fast  - run the seed-42 probe reproduction (uav-tda probe --seed 42)"
	@echo "  make manuscript  - regenerate the cited numbers and build the manuscript PDF"
	@echo "  make bundle      - regenerate the numbers and write the self-contained Overleaf zip"

test:
	python3 -m pytest -m "not slow" -q

test-all:
	python3 -m pytest -q

lint:
	ruff check .

repro-fast:
	uav-tda probe --seed 42

manuscript:
	uav-tda manuscript-numbers
	cd CLAUDE/SciTech2027_Manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

bundle:
	uav-tda manuscript-numbers
	uav-tda manuscript-bundle
