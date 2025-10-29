SHELL := bash
RUN := uv run

PRIMARY_BRANCH := main

SOURCES_TEST := tests
SOURCES_MAIN := src
SOURCES_TOOLS := tools

TESTS_PATTERN := $(SOURCES_TEST)/

SOURCES := $(SOURCES_MAIN) $(SOURCES_TEST) $(SOURCES_TOOLS)

SOURCES_MYPY := $(SOURCES_MAIN) $(SOURCES_TEST)
SOURCES_FORMAT_ALL := $(SOURCES)

REPO_NAME ?= $(notdir $(shell git rev-parse --show-toplevel))
TAG_NAME ?= $(shell git describe --tags --always)

install-deps:
ifeq (, $(shell which uv))
	@echo "No uv installation found in current path. Installing..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
else
	@echo "uv installation found. Skipping install."
endif

install:
	uv sync

test-coverage tests-coverage:
	MYPYPATH=src $(RUN) pytest --cov=condor --cov-branch --cov-report xml:coverage.xml --cov-report html:htmlcov -n 4 --junitxml=test-results.xml $(TESTS_PATTERN) --ignore=tests/local -vvv -o log_cli_level=CRITICAL --durations=40

diff-coverage:
	$(RUN) diff-cover -c pyproject.toml coverage.xml

test tests:
	MYPYPATH=src $(RUN) pytest --junitxml=test-results.xml -n auto --dist=loadfile $(TESTS_PATTERN) --ignore=tests/local -vvv -o log_cli_level=CRITICAL --durations=40

type-check:
	$(RUN) mypy --incremental $(SOURCES_MYPY)

format-lint-check:
	# Check whether python code is formatted correctly
	$(RUN) ruff format --check $(SOURCES_FORMAT_ALL)
	# Check whether python code matches linter rules, including import sorting
	$(RUN) ruff check $(SOURCES_FORMAT_ALL)

format-lint:
	# Make code follow linter rules, including import sorting
	$(RUN) ruff check --fix $(SOURCES_FORMAT_ALL)
	# Format python code
	$(RUN) ruff format $(SOURCES_FORMAT_ALL)

docs:
	$(RUN) tools/generate_doc.py
	$(RUN) mkdocs build

docs-check: docs
	git diff --exit-code

build wheel:
	uv build

shell: install
	source .venv/bin/activate

.PHONY: install-deps install test tests format-lint-check format-lint \
	type-check shell build wheel docs docs-check
