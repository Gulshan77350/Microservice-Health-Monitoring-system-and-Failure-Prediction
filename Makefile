# Linux/macOS/CI convenience targets. Windows equivalents are listed in run_command.txt.
PYTHON ?= python3.11
VENV   ?= .venv
PY     := $(VENV)/bin/python

.PHONY: help venv install data train test lint format bench up down logs clean dashboard-install dashboard-lint dashboard-build

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | sort | tr '\n' ' '; echo

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m model.generate_data --check

train: data
	$(PY) -m model.train

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

bench:
	$(PY) -m scripts.benchmark

dashboard-install:
	cd dashboard && npm ci

dashboard-lint:
	cd dashboard && npm run lint

dashboard-build:
	cd dashboard && npm run build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

clean:
	docker compose down -v
	rm -rf dashboard/dist .pytest_cache .ruff_cache
