set shell := ["bash", "-c"]

# Recipes mirror docs/01_architecture.md section 4 and docs/07_roadmap.md M0.

default:
    @just --list

# Create/refresh the 3.12 env from uv.lock. Safe to re-run.
setup:
    uv sync --python 3.12

# Format then lint.
format:
    uv run ruff format .
    uv run ruff check --fix .

# Lint only, no writes. This is what CI runs.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Strict type check on src/ (docs/01 section 4).
types:
    uv run mypy

# The gate: lint, types, unit tests. Must pass from a clean clone.
test: lint types
    uv run pytest -m "unit and not gpu"

# Local-only: anything touching a GPU or a simulator.
test-local:
    uv run pytest -m "integration or gpu or slow"

# Install the git hooks.
hooks:
    uv run pre-commit install
