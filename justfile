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

# Week-1 gates G1-G5 from docs/02 section 7. Local only, needs a CUDA host.
gates:
    @set -e; for g in scripts/gates/g*.py; do echo "== $g"; uv run python "$g" || true; done

# Install the git hooks.
hooks:
    uv run pre-commit install

# Download/convert PushT to Zarr (idempotent).
data:
    uv run python scripts/data/prepare_pusht.py

# E1: three policies x three seeds (GPU, ~90 min at the MVP budget).
train-e1:
    bash scripts/train_e1.sh

# Measured memory/throughput table (docs/03 section 4).
memory:
    uv run python scripts/bench_memory.py

# E9 latency rows: mode and N are passed as arguments.
latency mode="eager" steps="10":
    uv run python scripts/latency.py --ckpt checkpoints/e1_flow_s0/last.pt --mode {{mode}} --steps {{steps}} --config-id e9_{{mode}}_n{{steps}}

# Triton kernel equivalence + benchmark.
kernel:
    uv run python scripts/kernel_bench.py

# Tables and figures regenerate from results/*.jsonl; never hand-edited.
tables:
    uv run python scripts/make_tables.py

figures:
    uv run python scripts/make_figures.py

# E1/E3/E10 runners and the 8-episode overfit check (GPU, local only).
eval-e1:
    bash scripts/eval_e1.sh

eval-e3:
    bash scripts/eval_e3.sh

eval-e10:
    bash scripts/eval_e10.sh

latency-ladder:
    bash scripts/latency_ladder.sh

overfit:
    uv run python scripts/overfit.py

# Regenerate README.md from docs/README_template.md and results/tables/.
readme:
    uv run python scripts/make_readme.py
