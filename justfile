set shell := ["bash", "-c"]

# Recipes mirror docs/01_architecture.md section 4 and docs/07_roadmap.md M0.

default:
    @just --list

# Create/refresh the 3.12 env from uv.lock. Safe to re-run.
setup:
    uv sync --python 3.12

# Add the optional pretrained-policy backend on top of the sync. Re-run after `just setup`.
setup-pretrained:
    bash scripts/setup_pretrained.sh

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

# Read-only environment check: versions, CUDA, backend, checkpoint. No GPU work.
doctor:
    uv run python scripts/doctor.py

# One-screen summary of everything measured so far, from results/*.jsonl.
report:
    uv run python scripts/report.py

# Serve the policy over local HTTP (contract: docs/05 section 4). Ctrl-C to stop.
serve ckpt="hf:lerobot/diffusion_pusht" port="8000" steps="10":
    uv run python scripts/serve.py --ckpt {{ckpt}} --port {{port}} --steps {{steps}}

# Week-1 gates G1-G5 from docs/02 section 7. Local only, needs a CUDA host.
gates:
    @set -e; for g in scripts/gates/g*.py; do echo "== $g"; uv run python "$g" || true; done

# Install the git hooks.
hooks:
    uv run pre-commit install

# Download/convert PushT to Zarr (idempotent).
data:
    uv run python scripts/data/prepare_pusht.py

# Ablation: train the three from-scratch policies, 3 seeds at 12000 steps (~60 min each).
train-e1:
    bash scripts/train_e1.sh

# Measured memory/throughput table (docs/03 section 4).
memory:
    uv run python scripts/bench_memory.py

# E9 latency rows on the pretrained checkpoint: mode and N are passed as arguments.
latency mode="eager" steps="10":
    uv run python scripts/latency.py --ckpt hf:lerobot/diffusion_pusht --mode {{mode}} --steps {{steps}} --config-id e9_{{mode}}_n{{steps}}

# Triton kernel equivalence + benchmark.
kernel:
    uv run python scripts/kernel_bench.py

# Tables and figures regenerate from results/*.jsonl; never hand-edited.
tables:
    uv run python scripts/make_tables.py

figures:
    uv run python scripts/make_figures.py

# E0/E3/E10 runners plus the ablation re-run and the 8-episode overfit check (GPU, local).
# eval-e0 is the acceptance gate for the pretrained adapter: its rows must land near
# the model card's 65.4 % success, otherwise the observation/action mapping is wrong.
eval-e0:
    bash scripts/eval_e0.sh

eval-e3:
    bash scripts/eval_e3.sh

eval-e10:
    bash scripts/eval_e10.sh

# Ablation only: re-measure the from-scratch checkpoints (results/ablation/e1.jsonl).
eval-e1:
    bash scripts/eval_e1.sh

latency-ladder:
    bash scripts/latency_ladder.sh

overfit:
    uv run python scripts/overfit.py

# Regenerate README.md from docs/README_template.md and results/tables/.
readme:
    uv run python scripts/make_readme.py
