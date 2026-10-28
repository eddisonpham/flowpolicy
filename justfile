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

# CI smoke job (docs/01 section 4): 20 steps of real training on CPU, plus resume
# equivalence at rtol=1e-5. No GPU, no dataset, no network.
smoke:
    uv run pytest -m "smoke" -q

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

# E14: the same checkpoint run the published way, the stock way and our way.
# Latency ladder + real episodes/hour, with success and coverage on every arm.
bench-stack episodes="4":
    uv run python scripts/bench_stack.py --episodes {{episodes}}

# E8: does half precision buy a control period, and what does it cost the policy?
bench-precision dtypes="float32 bfloat16 float16" episodes="40":
    uv run python scripts/bench_precision.py --dtypes {{dtypes}} --episodes {{episodes}}

# M9.6: the same loop with wall-clock sleeps, so deadline misses become measured lateness.
# 12 episodes at each step count; run once per N (4 fits the period, 10 does not).
bench-paced steps="4":
    uv run python scripts/deploy_eval.py --ckpts hf:lerobot/diffusion_pusht --latencies 0 --seeds 10000 10001 10002 --episodes-per-seed 4 --steps {{steps}} --paced --out results/paced.jsonl

# Load-test the serving boundary (client round trip vs in-process GPU time).
bench-serve steps="4":
    uv run python scripts/bench_serve.py --steps {{steps}}

# Render one episode to an annotated GIF (packaging, not a measurement).
demo steps="4" seed="10000":
    uv run python scripts/demo_rollout.py --steps {{steps}} --seed {{seed}} --stride 1

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
