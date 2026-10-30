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

# The same commands CI runs, plus the artifacts it diffs: one command that proves the tree is
# self-consistent before a commit. `test` stays separate because it is what docs/01 promises.
verify: test
    uv run pytest -m "smoke" -q
    uv run python scripts/make_tables.py
    uv run python scripts/make_readme.py
    git diff --exit-code -- results/tables README.md

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

# Paired McNemar between any two results files, on the seeds both cover. One implementation of
# the test, so no number is computed by hand (docs/04 section 1).
paired a b="results/e0_published_seeds_ddpm.jsonl" label_a="a" label_b="b":
    uv run python scripts/paired_test.py --a {{a}} --b {{b}} --label-a {{label_a}} --label-b {{label_b}}

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

# E15: a second published policy family through the same harness (generality check).
# VQ-BeT has no sampler steps, so --steps is ignored by its adapter; --action-horizon is
# its own 5-action chunk.
eval-vqbet episodes="200" seed_start="100000":
    uv run python scripts/prune_results.py --path results/vqbet.jsonl --prefix vqbet_pusht_n
    uv run python scripts/evaluate.py --ckpt hf:lerobot/vqbet_pusht --config-id vqbet_pusht_n{{episodes}} --episodes {{episodes}} --seed-start {{seed_start}} --steps 1 --action-horizon 5 --workers 1 --out results/vqbet.jsonl

# Disjoint replication block for the VQ-BeT rows: seeds 100200-100399 are outside the block
# above, so this tests whether 54 % repeats on fresh episodes rather than resampling it.
eval-vqbet-replication episodes="200" seed_start="100200":
    uv run python scripts/prune_results.py --path results/vqbet_replication.jsonl --prefix vqbet_pusht_rep
    uv run python scripts/evaluate.py --ckpt hf:lerobot/vqbet_pusht --config-id vqbet_pusht_rep{{episodes}} --episodes {{episodes}} --seed-start {{seed_start}} --steps 1 --action-horizon 5 --workers 1 --out results/vqbet_replication.jsonl

# Pin both checkpoints' own per-episode eval blocks (500 seeds each) for the paired tests.
# Network only; the results are committed so `just tables` never needs them.
published-blocks:
    uv run python scripts/published_blocks.py

# E0 paired check: the diffusion checkpoint on the *published* block's own seeds, because
# E0's 10000-block is not inside it and an aggregate comparison cannot be paired.
eval-dp-published episodes="200" seed_start="1000":
    uv run python scripts/prune_results.py --path results/e0_published_seeds.jsonl --prefix dp_published_n
    uv run python scripts/evaluate.py --ckpt hf:lerobot/diffusion_pusht --config-id dp_published_n{{episodes}} --episodes {{episodes}} --seed-start {{seed_start}} --steps 10 --out results/e0_published_seeds.jsonl

# Is the diffusion gap a property of the policy or of the noise we drew? Same 200 published
# seeds, same sampler, same everything -- only the per-episode policy noise seed is shifted.
# A rate that stays near 52 % means the noise stream is not the explanation.
eval-dp-noise episodes="200" seed_start="1000" offset="500000":
    uv run python scripts/prune_results.py --path results/e0_published_seeds_noise2.jsonl --prefix dp_published_noise
    uv run python scripts/evaluate.py --ckpt hf:lerobot/diffusion_pusht --config-id dp_published_noise{{offset}} --episodes {{episodes}} --seed-start {{seed_start}} --steps 10 --noise-seed-offset {{offset}} --out results/e0_published_seeds_noise2.jsonl

# The same comparison the other way: the checkpoint's own DDPM@100 on the published seeds, to
# test whether the success gap is truncated sampling rather than the harness. N=100 costs ~4 s
# per chunk, and two rollout workers contend on one GPU (2.3x per chunk for 2x the work, i.e.
# slower in wall clock), so this recipe defaults to one worker.
eval-dp-published-ddpm episodes="50" seed_start="1000" workers="1":
    uv run python scripts/prune_results.py --path results/e0_published_seeds_ddpm.jsonl --prefix dp_published_ddpm100_n
    uv run python scripts/evaluate.py --ckpt hf:lerobot/diffusion_pusht --config-id dp_published_ddpm100_n{{episodes}} --scheduler ddpm --steps 100 --episodes {{episodes}} --seed-start {{seed_start}} --workers {{workers}} --out results/e0_published_seeds_ddpm.jsonl

# Adapter-free control (docs/04 section 4): the checkpoint through lerobot's *own* eval loop on
# the same seed block, with none of this repo's code in the path. Own venv, because lerobot
# caps torch<2.12 and av<16 against this project's pins.
lerobot-env:
    bash scripts/build_lerobot_env.sh

lerobot-control episodes="200" seed="1000" batch="8":
    EPISODES={{episodes}} SEED={{seed}} BATCH={{batch}} bash scripts/lerobot_control.sh

# The same control, but writing one row per episode so it pairs with this repo's rows by seed
# (McNemar instead of an aggregate Fisher comparison). Needs the migrated checkpoint that
# `lerobot-control` builds. Run under the isolated lerobot interpreter, not the project's.
lerobot-control-rows episodes="200" seed="1000" batch="8":
    .venv-lerobot/Scripts/python.exe scripts/lerobot_control_rows.py --episodes {{episodes}} --seed {{seed}} --batch {{batch}}

# How much does the adapter-free control move between runs? Reads the repeats written by
# `lerobot-control-rows` and states the like-for-like gap with an interval that includes them.
# Measured 2026-10-30: five runs span 61.5-64.0 % (sd 1.0 pp), so the repeats are part of the
# measurement, not a re-check (docs/04 section 4).
control-spread:
    uv run python scripts/control_spread.py

# Four repeats (r2-r5) of the identical control command, 200 episodes each (~38 min per run), on
# top of the original row file. Rerun only to extend the block: the reader above takes whatever
# files are on disk.
control-repeats from="2":
    bash scripts/control_repeats.sh {{from}}

# Option H step 0 (docs/10 section 3): do the checkpoint's eight buffers and lerobot's
# processor pipelines compute the same arithmetic? Deterministic, no sampler, no episodes, no
# GPU block; exits non-zero if they differ by more than 1e-6. Run it with the isolated lerobot
# interpreter, because only that env has `lerobot`.
processor-equiv checkpoint="checkpoints/diffusion_pusht_migrated":
    .venv-lerobot/Scripts/python.exe scripts/processor_equiv.py --checkpoint {{checkpoint}}

# Option H's episode arm (docs/10 section 5): the library's normalization in the measured path.
# 600 seeds, because 200 cannot resolve the ~8 pp the adoption is chasing; both arms are ours, so
# this needs no reference block. "proc" in the config_id is required by rollout.evaluate.
eval-dp-processors episodes="600" seed_start="1000":
    uv run python scripts/prune_results.py --path results/e0_processors.jsonl --prefix dp_proc
    uv run python scripts/evaluate.py --ckpt hf:lerobot/diffusion_pusht --config-id dp_proc_n{{episodes}} --episodes {{episodes}} --seed-start {{seed_start}} --steps 10 --normalization processors --workers 1 --out results/e0_processors.jsonl

# Do this repo's env wrapper and lerobot's produce the same coverage trace? R5 has been open
# since 2026-10-19, and the adapter-free control runs inside lerobot's env stack, so an env
# difference would sit inside every loop-to-loop comparison. Fixed action sequence, no policy.
env-equiv seed="1000" steps="300":
    .venv-lerobot/Scripts/python.exe scripts/env_equiv.py --seed {{seed}} --steps {{steps}}

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
