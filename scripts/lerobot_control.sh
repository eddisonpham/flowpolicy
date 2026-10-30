#!/usr/bin/env bash
# Adapter-free control for the diffusion gap (docs/04 section 4).
#
# Runs `lerobot/diffusion_pusht` through lerobot's *own* eval loop, on the same env seeds
# this repo's adapter is measured on (default 1000-1199), with none of this repo's code in
# the path. That is the only experiment that separates "our adapter loses precision" from
# "the checkpoint scores this under its own pipeline": the 200-seed McNemar test compares two
# *runs*, and pairs them only on the environment seed.
#
# Two steps, because a 2024 checkpoint cannot be loaded by a 2026 lerobot as shipped:
#   1. `scripts/lerobot_migrate.py` runs `lerobot.processor.migrate_policy_normalization`, which
#      reads the eight `normalize_inputs.*` / `unnormalize_outputs.*` buffers this checkpoint
#      kept inside `model.safetensors` and rebuilds them as `policy_preprocessor.json` /
#      `policy_postprocessor.json`. That script then finishes the model save the migration
#      cannot complete (its draccus encoder fails on `crop_shape: [84, 84]`).
#   2. `lerobot-eval` runs the documented loop on that migrated copy.
#
# Everything runs in `.venv-lerobot` (`scripts/build_lerobot_env.sh`), never in the project's
# locked environment: lerobot 0.6.1 caps `av<16`, while the project's `av>=15` resolved to
# 19.0.1, and lerobot's dataset module cannot import against that.
#
# The migrated copy runs the checkpoint's own configuration, i.e. DDPM@100 -- the setup its
# published 65.4 % block was measured with, not this repo's DDIM@10 deployment setting.
set -euo pipefail

PY=".venv-lerobot/Scripts/python.exe"
MIGRATED="checkpoints/diffusion_pusht_migrated"
EPISODES="${EPISODES:-200}"
SEED="${SEED:-1000}"
BATCH="${BATCH:-8}"
OUT="${OUT:-results/lerobot_control}"

# Both the tee target and lerobot's output directory: a missing directory would abort the run
# after the migration instead of before it.
mkdir -p logs "$OUT"

if [ ! -x "$PY" ]; then
  echo "missing $PY -- run: bash scripts/build_lerobot_env.sh" >&2
  exit 1
fi

if [ ! -f "$MIGRATED/policy_preprocessor.json" ] || [ ! -s "$MIGRATED/config.json" ]; then
  echo "== migrating $MIGRATED (one-off, needs the Hub)"
  "$PY" scripts/lerobot_migrate.py "$MIGRATED"
else
  echo "== reusing $MIGRATED"
fi

echo "== lerobot-eval: $EPISODES episodes from seed $SEED, batch $BATCH, own DDPM@100"
# `use_async_envs=false`: the async vector env asks for a `forkserver` multiprocessing context,
# which does not exist on Windows (`ValueError: cannot find context for 'forkserver'`). The sync
# vector env runs the same episodes in the same order with the same seeds, so the control is
# unaffected -- it is a platform requirement, not a change to what is measured.
"$PY" -m lerobot.scripts.lerobot_eval \
  --policy.path="$MIGRATED" \
  --env.type=pusht \
  --eval.n_episodes="$EPISODES" \
  --eval.batch_size="$BATCH" \
  --eval.use_async_envs=false \
  --seed="$SEED" \
  --output_dir="$OUT" 2>&1 | tee "logs/lerobot_control.log"
