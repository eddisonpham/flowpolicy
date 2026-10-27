#!/usr/bin/env bash
# E3: sampler-step sweep on the pretrained Diffusion Policy checkpoint.
#
# The sweep uses DDIM: the checkpoint's own scheduler is DDPM, whose every step injects
# noise, so truncating it to N=1 measures nothing about the policy (that is what E0's
# ddpm@100 row is for). Same model, same eval seeds, only N changes.
# 50 episodes per N, paired eval seeds, 2 workers. Missing rows are never reported as 0.
set -euo pipefail
HF_ID="${HF_ID:-lerobot/diffusion_pusht}"
EPISODES="${EPISODES:-50}"
WORKERS="${WORKERS:-2}"
SEED_START="${SEED_START:-20000}"
STEPS="${STEPS:-1 2 4 8 10 16}"

uv run python scripts/prune_results.py --path results/e3.jsonl --prefix e3_

for n in $STEPS; do
  uv run python scripts/evaluate.py \
    --ckpt "hf:${HF_ID}" --config-id "e3_${HF_ID##*/}_ddim_n${n}" \
    --scheduler ddim --steps "$n" \
    --episodes "$EPISODES" --seed-start "$SEED_START" --workers "$WORKERS" \
    --out results/e3.jsonl
done
