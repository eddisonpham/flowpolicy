#!/usr/bin/env bash
# E0: the pretrained Diffusion Policy checkpoint through our harness.
#
# This is the acceptance gate for models/pretrained.py: lerobot's own model card
# reports 65.4 % success and 0.955 mean max overlap on 500 gym-pusht episodes, so a
# run that lands near zero means our observation/action mapping is wrong, not that the
# policy is bad. Two rows, same 50 paired eval seeds:
#   - ddim@10 : the deployment setting used by every later experiment (178 ms/chunk)
#   - ddpm@100: the checkpoint's own scheduler and step count, i.e. the published setup
# Missing files are never reported as zero.
set -euo pipefail
HF_ID="${HF_ID:-lerobot/diffusion_pusht}"
EPISODES="${EPISODES:-50}"
WORKERS="${WORKERS:-2}"
SEED_START="${SEED_START:-10000}"
uv run python scripts/prune_results.py --path results/e0.jsonl --prefix e0_

uv run python scripts/evaluate.py \
  --ckpt "hf:${HF_ID}" --config-id "e0_${HF_ID##*/}_ddim10" \
  --scheduler ddim --steps 10 \
  --episodes "$EPISODES" --seed-start "$SEED_START" --workers "$WORKERS" \
  --out results/e0.jsonl

uv run python scripts/evaluate.py \
  --ckpt "hf:${HF_ID}" --config-id "e0_${HF_ID##*/}_ddpm100" \
  --scheduler ddpm --steps 100 \
  --episodes "$EPISODES" --seed-start "$SEED_START" --workers "$WORKERS" \
  --out results/e0.jsonl
