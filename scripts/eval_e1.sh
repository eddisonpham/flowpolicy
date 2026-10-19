#!/usr/bin/env bash
# E1: three policies x three seeds, paired eval seeds, 30 episodes each.
set -euo pipefail
rm -f results/e1.jsonl
for k in flow ddpm regression; do
  for s in 0 1 2; do
    uv run python scripts/evaluate.py \
      --ckpt "checkpoints/e1_${k}_s${s}/last.pt" \
      --config-id "e1_${k}_s${s}" \
      --episodes 30 --seed-start 10000 --workers 4 \
      --out results/e1.jsonl
  done
done
