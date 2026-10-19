#!/usr/bin/env bash
# E3: sampler-step sweep on the E1 checkpoints, same eval seeds, 20 episodes.
set -euo pipefail
rm -f results/e3.jsonl
for k in flow ddpm; do
  for s in 0 1 2; do
    for n in 1 2 4 8 10 16; do
      uv run python scripts/evaluate.py \
        --ckpt "checkpoints/e1_${k}_s${s}/last.pt" \
        --config-id "e3_${k}_n${n}" \
        --steps "$n" --episodes 20 --seed-start 20000 --workers 2 \
        --out results/e3.jsonl
    done
  done
done
