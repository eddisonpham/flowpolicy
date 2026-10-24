#!/usr/bin/env bash
# E3: sampler-step sweep on the E1 checkpoints, same eval seeds, 50 episodes.
# Regression is deterministic (no integration steps), so only flow and DDPM are swept.
# Default seed 0 is the 12000-step run; seeds 1 and 2 only hold 1800-step checkpoints.
set -euo pipefail
SEEDS="${SEEDS:-0}"
rm -f results/e3.jsonl
for k in flow ddpm; do
  for s in $SEEDS; do
    ckpt="checkpoints/e1_${k}_s${s}/last.pt"
    if [ ! -f "$ckpt" ]; then
      echo "skip $k seed $s (no $ckpt)"
      continue
    fi
    for n in 1 2 4 8 10 16; do
      uv run python scripts/evaluate.py \
        --ckpt "$ckpt" \
        --config-id "e3_${k}_n${n}" --train-seeds "$s" \
        --steps "$n" --episodes 50 --seed-start 20000 --workers 2 \
        --out results/e3.jsonl
    done
  done
done
