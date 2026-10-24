#!/usr/bin/env bash
# E1: three policies x SEEDS (default 0, the 12000-step run), paired eval seeds,
# 50 episodes each. Checkpoints that do not exist are skipped, so a
# single-training-seed run is reported as exactly that. Seeds 1 and 2 only hold
# 1800-step checkpoints, so they are excluded by default to keep one budget per table.
set -euo pipefail
SEEDS="${SEEDS:-0}"
rm -f results/e1.jsonl
for k in flow ddpm regression; do
  for s in $SEEDS; do
    ckpt="checkpoints/e1_${k}_s${s}/last.pt"
    if [ ! -f "$ckpt" ]; then
      echo "skip $k seed $s (no $ckpt)"
      continue
    fi
    uv run python scripts/evaluate.py \
      --ckpt "$ckpt" \
      --config-id "e1_${k}_s${s}" \
      --train-seeds "$s" \
      --episodes 50 --seed-start 10000 --workers 2 \
      --out results/e1.jsonl
  done
done
