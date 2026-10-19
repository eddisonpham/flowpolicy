#!/usr/bin/env bash
# Train E1: 3 policies x 3 seeds. Each run resumes from its last checkpoint if present.
set -euo pipefail
STEPS="${STEPS:-1800}"
mkdir -p logs
for s in 0 1 2; do
  for k in flow ddpm regression; do
    echo "[$(date +%H:%M:%S)] start $k seed $s"
    uv run python scripts/train.py "+experiment=e1_$k" seed=$s "run_id=e1_${k}_s${s}" \
      steps="$STEPS" ckpt_every=600 > "logs/train_${k}_s${s}.log" 2>&1
    echo "[$(date +%H:%M:%S)] done $k seed $s"
  done
done
