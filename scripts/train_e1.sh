#!/usr/bin/env bash
# Train E1: 3 policies x SEEDS (default 1). A finished run (last.pt) is skipped;
# an interrupted run resumes from its newest step_*.pt. Safe to re-run.
set -euo pipefail
STEPS="${STEPS:-12000}"
SEEDS="${SEEDS:-0}"
mkdir -p logs
for s in $SEEDS; do
  for k in flow ddpm regression; do
    run_dir="checkpoints/e1_${k}_s${s}"
    if [ -f "$run_dir/last.pt" ]; then
      echo "[$(date +%H:%M:%S)] skip $k seed $s (finished: last.pt exists)"
      continue
    fi
    resume=""
    latest="$(ls -1 "$run_dir"/step_*.pt 2>/dev/null | sort | tail -1 || true)"
    if [ -n "$latest" ]; then
      resume="$latest"
      echo "[$(date +%H:%M:%S)] resume $k seed $s from $latest"
    else
      echo "[$(date +%H:%M:%S)] start $k seed $s from scratch"
    fi
    uv run python scripts/train.py "+experiment=e1_$k" seed="$s" "run_id=e1_${k}_s${s}" \
      steps="$STEPS" ckpt_every=2000 val_every=1000 "resume=$resume" \
      > "logs/train_${k}_s${s}.log" 2>&1
    echo "[$(date +%H:%M:%S)] done $k seed $s"
  done
done
