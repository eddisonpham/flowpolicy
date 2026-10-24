#!/usr/bin/env bash
# E10: injected observation latency, flow + DDPM at the 12000-step budget,
# three eval seeds, 10 episodes per seed per row. Missing checkpoints are skipped.
set -euo pipefail
rm -f results/deploy.jsonl
CKPTS=()
for k in flow ddpm; do
  ckpt="checkpoints/e1_${k}_s0/last.pt"
  if [ -f "$ckpt" ]; then
    CKPTS+=("$ckpt")
  else
    echo "skip $k (no $ckpt)"
  fi
done
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "no checkpoints found; train first" >&2
  exit 1
fi
uv run python scripts/deploy_eval.py --ckpts "${CKPTS[@]}" \
  --train-seeds 0 \
  --latencies 0 50 100 150 200 --episodes-per-seed 10 --strategy naive \
  --out results/deploy.jsonl
# E12: temporal ensembling only differs from naive replace when chunks overlap,
# so both strategies are run at a 2-step replan rate (action_horizon is 8).
for s in naive-r2 temporal-ensemble-r2; do
  uv run python scripts/deploy_eval.py --ckpts "${CKPTS[@]}" \
    --train-seeds 0 \
    --latencies 100 200 --episodes-per-seed 10 --strategy "$s" \
    --out results/deploy.jsonl
done
