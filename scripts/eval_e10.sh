#!/usr/bin/env bash
# E10: injected observation latency, three checkpoints, three eval seeds, 5 episodes.
set -euo pipefail
rm -f results/deploy.jsonl
CKPTS=(
  checkpoints/e1_flow_s0/last.pt
  checkpoints/e1_flow_s1/last.pt
  checkpoints/e1_flow_s2/last.pt
)
uv run python scripts/deploy_eval.py --ckpts "${CKPTS[@]}" \
  --latencies 0 50 100 150 200 --episodes-per-seed 5 --strategy naive \
  --out results/deploy.jsonl
uv run python scripts/deploy_eval.py --ckpts "${CKPTS[@]}" \
  --latencies 100 200 --episodes-per-seed 5 --strategy temporal-ensemble \
  --out results/deploy.jsonl
