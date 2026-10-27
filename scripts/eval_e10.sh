#!/usr/bin/env bash
# E10/E12: injected observation latency in the fixed-rate loop, pretrained checkpoint.
#
# naive: replan when the 8-step action queue empties (the deployment default).
# The -r2 rows replan every 2 control steps so four chunks overlap, which is the only
# regime where temporal ensembling differs from naive replacement (docs/04 section 4).
# The loop runs in virtual time; chunk inference is timed for real.
set -euo pipefail
HF_ID="${HF_ID:-lerobot/diffusion_pusht}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-10}"
STEPS="${STEPS:-10}"

uv run python scripts/prune_results.py --path results/deploy.jsonl --prefix e10_

uv run python scripts/deploy_eval.py --ckpts "hf:${HF_ID}" \
  --scheduler ddim --steps "$STEPS" \
  --latencies 0 50 100 150 200 --episodes-per-seed "$EPISODES_PER_SEED" --strategy naive \
  --out results/deploy.jsonl

for s in naive-r2 temporal-ensemble-r2; do
  uv run python scripts/deploy_eval.py --ckpts "hf:${HF_ID}" \
    --scheduler ddim --steps "$STEPS" \
    --latencies 100 200 --episodes-per-seed "$EPISODES_PER_SEED" --strategy "$s" \
    --out results/deploy.jsonl
done
