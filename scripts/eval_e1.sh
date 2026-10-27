#!/usr/bin/env bash
# Ablation: the three from-scratch policies x SEEDS (default 0 1 2, 12000 steps each),
# paired eval seeds, 50 episodes each, written to results/ablation/. These rows are what
# the pretrained model replaced: at this budget they reach 4-7 % success, which is a floor
# effect (docs/04 section 3). A missing checkpoint is skipped and reported as missing.
SEEDS="${SEEDS:-0 1 2}"
POLICIES="${POLICIES:-flow ddpm regression}"
set -euo pipefail
prefixes=""
for k in $POLICIES; do prefixes="$prefixes e1_$k"; done
# shellcheck disable=SC2086
uv run python scripts/prune_results.py --path results/ablation/e1.jsonl --prefix $prefixes
for k in $POLICIES; do
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
      --out results/ablation/e1.jsonl
  done
done
