#!/usr/bin/env bash
# E9: eager/compile/CUDA-graph ladder at N=10, plus eager and graph N sweeps.
set -euo pipefail
rm -f results/latency.jsonl
CKPT="checkpoints/e1_flow_s0/last.pt"
for n in 1 2 4 8 10 16; do
  uv run python scripts/latency.py --ckpt "$CKPT" --mode eager --steps "$n" --config-id "e9_eager_n${n}"
done
for mode in compile-default compile-reduce-overhead compile-max-autotune cudagraph; do
  uv run python scripts/latency.py --ckpt "$CKPT" --mode "$mode" --steps 10 --config-id "e9_${mode}_n10"
done
for n in 1 2 4 8 10 16; do
  uv run python scripts/latency.py --ckpt "$CKPT" --mode cudagraph --steps "$n" --config-id "e9_cudagraph_n${n}"
done
