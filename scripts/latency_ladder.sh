#!/usr/bin/env bash
# E9: batch-1 latency ladder on the pretrained checkpoint, DDIM at N=10.
#
# eager N sweep first (the deployment-relevant curve), then the compile/CUDA-graph
# ladder at the N=10 operating point. The model is 262.7 M parameters fp32, so at N=10
# eager it is already far past the 100 ms control period -- that is the point of the table.
#
# Compile modes are attempted and reported honestly. On this host any graph containing a
# CPU partition needs MSVC `cl.exe` for Inductor's CPU codegen and fails with
# `InvalidCxxCompiler`; the checkpoint's DDIM scheduler builds host tensors inside the
# sampled function, so the pretrained path hits that partition where the from-scratch
# policies (pure CUDA graphs) did not. A failing mode is never written as a row.
set -euo pipefail
HF_ID="${HF_ID:-lerobot/diffusion_pusht}"
CKPT="hf:${HF_ID}"

uv run python scripts/prune_results.py --path results/latency.jsonl --prefix e9_

for n in 1 2 4 8 10 16; do
  uv run python scripts/latency.py --ckpt "$CKPT" --mode eager --steps "$n" \
    --config-id "e9_eager_n${n}"
done

for mode in compile-default compile-reduce-overhead compile-max-autotune; do
  if uv run python scripts/latency.py --ckpt "$CKPT" --mode "$mode" --steps 10 \
    --config-id "e9_${mode}_n10"; then
    echo "[e9] ${mode} measured"
  else
    echo "[e9] NOT RUN: ${mode} (InvalidCxxCompiler: cl.exe not found; see docs/05 section 1)"
  fi
done

for n in 1 2 4 8 10 16; do
  uv run python scripts/latency.py --ckpt "$CKPT" --mode cudagraph --steps "$n" \
    --config-id "e9_cudagraph_n${n}"
done
