"""Equivalence + benchmark for the fused Euler/de-normalization Triton kernel.

Pass criteria (docs/05 section 1): fp32 atol 1e-5 against the PyTorch
reference. The benchmark is honest: it reports the batch-1 case even if the
kernel loses to eager or compiled PyTorch, and writes results/kernel.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from flowpolicy.eval.rollout import load_policy
from flowpolicy.kernels.fused_euler import HAS_TRITON, fused_euler_denorm, reference_euler_denorm


def main() -> None:
    if not HAS_TRITON:
        raise SystemExit("triton is not installed; run on the Windows CUDA host")
    from triton.testing import do_bench

    policy, meta = load_policy("checkpoints/e1_flow_s0/last.pt", device="cuda")
    scale = (policy.normalizer.action_max - policy.normalizer.action_min) / 2.0
    bias = (policy.normalizer.action_max + policy.normalizer.action_min) / 2.0
    rows = []
    for batch in (1, 64):
        x = torch.randn(batch, policy.pred_horizon, policy.action_dim, device="cuda")
        v = torch.randn_like(x)
        dt = 1.0 / 10.0
        expected = reference_euler_denorm(x, v, dt, scale, bias)
        actual = fused_euler_denorm(x, v, dt, scale, bias)
        max_diff = float((actual - expected).abs().max())
        ok = bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))
        eager_ms = float(
            do_bench(lambda x=x, v=v, dt=dt: reference_euler_denorm(x, v, dt, scale, bias))
        )
        compiled_fn = torch.compile(
            lambda x=x, v=v, dt=dt: reference_euler_denorm(x, v, dt, scale, bias)
        )
        compiled_fn()
        compiled_ms = float(do_bench(compiled_fn))
        triton_ms = float(
            do_bench(lambda x=x, v=v, dt=dt: fused_euler_denorm(x, v, dt, scale, bias))
        )
        row = {
            "kernel": "fused_euler_denorm",
            "batch": batch,
            "elements": int(x.numel()),
            "equivalent": ok,
            "max_abs_diff": max_diff,
            "eager_ms": eager_ms,
            "compile_ms": compiled_ms,
            "triton_ms": triton_ms,
            "speedup_vs_eager": eager_ms / triton_ms,
            "speedup_vs_compile": compiled_ms / triton_ms,
            "policy": meta["policy"],
            "checkpoint": meta["checkpoint"],
        }
        rows.append(row)
        print(json.dumps(row))
    out = Path("results/kernel.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    if not all(row["equivalent"] for row in rows):
        raise SystemExit("FAIL: kernel does not match the reference")
    print("PASS: kernel equivalent, results/kernel.jsonl")


if __name__ == "__main__":
    main()
