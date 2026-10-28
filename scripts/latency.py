"""Latency ladder (E9): eager, compile modes, CUDA graph, and N sweep."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from flowpolicy.eval.latency import benchmark
from flowpolicy.eval.rollout import load_policy


def _warn_if_no_msvc() -> None:
    """Inductor needs `cl.exe` only when a graph has a CPU partition.

    The stock lerobot path has one (diffusers keeps `alphas_cumprod` on the host); the fast
    sampler does not. Saying so up front beats a 40-line C++ traceback.
    """
    if shutil.which("cl") is None:
        print(
            "[latency] cl.exe is not on PATH; if this compile run fails with "
            "InvalidCxxCompiler it needs MSVC (docs/05 section 1)",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    # Not type=Path: `hf:<repo_id>` is a pretrained checkpoint, not a path.
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "eager",
            "compile-default",
            "compile-reduce-overhead",
            "compile-max-autotune",
            "cudagraph",
        ],
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--scheduler", default="ddim", choices=["ddim", "ddpm"], help="pretrained only"
    )
    parser.add_argument(
        "--sampler",
        default="stock",
        choices=["stock", "fast"],
        help="pretrained DDIM loop: lerobot's (stock) or ours (fast, compilable)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="pretrained only; half precision needs --sampler fast (E8)",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("results/latency.jsonl"))
    parser.add_argument("--config-id", default="e9")
    args = parser.parse_args()

    policy, meta = load_policy(
        args.ckpt,
        device=args.device,
        use_ema=True,
        scheduler=args.scheduler,
        sampler=args.sampler,
        dtype=args.dtype,
    )
    if args.mode.startswith("compile-"):
        _warn_if_no_msvc()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    row = benchmark(
        policy,
        args.mode,
        steps=args.steps,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )
    row.update(
        {
            "config_id": args.config_id,
            "checkpoint": str(args.ckpt),
            "policy": meta["policy"],
            "train_seed": meta["train_seed"],
            "params": meta["params"],
            "dtype": meta.get("dtype", "float32"),
            "timestamp": datetime.now(UTC).isoformat(),
            "driver": torch.version.cuda,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
