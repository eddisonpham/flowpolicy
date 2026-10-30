"""Evaluate one checkpoint with the parallel rollout harness (docs/04 section 1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policybench.eval.rollout import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    # Not type=Path: `hf:<repo_id>` is a pretrained checkpoint, not a path.
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--train-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--scheduler", default="ddim", choices=["ddim", "ddpm"], help="pretrained only"
    )
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="pretrained only, and only with --sampler fast (E8)",
    )
    parser.add_argument(
        "--normalization",
        default="buffers",
        choices=["buffers", "processors"],
        help="pretrained diffusion only: whose normalization arithmetic (docs/10 option H)",
    )
    parser.add_argument(
        "--noise-seed-offset",
        type=int,
        default=0,
        help="shift the policy's per-episode noise seed, leaving the env seed unchanged",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("results/eval.jsonl"))
    args = parser.parse_args()
    if not str(args.ckpt).startswith("hf:") and not args.train_seeds:
        parser.error("--train-seeds is required for trained checkpoints")

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    rows = evaluate(
        args.ckpt,
        seeds,
        train_seeds=args.train_seeds,
        config_id=args.config_id,
        out_path=args.out,
        sampler_steps=args.steps,
        action_horizon=args.action_horizon,
        workers=args.workers,
        device=args.device,
        scheduler=args.scheduler,
        dtype=args.dtype,
        noise_seed_offset=args.noise_seed_offset,
        normalization=args.normalization,
    )
    success = np.array([r["success"] for r in rows])
    coverage = np.array([r["coverage"] for r in rows])
    summary = {
        "config_id": args.config_id,
        "episodes": len(rows),
        "noise_seed_offset": args.noise_seed_offset,
        "normalization": args.normalization,
        "success_rate": float(success.mean()),
        "coverage_mean": float(coverage.mean()),
        "latency_mean_ms": float(np.mean([r["latency_mean_ms"] for r in rows])),
        "latency_p95_ms": float(np.mean([r["latency_p95_ms"] for r in rows])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
