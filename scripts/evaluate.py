"""Evaluate one checkpoint with the parallel rollout harness (docs/04 section 1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flowpolicy.eval.rollout import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("results/eval.jsonl"))
    args = parser.parse_args()

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
    )
    success = np.array([r["success"] for r in rows])
    coverage = np.array([r["coverage"] for r in rows])
    summary = {
        "config_id": args.config_id,
        "episodes": len(rows),
        "success_rate": float(success.mean()),
        "coverage_mean": float(coverage.mean()),
        "latency_mean_ms": float(np.mean([r["latency_mean_ms"] for r in rows])),
        "latency_p95_ms": float(np.mean([r["latency_p95_ms"] for r in rows])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
