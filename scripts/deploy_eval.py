"""E10: fixed-rate loop with injected observation latency (docs/04 section 2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flowpolicy.deploy.control_loop import LoopConfig, run_episode
from flowpolicy.eval.rollout import assert_seed_disjoint, load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--latencies", type=float, nargs="+", default=[0, 50, 100, 150, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001, 10002])
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--jitter-ms", type=float, default=20.0)
    parser.add_argument("--strategy", default="naive", choices=["naive", "temporal-ensemble"])
    parser.add_argument("--period-ms", type=float, default=100.0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("results/deploy.jsonl"))
    args = parser.parse_args()

    assert_seed_disjoint(args.train_seeds, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for ckpt in args.ckpts:
            policy, meta = load_policy(ckpt, device=args.device)
            for latency in args.latencies:
                for seed in args.seeds:
                    for episode in range(args.episodes_per_seed):
                        env_seed = seed + episode
                        cfg = LoopConfig(
                            latency_ms=latency,
                            jitter_ms=args.jitter_ms,
                            strategy=args.strategy,
                            period_ms=args.period_ms,
                            action_horizon=args.action_horizon,
                            sampler_steps=args.steps,
                        )
                        rng = np.random.default_rng([env_seed, int(latency), episode])
                        row = run_episode(policy, env_seed, cfg, device=args.device, rng=rng)
                        row.update(
                            {
                                "config_id": f"e10_{args.strategy}_l{int(latency)}",
                                "policy": meta["policy"],
                                "train_seed": meta["train_seed"],
                                "eval_seed": env_seed,
                                "episode": episode,
                                "checkpoint": str(ckpt),
                                "task": "pusht",
                            }
                        )
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
            del policy
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
