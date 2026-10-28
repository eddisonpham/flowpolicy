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
    # Not type=Path: `hf:<repo_id>` is a pretrained checkpoint, not a path.
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--latencies", type=float, nargs="+", default=[0, 50, 100, 150, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001, 10002])
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--jitter-ms", type=float, default=20.0)
    # "-r2" variants replan every 2 control steps instead of every action_horizon
    # steps; temporal ensembling needs that overlap to differ from naive (E12).
    parser.add_argument(
        "--strategy",
        default="naive",
        choices=[
            "naive",
            "naive-r2",
            "temporal-ensemble",
            "temporal-ensemble-r2",
            "blend",
            "blend-r2",
        ],
    )
    parser.add_argument("--period-ms", type=float, default=100.0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--scheduler", default="ddim", choices=["ddim", "ddpm"], help="pretrained only"
    )
    parser.add_argument(
        "--paced",
        action="store_true",
        help="sleep to the control tick instead of running in virtual time (M9.6): turns the "
        "deadline-miss counter into measured wall-clock lateness",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="pretrained only, needs the fast sampler (E8)",
    )
    parser.add_argument(
        "--sampler",
        default="fast",
        choices=["stock", "fast"],
        help="pretrained DDIM loop; E10/E12 rows used fast, E0/E3 used stock",
    )
    parser.add_argument("--train-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("results/deploy.jsonl"))
    args = parser.parse_args()

    if not all(str(c).startswith("hf:") for c in args.ckpts) and not args.train_seeds:
        parser.error("--train-seeds is required for trained checkpoints")
    assert_seed_disjoint(args.train_seeds, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for ckpt in args.ckpts:
            policy, meta = load_policy(
                ckpt,
                device=args.device,
                scheduler=args.scheduler,
                sampler=args.sampler,
                dtype=args.dtype,
            )
            for latency in args.latencies:
                for seed in args.seeds:
                    for episode in range(args.episodes_per_seed):
                        env_seed = seed + episode
                        replan_every = 2 if args.strategy.endswith("-r2") else 0
                        cfg = LoopConfig(
                            latency_ms=latency,
                            jitter_ms=args.jitter_ms,
                            strategy=args.strategy.removesuffix("-r2"),
                            replan_every=replan_every,
                            period_ms=args.period_ms,
                            action_horizon=args.action_horizon,
                            sampler_steps=args.steps,
                            paced=args.paced,
                        )
                        rng = np.random.default_rng([env_seed, int(latency), episode])
                        row = run_episode(policy, env_seed, cfg, device=args.device, rng=rng)
                        row.update(
                            {
                                "config_id": f"e10_{args.strategy}_l{int(latency)}",
                                "strategy": args.strategy,
                                "replan_every": replan_every,
                                "policy": meta["policy"],
                                "train_seed": meta["train_seed"],
                                "eval_seed": env_seed,
                                "episode": episode,
                                "checkpoint": str(ckpt),
                                "params": meta["params"],
                                "scheduler": args.scheduler,
                                "sampler": args.sampler,
                                "dtype": args.dtype,
                                # `steps` in the row is the environment steps the episode took;
                                # the sampler step count is a different quantity and has to be
                                # recorded under its own name (it was missing before M9.6).
                                "sampler_steps": args.steps,
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
