"""E14: what this software buys you, measured against the obvious baseline.

The question a reader should ask about a "deployment stack" is: *compared to what?* This
script answers it with three baselines that need no straw man, all on the same checkpoint,
the same GPU and the same episodes:

``published``
    The checkpoint's own configuration -- DDPM, 100 steps, replan every 8. This is what the
    model card reports 65.4 % for, so it is the honest "just use the checkpoint as shipped"
    reference. Its per-chunk cost is over four seconds, which is the fact that makes the
    rest of this table interesting.

``stock-ddim{4,10}``
    DDIM at a usable step count, but through ``lerobot`` -> ``diffusers`` as written: the
    sampler keeps ``alphas_cumprod`` on the host, so every step pays a host/device sync and
    the loop cannot be compiled or captured as one graph.

``fast-ddim{4,10}``
    The same DDIM math through ``policybench.models.fast_sampler`` (coefficients precomputed
    on the device, proven equal to ``diffusers`` in tests). Same policy, same numbers, but
    the loop is one graph -- so CUDA graphs and ``torch.compile`` apply.

Two kinds of row are written, and both matter:

- ``kind: latency`` -- batch-1 chunk latency (200 timed calls after 50 warmups), converted
  to a control rate and to "fraction of a 100 ms control period".
- ``kind: throughput`` -- real episodes through the parallel rollout harness, timed end to
  end, with success and coverage recorded so the cheaper arm cannot quietly be a worse
  policy.

Usage:
    python scripts/bench_stack.py --episodes 3            # the full sweep (~25 min)
    python scripts/bench_stack.py --latency-only          # just the latency ladder
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from policybench.eval.latency import benchmark
from policybench.eval.rollout import evaluate, load_policy

DEFAULT_HF_ID = "lerobot/diffusion_pusht"

# (arm label, sampler, benchmark mode)
LATENCY_ARMS: list[tuple[str, str, str]] = [
    ("stock-eager", "stock", "eager"),
    ("fast-eager", "fast", "eager"),
    ("fast-cudagraph", "fast", "cudagraph"),
    ("fast-compile-default", "fast", "compile-default"),
    ("fast-compile-reduce-overhead", "fast", "compile-reduce-overhead"),
]

# (arm label, scheduler, steps, sampler, workers). Episode count comes from the CLI and is
# the same for every arm: a cheaper arm must not also get fewer, easier episodes.
THROUGHPUT_ARMS: list[tuple[str, str, int, str, int]] = [
    ("published", "ddpm", 100, "stock", 1),
    ("stock-ddim10", "ddim", 10, "stock", 1),
    ("fast-ddim10", "ddim", 10, "fast", 1),
    ("fast-ddim4", "ddim", 4, "fast", 1),
    ("fast-ddim4-2w", "ddim", 4, "fast", 2),
]


def _stamp(**row: Any) -> dict[str, Any]:
    return {"timestamp": datetime.now(UTC).isoformat(), "gpu": torch.cuda.get_device_name(), **row}


def latency_rows(
    checkpoint: str, steps_list: list[int], iters: int, warmup: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for steps in steps_list:
        for arm, sampler, mode in LATENCY_ARMS:
            load_start = time.perf_counter()
            policy, meta = load_policy(checkpoint, scheduler="ddim", sampler=sampler)
            load_s = time.perf_counter() - load_start
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            result = benchmark(policy, mode, steps=steps, warmup=warmup, iters=iters)
            mean_ms = float(result["mean_ms"])
            rows.append(
                _stamp(
                    kind="latency",
                    arm=arm,
                    sampler=sampler,
                    mode=mode,
                    scheduler="ddim",
                    steps=steps,
                    mean_ms=mean_ms,
                    p95_ms=float(result["p95_ms"]),
                    compile_s=float(result["compile_s"]),
                    control_hz=1000.0 / mean_ms if mean_ms > 0 else 0.0,
                    period_frac=mean_ms / 100.0,
                    fits_100ms=bool(mean_ms < 100.0),
                    peak_vram_mb=float(result["peak_vram_mb"]),
                    load_s=load_s,
                    params=int(meta["params"]),
                    checkpoint=str(checkpoint),
                )
            )
            print(
                f"[e14] {arm:>28s} N={steps:<3d} {mean_ms:8.1f} ms  "
                f"{1000.0 / mean_ms:5.2f} Hz  {'fits 100ms' if mean_ms < 100 else 'MISSES 100ms'}",
                flush=True,
            )
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def throughput_rows(
    checkpoint: str, episodes: int, seed_start: int, scratch: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = list(range(seed_start, seed_start + episodes))
    for arm, scheduler, steps, sampler, workers in THROUGHPUT_ARMS:
        wall_start = time.perf_counter()
        results = evaluate(
            checkpoint,
            seeds,
            train_seeds=[],
            config_id=f"e14_{arm}",
            out_path=scratch,
            sampler_steps=steps,
            workers=workers,
            scheduler=scheduler,
            sampler=sampler,
        )
        wall_s = time.perf_counter() - wall_start
        success = np.array([r["success"] for r in results], dtype=float)
        coverage = np.array([r["coverage"] for r in results], dtype=float)
        episode_wall = mean(float(r["wall_s"]) for r in results)
        chunk_ms = mean(float(r["latency_mean_ms"]) for r in results)
        chunks = mean(float(r["chunks"]) for r in results)
        # Episodes end early on success, so raw s/episode mixes cost with policy quality.
        # Cost per control step does not: it is the same quantity for a fast failure and a
        # quick success, and it is measured (episode wall time / steps actually taken).
        s_per_100 = 100.0 * mean(float(r["wall_s"]) / max(float(r["steps"]), 1.0) for r in results)
        rows.append(
            _stamp(
                kind="throughput",
                arm=arm,
                sampler=sampler,
                scheduler=scheduler,
                steps=steps,
                workers=workers,
                episodes=len(results),
                wall_s=wall_s,
                episode_wall_s=episode_wall,
                episodes_per_hour=3600.0 / episode_wall if episode_wall > 0 else 0.0,
                s_per_100_steps=s_per_100,
                chunk_ms=chunk_ms,
                chunks_per_episode=chunks,
                success_rate=float(success.mean()),
                coverage_mean=float(coverage.mean()),
                checkpoint=str(checkpoint),
            )
        )
        print(
            f"[e14] {arm:>16s} N={steps:<3d} w={workers}  {episode_wall:7.1f} s/episode  "
            f"{3600.0 / episode_wall:6.1f} episodes/h  {s_per_100:6.2f} s/100 steps  "
            f"success {success.mean():.0%}  coverage {coverage.mean():.3f}",
            flush=True,
        )
    return rows


def equivalence_rows(
    checkpoint: str, episodes: int, seed_start: int, scratch: Path
) -> list[dict[str, Any]]:
    """Same episodes, same policy, two DDIM loops: does the fast sampler change behaviour?

    The unit test proves it is the same *arithmetic*; this proves it is the same *policy*, on
    real rollouts, with the same seeds. A sampler that is subtly wrong (a mis-indexed schedule,
    a missing clip) would show up here as a coverage gap, not as a crash.
    """
    seeds = list(range(seed_start, seed_start + episodes))
    rows: list[dict[str, Any]] = []
    per_arm: dict[str, list[dict[str, Any]]] = {}
    for arm, sampler in (("stock-ddim10", "stock"), ("fast-ddim10", "fast")):
        results = evaluate(
            checkpoint,
            seeds,
            train_seeds=[],
            config_id=f"e14_equiv_{arm}",
            out_path=scratch,
            sampler_steps=10,
            workers=1,
            scheduler="ddim",
            sampler=sampler,
        )
        per_arm[arm] = results
        success = np.array([r["success"] for r in results], dtype=float)
        coverage = np.array([r["coverage"] for r in results], dtype=float)
        rows.append(
            _stamp(
                kind="equivalence",
                arm=arm,
                sampler=sampler,
                steps=10,
                workers=1,
                episodes=len(results),
                success_rate=float(success.mean()),
                coverage_mean=float(coverage.mean()),
                coverage_min=float(coverage.min()),
                chunk_ms=float(np.mean([r["latency_mean_ms"] for r in results])),
                checkpoint=str(checkpoint),
            )
        )
        print(
            f"[e14] equiv {arm:>12s} {len(results)} paired episodes: "
            f"success {success.mean():.0%} coverage {coverage.mean():.3f}",
            flush=True,
        )
    stock = per_arm["stock-ddim10"]
    fast = per_arm["fast-ddim10"]
    agree = sum(int(a["success"]) == int(b["success"]) for a, b in zip(stock, fast, strict=True))
    rows[0]["episodes_agreeing"] = agree
    rows[0]["coverage_max_abs_diff"] = float(
        max(abs(a["coverage"] - b["coverage"]) for a, b in zip(stock, fast, strict=True))
    )
    print(
        f"[e14] equivalence: {agree}/{len(stock)} episodes agree on success, "
        f"max coverage difference {rows[0]['coverage_max_abs_diff']:.3f}",
        flush=True,
    )
    return rows


def prune(path: Path, kinds: set[str]) -> None:
    """Drop rows of the kinds this run is about to rewrite, so re-runs are idempotent."""
    if not path.exists():
        return
    kept = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("kind") not in kinds
    ]
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", default=DEFAULT_HF_ID)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("results/stack.jsonl"))
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path("results/stack_episodes.jsonl"),
        help="per-episode rows for the throughput arms (not used by any table)",
    )
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--throughput-only", action="store_true")
    parser.add_argument(
        "--equivalence",
        type=int,
        default=0,
        metavar="EPISODES",
        help="run only the stock-vs-fast sampler equivalence check on N paired episodes",
    )
    args = parser.parse_args()

    checkpoint = args.hf_id if args.hf_id.startswith("hf:") else f"hf:{args.hf_id}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.equivalence:
        rows = equivalence_rows(checkpoint, args.equivalence, args.seed_start, args.scratch)
        prune(args.out, {"equivalence"})
    else:
        rows = []
        kinds = set()
        if not args.throughput_only:
            kinds.add("latency")
        if not args.latency_only:
            kinds.add("throughput")
        prune(args.out, kinds)
        if not args.throughput_only:
            rows += latency_rows(checkpoint, args.steps, args.iters, args.warmup)
        if not args.latency_only:
            rows += throughput_rows(checkpoint, args.episodes, args.seed_start, args.scratch)

    with args.out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\n[e14] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
