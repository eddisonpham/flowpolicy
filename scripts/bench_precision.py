"""E8: does half precision buy a control period, and what does it cost the policy?

The deployment envelope is set by one number: at 262.7 M parameters and batch 1, a chunk
costs ~73 ms of fp32 compute at N=4 and ~180 ms at N=10 (E9/E14), so only N <= 4 fits a
100 ms control period. Precision is the remaining lever that nobody has pulled on this
checkpoint: bf16 and fp16 halve the weight traffic and engage the tensor cores, and E14
already showed that launch overhead is not where the time goes.

This script measures both halves of that trade, because either half alone is a bad result:

- ``kind: latency`` -- batch-1 chunk cost per dtype, step count and mode (eager, CUDA
  graph), plus peak VRAM. Same protocol as E9 (50 warmups, 200 timed calls, sync around
  the timer, one process, nothing else on the GPU).
- ``kind: episode`` -- the same policy, in a *different* dtype, on the same paired eval
  seeds, so a speed win cannot be a quietly worse policy. Half precision in a diffusion
  loop is exactly the kind of change that produces plausible-looking actions: fp16's
  exponent range is small and the sampler's ``x0`` division amplifies error, so the
  correctness check is a real check.

Honest resolution limit, stated before the numbers: 40 paired episodes per arm resolves a
gross regression (tens of percentage points) and nothing at the 2 pp level the pre-registered
E8 hypothesis named. A null result here is "no gross regression", not "equal to fp32".

Usage:
    python scripts/bench_precision.py                # latency + 40 paired episodes per dtype
    python scripts/bench_precision.py --latency-only  # just the ladder
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from policybench.eval.latency import benchmark
from policybench.eval.metrics import bootstrap_ci
from policybench.eval.rollout import evaluate, load_policy

DEFAULT_HF_ID = "lerobot/diffusion_pusht"
DTYPES = ("float32", "bfloat16", "float16")
MODES = ("eager", "cudagraph")


def _stamp(**row: Any) -> dict[str, Any]:
    return {"timestamp": datetime.now(UTC).isoformat(), "gpu": torch.cuda.get_device_name(), **row}


def latency_rows(
    checkpoint: str, dtypes: list[str], steps_list: list[int], modes: list[str], iters: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        for steps in steps_list:
            for mode in modes:
                policy, meta = load_policy(
                    checkpoint, scheduler="ddim", sampler="fast", dtype=dtype
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                result = benchmark(policy, mode, steps=steps, warmup=50, iters=iters)
                mean_ms = float(result["mean_ms"])
                rows.append(
                    _stamp(
                        kind="latency",
                        dtype=dtype,
                        mode=mode,
                        steps=steps,
                        sampler="fast",
                        mean_ms=mean_ms,
                        p95_ms=float(result["p95_ms"]),
                        compile_s=float(result["compile_s"]),
                        peak_vram_mb=float(result["peak_vram_mb"]),
                        period_frac=mean_ms / 100.0,
                        fits_100ms=bool(mean_ms < 100.0),
                        params=int(meta["params"]),
                        checkpoint=str(checkpoint),
                    )
                )
                fits = "fits 100ms" if mean_ms < 100 else "misses 100ms"
                print(
                    f"[e8] {dtype:>9s} {mode:>9s} N={steps:<3d} {mean_ms:8.1f} ms  "
                    f"{1000.0 / mean_ms:5.2f} Hz  {fits}",
                    flush=True,
                )
                del policy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return rows


def episode_rows(
    checkpoint: str, dtypes: list[str], steps: int, episodes: int, seed_start: int, scratch: Path
) -> list[dict[str, Any]]:
    seeds = list(range(seed_start, seed_start + episodes))
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        start = time.perf_counter()
        results = evaluate(
            checkpoint,
            seeds,
            train_seeds=[],
            config_id=f"e8_{dtype}_n{steps}",
            out_path=scratch,
            sampler_steps=steps,
            workers=1,
            scheduler="ddim",
            sampler="fast",
            dtype=dtype,
        )
        wall_s = time.perf_counter() - start
        point, lo, hi = bootstrap_ci(results)
        rows.append(
            _stamp(
                kind="episode",
                dtype=dtype,
                steps=steps,
                episodes=len(results),
                success_rate=float(point),
                success_lo=float(lo),
                success_hi=float(hi),
                coverage_mean=float(np.mean([r["coverage"] for r in results])),
                chunk_ms=float(np.mean([r["latency_mean_ms"] for r in results])),
                wall_s=wall_s,
                seed_start=seed_start,
                checkpoint=str(checkpoint),
            )
        )
        print(
            f"[e8] {dtype:>9s} N={steps} {len(results)} paired episodes: "
            f"success {point:.1%} [{lo:.1%}, {hi:.1%}]  coverage "
            f"{rows[-1]['coverage_mean']:.3f}  {rows[-1]['chunk_ms']:.1f} ms/chunk",
            flush=True,
        )
    return rows


def prune(path: Path, kinds: set[str]) -> None:
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
    parser.add_argument("--dtypes", nargs="+", default=list(DTYPES))
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--episode-steps", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--out", type=Path, default=Path("results/precision.jsonl"))
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path("results/precision_episodes.jsonl"),
        help="per-episode rows (not used by any table)",
    )
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--episodes-only", action="store_true")
    args = parser.parse_args()

    checkpoint = args.hf_id if args.hf_id.startswith("hf:") else f"hf:{args.hf_id}"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kinds = set()
    if not args.episodes_only:
        kinds.add("latency")
    if not args.latency_only:
        kinds.add("episode")
    prune(args.out, kinds)

    rows: list[dict[str, Any]] = []
    if not args.episodes_only:
        rows += latency_rows(checkpoint, args.dtypes, args.steps, args.modes, args.iters)
    if not args.latency_only:
        rows += episode_rows(
            checkpoint,
            args.dtypes,
            args.episode_steps,
            args.episodes,
            args.seed_start,
            args.scratch,
        )

    with args.out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\n[e8] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
