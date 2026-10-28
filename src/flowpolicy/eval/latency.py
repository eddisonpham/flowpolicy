"""Batch-1 latency protocol (docs/05 section 1).

50 warmup calls, 200 timed calls, synchronize around timing, same input every
call, report mean and p95. The noise tensor is a static buffer so CUDA graph
capture does not touch the RNG.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from flowpolicy.models.policies import PolicyLike


def static_obs(policy: PolicyLike, device: str, batch: int = 1) -> dict[str, torch.Tensor]:
    to = policy.obs_horizon
    image = torch.rand(batch, to, 3, 96, 96, device=device)
    state = torch.rand(batch, to, 2, device=device) * 512.0
    return {"image": image, "state": state}


def static_noise(policy: PolicyLike, device: str, batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, policy.pred_horizon, policy.action_dim, device=device)


def _timeit(fn: Callable[[], Any], warmup: int, iters: int, cuda: bool) -> np.ndarray:
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    times = np.empty(iters)
    for i in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def benchmark(
    policy: PolicyLike,
    mode: str,
    *,
    steps: int = 10,
    warmup: int = 50,
    iters: int = 200,
    device: str = "cuda",
) -> dict[str, Any]:
    policy.eval()
    obs = static_obs(policy, device)
    noise = static_noise(policy, device)
    cuda = device.startswith("cuda")
    compile_s = 0.0
    first_call_ms = 0.0

    fn: Callable[[], Any]
    if mode == "eager":
        fn = lambda: policy.predict_chunk(obs, steps=steps, noise=noise)  # noqa: E731
    elif mode.startswith("compile-"):
        compile_mode = mode.removeprefix("compile-")
        warm = getattr(policy, "warm", None)
        if callable(warm):
            warm(steps)
        target = torch.compile(
            lambda: policy.predict_chunk(obs, steps=steps, noise=noise),
            mode=compile_mode,
            fullgraph=False,
        )
        t0 = time.perf_counter()
        target()
        if cuda:
            torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        first_call_ms = compile_s * 1000.0
        fn = target
    elif mode == "cudagraph":
        target = lambda: policy.predict_chunk(obs, steps=steps, noise=noise)  # noqa: E731
        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                target()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = target()
        assert static_out is not None
        fn = graph.replay
    else:
        raise ValueError(f"unknown mode {mode!r}")

    t0 = time.perf_counter()
    times = _timeit(fn, warmup, iters, cuda)
    elapsed = time.perf_counter() - t0
    return {
        "mode": mode,
        "steps": steps,
        "mean_ms": float(times.mean()),
        "p95_ms": float(np.percentile(times, 95)),
        "n": iters,
        "warmup": warmup,
        "compile_s": compile_s,
        "first_call_ms": first_call_ms,
        "bench_s": elapsed,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 2**20) if cuda else 0.0,
        "gpu": torch.cuda.get_device_name() if cuda else "cpu",
    }
