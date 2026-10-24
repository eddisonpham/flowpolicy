"""Fixed-rate control loop with latency injection (docs/05 section 2, M7).

ROS 2 is unavailable on this host, so this is the documented plain-Python
fallback (D7): same Policy, env and logging interfaces.

Latency is injected on the observation-to-action path as a delay in control
ticks (`ceil(latency_ms / period_ms)`), which is what the docs ask for and is
reproducible without sleeping. Inference time is measured for real on the GPU;
the loop itself runs in virtual time (no wall-clock sleeps), which is recorded
in every result row as `virtual_time=true`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from flowpolicy.envs.pusht import MAX_STEPS, PushTEnv
from flowpolicy.models.policies import BasePolicy


@dataclass
class LoopConfig:
    period_ms: float = 100.0
    action_horizon: int = 8
    sampler_steps: int = 10
    max_steps: int = MAX_STEPS
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    strategy: str = "naive"  # naive | temporal-ensemble
    # Replan every N control steps. 0 means replan only when the queued
    # predictions run out (action_horizon). Temporal ensembling needs N <
    # action_horizon so consecutive chunks overlap; otherwise every step is
    # covered by exactly one chunk and ensembling is identical to naive.
    replan_every: int = 0


def _obs_batch(
    frames: deque[np.ndarray], states: deque[np.ndarray], device: str
) -> dict[str, torch.Tensor]:
    image = np.stack(frames)
    state = np.stack(states).astype(np.float32)
    return {
        "image": torch.from_numpy(image).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        / 255.0,
        "state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def run_episode(
    policy: BasePolicy,
    seed: int,
    cfg: LoopConfig,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if cfg.strategy not in {"naive", "temporal-ensemble"}:
        raise ValueError(f"unknown chunk strategy {cfg.strategy!r}")
    rng = rng or np.random.default_rng(seed)
    # Same per-episode seeding as the eval harness: the chunk sampler must not
    # depend on the global RNG state left over from earlier episodes.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = PushTEnv()
    try:
        obs = env.reset(seed=seed)
        # Delayed observation buffer: the loop consumes the observation that
        # was current `delay + jitter` ticks ago.
        base_delay = math.ceil(cfg.latency_ms / cfg.period_ms)
        jitter_steps = math.ceil(cfg.jitter_ms / cfg.period_ms)
        delay = base_delay + int(rng.integers(0, jitter_steps + 1))
        buffer: deque[dict[str, np.ndarray]] = deque([obs] * (delay + 1), maxlen=delay + 1)

        frames: deque[np.ndarray] = deque(
            [obs["image"]] * policy.obs_horizon, maxlen=policy.obs_horizon
        )
        states: deque[np.ndarray] = deque(
            [obs["state"]] * policy.obs_horizon, maxlen=policy.obs_horizon
        )
        queue: list[np.ndarray] = []
        # Temporal ensembling: every chunk contributes a prediction for the
        # absolute control step it targets; the action executed at step t is the
        # mean over all chunks that predicted t (ACT, Zhao et al. 2023).
        ensemble_sums: dict[int, np.ndarray] = {}
        ensemble_counts: dict[int, int] = {}
        # Warm up outside the timed region (docs/05: latency is measured warmed
        # up). Without this, the first chunk includes CUDA context and kernel
        # load and biases the mean.
        warm = _obs_batch(frames, states, device)
        policy.predict_chunk(warm, steps=cfg.sampler_steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_ms: list[float] = []
        deadline_misses = 0
        steps = 0
        coverage = 0.0
        success = 0.0
        actions_returned = 0
        while steps < cfg.max_steps:
            delayed = buffer[0]
            frames.append(delayed["image"])
            states.append(delayed["state"])
            needs_chunk = (
                steps not in ensemble_sums if cfg.strategy == "temporal-ensemble" else not queue
            )
            if cfg.replan_every and steps % cfg.replan_every == 0:
                needs_chunk = True
            if needs_chunk:
                batch = _obs_batch(frames, states, device)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                chunk = policy.predict_chunk(batch, steps=cfg.sampler_steps)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                inference_ms.append(elapsed_ms)
                if elapsed_ms > cfg.period_ms:
                    deadline_misses += 1
                new = chunk[0, : cfg.action_horizon].float().cpu().numpy()
                if cfg.strategy == "temporal-ensemble":
                    for i, value in enumerate(new):
                        key = steps + i
                        ensemble_sums[key] = ensemble_sums.get(key, np.zeros_like(value)) + value
                        ensemble_counts[key] = ensemble_counts.get(key, 0) + 1
                else:
                    queue = list(new)
                actions_returned += 1
            if cfg.strategy == "temporal-ensemble":
                action = (ensemble_sums.pop(steps) / ensemble_counts.pop(steps)).astype(np.float32)
                for key in [k for k in ensemble_sums if k < steps]:
                    ensemble_sums.pop(key)
                    ensemble_counts.pop(key)
            else:
                action = queue.pop(0)
            action = np.clip(action, 0.0, 512.0)
            obs, _reward, done, info = env.step(action)
            buffer.append(obs)
            steps += 1
            coverage = max(coverage, info["coverage"])
            success = max(success, info["success"])
            if done:
                break
    finally:
        env.close()
    lat = np.asarray(inference_ms) if inference_ms else np.zeros(1)
    return {
        "success": float(success),
        "coverage": float(coverage),
        "steps": steps,
        "chunks": actions_returned,
        "deadline_misses": deadline_misses,
        "inference_mean_ms": float(lat.mean()),
        "inference_p95_ms": float(np.percentile(lat, 95)),
        "inference_samples_ms": [float(v) for v in lat.tolist()],
        "latency_ms": cfg.latency_ms,
        "jitter_ms": cfg.jitter_ms,
        "delay_steps": delay,
        "period_ms": cfg.period_ms,
        "strategy": cfg.strategy,
        "virtual_time": True,
    }
