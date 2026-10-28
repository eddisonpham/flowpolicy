"""Fixed-rate control loop with latency injection (docs/05 section 2, M7).

ROS 2 is unavailable on this host, so this is the documented plain-Python
fallback (D7): same Policy, env and logging interfaces.

Latency is injected on the observation-to-action path as a delay in control
ticks (`ceil(latency_ms / period_ms)`), which is what the docs ask for and is
reproducible without sleeping. Inference time is measured for real on the GPU.

The loop runs in virtual time by default (no sleeps), which is what E10/E12
measured and what every one of those rows records as `virtual_time=true`. With
`paced=True` it really sleeps to the next tick, so an over-long chunk makes the
loop fall behind on the wall clock; the per-tick lateness is then reported
alongside the miss count (M9.6).
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from flowpolicy.envs.pusht import MAX_STEPS, PushTEnv
from flowpolicy.models.policies import PolicyLike


@dataclass
class LoopConfig:
    period_ms: float = 100.0
    action_horizon: int = 8
    sampler_steps: int = 10
    max_steps: int = MAX_STEPS
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    strategy: str = "naive"  # naive | temporal-ensemble | blend
    # Only for strategy="blend": how many actions to cross-fade from the old chunk into the
    # new one. 0 would be naive replacement.
    blend_steps: int = 4
    # Replan every N control steps. 0 means replan only when the queued
    # predictions run out (action_horizon). Temporal ensembling needs N <
    # action_horizon so consecutive chunks overlap; otherwise every step is
    # covered by exactly one chunk and ensembling is identical to naive.
    replan_every: int = 0
    # False = virtual time (no sleeps), which is what E10 and E12 measured. True = the loop
    # really sleeps to the next tick, so a chunk that takes longer than the period makes the
    # loop fall behind on the wall clock. M9.6 measures that because "deadline miss" in
    # virtual time is a counter, not a consequence.
    paced: bool = False


def blend_chunks(
    old: Sequence[np.ndarray], new: Sequence[np.ndarray], steps: int
) -> list[np.ndarray]:
    """Cross-fade the first `steps` actions of `new` with what was already queued.

    The weight ramps 1/steps, 2/steps, ... so the first action still leans on the old chunk
    and the last is pure new. Where the chunks agree this is a no-op; where they disagree it
    spreads the correction over a few ticks instead of taking the jump that naive
    replacement takes (E12's third arm, `docs/05` section 2).
    """
    blended = [np.asarray(action, dtype=np.float32) for action in new]
    overlap = min(
        int(steps),
        len(blended),
        len(old),
    )
    for i in range(overlap):
        weight = (i + 1) / max(int(steps), 1)
        blended[i] = (1.0 - weight) * np.asarray(old[i], dtype=np.float32) + weight * blended[i]
    return blended


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
    policy: PolicyLike,
    seed: int,
    cfg: LoopConfig,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if cfg.strategy not in {"naive", "temporal-ensemble", "blend"}:
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
        lateness_ms: list[float] = []
        deadline_misses = 0
        steps = 0
        tick_start = time.perf_counter()
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
                elif cfg.strategy == "blend":
                    # E12's third arm. A model-level "inpainting" pass would condition the new
                    # chunk's sampling on the actions already committed, which this borrowed
                    # checkpoint cannot do (it is unconditional at inference). The achievable
                    # version is a receding-horizon cross-fade, so a large disagreement costs
                    # a smooth few steps rather than naive replacement's jump.
                    queue = blend_chunks(queue, list(new), cfg.blend_steps)
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
            if cfg.paced:
                # The tick that should have started at tick_start + (steps-1) periods has
                # already started; sleep only if there is time left. Lateness is recorded
                # per tick, so a loop that never catches up shows up as a growing series
                # rather than a single "miss" count.
                # `steps` periods after the clock started, not `steps - 1`: the work of the
                # first tick (planning the first chunk) is part of the first period, so a
                # chunk that fits should not register as late.
                target = tick_start + steps * cfg.period_ms / 1000.0
                remaining = target - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    lateness_ms.append(-remaining * 1000.0)
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
        "paced": cfg.paced,
        "virtual_time": not cfg.paced,
        "late_ticks": len(lateness_ms),
        "lateness_mean_ms": float(np.mean(lateness_ms)) if lateness_ms else 0.0,
        "lateness_p95_ms": float(np.percentile(lateness_ms, 95)) if lateness_ms else 0.0,
        "lateness_max_ms": float(np.max(lateness_ms)) if lateness_ms else 0.0,
        "wall_s": float(time.perf_counter() - tick_start),
    }
