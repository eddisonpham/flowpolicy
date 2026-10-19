"""Parallel rollout harness (docs/04 section 1).

One JSON line per episode. Eval seeds are environment seeds and must be
disjoint from training seeds; the harness asserts it rather than trusting the
caller. Each worker loads the checkpoint once and runs its assigned episodes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flowpolicy.config import ModelConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.envs.pusht import MAX_STEPS, PushTEnv
from flowpolicy.eval.metrics import assert_seed_disjoint
from flowpolicy.models.policies import BasePolicy, build_policy

__all__ = ["assert_seed_disjoint", "evaluate", "load_policy", "rollout_episode"]


def load_policy(
    checkpoint: str | Path, device: str = "cuda", use_ema: bool = True
) -> tuple[BasePolicy, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    stats = Stats(**payload["stats"])
    model = ModelConfig(**cfg["model"])
    policy = build_policy(
        cfg["policy"]["kind"],
        stats,
        cfg["data"]["obs_horizon"],
        cfg["data"]["pred_horizon"],
        cfg["data"]["state_dim"],
        cfg["data"]["action_dim"],
        model,
        ddpm_train_steps=cfg["policy"]["ddpm_train_steps"],
        flow_time_sampling=cfg["policy"]["flow_time_sampling"],
    )
    policy.load_state_dict(payload["ema"] if use_ema else payload["model"])
    policy.to(device).eval()
    meta = {
        "policy": cfg["policy"]["kind"],
        "train_seed": int(cfg["seed"]),
        "checkpoint": str(checkpoint),
        "step": int(payload["step"]),
        "params": policy.num_parameters(),
    }
    return policy, meta


def _obs_batch(
    frames: deque[np.ndarray], states: deque[np.ndarray], device: str
) -> dict[str, torch.Tensor]:
    image = np.stack(list(frames))  # [To,H,W,3] uint8
    state = np.stack(states).astype(np.float32)
    return {
        "image": torch.from_numpy(image).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        / 255.0,
        "state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def rollout_episode(
    policy: BasePolicy,
    env_seed: int,
    *,
    sampler_steps: int = 10,
    action_horizon: int = 8,
    device: str = "cuda",
    max_steps: int = MAX_STEPS,
    record_latency: bool = True,
) -> dict[str, float]:
    obs_horizon = policy.obs_horizon
    env = PushTEnv()
    try:
        obs = env.reset(seed=env_seed)
        frames = deque([obs["image"]] * obs_horizon, maxlen=obs_horizon)
        states = deque([obs["state"]] * obs_horizon, maxlen=obs_horizon)
        queue: list[np.ndarray] = []
        latencies: list[float] = []
        steps = 0
        coverage = 0.0
        success = 0.0
        while steps < max_steps:
            if not queue:
                batch = _obs_batch(frames, states, device)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                chunk = policy.predict_chunk(batch, steps=sampler_steps)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                if record_latency:
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                queue = list(chunk[0, :action_horizon].float().cpu().numpy())
            action = np.clip(queue.pop(0), 0.0, 512.0)
            obs, _reward, done, info = env.step(action)
            frames.append(obs["image"])
            states.append(obs["state"])
            steps += 1
            coverage = max(coverage, info["coverage"])
            if info["success"] > 0:
                success = 1.0
            if done:
                break
    finally:
        env.close()
    lat = np.asarray(latencies) if latencies else np.zeros(1)
    return {
        "success": success,
        "coverage": coverage,
        "steps": float(steps),
        "latency_mean_ms": float(lat.mean()),
        "latency_p95_ms": float(np.percentile(lat, 95)),
    }


_WORKER: dict[str, Any] = {}


def _init_worker(checkpoint: str, device: str, sampler_steps: int, action_horizon: int) -> None:
    torch.set_num_threads(1)
    policy, meta = load_policy(checkpoint, device=device)
    _WORKER.update(
        policy=policy, meta=meta, device=device, steps=sampler_steps, horizon=action_horizon
    )


def _run_episodes(seeds: list[int]) -> list[dict[str, Any]]:
    torch.manual_seed(seeds[0])
    out = []
    for seed in seeds:
        result = rollout_episode(
            _WORKER["policy"],
            seed,
            sampler_steps=_WORKER["steps"],
            action_horizon=_WORKER["horizon"],
            device=_WORKER["device"],
        )
        out.append({"eval_seed": seed, **result, **_WORKER["meta"]})
    return out


def evaluate(
    checkpoint: str | Path,
    eval_seeds: list[int],
    *,
    train_seeds: list[int],
    config_id: str,
    out_path: str | Path,
    sampler_steps: int = 10,
    action_horizon: int = 8,
    workers: int = 4,
    device: str = "cuda",
    use_ema: bool = True,
) -> list[dict[str, Any]]:
    assert_seed_disjoint(train_seeds, eval_seeds)
    checkpoint = str(checkpoint)
    if not use_ema:
        raise ValueError("evaluation always uses EMA weights (docs/03 section 6)")
    chunks = [eval_seeds[i::workers] for i in range(workers)]
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(checkpoint, device, sampler_steps, action_horizon),
    ) as pool:
        for part in pool.map(_run_episodes, [c for c in chunks if c]):
            results.extend(part)
    for row in results:
        row["config_id"] = config_id
        row["task"] = "pusht"
        row["sampler_steps"] = sampler_steps
        row["action_horizon"] = action_horizon
        row["weights"] = "ema"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")
    return results
