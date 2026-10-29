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

from policybench.config import ModelConfig
from policybench.data.normalizer import Stats
from policybench.envs.pusht import MAX_STEPS, PushTEnv
from policybench.eval.metrics import assert_seed_disjoint
from policybench.models.policies import PolicyLike, build_policy
from policybench.models.pretrained import PretrainedDiffusionPolicy

__all__ = ["assert_seed_disjoint", "evaluate", "load_policy", "rollout_episode"]


def load_policy(
    checkpoint: str | Path,
    device: str = "cuda",
    use_ema: bool = True,
    *,
    scheduler: str = "ddim",
    sampler: str = "stock",
    dtype: str = "float32",
) -> tuple[PolicyLike, dict[str, Any]]:
    """Load a trained checkpoint (`path/last.pt`) or an HF one (`hf:<repo_id>`).

    Pretrained rows carry `train_seed = -1`: there is no training seed to resample, so
    their intervals are episode-level (docs/04 section 1). `sampler` selects the DDIM loop
    for the pretrained path ("stock" = lerobot's, "fast" = ours, which compiles and
    captures); it has no effect on trained checkpoints.
    """
    if str(checkpoint).startswith("hf:"):
        hf_id = str(checkpoint).removeprefix("hf:")
        pretrained = PretrainedDiffusionPolicy(
            hf_id, device=device, scheduler=scheduler, sampler=sampler, dtype=dtype
        )
        pretrained.eval()
        return pretrained, {
            "policy": "pretrained-dp",
            "source": hf_id,
            "train_seed": -1,
            "checkpoint": f"hf:{hf_id}",
            "step": pretrained.pretrained_step,
            "params": pretrained.num_parameters(),
            "weights": "pretrained",
            "scheduler": scheduler,
            "sampler": sampler,
            "dtype": dtype,
        }
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
        "weights": "ema",
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
    policy: PolicyLike,
    env_seed: int,
    *,
    sampler_steps: int = 10,
    action_horizon: int = 8,
    device: str = "cuda",
    max_steps: int = MAX_STEPS,
    record_latency: bool = True,
) -> dict[str, float]:
    # Seed the policy's stochastic sampler per episode. Without this the flow
    # sampler draws from whatever global RNG state the worker happens to hold,
    # so the same eval seed gave different rollouts on different runs and the
    # number of episodes per worker silently changed the result.
    torch.manual_seed(env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_seed)
    obs_horizon = policy.obs_horizon
    env = PushTEnv()
    wall_start = time.perf_counter()
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
    wall_s = time.perf_counter() - wall_start
    lat = np.asarray(latencies) if latencies else np.zeros(1)
    return {
        "success": success,
        "coverage": coverage,
        "steps": float(steps),
        "latency_mean_ms": float(lat.mean()),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        # Wall-clock for the episode body (reset -> done), excluding checkpoint load and
        # worker start-up. This is what `scripts/bench_stack.py` converts to episodes/hour.
        "wall_s": float(wall_s),
        "chunks": float(len(latencies)),
    }


_WORKER: dict[str, Any] = {}


def _init_worker(
    checkpoint: str,
    device: str,
    sampler_steps: int,
    action_horizon: int,
    scheduler: str,
    sampler: str,
    dtype: str,
) -> None:
    torch.set_num_threads(1)
    policy, meta = load_policy(
        checkpoint, device=device, scheduler=scheduler, sampler=sampler, dtype=dtype
    )
    _WORKER.update(
        policy=policy, meta=meta, device=device, steps=sampler_steps, horizon=action_horizon
    )


def _run_episodes(seeds: list[int]) -> list[dict[str, Any]]:
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
    scheduler: str = "ddim",
    sampler: str = "stock",
    dtype: str = "float32",
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
        initargs=(checkpoint, device, sampler_steps, action_horizon, scheduler, sampler, dtype),
    ) as pool:
        for part in pool.map(_run_episodes, [c for c in chunks if c]):
            results.extend(part)
    for row in results:
        row["config_id"] = config_id
        row["task"] = "pusht"
        row["sampler_steps"] = sampler_steps
        row["action_horizon"] = action_horizon
        row["weights"] = str(row.get("weights", "ema"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")
    return results
