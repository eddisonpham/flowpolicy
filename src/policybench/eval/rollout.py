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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        # Which family the checkpoint is comes from its own config.json, not from a CLI flag:
        # a second published policy has to plug in here without touching eval or deploy.
        kind = _hf_policy_type(hf_id)
        if kind == "vqbet":
            from policybench.models.pretrained_vqbet import PretrainedVQBeTPolicy

            vqbet = PretrainedVQBeTPolicy(hf_id, device=device)
            return vqbet, {
                "policy": "pretrained-vqbet",
                "source": hf_id,
                "train_seed": -1,
                "checkpoint": f"hf:{hf_id}",
                # The checkpoint ships no step counter; -1 is this repo's "unknown", the
                # same convention train_seed uses for a borrowed policy.
                "step": -1,
                "params": vqbet.num_parameters(),
                "weights": "pretrained",
                "scheduler": "none",
                "sampler": "none",
                "dtype": "float32",
            }
        if kind != "diffusion":
            raise ValueError(
                f"hf:{hf_id} is a {kind!r} policy; "
                "this repo has adapters for 'diffusion' and 'vqbet'"
            )
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


def _hf_policy_type(hf_id: str) -> str:
    """Read `type` from the checkpoint's `config.json` ("diffusion", "vqbet", ...)."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(hf_id, "config.json")
    with open(path, encoding="utf-8") as fh:
        return str(json.load(fh).get("type", ""))


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
    noise_seed_offset: int = 0,
) -> dict[str, float]:
    # Seed the policy's stochastic sampler per episode. Without this the flow
    # sampler draws from whatever global RNG state the worker happens to hold,
    # so the same eval seed gave different rollouts on different runs and the
    # number of episodes per worker silently changed the result.
    #
    # `noise_seed_offset` shifts that seed without touching the environment seed,
    # which is what makes "is the success rate a property of the policy or of the
    # noise stream we drew" a measurable question: env episodes stay identical,
    # only the sampler's initial noise changes. Offset 0 is the historical
    # behaviour, so every earlier row keeps its exact number.
    noise_seed = int(env_seed) + int(noise_seed_offset)
    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(noise_seed)
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
    noise_seed_offset: int,
) -> None:
    torch.set_num_threads(1)
    policy, meta = load_policy(
        checkpoint, device=device, scheduler=scheduler, sampler=sampler, dtype=dtype
    )
    _WORKER.update(
        policy=policy,
        meta=meta,
        device=device,
        steps=sampler_steps,
        horizon=action_horizon,
        noise_offset=noise_seed_offset,
    )


def _run_episodes(seeds: list[int]) -> list[dict[str, Any]]:
    out = []
    for seed in seeds:
        offset = int(_WORKER["noise_offset"])
        result = rollout_episode(
            _WORKER["policy"],
            seed,
            sampler_steps=_WORKER["steps"],
            action_horizon=_WORKER["horizon"],
            device=_WORKER["device"],
            noise_seed_offset=offset,
        )
        out.append(
            {
                "eval_seed": seed,
                "noise_seed": int(seed) + offset,
                **result,
                **_WORKER["meta"],
            }
        )
    return out


def _stamp(
    row: dict[str, Any],
    *,
    config_id: str,
    sampler_steps: int,
    action_horizon: int,
    noise_seed_offset: int,
) -> None:
    """Attach the run's metadata to one episode row.

    Applied per shard so rows can be flushed as they complete (`evaluate`), and so a row's
    meaning never depends on the row that follows it.
    """
    row["config_id"] = config_id
    row["task"] = "pusht"
    row["sampler_steps"] = sampler_steps
    row["action_horizon"] = action_horizon
    row["weights"] = str(row.get("weights", "ema"))
    row["noise_seed_offset"] = int(noise_seed_offset)


# Chunks per worker: the unit of "what a crash costs". One chunk per worker (the old shape) meant
# a single-worker block flushed nothing until its last episode.
CHUNKS_PER_WORKER = 8


def _chunks(seeds: list[int], workers: int, per_worker: int = CHUNKS_PER_WORKER) -> list[list[int]]:
    """Interleave seeds into `workers * per_worker` chunks, dropping empty ones.

    Interleaving (rather than slicing the block into consecutive runs) keeps every chunk a spread
    of the eval seeds, so a partial block is a sample of the block and not one corner of it.
    """
    n_chunks = max(1, workers * per_worker)
    return [chunk for chunk in (seeds[i::n_chunks] for i in range(n_chunks)) if chunk]


def completed_seeds(partial: Path, config_id: str) -> set[int]:
    """Seeds already written to `partial` under this `config_id` (resume support)."""
    if not partial.exists():
        return set()
    done: set[int] = set()
    for line in partial.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # A `.partial` from a different run must not be spliced into this one.
        if row.get("config_id") == config_id and "eval_seed" in row:
            done.add(int(row["eval_seed"]))
    return done


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
    noise_seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Run `eval_seeds` and append one JSON line per episode to `out_path`.

    `noise_seed_offset` shifts the policy's per-episode sampling seed while leaving the
    environment seed alone, so two blocks over the same seeds differ only in the noise
    drawn. Default 0 reproduces every pre-existing result exactly.

    Rows are flushed to `<out_path>.partial` as each chunk finishes and only appended to
    `out_path` once every chunk is done. A multi-hour block therefore loses at most one chunk to
    a crash, and a half-finished block cannot be read as a complete one: either the file is
    whole or it does not exist. Re-running the same `config_id` resumes: seeds already in the
    `.partial` are not re-run, and seeds from any other `config_id` are ignored.
    """
    assert_seed_disjoint(train_seeds, eval_seeds)
    checkpoint = str(checkpoint)
    if not use_ema:
        raise ValueError("evaluation always uses EMA weights (docs/03 section 6)")
    ctx = mp.get_context("spawn")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    done = completed_seeds(partial, config_id) & set(eval_seeds)
    todo = [seed for seed in eval_seeds if seed not in done]
    if done:
        print(
            f"resuming {config_id}: {len(done)}/{len(eval_seeds)} seeds already in {partial}, "
            f"{len(todo)} to run"
        )
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                checkpoint,
                device,
                sampler_steps,
                action_horizon,
                scheduler,
                sampler,
                dtype,
                int(noise_seed_offset),
            ),
        ) as pool,
        partial.open("a" if done else "w", encoding="utf-8", newline="\n") as partial_fh,
    ):
        futures = [pool.submit(_run_episodes, chunk) for chunk in _chunks(todo, workers)]
        for future in as_completed(futures):
            for row in future.result():
                _stamp(
                    row,
                    config_id=config_id,
                    sampler_steps=sampler_steps,
                    action_horizon=action_horizon,
                    noise_seed_offset=int(noise_seed_offset),
                )
                partial_fh.write(json.dumps(row) + "\n")
            partial_fh.flush()
    # Sorted before the append: chunks finish out of order now, and the block's row order should
    # not depend on which worker got there first.
    rows: list[dict[str, Any]] = sorted(
        (
            json.loads(line)
            for line in partial.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        key=lambda row: (int(row["eval_seed"]), int(row.get("noise_seed", 0))),
    )
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    partial.unlink()
    return rows
