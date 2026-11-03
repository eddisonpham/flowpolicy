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
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import spawn as mp_spawn
from pathlib import Path
from typing import Any

import numpy as np
import torch

from policybench.config import ModelConfig
from policybench.data.normalizer import Stats
from policybench.envs.base import EnvLike
from policybench.envs.pusht import MAX_STEPS, PushTEnv
from policybench.eval.metrics import assert_seed_disjoint
from policybench.logging import Progress, RunLog, default_log_path
from policybench.models.normalization import config_id_marks_path
from policybench.models.observation import (
    DEFAULT_CAMERA,
    check_cameras,
    check_instruction,
    spec_for,
)
from policybench.models.policies import PolicyLike, build_policy
from policybench.models.pretrained import PretrainedDiffusionPolicy
from policybench.provenance import provenance

__all__ = ["assert_seed_disjoint", "evaluate", "load_policy", "rollout_episode"]


def load_policy(
    checkpoint: str | Path,
    device: str = "cuda",
    use_ema: bool = True,
    *,
    scheduler: str = "ddim",
    sampler: str = "stock",
    dtype: str = "float32",
    normalization: str = "buffers",
) -> tuple[PolicyLike, dict[str, Any]]:
    """Load a trained checkpoint (`path/last.pt`) or an HF one (`hf:<repo_id>`).

    Pretrained rows carry `train_seed = -1`: there is no training seed to resample, so
    their intervals are episode-level (docs/04 section 1). `sampler` selects the DDIM loop
    for the pretrained path ("stock" = lerobot's, "fast" = ours, which compiles and
    captures); it has no effect on trained checkpoints. `normalization` selects whose
    arithmetic maps units (docs/10, option H); it is default-off and only the diffusion
    adapter implements the second path.
    """
    if str(checkpoint).startswith("hf:"):
        hf_id = str(checkpoint).removeprefix("hf:")
        # Which family the checkpoint is comes from its own config.json, not from a CLI flag:
        # a second published policy has to plug in here without touching eval or deploy.
        kind = _hf_policy_type(hf_id)
        if kind == "vqbet":
            if normalization != "buffers":
                raise ValueError(
                    "normalization='processors' is implemented for the diffusion adapter only "
                    "(docs/10 section 4.3); VQ-BeT carries its own statistics"
                )
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
            hf_id,
            device=device,
            scheduler=scheduler,
            sampler=sampler,
            dtype=dtype,
            normalization=normalization,
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
    instruction: str = "",
    env_factory: Callable[[], EnvLike] = PushTEnv,
) -> dict[str, float]:
    """One episode. `env_factory` is the environment seam: default is the PushT wrapper, and a
    caller that supplies its own (a test, or a second task) gets the same loop unchanged.
    """
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
    # The observation contract is checked before the episode starts, not per chunk: this env
    # supplies one view (`image`), so a policy declaring more is refused here rather than fed
    # the same frame twice (models/observation.py, docs/11 section 3).
    spec = spec_for(policy)
    check_cameras(spec, (DEFAULT_CAMERA,))
    check_instruction(spec, instruction)
    noise_seed = int(env_seed) + int(noise_seed_offset)
    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(noise_seed)
    obs_horizon = policy.obs_horizon
    env = env_factory()
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
                chunk = policy.predict_chunk(batch, steps=sampler_steps, instruction=instruction)
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


def _load_worker(
    checkpoint: str,
    device: str,
    sampler_steps: int,
    action_horizon: int,
    scheduler: str,
    sampler: str,
    dtype: str,
    noise_seed_offset: int,
    normalization: str,
    instruction: str,
) -> None:
    policy, meta = load_policy(
        checkpoint,
        device=device,
        scheduler=scheduler,
        sampler=sampler,
        dtype=dtype,
        normalization=normalization,
    )
    _WORKER.update(
        policy=policy,
        meta=meta,
        device=device,
        steps=sampler_steps,
        horizon=action_horizon,
        noise_offset=noise_seed_offset,
        instruction=instruction,
    )


def _init_worker(
    checkpoint: str,
    device: str,
    sampler_steps: int,
    action_horizon: int,
    scheduler: str,
    sampler: str,
    dtype: str,
    noise_seed_offset: int,
    normalization: str,
    instruction: str,
) -> None:
    """Pool initializer: one BLAS thread per worker, then load the checkpoint once."""
    # N workers on a laptop otherwise contend on the CPU half of the pipeline, which is what made
    # four CUDA workers freeze the host (docs/04 section 4).
    torch.set_num_threads(1)
    _load_worker(
        checkpoint,
        device,
        sampler_steps,
        action_horizon,
        scheduler,
        sampler,
        dtype,
        noise_seed_offset,
        normalization,
        instruction,
    )


class RolloutError(RuntimeError):
    """An episode raised, named by the seed that produced it.

    The pool reports a worker crash as `BrokenProcessPool`, which says nothing about *which*
    episode died -- and under `spawn` it does not even carry the worker's traceback to the parent.
    A 200-episode block then has to be debugged from the log rather than from the seed.
    """

    def __init__(self, eval_seed: int, config_id: str, cause: BaseException) -> None:
        super().__init__(
            f"episode {eval_seed} of {config_id!r} failed: {type(cause).__name__}: {cause}"
        )
        self.eval_seed = eval_seed
        self.config_id = config_id


def _run_episodes(seeds: list[int], out_path: str, log_path: str, config_id: str) -> int:
    """Run one chunk, appending each row to `out_path` as it finishes.

    The worker writes the row rather than returning it. A chunk that raised on its 12th of 25
    episodes used to lose all 25, because the parent only ever saw `future.result()`; now the
    block keeps everything that completed, and the failure is logged by the process that saw it.

    `out_path` is this worker's **own** shard, not a shared file. Appends from several processes
    to one file are not atomic for a ~600-byte line: four workers at E3 N=1 produced interleaved
    fragments, and the assembly step then died on `json.loads` at the end of a completed block
    (2026-11-03, found by `just pipeline`). One writer per shard removes the question, and the
    parent merges and sorts the shards at the end.
    """
    log = RunLog(Path(log_path), run_id=config_id, echo=False)
    written = 0
    try:
        with open(out_path, "a", encoding="utf-8", newline="\n") as fh:
            for seed in seeds:
                offset = int(_WORKER["noise_offset"])
                started = time.perf_counter()
                try:
                    result = rollout_episode(
                        _WORKER["policy"],
                        seed,
                        sampler_steps=_WORKER["steps"],
                        action_horizon=_WORKER["horizon"],
                        device=_WORKER["device"],
                        noise_seed_offset=offset,
                        instruction=_WORKER["instruction"],
                    )
                except Exception as exc:
                    log.error(exc, stage="episode", eval_seed=seed, config_id=config_id)
                    raise RolloutError(seed, config_id, exc) from exc
                row: dict[str, Any] = {
                    "eval_seed": seed,
                    "noise_seed": int(seed) + offset,
                    **result,
                    **_WORKER["meta"],
                }
                _stamp(
                    row,
                    config_id=config_id,
                    sampler_steps=int(_WORKER["steps"]),
                    action_horizon=int(_WORKER["horizon"]),
                    noise_seed_offset=offset,
                    instruction=str(_WORKER["instruction"]),
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                written += 1
                log.event(
                    "episode_done",
                    eval_seed=seed,
                    success=row["success"],
                    coverage=round(float(row["coverage"]), 4),
                    steps=int(row["steps"]),
                    wall_s=round(float(row["wall_s"]), 2),
                    secs=round(time.perf_counter() - started, 2),
                )
    finally:
        log.close()
    return written


def _stamp(
    row: dict[str, Any],
    *,
    config_id: str,
    sampler_steps: int,
    action_horizon: int,
    noise_seed_offset: int,
    instruction: str = "",
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
    # Nested, so no reader can mistake a hash for a measurement (agents/05). Old rows have no
    # key at all, which reads as "unknown" -- the same convention `step` and `train_seed` use.
    row["provenance"] = provenance()
    # Stamped only when an instruction was actually used. Every row written before this seam
    # existed has no instruction, and an empty string is that absence, not a measurement.
    if instruction:
        row["instruction"] = instruction


def _can_spawn() -> bool:
    """Whether `spawn` can start a worker from this caller.

    Rather than guessing from `__main__`, ask `spawn` what it would do: `get_preparation_data`
    reports an `init_main_from_path` only when there is a main file to re-import, and a path that
    is not on disk (`<stdin>`, `<string>`, a deleted script) is exactly the case that dies as
    `BrokenProcessPool` after the first submit. `python -` / `python -c` also fail this by raising
    inside `get_preparation_data` itself, which is why the probe is guarded.
    """
    try:
        data = mp_spawn.get_preparation_data("policybench-probe")
    except (AttributeError, RuntimeError):
        return False
    main_path = data.get("init_main_from_path")
    return not (isinstance(main_path, str) and not Path(main_path).is_file())


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


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Rows from a JSONL file, skipping a line that does not decode.

    A killed worker leaves at most one torn line per shard, and a torn line must not make a
    finished block unreadable: the seed it belonged to is simply not complete, so it is re-run.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _shards(partial: Path) -> list[Path]:
    """The block's per-worker files: the bare `.partial` plus `.partial.<worker>`."""
    return [partial, *sorted(partial.parent.glob(partial.name + ".*"))]


def completed_seeds(partial: Path, config_id: str) -> set[int]:
    """Seeds already written under this `config_id`, across every shard (resume support)."""
    done: set[int] = set()
    for path in _shards(partial):
        for row in _iter_rows(path):
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
    normalization: str = "buffers",
    instruction: str = "",
    log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run `eval_seeds` and append one JSON line per episode to `out_path`.

    `noise_seed_offset` shifts the policy's per-episode sampling seed while leaving the
    environment seed alone, so two blocks over the same seeds differ only in the noise
    drawn. Default 0 reproduces every pre-existing result exactly.

    Rows are written by the workers straight into `<out_path>.partial` as each episode finishes
    and only appended to `out_path` once every chunk is done. A multi-hour block therefore loses
    at most the episode that was running to a crash, and a half-finished block cannot be read as
    a complete one: either the file is whole or it does not exist. Re-running the same
    `config_id` resumes: seeds already in the `.partial` are not re-run, and seeds from any other
    `config_id` are ignored.

    `log_path` defaults to `results/logs/eval_<config_id>_<stamp>.jsonl`. It carries a
    `run_start`, one `episode_done` per episode from the worker that ran it, `chunk_done` with
    rate and ETA, and `error` with the seed and traceback when something raises -- which is the
    difference between diagnosing a 200-episode block and re-running it.
    """
    assert_seed_disjoint(train_seeds, eval_seeds)
    # The row schema has no normalization field (docs/10 section 4.3), so a non-default path has
    # to be identifiable from the config_id or two paths' rows become one sample.
    if not config_id_marks_path(config_id, normalization):
        raise ValueError(
            f"config_id {config_id!r} does not mark the {normalization!r} path; "
            "include 'proc' in the id (docs/10 section 4.3)"
        )
    checkpoint = str(checkpoint)
    if not use_ema:
        raise ValueError("evaluation always uses EMA weights (docs/03 section 6)")
    ctx = mp.get_context("spawn")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    started = time.perf_counter()
    log = RunLog(
        default_log_path(f"eval_{config_id}") if log_path is None else log_path,
        run_id=config_id,
    )
    done = completed_seeds(partial, config_id) & set(eval_seeds)
    todo = [seed for seed in eval_seeds if seed not in done]
    log.progress(
        "run_start",
        checkpoint=checkpoint,
        episodes=len(eval_seeds),
        todo=len(todo),
        resumed=len(done),
        workers=workers,
        device=device,
        sampler_steps=sampler_steps,
        action_horizon=action_horizon,
        scheduler=scheduler,
        sampler=sampler,
        dtype=dtype,
        noise_seed_offset=int(noise_seed_offset),
        normalization=normalization,
        out=str(out),
        log=str(log.path),
        git=provenance()["git"],
    )
    if done:
        log.progress("resume", partial=str(partial), completed=len(done), of=len(eval_seeds))
    # The `.partial` files belong to one `config_id`: starting a fresh block deletes them,
    # resuming keeps them. Each worker appends to its own shard, so the bare file is only a
    # marker that the block started.
    if not done:
        for stale in _shards(partial):
            stale.unlink(missing_ok=True)
    partial.touch()
    progress = Progress(total=len(todo))
    chunks = _chunks(todo, workers)
    if todo and not _can_spawn():
        # Serial fallback: same rows, one worker. `spawn` cannot import a main module that is not
        # on disk (REPL, notebook, `python -c`), and the failure it produced named `spawn_main`.
        log.progress("spawn_unavailable", workers_requested=workers, workers_used=1)
        _load_worker(
            checkpoint,
            device,
            sampler_steps,
            action_horizon,
            scheduler,
            sampler,
            dtype,
            int(noise_seed_offset),
            normalization,
            instruction,
        )
        for index, chunk in enumerate(chunks):
            _run_episodes(chunk, f"{partial}.{index}", str(log.path), config_id)
            log.progress("chunk_done", **progress.advance(len(chunk)))
    elif todo:
        try:
            with ProcessPoolExecutor(
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
                    normalization,
                    instruction,
                ),
            ) as pool:
                futures = [
                    pool.submit(
                        _run_episodes, chunk, f"{partial}.{index}", str(log.path), config_id
                    )
                    for index, chunk in enumerate(chunks)
                ]
                for future in as_completed(futures):
                    log.progress("chunk_done", **progress.advance(future.result()))
        except BaseException as exc:
            # Log the state the block is in, then let the caller see the failure. Rows already in
            # `.partial` are kept, so re-running the same `config_id` resumes instead of restarting.
            log.error(exc, stage="block", completed=progress.done, of=progress.total)
            log.close()
            raise
    # Sorted before the append: chunks finish out of order now, and the block's row order should
    # not depend on which worker got there first. Filtered to this run's `config_id` and seeds:
    # `completed_seeds` already refuses another run's rows when deciding what to *run*, and
    # appending them here would splice a foreign block into this one -- a half-finished run from a
    # different `config_id` is exactly what the `.partial` file is supposed to keep out.
    requested = set(eval_seeds)
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for path in _shards(partial):
        for row in _iter_rows(path):
            if row.get("config_id") == config_id and int(row["eval_seed"]) in requested:
                merged.setdefault((int(row["eval_seed"]), int(row.get("noise_seed", 0))), row)
    rows = [merged[key] for key in sorted(merged)]
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    for path in _shards(partial):
        path.unlink(missing_ok=True)
    successes = sum(1 for row in rows if row["success"])
    log.progress(
        "run_end",
        rows=len(rows),
        success_rate=round(successes / len(rows), 4) if rows else 0.0,
        wall_s=round(time.perf_counter() - started, 1),
        out=str(out),
    )
    log.close()
    return rows
