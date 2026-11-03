"""Same seed, same episode -- the guard on per-episode seeding.

`rollout_episode` seeds the policy's sampler with the eval seed before the first chunk, because
without it the sampler draws from whatever global RNG state the worker happens to hold: the same
eval seed then gave different rollouts, and the number of episodes assigned to a worker silently
changed the answer. A paired comparison cannot survive that, so the property is a test rather
than a comment.

The stub policy draws its chunk from the global RNG on purpose. If the `torch.manual_seed(...)`
line at the top of `rollout_episode` is removed, the second run continues the first run's stream
and `test_the_same_seed_reproduces_the_chunk_trace` fails -- which is the regression this file
exists to catch. A deterministic stub would have kept it green either way.

The environment is a stub too (`GridEnv`), so this runs in CI: no simulator, no display, no GPU.
`EnvLike` is the seam that makes that possible (envs/base.py).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from policybench.envs.base import EnvLike
from policybench.eval.rollout import rollout_episode

pytestmark = pytest.mark.unit


class RngPolicy(torch.nn.Module):
    """`PolicyLike` whose chunk is random, and which records what it returned."""

    obs_horizon = 2
    pred_horizon = 4
    action_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[NDArray[np.float32]] = []

    def predict_chunk(self, obs: dict[str, Tensor], **_: Any) -> Tensor:
        chunk = torch.randn(1, self.pred_horizon, self.action_dim) + 100.0
        self.chunks.append(chunk.numpy().copy())
        return chunk

    def num_parameters(self) -> int:
        return 0


class GridEnv:
    """Minimal `EnvLike`: a state that moves by the action, fixed length, no rendering."""

    fps = 10
    task_description = "unit-test task"

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, seed: int | None = None) -> dict[str, NDArray[np.generic]]:
        self.steps = 0
        return {
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
            "state": np.zeros(2, dtype=np.float32),
        }

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, dict[str, float]]:
        self.steps += 1
        obs = {
            "image": np.full((16, 16, 3), self.steps % 256, dtype=np.uint8),
            "state": np.asarray(action, dtype=np.float32),
        }
        done = self.steps >= 6
        return obs, 0.0, done, {"success": 0.0, "coverage": 0.0}

    def close(self) -> None:
        pass


def _run(
    seed: int = 1000, *, offset: int = 0
) -> tuple[list[NDArray[np.float32]], dict[str, float]]:
    policy = RngPolicy()
    row = rollout_episode(
        policy,
        seed,
        action_horizon=2,
        device="cpu",
        max_steps=6,
        noise_seed_offset=offset,
        env_factory=GridEnv,
    )
    return policy.chunks, row


def test_the_same_seed_reproduces_the_chunk_trace() -> None:
    first_chunks, first_row = _run()
    second_chunks, second_row = _run()
    assert len(first_chunks) == len(second_chunks) == 3  # 6 steps, 2 actions per chunk
    for first, second in zip(first_chunks, second_chunks, strict=True):
        assert np.array_equal(first, second)
    # Everything that is a measurement of the episode, not of the clock.
    for key in ("success", "coverage", "steps", "chunks"):
        assert first_row[key] == second_row[key]


def test_a_different_seed_draws_different_noise() -> None:
    """Without this, the test above could pass on a policy that ignores the RNG."""
    first, _ = _run(seed=1000)
    second, _ = _run(seed=2000)
    assert not np.array_equal(first[0], second[0])


def test_the_noise_offset_moves_the_noise_and_not_the_episode() -> None:
    """The seam the noise-stream experiment relied on: same env episodes, different draws."""
    base, base_row = _run()
    shifted, shifted_row = _run(offset=500_000)
    assert not np.array_equal(base[0], shifted[0])
    # The environment stream is untouched, so the episode's own trajectory is identical.
    assert base_row["steps"] == shifted_row["steps"]
    assert base_row["coverage"] == shifted_row["coverage"]


def test_the_default_environment_is_the_real_one() -> None:
    """`env_factory` is a seam, not a behaviour change: the default is still PushTEnv."""
    from policybench.envs.pusht import PushTEnv

    params = inspect.signature(rollout_episode).parameters
    assert params["env_factory"].default is PushTEnv


def test_the_stub_satisfies_the_environment_protocol() -> None:
    assert isinstance(GridEnv(), EnvLike)
