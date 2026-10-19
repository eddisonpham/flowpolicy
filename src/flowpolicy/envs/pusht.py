"""gym-pusht wrapper: observation dict {image, state}, raw units for the policy.

docs/04 section 1: success = max coverage >= 95 %; also log mean max coverage.
The env itself reports `is_success` and `coverage` per step.
"""

from __future__ import annotations

import os

import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

# Headless rendering on Windows: pygame/SDL must not open a window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

SUCCESS_THRESHOLD = 0.95
MAX_STEPS = 300


class PushTEnv:
    """Minimal Gymnasium-style wrapper used by eval, gates and deployment."""

    def __init__(self, seed: int | None = None) -> None:
        self._env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        ).unwrapped
        self._seed = seed
        self._steps = 0

    @property
    def fps(self) -> int:
        return int(self._env.metadata["render_fps"])

    def reset(self, seed: int | None = None) -> dict[str, NDArray[np.generic]]:
        if seed is None:
            seed = self._seed
        obs, _info = self._env.reset(seed=seed)
        self._steps = 0
        return self._obs(obs)

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, dict[str, float]]:
        obs, reward, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32)
        )
        self._steps += 1
        done = bool(terminated or truncated or self._steps >= MAX_STEPS)
        return (
            self._obs(obs),
            float(reward),
            done,
            {
                "success": float(bool(info.get("is_success", False))),
                "coverage": float(info.get("coverage", 0.0)),
            },
        )

    @staticmethod
    def _obs(obs: dict[str, NDArray[np.generic]]) -> dict[str, NDArray[np.generic]]:
        return {
            "image": np.asarray(obs["pixels"], dtype=np.uint8),
            "state": np.asarray(obs["agent_pos"], dtype=np.float32),
        }

    def close(self) -> None:
        self._env.close()
