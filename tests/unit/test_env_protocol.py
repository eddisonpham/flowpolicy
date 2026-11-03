"""The environment seam: `PushTEnv` is what `EnvLike` describes (envs/base.py).

Why this file exists at all: the rollout loop, the control loop and the servers name
`PushTEnv` directly, which is fine while PushT is the only task and wrong the moment a second
one is added -- changing the harness that produced every row in `results/` is not a refactor.
Writing the interface down makes the next step possible without touching those call sites, and
makes a mismatch visible today.

Two levels, deliberately:

* the protocol's runtime check, exercised on stubs, so a mistake in the protocol itself (a member
  nobody declared) fails in CI;
* `PushTEnv`'s conformance, checked at class level in CI and by construction locally, because
  instantiating it needs the simulator and a display device that CI does not have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from policybench.envs.base import EnvLike
from policybench.envs.pusht import PushTEnv

pytestmark = pytest.mark.unit


class FullEnv:
    fps = 10
    task_description = "stub"

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return {"image": np.zeros((8, 8, 3), np.uint8), "state": np.zeros(2, np.float32)}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, float]]:
        return self.reset(), 0.0, True, {"success": 0.0, "coverage": 0.0}

    def close(self) -> None:
        pass


class NoClose:
    """Everything but `close`: a wrapper that leaks the simulator handle is not an `EnvLike`."""

    fps = 10
    task_description = "stub"

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return {"image": np.zeros((8, 8, 3), np.uint8), "state": np.zeros(2, np.float32)}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, float]]:
        return self.reset(), 0.0, True, {"success": 0.0, "coverage": 0.0}


class NoDescription:
    """A task string is part of the contract: a VLA needs one, and a client guessing is a bug."""

    fps = 10

    def __init__(self) -> None:
        self.inner = FullEnv()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return self.inner.reset(seed)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, float]]:
        return self.inner.step(action)

    def close(self) -> None:
        self.inner.close()


def test_the_protocol_is_runtime_checkable() -> None:
    assert isinstance(FullEnv(), EnvLike)


def test_a_missing_close_is_not_an_environment() -> None:
    assert not isinstance(NoClose(), EnvLike)


def test_a_missing_task_description_is_not_an_environment() -> None:
    assert not isinstance(NoDescription(), EnvLike)


def test_pusht_declares_every_member_of_the_protocol() -> None:
    """Class-level, so this runs in CI where the simulator is available but a display is not."""
    for name in ("reset", "step", "close", "fps", "task_description"):
        assert hasattr(PushTEnv, name), name


@pytest.mark.integration
def test_pusht_satisfies_the_protocol_by_construction() -> None:
    """The real instance check. Local only: it builds the simulator and a pygame surface."""
    env = PushTEnv()
    try:
        assert isinstance(env, EnvLike)
        assert env.fps > 0
        assert env.task_description
        obs = env.reset(seed=0)
        assert set(obs) == {"image", "state"}
        assert obs["image"].dtype == np.uint8
    finally:
        env.close()
