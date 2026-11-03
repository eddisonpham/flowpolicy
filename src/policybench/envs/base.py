"""The environment seam: what the harness needs, without naming gym-pusht.

`PushTEnv` is the only environment in this repo, and the rollout loop, the control loop and the
servers name it directly. That is deliberate for this stage -- those files produced every row in
`results/` -- but it means a second task cannot be added without editing them.

`EnvLike` is the structural half of the fix: the *interface* is written down where a reader looks
for it, and `PushTEnv` is checked against it. The call sites keep using the concrete class, so no
measured path changes; a later step can swap the constructor for this protocol with its own
golden test (agents/02_generic_adapter.md).

Declared here rather than next to `PolicyLike` (models/policies.py) because it describes an
environment, not a policy, and a reader looking for "how do I add a task" starts in `envs/`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["EnvLike"]

Observation = dict[str, NDArray[np.generic]]


@runtime_checkable
class EnvLike(Protocol):
    """A Gymnasium-style environment with named observation fields.

    `reset` returns `{"image": uint8 [H, W, 3], "state": float32 [Ds], ...}`: named, because a
    positional observation is how a second view gets bound to the wrong slot. `step` returns
    `(obs, reward, done, info)` and `info` carries `coverage` and `success`, which are the two
    quantities every result in this repo is aggregated from. `fps` is the control rate the
    dataset was recorded at, and `task_description` is the string a language-conditioned policy
    receives -- both as attributes, because they are properties of the task rather than of a
    particular observation.
    """

    # Read-only properties in the protocol, because the implementations expose them as
    # `@property` over the gym handle: a settable-variable declaration would reject them
    # (mypy: "expected settable variable, got read-only attribute").
    @property
    def fps(self) -> int: ...

    @property
    def task_description(self) -> str: ...

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[Observation, float, bool, dict[str, float]]: ...

    def close(self) -> None: ...
