"""The observation contract: named camera views plus a language slot.

Until now the contract was implicit and single-view: `PolicyLike.predict_chunk` took a dict
with exactly `image` (`[B, To, C, H, W]`) and `state` (`[B, To, Ds]`), and nothing carried a
task string. Every VLA worth wrapping is conditioned on both -- two or three *named* views
(`observation.images.<name>`) and an instruction -- so a policy written against the old
contract cannot express one, and an adapter written first would have decided the protocol by
accident (docs/11 section 3).

This module makes that contract explicit and checkable, and deliberately owns no model:

- **Cameras are named.** The observation dict's keys, other than `state`, are camera names, and
  a policy declares the ones it consumes in an optional `camera_names` attribute. Default
  `("image",)`, which is the historical single-camera contract, so every existing row, adapter
  and test is unchanged.
- **A missing view is an error, never a duplicate.** `check_cameras` refuses a policy that
  wants two views in an environment that supplies one. Feeding the same frame into both slots
  produces plausible actions and a broken success rate -- the failure mode docs/10 was written
  to prevent -- and it is invisible in the logs, which is exactly why it is a hard error here.
- **The instruction travels outside the tensor dict.** `instruction: str` is a call argument to
  `predict_chunk`, not a key in `obs`. An empty string means "unconditional", which is what the
  diffusion and VQ-BeT checkpoints are, so a policy that never sees a language slot keeps
  working and existing rows keep their meaning.

Nothing in this module imports `lerobot` or torch's CUDA side, so the contract is testable in
CI and decided before any GPU hour is spent on an adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

__all__ = [
    "DEFAULT_CAMERA",
    "STATE_KEY",
    "ObservationSpec",
    "check_cameras",
    "check_frames",
    "check_instruction",
    "spec_for",
]

# The single-camera key every row in `results/` was measured with, and the key the env supplies.
DEFAULT_CAMERA = "image"
STATE_KEY = "state"


@dataclass(frozen=True)
class ObservationSpec:
    """What one policy needs from the observation path.

    `cameras` is ordered: the order in which an adapter maps views onto the checkpoint's own
    feature keys is a property of the adapter, but an unordered set would let a translation
    table silently swap two views.
    """

    cameras: tuple[str, ...] = (DEFAULT_CAMERA,)
    requires_instruction: bool = False

    @property
    def single_view(self) -> bool:
        """True for the historical contract (`image` only, no language slot)."""
        return self.cameras == (DEFAULT_CAMERA,) and not self.requires_instruction


def spec_for(policy: Any) -> ObservationSpec:
    """Read a policy's declared observation needs, defaulting to the historical contract.

    The two attributes are optional on purpose. `PolicyLike` is structural and implemented by
    third-party-backed adapters; adding required members would break every implementation for a
    feature none of them uses yet. A policy that consumes one view and no instruction stays
    silent and gets the default.
    """
    declared = getattr(policy, "camera_names", (DEFAULT_CAMERA,))
    cameras = tuple(str(name) for name in declared)
    if not cameras:
        raise ValueError(
            f"{type(policy).__name__}.camera_names is empty; a policy needs at least one view"
        )
    if len(set(cameras)) != len(cameras):
        raise ValueError(
            f"{type(policy).__name__}.camera_names has repeated views: {list(cameras)}; "
            "two slots with one name is the duplicate-frame bug this contract exists to prevent"
        )
    return ObservationSpec(
        cameras=cameras,
        requires_instruction=bool(getattr(policy, "requires_instruction", False)),
    )


def check_cameras(spec: ObservationSpec, available: Iterable[str]) -> None:
    """Fail if something can supply fewer views than `spec` needs.

    Used on the *source* side (an environment, a client payload), so extra views are not an
    error: an env may expose more than this policy reads. The batch actually handed to the model
    is checked by `check_frames`, which is strict in both directions.
    """
    supplied = sorted(set(available))
    missing = [name for name in spec.cameras if name not in supplied]
    if missing:
        raise ValueError(
            f"observation contract mismatch: the policy needs cameras {list(spec.cameras)} "
            f"but only {supplied} are available (missing {missing}). One view is never "
            "duplicated to fill a second slot: a checkpoint fed the same frame twice returns "
            "plausible actions and a broken success rate (docs/11 section 3)."
        )


def check_frames(spec: ObservationSpec, obs: Mapping[str, Tensor]) -> None:
    """Validate a batch about to be passed to `predict_chunk`.

    Strict both ways. A missing view is the duplicate-frame failure; an *extra* view means the
    caller and the policy disagree about the contract, and one of them is measuring something
    other than what it reports.
    """
    if STATE_KEY not in obs:
        raise ValueError(f"observation has no {STATE_KEY!r} (got {sorted(obs)})")
    supplied = [name for name in obs if name != STATE_KEY]
    check_cameras(spec, supplied)
    extra = [name for name in supplied if name not in spec.cameras]
    if extra:
        raise ValueError(
            f"observation carries cameras {extra} that the policy does not consume; it declares "
            f"{list(spec.cameras)}. The contract is named views, not 'any image-like key'."
        )
    state = obs[STATE_KEY]
    if state.ndim != 3:
        raise ValueError(f"state must be [B, To, Ds], got {tuple(state.shape)}")
    for name in spec.cameras:
        frame = obs[name]
        if frame.ndim != 5:
            raise ValueError(f"camera {name!r} must be [B, To, C, H, W], got {tuple(frame.shape)}")
        if frame.shape[1] != state.shape[1]:
            raise ValueError(
                f"camera {name!r} has {frame.shape[1]} frames but state has {state.shape[1]}; "
                "history lengths must agree"
            )


def check_instruction(spec: ObservationSpec, instruction: str) -> None:
    """Fail if a language-conditioned policy is handed an empty string.

    An empty instruction means "unconditional" and is therefore a valid value for a policy that
    does not need one -- but a VLA given `""` returns actions for no task at all, which looks
    like a policy-quality result rather than the configuration error it is.
    """
    if spec.requires_instruction and not instruction.strip():
        raise ValueError(
            f"{type(spec).__name__} says this policy requires an instruction "
            "(`requires_instruction = True`) and none was supplied; an unconditional call to a "
            "language-conditioned policy is not a measurement of that policy"
        )
