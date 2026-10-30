"""Where the diffusion adapter maps raw units to network units, and back.

One *declared* mapping -- MEAN_STD for images, MIN_MAX for state and actions, as spelled out in
the checkpoint's own `normalization_mapping` -- has two implementations:

* `BuffersNorm` applies the eight `normalize_inputs.*` / `normalize_targets.*` /
  `unnormalize_outputs.*` tensors by hand. This is the default, and the path every measured row
  in `results/` used.
* `ProcessorsNorm` calls `lerobot`'s processor pipelines, built from the migrated checkpoint's
  `policy_preprocessor.json` / `policy_postprocessor.json` -- the library's own arithmetic for
  the same mapping (docs/10, option H).

They are *supposed* to agree, and nobody has ever checked numerically. That check is
`scripts/processor_equiv.py`, and its result -- not this module -- decides which one ships
(docs/10 section 3).

`PolicyLike` does not know this module exists: normalization stays inside the policy, so
observations and actions still cross every boundary in raw units (docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

# The keys the checkpoint's `policy_preprocessor.json` declares. Ours are the short names the
# rollout harness puts in the observation dict; the pipeline's are the checkpoint's features.
LIB_IMAGE = "observation.image"
LIB_STATE = "observation.state"
LIB_ACTION = "action"

Transition = dict[str, Any]
Pipeline = Callable[[Transition], Transition]

__all__ = [
    "BuffersNorm",
    "Normalization",
    "ProcessorsNorm",
    "config_id_marks_path",
    "denormalize_action",
    "flatten_history",
    "load_processor_norm",
    "normalize_image",
    "normalize_state",
    "override_processor_device",
    "restore_history",
]


class Normalization(Protocol):
    """The three transforms the diffusion adapter needs, in its own units.

    `image` takes `[B, T, 3, H, W]` in [0, 1] and returns network units; `state` takes
    `[B, T, 2]` in pixels; `actions` takes `[B, H, 2]` in network units and returns pixels.
    """

    def image(self, image: Tensor) -> Tensor: ...

    def state(self, state: Tensor) -> Tensor: ...

    def actions(self, chunk: Tensor) -> Tensor: ...


def normalize_image(image: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """VISUAL = MEAN_STD: uint8-derived [0, 1] image to zero-mean unit-variance."""
    return (image - mean) / std


def normalize_state(state: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """STATE = MIN_MAX: raw pixels to [-1, 1] with the dataset's own extremes."""
    return 2.0 * (state - low) / (high - low) - 1.0


def denormalize_action(action: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """ACTION = MIN_MAX: [-1, 1] back to pixels, the inverse of `normalize_state`."""
    return (action + 1.0) * 0.5 * (high - low) + low


class BuffersNorm:
    """The checkpoint's eight buffers, applied by hand (the default path)."""

    def __init__(
        self,
        *,
        image_mean: Tensor,
        image_std: Tensor,
        state_low: Tensor,
        state_high: Tensor,
        action_low: Tensor,
        action_high: Tensor,
    ) -> None:
        self.image_mean = image_mean
        self.image_std = image_std
        self.state_low = state_low
        self.state_high = state_high
        self.action_low = action_low
        self.action_high = action_high

    def image(self, image: Tensor) -> Tensor:
        return normalize_image(image, self.image_mean, self.image_std)

    def state(self, state: Tensor) -> Tensor:
        return normalize_state(state, self.state_low, self.state_high)

    def actions(self, chunk: Tensor) -> Tensor:
        return denormalize_action(chunk, self.action_low, self.action_high)


def flatten_history(tensor: Tensor) -> Tensor:
    """`[B, T, ...]` -> `[B*T, ...]`, row-major over batch then time."""
    return tensor.reshape(-1, *tensor.shape[2:])


def restore_history(flat: Tensor, batch: int, horizon: int) -> Tensor:
    """The inverse of `flatten_history`, and the reason it is a function.

    `[B, T]` and `[T, B]` have the same number of elements, so a transposed reshape runs and
    silently shuffles history frames. `tests/unit/test_normalization_protocol.py` pins the
    element mapping rather than the shape.
    """
    return flat.reshape(batch, horizon, *flat.shape[1:])


def _run_frames(pipeline: Pipeline, key: str, flat: Tensor) -> Tensor:
    """Apply a pipeline to each sample of `flat` separately, as `lerobot-eval` does.

    The pipelines are written for one unbatched observation per call (`to_batch_processor`
    exists because of that), so batching here would be guessing. `B*T` calls of a
    normalization step is microseconds against a 262.7 M-parameter network.
    """
    outputs: list[Tensor] = []
    for index in range(flat.shape[0]):
        sample = flat[index]
        transition = pipeline({key: sample})
        if key not in transition:
            raise KeyError(f"processor pipeline returned no {key!r} (got {sorted(transition)})")
        value = transition[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"processor pipeline returned {type(value).__name__} for {key!r}")
        # The pipelines own `to_batch_processor`, so one frame in comes back with a leading 1.
        # Squeeze it rather than asserting a shape: that batch dim is how the library uses them
        # (`lerobot-eval` feeds one env observation at a time), and it is not part of the mapping.
        if value.ndim == sample.ndim + 1 and value.shape[0] == 1:
            value = value[0]
        outputs.append(value)
    return torch.stack(outputs)


class ProcessorsNorm:
    """lerobot's two pipelines behind the same three methods `BuffersNorm` implements.

    Injected as callables rather than imported here, so the tensor plumbing is testable in CI
    (where `lerobot` is not installed) and the library-facing part is one function,
    `load_processor_norm`.
    """

    def __init__(self, pre: Pipeline, post: Pipeline) -> None:
        self.pre = pre
        self.post = post

    def image(self, image: Tensor) -> Tensor:
        batch, horizon = image.shape[0], image.shape[1]
        flat = _run_frames(self.pre, LIB_IMAGE, flatten_history(image))
        return restore_history(flat, batch, horizon)

    def state(self, state: Tensor) -> Tensor:
        batch, horizon = state.shape[0], state.shape[1]
        flat = _run_frames(self.pre, LIB_STATE, flatten_history(state))
        return restore_history(flat, batch, horizon)

    def actions(self, chunk: Tensor) -> Tensor:
        batch, horizon = chunk.shape[0], chunk.shape[1]
        flat = _run_frames(self.post, LIB_ACTION, flatten_history(chunk))
        return restore_history(flat, batch, horizon)


def override_processor_device(config: dict[str, Any], device: str) -> dict[str, Any]:
    """Copy a processor config with its `device_processor` steps moved to `device`.

    The shipped postprocessor ends with `device="cpu"` because `lerobot-eval` wants numpy-ready
    output. In this repo's loop that would be a device round trip per chunk, which shows up in
    the latency rows and in E10's deadline-miss count. The deviation is made *here*, in our
    code, never by editing the checkpoint's JSON (docs/10 condition 4).
    """
    out: dict[str, Any] = dict(config)
    steps: list[dict[str, Any]] = []
    for step in config.get("steps", []):
        step = dict(step)
        if step.get("registry_name") == "device_processor":
            step["config"] = {**step.get("config", {}), "device": device}
        steps.append(step)
    out["steps"] = steps
    return out


def load_processor_norm(directory: str | Path, *, device: str = "cuda") -> ProcessorsNorm:
    """Build `ProcessorsNorm` from a migrated checkpoint directory.

    Needs `lerobot` (this project's lockfile does not contain it: it caps torch < 2.12, R13), so
    the import is local and the failure names the recipe that installs it.
    """
    # Local import: the hint lives next to the rest of the backend's error text, and importing
    # `pretrained` at module scope here would be a cycle.
    from policybench.models.pretrained import _BACKEND_HINT

    try:
        from lerobot.processor import pipeline as processor_pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_BACKEND_HINT) from exc

    # 0.6.1 names the generic container `DataProcessorPipeline`; other releases use a policy
    # subclass. Resolved at call time so a rename is a message here, not an import error at the
    # top of a module that CI imports.
    pipeline_cls = getattr(processor_pipeline, "PolicyProcessorPipeline", None) or getattr(
        processor_pipeline, "DataProcessorPipeline", None
    )
    if pipeline_cls is None:  # pragma: no cover - depends on the installed lerobot
        raise ImportError(
            "no processor pipeline class in lerobot.processor.pipeline; docs/02 section 3.4 pins "
            "lerobot 0.6.1, so this is a version change, not a configuration problem"
        )

    root = Path(directory)
    required = ("policy_preprocessor.json", "policy_postprocessor.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root} has no {missing}; run `just lerobot-control` once to build the processors "
            "(scripts/lerobot_migrate.py)"
        )

    pre = pipeline_cls.from_pretrained(str(root), config_filename="policy_preprocessor.json")
    post = pipeline_cls.from_pretrained(str(root), config_filename="policy_postprocessor.json")
    _move_pipeline_device(pre, device)
    _move_pipeline_device(post, device)
    return ProcessorsNorm(pre, post)


def _move_pipeline_device(pipeline: Any, device: str) -> None:
    """Point any `device_processor` step at `device`, if the loaded pipeline kept one.

    Duck-typed on purpose: the step objects' attribute names are not part of anything this repo
    pins, and a pipeline that ignores this still *runs* (it just returns CPU tensors, which is
    slower and visible in the latency column, not wrong).
    """
    for step in getattr(pipeline, "steps", []):
        config = getattr(step, "config", None)
        if config is not None and hasattr(config, "device"):
            config.device = device


def config_id_marks_path(config_id: str, normalization: str) -> bool:
    """A non-default normalization path must be identifiable from the row's `config_id`.

    The row schema stays unchanged (docs/10 section 4.3), so provenance is the `config_id`
    prefix. This is the cheap guard against a processor-path block being appended next to a
    buffer-path block and silently read as one sample.
    """
    return normalization == "buffers" or "proc" in config_id
