"""The chunk space an RTC-guided sampler works in, and the map to the harness's raw units.

`PolicyLike` makes one promise about units: observations and actions cross the boundary in the
environment's own units (`docs/ARCHITECTURE.md`). RTC is the one path where that is not enough,
because its guidance acts on `x_t` *inside* the flow loop -- the checkpoint's space, not the
environment's: the action normalized over its 7 dims and zero-padded to the flow head's 32.
`lerobot` never hands that space back: `_get_action_chunk` slices the pad off before returning
(`modeling_smolvla.py` 217) and `lerobot_eval` unnormalizes afterwards (line 296). So an adapter
that wants to guide a chunk has to rebuild the previous chunk's unexecuted tail in that space
from the raw units `deploy/control_loop.py` already holds.

Rebuilding it wrongly is silent: guidance pulls the new chunk toward the wrong region, the policy
returns plausible actions, and the success rate means nothing -- the defect shape `docs/10` was
written about. So the mapping lives here once, and `scripts/smolvla_leftover_probe.py` measures
*this* arithmetic rather than a copy of it. When the VLA adapter lands it must import this class
for the same reason.

Two conventions, read off `lerobot` 0.6.1 rather than assumed:

* **Normalize, then pad.** The processor normalizes the 7-wide `action` feature; `prepare_action`
  pads afterwards with `pad_vector(batch[ACTION], max_action_dim)`, and `pad_vector` fills with
  zeros. Both directions therefore unpad before unnormalizing.
* **The pad is zeros, and it is discarded on the way back.** `RTCProcessor.denoise_step` fills a
  short leftover with `torch.zeros` up to the chunk's shape, so zero is the value the library
  itself uses for "nothing was executed here". On the way out the pad carries whatever the
  denoiser produced and must not reach an actuator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["PaddedActionSpace"]


@dataclass(frozen=True)
class PaddedActionSpace:
    """`raw env units -> normalized, zero-padded model space`, and its exact inverse.

    The transform is one affine per action dimension, `(x - center) / spread`, which is how both
    of `lerobot`'s declared normalizations are written:

    * `MEAN_STD`: `center` is the mean, `spread` the standard deviation.
    * `MIN_MAX`: `center = (low + high) / 2`, `spread = (high - low) / 2`, since
      `2 * (x - low) / (high - low) - 1` is that same affine. One form, not two code paths.

    Dimension 7 of the environment's action becomes 32 of the flow head's; the same 32 come back
    and only the first 7 are meaningful.
    """

    center: Tensor
    spread: Tensor
    model_dim: int

    def __post_init__(self) -> None:
        if self.center.ndim != 1 or self.spread.ndim != 1:
            raise ValueError(
                f"center and spread must be 1-D per action dimension, got "
                f"{tuple(self.center.shape)} and {tuple(self.spread.shape)}"
            )
        if self.center.shape != self.spread.shape:
            raise ValueError(
                f"center has {self.center.numel()} dims and spread has {self.spread.numel()}; "
                "one pair per action dimension"
            )
        if self.model_dim < self.center.numel():
            raise ValueError(
                f"model_dim {self.model_dim} is narrower than the {self.center.numel()}-dim "
                "action; the pad only ever adds dims"
            )
        if not torch.isfinite(self.center).all() or not torch.isfinite(self.spread).all():
            raise ValueError("center and spread must be finite")
        if (self.spread <= 0).any():
            # A zero spread is a constant action dimension in the training data, which makes the
            # affine non-invertible. Round-tripping would then return `center` for every input and
            # silently describe a different action, so this is refused rather than clamped.
            raise ValueError(
                f"spread must be positive per dim, got {self.spread.tolist()}; a zero spread "
                "makes the raw<->model map non-invertible"
            )
        if self.model_dim == self.center.numel():
            # Not an error, but the probe's padding checks become vacuous, so say it once.
            pass

    @property
    def action_dim(self) -> int:
        """The environment's action width (7 for LIBERO, 2 for PushT)."""
        return int(self.center.numel())

    def to_model(self, chunk: Tensor) -> Tensor:
        """Raw env units -> normalized, zero-padded chunk space: the units of a leftover."""
        self._check(chunk, self.action_dim, "raw")
        scaled = (chunk - self._as(chunk, self.center)) / self._as(chunk, self.spread)
        pad = chunk.new_zeros(*chunk.shape[:-1], self.model_dim - self.action_dim)
        return torch.cat([scaled, pad], dim=-1)

    def to_raw(self, chunk: Tensor) -> Tensor:
        """Chunk space -> raw env units. The pad is sliced off first, so it cannot leak out."""
        self._check(chunk, self.model_dim, "model")
        valid = chunk[..., : self.action_dim]
        return valid * self._as(chunk, self.spread) + self._as(chunk, self.center)

    @staticmethod
    def _as(chunk: Tensor, stats: Tensor) -> Tensor:
        """Move a statistic alongside the chunk: device and dtype both, never just one."""
        return stats.to(device=chunk.device, dtype=chunk.dtype)

    @staticmethod
    def _check(chunk: Tensor, width: int, space: str) -> None:
        if chunk.ndim < 2:
            raise ValueError(
                f"expected a chunk of at least [T, {width}] ({space} units), "
                f"got {tuple(chunk.shape)}"
            )
        if chunk.shape[-1] != width:
            raise ValueError(
                f"expected {width} {space}-space dims on the last axis, got {chunk.shape[-1]}; "
                "the model axis is normalized-then-padded and the raw axis is neither"
            )
