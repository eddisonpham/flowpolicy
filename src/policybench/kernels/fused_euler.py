"""One Triton kernel (docs/05 section 1 step 5): fused Euler update + de-normalization.

Replaces the last sampler update and the de-normalization pass with one kernel:
    out = (x + dt * v) * scale[channel] + bias[channel]
where scale/bias come from the train-only min-max action stats. It is a
drop-in for the last step only; the honest benchmark compares it against the
same operation in eager and compiled PyTorch.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:  # triton is unavailable on some platforms; callers report that honestly
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the platform
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _euler_denorm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        v_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        dt: tl.constexpr,
        n_elements: tl.constexpr,
        n_channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Triton rejects annotations that resolve to runtime globals (typing.Any),
        # so the pointer arguments stay unannotated.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        v = tl.load(v_ptr + offs, mask=mask)
        ch = offs % n_channels
        scale = tl.load(scale_ptr + ch, mask=mask)
        bias = tl.load(bias_ptr + ch, mask=mask)
        tl.store(out_ptr + offs, (x + dt * v) * scale + bias, mask=mask)


def reference_euler_denorm(x: Tensor, v: Tensor, dt: float, scale: Tensor, bias: Tensor) -> Tensor:
    return (x + dt * v) * scale + bias


def fused_euler_denorm(x: Tensor, v: Tensor, dt: float, scale: Tensor, bias: Tensor) -> Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed")
    if x.shape != v.shape or x.ndim != 3:
        raise ValueError("x and v must be [B, T, D] and equal")
    x = x.contiguous()
    v = v.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _euler_denorm_kernel[grid](
        x,
        v,
        scale.contiguous(),
        bias.contiguous(),
        out,
        dt=float(dt),
        n_elements=n,
        n_channels=x.shape[-1],
        BLOCK=block,
    )
    return out
