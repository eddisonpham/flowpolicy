"""A device-resident DDIM sampler for the pretrained Diffusion Policy.

Why this file exists
--------------------
`lerobot`'s `DiffusionModel.conditional_sample` calls `diffusers`' DDIM step once per
denoising step, and that step does its arithmetic on **host** tensors: `diffusers` keeps
`alphas_cumprod` on the CPU and indexes it with a Python int, so every step produces a
0-dim CPU tensor that is then mixed with a CUDA sample. In eager PyTorch that works (a
0-dim tensor is allowed as a scalar), but it costs

1. a host/device synchronization per step, since the CPU scalar must be read while the
   GPU queue is still filling, and
2. a graph break for `torch.compile`, which puts the scalar math into a C++ CPU partition
   and therefore needs MSVC `cl.exe` on the build path. On this host that surfaced as
   `InvalidCxxCompiler: cl is not found` and left the largest optimization class in E9
   unmeasured.

Both problems have the same fix: precompute the DDIM coefficients **once**, as device
tensors, and run the loop with tensor-only math. The result is

- numerically the same sampler (eta=0 DDIM, `set_alpha_to_one=False`, same timestep grid),
  checked against `diffusers` in `tests/unit/test_fast_sampler.py`;
- capturable by `torch.cuda.CUDAGraph` and compilable by `torch.compile` with no CPU
  partition, which is what `results/tables/e9_latency.md` measures.

The math (DDIM, eta = 0, epsilon prediction):

    x0 = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)          clipped to [-c, c]
    x_{t-1} = sqrt(a_prev) * x0 + sqrt(1 - a_prev) * eps

so each step needs four scalars, all of which are known before the loop starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DDIMCoefficients:
    """Per-step DDIM scalars, already on the sampling device.

    One entry per denoising step, in sampling order (largest timestep first).
    """

    inv_sqrt_alpha: Tensor  # 1 / sqrt(a_t)
    sqrt_beta_over_sqrt_alpha: Tensor  # sqrt(1 - a_t) / sqrt(a_t)
    sqrt_alpha_prev: Tensor  # sqrt(a_prev)
    sqrt_beta_prev: Tensor  # sqrt(1 - a_prev)
    timesteps: Tensor  # int64 model timesteps, same order

    def __len__(self) -> int:
        return int(self.timesteps.numel())


def ddim_timesteps(num_train_timesteps: int, steps: int) -> list[int]:
    """The timestep grid `DDIMScheduler.set_timesteps` builds, as Python ints."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps > num_train_timesteps:
        raise ValueError("steps cannot exceed num_train_timesteps")
    ratio = num_train_timesteps // steps
    return [i * ratio for i in range(steps)][::-1]


def ddim_coefficients(
    alphas_cumprod: Tensor,
    num_train_timesteps: int,
    steps: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    set_alpha_to_one: bool = False,
) -> DDIMCoefficients:
    """Precompute the DDIM scalars for `steps` steps of a `num_train_timesteps` schedule.

    `alphas_cumprod` is the training schedule (length `num_train_timesteps`); it stays
    where the caller has it and only the derived scalars are moved to `device`.
    """
    ac = alphas_cumprod.detach().to(device="cpu", dtype=torch.float64).flatten()
    ratio = num_train_timesteps // steps
    final_alpha = torch.tensor(1.0, dtype=torch.float64) if set_alpha_to_one else ac[0]

    inv_sqrt_alpha: list[float] = []
    sqrt_beta_over_sqrt_alpha: list[float] = []
    sqrt_alpha_prev: list[float] = []
    sqrt_beta_prev: list[float] = []
    for t in ddim_timesteps(num_train_timesteps, steps):
        prev_t = t - ratio
        alpha_t = ac[t]
        alpha_prev = ac[prev_t] if prev_t >= 0 else final_alpha
        sqrt_alpha = alpha_t.sqrt()
        inv_sqrt_alpha.append(float(1.0 / sqrt_alpha))
        sqrt_beta_over_sqrt_alpha.append(float((1.0 - alpha_t).sqrt() / sqrt_alpha))
        sqrt_alpha_prev.append(float(alpha_prev.sqrt()))
        sqrt_beta_prev.append(float((1.0 - alpha_prev).sqrt()))

    return DDIMCoefficients(
        inv_sqrt_alpha=torch.tensor(inv_sqrt_alpha, device=device, dtype=dtype),
        sqrt_beta_over_sqrt_alpha=torch.tensor(
            sqrt_beta_over_sqrt_alpha, device=device, dtype=dtype
        ),
        sqrt_alpha_prev=torch.tensor(sqrt_alpha_prev, device=device, dtype=dtype),
        sqrt_beta_prev=torch.tensor(sqrt_beta_prev, device=device, dtype=dtype),
        timesteps=torch.tensor(
            ddim_timesteps(num_train_timesteps, steps), device=device, dtype=torch.long
        ),
    )


def ddim_sample(
    predict_epsilon: object,
    noise: Tensor,
    coeffs: DDIMCoefficients,
    *,
    clip_sample: bool = True,
    clip_range: float = 1.0,
) -> Tensor:
    """Run the DDIM loop with tensor-only math.

    `predict_epsilon(x_t, t)` must return the predicted noise for timestep `t` (a 0-dim
    int64 tensor on the same device as `x_t`). Everything else here is elementwise
    arithmetic, so the whole loop is a single compilable/capturable graph.
    """
    assert callable(predict_epsilon)
    sample = noise
    batch = sample.shape[0]
    for i in range(len(coeffs)):
        t = coeffs.timesteps[i].expand(batch)
        eps = predict_epsilon(sample, t)
        # x0 = x_t / sqrt(a_t) - eps * sqrt(1 - a_t) / sqrt(a_t)
        x0 = coeffs.inv_sqrt_alpha[i] * sample - coeffs.sqrt_beta_over_sqrt_alpha[i] * eps
        if clip_sample:
            x0 = x0.clamp(-clip_range, clip_range)
        sample = coeffs.sqrt_alpha_prev[i] * x0 + coeffs.sqrt_beta_prev[i] * eps
    return sample
