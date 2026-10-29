"""The fast DDIM sampler must be the same sampler, not merely a similar one.

`policybench.models.fast_sampler` exists so the denoising loop has no host scalar math
(see its module docstring). That is only worth having if the numbers are unchanged, so
these tests run a toy epsilon-predictor through `diffusers`' own `DDIMScheduler` and
through our loop and require the trajectories to agree. CPU only, no checkpoint, no GPU:
this is what CI runs.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from policybench.models.fast_sampler import ddim_coefficients, ddim_sample, ddim_timesteps


def _scheduler(**overrides: object) -> DDIMScheduler:
    kwargs: dict[str, object] = {
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "clip_sample_range": 1.0,
        "prediction_type": "epsilon",
        "set_alpha_to_one": False,
        "steps_offset": 0,
    }
    kwargs.update(overrides)
    return DDIMScheduler(**kwargs)


def _predict_epsilon(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """A deterministic stand-in for the U-Net: depends on both x and t."""
    tf = t.float().reshape(-1, 1, 1) / 100.0
    return 0.1 * x + 0.05 * torch.sin(tf) + 0.01 * torch.ones_like(x)


def _diffusers_trajectory(
    scheduler: DDIMScheduler, noise: torch.Tensor, steps: int
) -> torch.Tensor:
    scheduler.set_timesteps(steps)
    sample = noise
    for t in scheduler.timesteps:
        eps = _predict_epsilon(sample, t.expand(sample.shape[0]))
        sample = scheduler.step(eps, t, sample, generator=None).prev_sample
    return sample


def test_timestep_grid_matches_diffusers() -> None:
    for steps in (1, 2, 4, 10, 16):
        scheduler = _scheduler()
        scheduler.set_timesteps(steps)
        assert ddim_timesteps(100, steps) == [int(t) for t in scheduler.timesteps], steps


@pytest.mark.parametrize("steps", [1, 2, 4, 10, 16])
def test_loop_matches_diffusers_ddim(steps: int) -> None:
    torch.manual_seed(0)
    noise = torch.randn(2, 16, 2)
    scheduler = _scheduler()
    expected = _diffusers_trajectory(scheduler, noise, steps)

    coeffs = ddim_coefficients(scheduler.alphas_cumprod, 100, steps)
    got = ddim_sample(_predict_epsilon, noise, coeffs, clip_sample=True, clip_range=1.0)

    # fp32 arithmetic in a slightly different order; both agree to ~1e-6.
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_linear_schedule_matches_diffusers() -> None:
    """The checkpoint's schedule is configurable; do not bake in one beta schedule."""
    torch.manual_seed(1)
    noise = torch.randn(1, 16, 2)
    scheduler = _scheduler(beta_schedule="linear")
    expected = _diffusers_trajectory(scheduler, noise, 10)
    coeffs = ddim_coefficients(scheduler.alphas_cumprod, 100, 10)
    got = ddim_sample(_predict_epsilon, noise, coeffs)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_no_host_tensors_in_the_loop() -> None:
    """The point of the module: every coefficient is on the sampling device.

    With a CPU 'device' this cannot tell the two apart, so it checks the property that
    matters for compile/capture: the loop reads only tensors, never Python floats.
    """
    coeffs = ddim_coefficients(_scheduler().alphas_cumprod, 100, 10)
    for name in (
        "inv_sqrt_alpha",
        "sqrt_beta_over_sqrt_alpha",
        "sqrt_alpha_prev",
        "sqrt_beta_prev",
    ):
        tensor = getattr(coeffs, name)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (10,)
    assert coeffs.timesteps.tolist() == ddim_timesteps(100, 10)


def test_rejects_bad_step_counts() -> None:
    with pytest.raises(ValueError):
        ddim_timesteps(100, 0)
    with pytest.raises(ValueError):
        ddim_timesteps(100, 101)
