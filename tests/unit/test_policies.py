import math

import pytest
import torch
from torch import nn

from flowpolicy.config import ModelConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.models.policies import build_policy

STATS = Stats(
    state_min=[0.0, 0.0],
    state_max=[512.0, 512.0],
    action_min=[0.0, 0.0],
    action_max=[512.0, 512.0],
)
MODEL = ModelConfig(
    encoder_width=0.25,
    feature_dim=32,
    head_dim=32,
    head_depth=2,
    transformer_dim=32,
    transformer_heads=4,
    transformer_layers=1,
)

pytestmark = pytest.mark.unit


def batch(b: int = 3) -> dict[str, torch.Tensor]:
    return {
        "image": torch.rand(b, 2, 3, 8, 8),
        "state": torch.rand(b, 2, 2) * 512,
        "action": torch.rand(b, 4, 2) * 512,
    }


@pytest.mark.parametrize("kind", ["flow", "ddpm", "regression"])
def test_loss_and_chunk_interface(kind):
    policy = build_policy(kind, STATS, 2, 4, 2, 2, MODEL)
    out = policy.loss(batch())
    assert "loss" in out and torch.isfinite(out["loss"])
    obs = {"image": batch()["image"], "state": batch()["state"]}
    chunk = policy.predict_chunk(obs, steps=2)
    assert chunk.shape == (3, 4, 2)
    assert torch.isfinite(chunk).all()


@pytest.mark.parametrize("kind", ["flow", "ddpm", "regression"])
def test_no_batchnorm_in_encoder(kind):
    policy = build_policy(kind, STATS, 2, 4, 2, 2, MODEL)
    assert not any(
        isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d) for m in policy.modules()
    )


def test_normalizer_travels_with_state_dict():
    policy = build_policy("flow", STATS, 2, 4, 2, 2, MODEL)
    clone = build_policy("flow", STATS, 2, 4, 2, 2, MODEL)
    clone.load_state_dict(policy.state_dict())
    action = torch.full((1, 1, 2), 512.0)
    assert torch.allclose(
        clone.normalizer.normalize_action(action), policy.normalizer.normalize_action(action)
    )


def test_ddpm_schedule_matches_closed_form():
    steps = 100
    policy = build_policy("ddpm", STATS, 2, 4, 2, 2, MODEL, ddpm_train_steps=steps)
    ab = policy.alphas_cumprod
    t = torch.linspace(0, steps - 1, steps, dtype=torch.float64)
    expected = torch.cos(((t / steps + 0.008) / 1.008) * math.pi / 2) ** 2
    expected = (expected / expected[0]).float()
    assert ab.shape == (steps,)
    assert torch.allclose(ab, expected, atol=1e-6)
    assert float(ab[0]) == pytest.approx(1.0)
    assert bool((ab[1:] < ab[:-1]).all())
    # A per-step ratio schedule (the bug fixed 2026-10-24) stays at ~0.9999 for
    # every t, i.e. x_t is almost clean and the DDIM update is meaningless.
    assert float(ab[-1]) < 1e-3


def test_ddpm_sampler_recovers_target_with_oracle_epsilon():
    """Sampler and forward process must agree: oracle eps -> x0 back, exactly."""
    steps = 100
    policy = build_policy("ddpm", STATS, 2, 4, 2, 2, MODEL, ddpm_train_steps=steps)
    ab = policy.alphas_cumprod
    x0 = torch.rand(2, 4, 2) * 2 - 1

    class Oracle(nn.Module):
        def forward(self, x_t, t, cond):
            xt = x_t.transpose(1, 2)
            ab_t = ab[(t * steps).round().long()].view(-1, 1, 1)
            denom = (1.0 - ab_t).clamp_min(1e-8).sqrt()
            eps = (xt - ab_t.sqrt() * x0) / denom
            eps = torch.where(ab_t < 1.0, eps, torch.zeros_like(eps))
            return eps.transpose(1, 2)

    policy.net = Oracle()  # type: ignore[assignment]
    eps = torch.randn_like(x0)
    x_t = ab[-1].sqrt() * x0 + (1.0 - ab[-1]).sqrt() * eps
    out = policy._sample(torch.zeros(2, 1), steps=steps, noise=x_t)
    assert torch.allclose(out, x0, atol=1e-4)


def test_ddpm_step_count_is_effective():
    """Every requested step must denoise: steps=1 and steps=2 cannot be equal.

    alphas_cumprod[0] == 1.0, so a ladder that reaches t=0 ends on a no-op and
    makes steps=1 and steps=2 identical (the bug fixed 2026-10-24).
    """
    policy = build_policy("ddpm", STATS, 2, 4, 2, 2, MODEL, ddpm_train_steps=100)
    obs = batch()
    cond = policy.encode({"image": obs["image"], "state": obs["state"]})
    noise = torch.randn(3, 4, 2)
    one = policy._sample(cond, steps=1, noise=noise)
    two = policy._sample(cond, steps=2, noise=noise)
    assert not torch.allclose(one, two)


def test_ddpm_sampling_is_bounded_for_untrained_network():
    policy = build_policy("ddpm", STATS, 2, 4, 2, 2, MODEL)
    obs = {"image": batch()["image"], "state": batch()["state"]}
    chunk = policy.predict_chunk(obs, steps=4)
    assert chunk.abs().max() < 1000.0
