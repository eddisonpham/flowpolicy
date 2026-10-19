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


def test_ddpm_sampling_is_bounded_for_untrained_network():
    policy = build_policy("ddpm", STATS, 2, 4, 2, 2, MODEL)
    obs = {"image": batch()["image"], "state": batch()["state"]}
    chunk = policy.predict_chunk(obs, steps=4)
    assert chunk.abs().max() < 1000.0
