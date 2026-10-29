import numpy as np
import pytest
import torch

from policybench.data.normalizer import Normalizer, Stats

pytestmark = pytest.mark.unit


def test_stats_ignore_held_out_frames():
    state = np.array([[0.0, 10.0], [1.0, 11.0], [99.0, 99.0]], dtype=np.float32)
    action = np.array([[0.0, 0.0], [2.0, 4.0], [500.0, 500.0]], dtype=np.float32)
    train_frames = np.array([0, 1])
    stats = Stats.compute(state[train_frames], action[train_frames])
    assert stats.state_max == [1.0, 11.0]
    assert stats.action_max == [2.0, 4.0]


def test_normalize_round_trip_and_range():
    stats = Stats(
        state_min=[0.0, -2.0],
        state_max=[10.0, 2.0],
        action_min=[10.0, 20.0],
        action_max=[510.0, 520.0],
    )
    normalizer = Normalizer(stats)
    action = torch.tensor([[[10.0, 20.0], [510.0, 520.0], [260.0, 270.0]]])
    normalized = normalizer.normalize_action(action)
    assert torch.allclose(normalized[0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(normalized[0, 1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(normalizer.denormalize_action(normalized), action)
    state = torch.tensor([[[0.0, -2.0], [10.0, 2.0]]])
    assert torch.allclose(normalizer.denormalize_state(normalizer.normalize_state(state)), state)


def test_constant_dimension_does_not_divide_by_zero():
    stats = Stats(state_min=[1.0], state_max=[1.0], action_min=[0.0], action_max=[1.0])
    normalizer = Normalizer(stats)
    out = normalizer.normalize_state(torch.ones(2, 1))
    assert torch.isfinite(out).all()


def test_image_normalization_round_trip():
    image = torch.tensor([[[[0.0, 255.0]]]], dtype=torch.uint8)
    normalized = Normalizer.normalize_image(image)
    assert normalized.min() == -1.0 and normalized.max() == 1.0
    assert torch.allclose(Normalizer.denormalize_image(normalized) * 255.0, image.float())
