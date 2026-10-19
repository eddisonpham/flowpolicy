import numpy as np
import pytest
import torch

from flowpolicy.data.window import WindowDataset, ZarrArrays, split_episodes

pytestmark = pytest.mark.unit


def tiny_arrays() -> ZarrArrays:
    # two episodes, 5 and 4 frames, 2x2 images
    n = 9
    image = np.arange(n * 2 * 2 * 3, dtype=np.uint8).reshape(n, 2, 2, 3)
    state = np.stack([np.arange(n), np.arange(n) * 2], axis=1).astype(np.float32)
    action = np.stack([np.arange(n) * 3, np.arange(n) * 4], axis=1).astype(np.float32)
    return ZarrArrays(
        image=image, state=state, action=action, episode_ends=np.array([5, 9], dtype=np.int64)
    )


def test_episode_split_is_disjoint_and_deterministic():
    train, val = split_episodes(10, 0.2, seed=3)
    assert len(train) == 8 and len(val) == 2
    assert not set(train.tolist()) & set(val.tolist())
    train2, val2 = split_episodes(10, 0.2, seed=3)
    assert np.array_equal(train, train2) and np.array_equal(val, val2)


def test_window_pads_edges_by_repeating():
    ds = WindowDataset(tiny_arrays(), np.array([0, 1]), obs_horizon=2, pred_horizon=4)
    obs, act = ds.window_indices(0)  # first frame of episode 0
    assert obs.tolist() == [0, 0]
    assert act.tolist() == [0, 1, 2, 3]
    obs, act = ds.window_indices(4)  # last frame of episode 0
    assert obs.tolist() == [3, 4]
    assert act.tolist() == [4, 4, 4, 4]


def test_window_respects_episode_boundary():
    ds = WindowDataset(tiny_arrays(), np.array([0, 1]), obs_horizon=2, pred_horizon=4)
    obs, act = ds.window_indices(5)  # first frame of episode 1
    assert obs.tolist() == [5, 5]
    assert act.tolist() == [5, 6, 7, 8]


def test_batch_shapes_and_normalization_range():
    ds = WindowDataset(tiny_arrays(), np.array([0, 1]), obs_horizon=2, pred_horizon=4)
    batch = ds.get_batch(np.array([0, 5, 8]))
    assert batch["image"].shape == (3, 2, 3, 2, 2)
    assert batch["state"].shape == (3, 2, 2)
    assert batch["action"].shape == (3, 4, 2)
    assert torch.all(batch["image"] >= 0) and torch.all(batch["image"] <= 1)
    assert torch.equal(batch["action"][0, 0], torch.tensor([0.0, 0.0]))


def test_action_horizon_bigger_than_pred_horizon_is_rejected():
    from flowpolicy.config import TrainConfig

    cfg = TrainConfig()
    cfg.data.action_horizon = cfg.data.pred_horizon + 1
    with pytest.raises(ValueError):
        cfg.validate()
