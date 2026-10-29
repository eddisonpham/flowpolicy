import json

import numpy as np
import pytest

from policybench.config import TrainConfig
from policybench.data.normalizer import Stats
from policybench.data.window import WindowDataset, ZarrArrays
from policybench.training.trainer import Trainer

pytestmark = pytest.mark.unit


def synthetic() -> tuple[WindowDataset, Stats]:
    episodes = 4
    length = 8
    n = episodes * length
    rng = np.random.default_rng(0)
    image = (rng.random((n, 8, 8, 3)) * 255).astype(np.uint8)
    state = rng.random((n, 2)).astype(np.float32) * 10
    action = rng.random((n, 2)).astype(np.float32) * 10
    arrays = ZarrArrays(
        image=image,
        state=state,
        action=action,
        episode_ends=np.arange(length, n + 1, length, dtype=np.int64),
    )
    stats = Stats.compute(state, action)
    return WindowDataset(arrays, np.arange(episodes), obs_horizon=2, pred_horizon=4), stats


def config(tmp_path) -> TrainConfig:
    cfg = TrainConfig()
    cfg.data.obs_horizon = 2
    cfg.data.pred_horizon = 4
    cfg.data.action_horizon = 2
    cfg.data.state_dim = 2
    cfg.data.action_dim = 2
    cfg.model.encoder_width = 0.25
    cfg.model.feature_dim = 32
    cfg.model.head_dim = 32
    cfg.model.head_depth = 2
    cfg.policy.kind = "flow"
    cfg.steps = 6
    cfg.batch_size = 4
    cfg.lr = 1e-3
    cfg.warmup_steps = 1
    cfg.ema_decay = 0.9
    cfg.precision = "fp32"
    cfg.device = "cpu"
    cfg.log_every = 1
    cfg.ckpt_every = 1000
    cfg.val_every = 0
    cfg.out_dir = str(tmp_path / "ckpt")
    cfg.results_dir = str(tmp_path / "results")
    cfg.run_id = "resume"
    return cfg


def losses(path) -> list[float]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [
        row["loss"]
        for row in rows
        if "loss" in row and "event" not in row and "val_loss" not in row
    ]


def test_resume_matches_uninterrupted_run(tmp_path):
    data, stats = synthetic()
    cfg = config(tmp_path)
    uninterrupted = Trainer(cfg, stats, data, device="cpu")
    uninterrupted.run()
    expected = losses(uninterrupted.log_path)

    interrupted_cfg = config(tmp_path / "interrupted")
    interrupted_cfg.run_id = "interrupted"
    first = Trainer(interrupted_cfg, stats, data, device="cpu")
    first.run(max_steps=3)
    first_half = losses(first.log_path)

    resumed = Trainer(interrupted_cfg, stats, data, device="cpu")
    resumed.resume(interrupted_cfg.out_dir + "/interrupted/last.pt")
    resumed.log_path.unlink()  # keep only the resumed half in this file
    resumed.run()
    second_half = losses(resumed.log_path)

    assert len(expected) == 6 and len(first_half) == 3 and len(second_half) == 3
    actual = first_half + second_half
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-7), (actual, expected)
