import json

import numpy as np
import pytest

from flowpolicy.config import TrainConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.data.window import WindowDataset, ZarrArrays
from flowpolicy.training.trainer import Trainer

pytestmark = pytest.mark.unit


def test_tiny_run_reduces_loss(tmp_path):
    episodes, length = 4, 8
    n = episodes * length
    rng = np.random.default_rng(0)
    state = rng.random((n, 2)).astype(np.float32) * 10
    action = rng.random((n, 2)).astype(np.float32) * 10
    arrays = ZarrArrays(
        image=(rng.random((n, 8, 8, 3)) * 255).astype(np.uint8),
        state=state,
        action=action,
        episode_ends=np.arange(length, n + 1, length, dtype=np.int64),
    )
    data = WindowDataset(arrays, np.arange(episodes), obs_horizon=2, pred_horizon=4)
    stats = Stats.compute(state, action)

    cfg = TrainConfig()
    cfg.data.pred_horizon = 4
    cfg.data.action_horizon = 2
    cfg.model.encoder_width = 0.25
    cfg.model.feature_dim = 32
    cfg.model.head_dim = 32
    cfg.model.head_depth = 2
    cfg.steps = 10
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
    cfg.run_id = "smoke"

    trainer = Trainer(cfg, stats, data, device="cpu")
    trainer.run()
    rows = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
    values = [r["loss"] for r in rows if "loss" in r and "event" not in r]
    assert len(values) == 10
    assert values[-1] < values[0], values
