"""End-to-end CPU smoke run (docs/01 section 4, the `smoke` CI job).

The unit suite checks components; this checks the thing they add up to: a model trained for
20 steps on synthetic data must produce a *decreasing* loss, and an interrupted-and-resumed
run must match an uninterrupted one. It is deliberately small enough for a CI runner (no GPU,
no dataset, no network) and deliberately end-to-end (real `Trainer`, real checkpoints, real
JSONL log), so a broken trainer fails here rather than in a GPU session days later.

Docs/03 section 6 calls the resume comparison tolerance-based: `rtol=1e-5`, never "bit-exact"
unless a `torch.equal` test passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flowpolicy.config import TrainConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.data.window import WindowDataset, ZarrArrays
from flowpolicy.training.trainer import Trainer

pytestmark = [pytest.mark.smoke, pytest.mark.slow]

STEPS = 20


def synthetic(episodes: int = 4, length: int = 8) -> tuple[WindowDataset, Stats]:
    """Fixed synthetic PushT-shaped data: no download, no simulator, deterministic."""
    n = episodes * length
    rng = np.random.default_rng(0)
    image = (rng.random((n, 8, 8, 3)) * 255).astype(np.uint8)
    # A learnable signal rather than pure noise: action is a scaled function of state, so a
    # working optimisation loop drives the loss down instead of fitting noise.
    state = rng.random((n, 2)).astype(np.float32) * 10
    action = (state * 0.5 + rng.normal(0, 0.1, state.shape)).astype(np.float32)
    arrays = ZarrArrays(
        image=image,
        state=state,
        action=action,
        episode_ends=np.arange(length, n + 1, length, dtype=np.int64),
    )
    return WindowDataset(arrays, np.arange(episodes), obs_horizon=2, pred_horizon=4), Stats.compute(
        state, action
    )


def config(root: Path, run_id: str = "smoke") -> TrainConfig:
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
    cfg.steps = STEPS
    cfg.batch_size = 4
    cfg.lr = 3e-3
    cfg.warmup_steps = 1
    cfg.ema_decay = 0.9
    cfg.precision = "fp32"
    cfg.device = "cpu"
    cfg.log_every = 1
    cfg.ckpt_every = 1000
    cfg.val_every = 0
    cfg.out_dir = str(root / "ckpt")
    cfg.results_dir = str(root / "results")
    cfg.run_id = run_id
    return cfg


def losses(path: Path) -> list[float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        float(row["loss"])
        for row in rows
        if "loss" in row and "event" not in row and "val_loss" not in row
    ]


def test_twenty_steps_lower_the_loss_and_resume_matches(tmp_path: Path) -> None:
    data, stats = synthetic()

    straight = Trainer(config(tmp_path / "straight"), stats, data, device="cpu")
    straight.run()
    reference = losses(straight.log_path)
    assert len(reference) == STEPS

    # Loss decreases: the last five steps must be clearly below the first five. A tolerance
    # rather than a threshold, because the absolute value depends on the toy data.
    first, last = float(np.mean(reference[:5])), float(np.mean(reference[-5:]))
    assert last < 0.95 * first, (first, last)

    # Resume equivalence: 10 steps, checkpoint, fresh Trainer, 10 more steps.
    split_cfg = config(tmp_path / "split", run_id="split")
    first_half = Trainer(split_cfg, stats, data, device="cpu")
    first_half.run(max_steps=STEPS // 2)
    head = losses(first_half.log_path)

    resumed = Trainer(split_cfg, stats, data, device="cpu")
    resumed.resume(str(Path(split_cfg.out_dir) / "split/last.pt"))
    resumed.log_path.unlink()  # keep only the resumed half in this file
    resumed.run()
    tail = losses(resumed.log_path)

    assert len(head) == STEPS // 2 and len(tail) == STEPS - STEPS // 2
    assert np.allclose(head + tail, reference, rtol=1e-5, atol=1e-7), (head + tail, reference)
