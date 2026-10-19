"""M3 acceptance: each policy overfits 8 episodes to near-zero loss.

Writes results/overfit/<kind>.jsonl and summary.json. Run on GPU, local only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowpolicy.config import TrainConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.data.window import WindowDataset, load_zarr, split_episodes
from flowpolicy.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["flow", "ddpm", "regression"])
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()

    arrays = load_zarr("data/pusht/pusht.zarr")
    train_eps, _ = split_episodes(len(arrays.episode_ends), 0.1, seed=0)
    stats = Stats.from_json("data/pusht/stats.json")
    out_dir = Path("results/overfit")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for kind in args.policies:
        data = WindowDataset(arrays, train_eps[:8], obs_horizon=2, pred_horizon=16)
        cfg = TrainConfig()
        cfg.policy.kind = kind
        cfg.run_id = f"overfit_{kind}"
        cfg.steps = args.steps
        cfg.batch_size = 32
        cfg.lr = 3e-4
        cfg.warmup_steps = 50
        cfg.log_every = 10
        cfg.ckpt_every = 10_000
        cfg.val_every = 0
        cfg.out_dir = "checkpoints"
        cfg.results_dir = "results"
        trainer = Trainer(cfg, stats, data, val_data=None)
        run = trainer.run()
        rows = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
        losses = [r["loss"] for r in rows if "loss" in r and "event" not in r]
        summary[kind] = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "ratio": losses[-1] / losses[0],
            "steps": run["step"],
            "overfit_8_episodes": bool(losses[-1] < 0.1 * losses[0]),
        }
        (out_dir / f"{kind}.jsonl").write_text(trainer.log_path.read_text())
        print(json.dumps({kind: summary[kind]}))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
