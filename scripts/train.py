"""Train one policy configuration. Hydra configs live in conf/.

Examples:
  uv run python scripts/train.py policy.kind=flow run_id=e1_flow_s0 seed=0
  uv run python scripts/train.py --multirun policy.kind=flow,ddpm,regression seed=0,1,2 \
      'run_id=e1_${policy.kind}_s${seed}'
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from flowpolicy.config import TrainConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.data.window import WindowDataset, load_zarr, split_episodes
from flowpolicy.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg)
    conf: TrainConfig = OmegaConf.to_object(merged)
    conf.validate()

    stats = Stats.from_json(conf.data.stats_path)
    arrays = load_zarr(conf.data.zarr_path)
    train_eps, val_eps = split_episodes(len(arrays.episode_ends), conf.data.val_fraction, seed=0)
    if conf.limit_train_episodes is not None:
        train_eps = train_eps[: conf.limit_train_episodes]
        val_eps = val_eps
    train_data = WindowDataset(arrays, train_eps, conf.data.obs_horizon, conf.data.pred_horizon)
    val_data = WindowDataset(arrays, val_eps, conf.data.obs_horizon, conf.data.pred_horizon)

    trainer = Trainer(conf, stats, train_data, val_data)
    if conf.resume:
        trainer.resume(conf.resume)
        print(json.dumps({"resumed_from": conf.resume, "step": trainer.step}), flush=True)
    summary = trainer.run()
    Path(conf.results_dir, "train", f"{conf.run_id}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
