"""Structured configs (docs/01 section 3: Hydra, dataclasses, experiment files)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    zarr_path: str = "data/pusht/pusht.zarr"
    stats_path: str = "data/pusht/stats.json"
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    image_size: int = 96
    state_dim: int = 2
    action_dim: int = 2
    val_fraction: float = 0.1


@dataclass
class ModelConfig:
    # Encoder: ResNet-18 with GroupNorm (docs/03 section 3). width scales all channels.
    encoder_width: float = 0.75
    feature_dim: int = 256
    # 1-D U-Net head, FiLM conditioning.
    head_dim: int = 256
    head_depth: int = 2
    # Transformer head used by the ACT-style regression baseline.
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 3


@dataclass
class PolicyConfig:
    kind: str = "flow"  # flow | ddpm | regression
    flow_steps: int = 10
    flow_solver: str = "euler"  # euler | heun
    flow_time_sampling: str = "uniform"  # uniform | beta
    ddpm_train_steps: int = 100
    ddpm_sample_steps: int = 10


@dataclass
class TrainConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run_id: str = "debug"
    seed: int = 0
    steps: int = 1800
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 200
    grad_clip: float = 10.0
    ema_decay: float = 0.999
    precision: str = "bf16"  # bf16 | fp32
    log_every: int = 25
    ckpt_every: int = 1000
    val_every: int = 250
    out_dir: str = "checkpoints"
    results_dir: str = "results"
    device: str = "cuda"
    limit_train_episodes: int | None = None
    resume: str = ""

    def validate(self) -> None:
        d = self.data
        if d.action_horizon > d.pred_horizon:
            raise ValueError("action_horizon must be <= pred_horizon (docs/03 section 6)")
        if d.obs_horizon < 1 or d.pred_horizon < 1:
            raise ValueError("horizons must be positive")
        if self.policy.kind not in {"flow", "ddpm", "regression"}:
            raise ValueError(f"unknown policy kind {self.policy.kind!r}")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"unknown precision {self.precision!r}")
