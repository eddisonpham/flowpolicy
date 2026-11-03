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
    # Train-time random crop: side length in pixels. 0 disables. 84 on 96px
    # images mirrors the reference recipe; eval always uses the full image.
    augment_crop: int = 0
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
        # Fields the loop divides by, indexes with, or uses as a modulus. A zero here is not a
        # configuration the trainer can survive -- `step % log_every` is a ZeroDivisionError an
        # hour into a run -- so they are rejected before the model is built.
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1 (the loop takes step % log_every)")
        if self.val_every < 0 or self.ckpt_every < 0:
            raise ValueError("val_every and ckpt_every must be >= 0 (0 disables them)")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if (
            min(self.policy.flow_steps, self.policy.ddpm_sample_steps, self.policy.ddpm_train_steps)
            < 1
        ):
            raise ValueError("sampler step counts must be >= 1")
        if self.policy.kind not in {"flow", "ddpm", "regression"}:
            raise ValueError(f"unknown policy kind {self.policy.kind!r}")
        if self.data.augment_crop < 0 or self.data.augment_crop > self.data.image_size:
            raise ValueError("augment_crop must be 0 (off) or within the image size")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"unknown precision {self.precision!r}")
