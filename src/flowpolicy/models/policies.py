"""Policies implementing the single interface in docs/01 section 2.

`loss(batch)` takes raw (un-normalized) data from the dataset; `predict_chunk`
takes raw observations and returns de-normalized actions. Normalization lives
inside the policy and in the checkpoint.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, cast

import torch
from torch import Tensor, nn

from flowpolicy.config import ModelConfig
from flowpolicy.data.normalizer import Normalizer, Stats
from flowpolicy.models.encoders import MultiFrameEncoder
from flowpolicy.models.heads import ConditionalUnet1D, TransformerChunkHead


class BasePolicy(nn.Module, ABC):
    def __init__(
        self,
        *,
        stats: Stats,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        model: ModelConfig,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.normalizer: Normalizer = Normalizer(stats)
        self.encoder: MultiFrameEncoder = MultiFrameEncoder(
            obs_horizon, state_dim, model.encoder_width, model.feature_dim
        )

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        image = self.normalizer.normalize_image(obs["image"])
        state = self.normalizer.normalize_state(obs["state"])
        return cast(Tensor, self.encoder(image, state))

    @abstractmethod
    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Training loss; must include the key \"loss\"."""

    @abstractmethod
    def _sample(self, cond: Tensor, **kwargs: Any) -> Tensor:
        """Return normalized actions [B, Tp, Da]."""

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, Tensor], **sampler_kw: Any) -> Tensor:
        cond = self.encode(obs)
        return self.normalizer.denormalize_action(self._sample(cond, **sampler_kw))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FlowPolicy(BasePolicy):
    """Conditional flow matching, t=0 noise, t=1 data (docs/03 section 3)."""

    def __init__(
        self,
        *,
        stats: Stats,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        model: ModelConfig,
    ) -> None:
        super().__init__(
            stats=stats,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
        )
        self.net: ConditionalUnet1D = ConditionalUnet1D(
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            cond_dim=model.feature_dim,
            hidden_dim=model.head_dim,
            depth=model.head_depth,
        )
        self.time_sampling = "uniform"

    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        action = self.normalizer.normalize_action(batch["action"])
        cond = self.encode(batch)
        noise = torch.randn_like(action)
        b = action.shape[0]
        if self.time_sampling == "beta":
            t = torch.distributions.Beta(1.5, 1.0).sample((b,)).to(action.device)
        else:
            t = torch.rand((b,), device=action.device)
        tt = t[:, None, None]
        x_t = (1.0 - tt) * noise + tt * action
        velocity = action - noise
        pred = self.net(x_t.transpose(1, 2), t, cond).transpose(1, 2)
        loss = nn.functional.mse_loss(pred, velocity)
        return {"loss": loss, "mse": loss.detach()}

    def _sample(
        self,
        cond: Tensor,
        steps: int = 10,
        solver: str = "euler",
        noise: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        x = (
            torch.randn((cond.shape[0], self.pred_horizon, self.action_dim), device=cond.device)
            if noise is None
            else noise.clone()
        )
        if solver == "heun":
            for k in range(steps):
                t0 = k / steps
                t1 = (k + 1) / steps
                t0_batch = torch.full((cond.shape[0],), t0, device=cond.device)
                t1_batch = torch.full((cond.shape[0],), t1, device=cond.device)
                v0 = self.net(x.transpose(1, 2), t0_batch, cond)
                x_euler = x + v0.transpose(1, 2) / steps
                v1 = self.net(x_euler.transpose(1, 2), t1_batch, cond)
                x = x + (v0 + v1).transpose(1, 2) * (0.5 / steps)
            return x
        for k in range(steps):
            t = torch.full((cond.shape[0],), k / steps, device=cond.device)
            v = self.net(x.transpose(1, 2), t, cond)
            x = x + v.transpose(1, 2) / steps
        return x


class DDPMPolicy(BasePolicy):
    """Epsilon-prediction DDPM, squared-cosine schedule, DDIM sampling."""

    def __init__(
        self,
        *,
        stats: Stats,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        model: ModelConfig,
        train_steps: int = 100,
    ) -> None:
        super().__init__(
            stats=stats,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
        )
        self.net: ConditionalUnet1D = ConditionalUnet1D(
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            cond_dim=model.feature_dim,
            hidden_dim=model.head_dim,
            depth=model.head_depth,
        )
        self.train_steps = train_steps
        self.alphas_cumprod: Tensor
        self.register_buffer("alphas_cumprod", _squared_cosine_schedule(train_steps))

    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        action = self.normalizer.normalize_action(batch["action"])
        cond = self.encode(batch)
        noise = torch.randn_like(action)
        t = torch.randint(0, self.train_steps, (action.shape[0],), device=action.device)
        ab = self.alphas_cumprod[t].view(-1, 1, 1)
        x_t = ab.sqrt() * action + (1.0 - ab).sqrt() * noise
        pred = self.net(x_t.transpose(1, 2), t.float() / self.train_steps, cond).transpose(1, 2)
        loss = nn.functional.mse_loss(pred, noise)
        return {"loss": loss, "mse": loss.detach()}

    def _sample(
        self, cond: Tensor, steps: int = 10, noise: Tensor | None = None, **_: Any
    ) -> Tensor:
        x = (
            torch.randn((cond.shape[0], self.pred_horizon, self.action_dim), device=cond.device)
            if noise is None
            else noise.clone()
        )
        values = torch.linspace(self.train_steps - 1, 0, steps).round().long().tolist()
        timesteps = sorted(set(values), reverse=True)
        for i, t in enumerate(timesteps):
            ab_t = self.alphas_cumprod[t]
            prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            ab_prev = (
                self.alphas_cumprod[prev] if prev >= 0 else torch.tensor(1.0, device=cond.device)
            )
            t_batch = torch.full((cond.shape[0],), float(t) / self.train_steps, device=cond.device)
            eps = self.net(x.transpose(1, 2), t_batch, cond).transpose(1, 2)
            x0 = ((x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1.0, 1.0)
            x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
        return x


class RegressionPolicy(BasePolicy):
    """ACT-style deterministic chunk regression with L1 loss."""

    def __init__(
        self,
        *,
        stats: Stats,
        obs_horizon: int,
        pred_horizon: int,
        state_dim: int,
        action_dim: int,
        model: ModelConfig,
    ) -> None:
        super().__init__(
            stats=stats,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
        )
        self.head: TransformerChunkHead = TransformerChunkHead(
            cond_dim=model.feature_dim,
            pred_horizon=self.pred_horizon,
            action_dim=self.action_dim,
            d_model=model.transformer_dim,
            n_heads=model.transformer_heads,
            n_layers=model.transformer_layers,
        )

    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        action = self.normalizer.normalize_action(batch["action"])
        cond = self.encode(batch)
        pred = self.head(cond)
        loss = nn.functional.l1_loss(pred, action)
        return {"loss": loss, "l1": loss.detach()}

    def _sample(self, cond: Tensor, **_: Any) -> Tensor:
        return cast(Tensor, self.head(cond))


def _squared_cosine_schedule(steps: int, s: float = 0.008) -> Tensor:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    acp = torch.cos(((t / steps + s) / (1.0 + s)) * math.pi / 2) ** 2
    acp = acp / acp[0]
    return (acp[1:] / acp[:-1]).float()


def build_policy(
    kind: str,
    stats: Stats,
    obs_horizon: int,
    pred_horizon: int,
    state_dim: int,
    action_dim: int,
    model: ModelConfig,
    *,
    ddpm_train_steps: int = 100,
    flow_time_sampling: str = "uniform",
) -> BasePolicy:
    if kind == "flow":
        policy = FlowPolicy(
            stats=stats,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
        )
        policy.time_sampling = flow_time_sampling
        return policy
    if kind == "ddpm":
        return DDPMPolicy(
            stats=stats,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
            train_steps=ddpm_train_steps,
        )
    if kind == "regression":
        return RegressionPolicy(
            stats=stats,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
        )
    raise ValueError(f"unknown policy kind {kind!r}")
