"""Action heads (docs/03 section 3): FiLM 1-D U-Net primary, transformer ablation."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("embedding dim must be even")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv(self.act(self.norm(x)))
        scale, shift = self.film(emb).chunk(2, dim=-1)
        h = h * (1.0 + scale[..., None]) + shift[..., None]
        return cast(Tensor, x + h)


class ConditionalUnet1D(nn.Module):
    """Small 1-D U-Net with FiLM conditioning on (time, observation)."""

    def __init__(
        self,
        action_dim: int,
        pred_horizon: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if pred_horizon % (2**depth) != 0:
            raise ValueError("pred_horizon must be divisible by 2**depth")
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.in_proj = nn.Conv1d(action_dim, hidden_dim, 1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = hidden_dim
        for _ in range(depth):
            self.down_blocks.append(
                nn.ModuleList([FiLMBlock(ch, hidden_dim), FiLMBlock(ch, hidden_dim)])
            )
            self.downsamples.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
        self.mid = nn.ModuleList([FiLMBlock(ch, hidden_dim), FiLMBlock(ch, hidden_dim)])
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            self.upsamples.append(nn.Conv1d(ch, ch, 3, padding=1))
            self.up_blocks.append(
                nn.ModuleList([FiLMBlock(ch, hidden_dim), FiLMBlock(ch, hidden_dim)])
            )
        self.out = nn.Conv1d(ch, action_dim, 1)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        # x [B,Da,T], t [B] (in [0,1]), cond [B,cond_dim]
        emb = self.time_mlp(self.time_emb(t)) + self.cond_mlp(cond)
        h = self.in_proj(x)
        skips: list[Tensor] = []
        for i in range(len(self.down_blocks)):
            blocks = cast(nn.ModuleList, self.down_blocks[i])
            for j in range(len(blocks)):
                h = cast(Tensor, blocks[j](h, emb))
            skips.append(h)
            h = cast(Tensor, self.downsamples[i](h))
        for i in range(len(self.mid)):
            h = cast(Tensor, self.mid[i](h, emb))
        for i, skip in enumerate(reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = cast(Tensor, self.upsamples[i](h)) + skip
            blocks = cast(nn.ModuleList, self.up_blocks[i])
            for j in range(len(blocks)):
                h = cast(Tensor, blocks[j](h, emb))
        return cast(Tensor, self.out(h))


class TransformerChunkHead(nn.Module):
    """ACT-style transformer: observation tokens -> learned action queries."""

    def __init__(
        self,
        cond_dim: int,
        pred_horizon: int,
        action_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.pred_horizon = pred_horizon
        self.in_proj = nn.Linear(cond_dim, d_model)
        self.queries = nn.Parameter(torch.randn(pred_horizon, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, action_dim)

    def forward(self, cond: Tensor) -> Tensor:
        memory = self.in_proj(cond)[:, None, :]
        query = self.queries[None].expand(cond.shape[0], -1, -1)
        return cast(Tensor, self.out(self.decoder(query, memory)))
