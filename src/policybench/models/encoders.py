"""Shared encoder (docs/03 section 3): ResNet-18 with GroupNorm, no BatchNorm.

Instantiated per policy; eval and deploy never branch on policy type.
Images enter raw in [0, 1] and are normalized by the policy's Normalizer.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


def _gn(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = _gn(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), _gn(out_ch)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return cast(Tensor, self.act(out + identity))


class FrameEncoder(nn.Module):
    """One image frame -> feature vector. 96x96 needs no stem downsampling."""

    def __init__(self, width: float = 1.0, feature_dim: int = 256) -> None:
        super().__init__()
        channels = [int(64 * width), int(128 * width), int(256 * width), int(512 * width)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[1], stride=2),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[2], stride=2),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[3], stride=2),
            BasicBlock(channels[3], channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[3], feature_dim)

    def forward(self, image: Tensor) -> Tensor:
        x = self.stem(image)
        x = self.blocks(x)
        return cast(Tensor, self.proj(self.pool(x).flatten(1)))


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, state: Tensor) -> Tensor:
        return cast(Tensor, self.net(state))


class MultiFrameEncoder(nn.Module):
    """Encode To frames plus their states, then mix into one conditioning vector."""

    def __init__(self, obs_horizon: int, state_dim: int, width: float, feature_dim: int) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.feature_dim = feature_dim
        self.image_encoder = FrameEncoder(width=width, feature_dim=feature_dim)
        self.state_encoder = StateEncoder(state_dim, feature_dim)
        self.mix = nn.Sequential(
            nn.Linear(obs_horizon * feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, image: Tensor, state: Tensor) -> Tensor:
        # image [B,To,3,H,W], state [B,To,Ds]
        b, to = image.shape[:2]
        feats = self.image_encoder(image.flatten(0, 1)).view(b, to, -1)
        feats = feats + self.state_encoder(state)
        return cast(Tensor, self.mix(feats.flatten(1)))
