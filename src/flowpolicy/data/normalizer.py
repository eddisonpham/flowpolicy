"""Normalizer stored inside the checkpoint (docs/01 section 2, docs/03 section 1).

Stats are fit on the train split only. The module holds the buffers so a
checkpoint is self-sufficient: `policy.state_dict()` carries them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Stats:
    state_min: list[float]
    state_max: list[float]
    action_min: list[float]
    action_max: list[float]

    @staticmethod
    def from_json(path: str | Path) -> Stats:
        payload = json.loads(Path(path).read_text())
        return Stats(
            state_min=list(payload["state_min"]),
            state_max=list(payload["state_max"]),
            action_min=list(payload["action_min"]),
            action_max=list(payload["action_max"]),
        )

    @staticmethod
    def compute(state: Any, action: Any) -> Stats:
        s = torch.as_tensor(state, dtype=torch.float64)
        a = torch.as_tensor(action, dtype=torch.float64)
        return Stats(
            state_min=s.min(dim=0).values.tolist(),
            state_max=s.max(dim=0).values.tolist(),
            action_min=a.min(dim=0).values.tolist(),
            action_max=a.max(dim=0).values.tolist(),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n")


def _safe_range(low: Tensor, high: Tensor) -> Tensor:
    # A constant dimension must not divide by zero.
    return torch.where((high - low).abs() < 1e-12, torch.ones_like(high), high - low)


class Normalizer(nn.Module):
    """Min-max normalizes state and actions to [-1, 1]; images to [0, 1] -> [-1, 1]."""

    state_min: Tensor
    state_max: Tensor
    action_min: Tensor
    action_max: Tensor

    def __init__(self, stats: Stats) -> None:
        super().__init__()
        self.register_buffer("state_min", torch.tensor(stats.state_min, dtype=torch.float32))
        self.register_buffer("state_max", torch.tensor(stats.state_max, dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(stats.action_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(stats.action_max, dtype=torch.float32))

    def normalize_state(self, state: Tensor) -> Tensor:
        return 2.0 * (state - self.state_min) / _safe_range(self.state_min, self.state_max) - 1.0

    def denormalize_state(self, state: Tensor) -> Tensor:
        return (state + 1.0) * 0.5 * _safe_range(self.state_min, self.state_max) + self.state_min

    def normalize_action(self, action: Tensor) -> Tensor:
        return 2.0 * (action - self.action_min) / _safe_range(self.action_min, self.action_max) - 1.0

    def denormalize_action(self, action: Tensor) -> Tensor:
        return (action + 1.0) * 0.5 * _safe_range(self.action_min, self.action_max) + self.action_min

    @staticmethod
    def normalize_image(image: Tensor) -> Tensor:
        """uint8 [0, 255] or float [0, 1] -> float [-1, 1]."""
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return image * 2.0 - 1.0

    @staticmethod
    def denormalize_image(image: Tensor) -> Tensor:
        return (image + 1.0) * 0.5
