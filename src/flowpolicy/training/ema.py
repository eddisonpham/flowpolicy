"""Exponential moving average of weights, used for every evaluation (docs/03 section 5)."""

from __future__ import annotations

import copy

from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module, step: int) -> None:
        # Warmup: docs/03 section 5 asks for a decay warmup, the exact form is [U].
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        with_mod = self.module.state_dict()
        with_src = model.state_dict()
        for key, value in with_mod.items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(with_src[key].detach(), alpha=1.0 - decay)
            else:
                value.copy_(with_src[key])
