"""Run a pretrained VQ-BeT checkpoint behind the same `PolicyLike` interface.

This file exists to test one claim in `docs/01`: the harness does not know which policy it
holds. `lerobot/vqbet_pusht` (Apache-2.0) is a different family from the Diffusion Policy
checkpoint -- a behaviour transformer over discretized action tokens, 5 observation steps,
a 5-action chunk, no denoising sampler -- and the only new code is this adapter. The rollout
harness, the latency protocol, the control loop and every table generator are unchanged.

Two seams need glue, and both are visible in the checkpoint rather than invented here:

- `lerobot` 0.6.1 reads `config.json` with `draccus`, which rejects the 2024 checkpoint's
  extra `mlp_hidden_dim` field. The field is dropped before decoding; the weights prove the
  rest of the config is right (zero missing keys on load).
- 0.6.1 moved normalization out of the policy, so the six `normalize_inputs.*` /
  `unnormalize_outputs.*` buffers load as *unexpected* keys and are applied here, exactly
  like `models/pretrained.py` does for the diffusion checkpoint. The mapping is the
  checkpoint's own: `VISUAL = IDENTITY` (images stay in `[0, 1]`), `STATE` and `ACTION`
  `MIN_MAX` with the dataset's min/max buffers.

VQ-BeT predicts a chunk in one forward pass, so `steps` and `noise` in `predict_chunk` are
accepted and ignored: the interface is per-chunk, not per-sampler. E0-style validation of
this file is in `results/vqbet.jsonl` (published reference: 57.0 % success over 500 episodes).
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from policybench.models.pretrained import denormalize_action, normalize_state

DEFAULT_HF_ID = "lerobot/vqbet_pusht"

# Buffers the modern library drops on load; we apply them by hand (module docstring).
_NORM_PREFIXES = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")


def download_config(repo_id: str) -> dict[str, Any]:
    """Fetch the checkpoint's `config.json` as a dict."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, "config.json"))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def filter_config(raw: dict[str, Any], config_cls: type) -> dict[str, Any]:
    """Drop keys the installed library's config dataclass no longer has.

    The 2024 checkpoint carries `mlp_hidden_dim`, which `VQBeTConfig` 0.6.1 rejects at parse
    time. Dropping unknown fields is safe for *this* direction only (old config, new class);
    a missing required field still fails, which is the check we want.
    """
    known = {field.name for field in dataclasses.fields(config_cls)}
    # `type` is the discriminator `from_pretrained` requires; it is not a dataclass field.
    keep = known | {"type"}
    return {key: value for key, value in raw.items() if key in keep}


class PretrainedVQBeTPolicy(nn.Module):
    """`lerobot/vqbet_pusht` wrapped as `PolicyLike` (docs/01 section 2)."""

    # Declared for mypy: `nn.Module.__getattr__` types registered buffers as Tensor | Module.
    state_min: Tensor
    state_max: Tensor
    action_min: Tensor
    action_max: Tensor

    def __init__(
        self,
        hf_id: str = DEFAULT_HF_ID,
        *,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        try:
            from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
            from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the pretrained-policy backend needs `lerobot` and `diffusers`; "
                "run `just setup-pretrained` (see docs/02 section 3.4)"
            ) from exc

        raw = download_config(hf_id)
        filtered = filter_config(raw, VQBeTConfig)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps(filtered), encoding="utf-8")
            self.config = VQBeTConfig.from_pretrained(tmp)

        net = VQBeTPolicy(self.config)
        from huggingface_hub import hf_hub_download

        weights = load_file(hf_hub_download(hf_id, "model.safetensors"))
        missing, unexpected = net.load_state_dict(weights, strict=False)
        if missing:
            raise ValueError(f"{hf_id} is missing weights: {missing[:5]}")
        stray = [key for key in unexpected if not key.startswith(_NORM_PREFIXES)]
        if stray:
            raise ValueError(f"{hf_id} has unexpected weights: {stray[:5]}")

        self.net = net
        self.source = hf_id
        self.obs_horizon = int(self.config.n_obs_steps)
        self.pred_horizon = int(self.config.action_chunk_size)
        self.action_dim = int(self.config.action_feature.shape[0])
        # MIN_MAX statistics from the checkpoint, registered so `.to()` moves them.
        state_stats = self._stats(weights, "normalize_inputs.buffer_observation_state")
        action_stats = self._stats(weights, "unnormalize_outputs.buffer_action")
        self.register_buffer("state_min", state_stats[0])
        self.register_buffer("state_max", state_stats[1])
        self.register_buffer("action_min", action_stats[0])
        self.register_buffer("action_max", action_stats[1])
        self.to(device)
        self.eval()

    @staticmethod
    def _stats(weights: dict[str, Tensor], prefix: str) -> tuple[Tensor, Tensor]:
        low = weights[f"{prefix}.min"].detach().clone().float()
        high = weights[f"{prefix}.max"].detach().clone().float()
        return low, high

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, Tensor], **sampler_kw: Any) -> Tensor:
        """Return raw-pixel actions `[B, pred_horizon, 2]`; `steps`/`noise` are ignored.

        VQ-BeT has no sampler to truncate, so a caller passing `steps=` (the eval and latency
        protocols both do) gets the same chunk. That is a property of the policy, not a
        silently dropped argument: `results/tables/generality.md` reports it.
        """
        image = obs["image"]
        state = obs["state"].to(self.state_min.dtype)
        if image.ndim != 5:
            raise ValueError(f"image must be [B, To, 3, H, W], got {tuple(image.shape)}")
        batch, horizon = image.shape[:2]
        if horizon != self.obs_horizon:
            raise ValueError(f"expected {self.obs_horizon} observation steps, got {horizon}")
        # The model's own layout: (batch, obs steps, cameras, C, H, W) -- one camera here.
        batch_dict = {
            "observation.images": image.to(self.state_min.dtype).unsqueeze(2),
            "observation.state": normalize_state(state, self.state_min, self.state_max),
        }
        normalized = self.net.vqbet(batch_dict, rollout=True)
        return denormalize_action(normalized, self.action_min, self.action_max).reshape(
            batch, self.pred_horizon, self.action_dim
        )
