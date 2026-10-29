"""Run a pretrained Diffusion Policy checkpoint behind the repo's Policy interface.

The checkpoint is `lerobot/diffusion_pusht` (Apache-2.0): Diffusion Policy trained on
gym-pusht with n_obs_steps=2, horizon=16, n_action_steps=8 -- the same observation
horizon and action chunk this repo uses, and the same environment and success metric
(max overlap >= 95 %) that `policybench.envs.pusht` measures.

Model code is the `lerobot` package's. What this module owns is the glue, because the
2024 checkpoint and the 2026 library disagree about where normalization lives:

- `lerobot` >= 0.6 moved normalization out of the policy into a processor pipeline that
  this checkpoint does not ship, so `from_pretrained` logs the eight `normalize_inputs.*`
  / `unnormalize_outputs.*` tensors as unexpected keys and drops them. They are still the
  statistics the weights were trained with, so we read them from `model.safetensors`.
- The three transforms below (`MEAN_STD` for images, `MIN_MAX` for state and actions)
  are the checkpoint's `normalization_mapping`, not this repo's train-time min/max.

Getting any of that wrong produces plausible-looking actions and a success rate near
zero, so docs/04 keeps the published 65.4 % as the acceptance gate for this file.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from policybench.models.fast_sampler import DDIMCoefficients, ddim_coefficients, ddim_sample

DEFAULT_HF_ID = "lerobot/diffusion_pusht"

# E8 (docs/04 section 2): the checkpoint is fp32, so mixed precision is a *choice* with a
# correctness risk, not a free win. Names are accepted in both spellings so a runner flag
# and a docs table can use whichever reads better.
DTYPE_NAMES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a CLI/docs name to a `torch.dtype`, or say what is allowed."""
    try:
        return DTYPE_NAMES[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(DTYPE_NAMES)}") from exc


# `lerobot` 0.6.1 requires torch < 2.12 and this project pins torch 2.14.1. The import
# happens lazily so that the rest of the package (eval of trained checkpoints, tests,
# CI) does not depend on the optional backend being installed.
_BACKEND_HINT = (
    "the pretrained-policy backend needs `lerobot` and `diffusers`; "
    "run `just setup-pretrained` (see docs/02 section 3.4)"
)


def import_backend() -> tuple[Any, Any]:
    """Import the optional backend, with a message that says what to run."""
    try:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_BACKEND_HINT) from exc
    return DiffusionPolicy, DDIMScheduler


def normalize_image(image: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """VISUAL = MEAN_STD: uint8-derived [0, 1] image to zero-mean unit-variance."""
    return (image - mean) / std


def normalize_state(state: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """STATE = MIN_MAX: raw pixels to [-1, 1] with the dataset's own extremes."""
    return 2.0 * (state - low) / (high - low) - 1.0


def denormalize_action(action: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """ACTION = MIN_MAX: [-1, 1] back to pixels, the inverse of `normalize_state`."""
    return (action + 1.0) * 0.5 * (high - low) + low


def load_normalization(hf_id: str = DEFAULT_HF_ID) -> dict[str, Tensor]:
    """Read the checkpoint's normalization buffers (see the module docstring)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(hf_id, "model.safetensors")
    keys = {
        "image_mean": "normalize_inputs.buffer_observation_image.mean",
        "image_std": "normalize_inputs.buffer_observation_image.std",
        "state_low": "normalize_inputs.buffer_observation_state.min",
        "state_high": "normalize_inputs.buffer_observation_state.max",
        "action_low": "unnormalize_outputs.buffer_action.min",
        "action_high": "unnormalize_outputs.buffer_action.max",
    }
    stats: dict[str, Tensor] = {}
    with safe_open(path, framework="pt") as handle:
        available = set(handle.keys())
        missing = [key for key in keys.values() if key not in available]
        if missing:
            raise ValueError(f"{hf_id} is missing normalization buffers: {missing}")
        for name, key in keys.items():
            stats[name] = handle.get_tensor(key).clone()
    return stats


class PretrainedDiffusionPolicy(nn.Module):
    """Diffusion Policy checkpoint wrapped as `PolicyLike` (docs/01 section 2).

    `scheduler` picks how the reverse process is truncated: "ddim" (deterministic, the
    only sane choice when `steps` < num_train_timesteps) or "ddpm" (the checkpoint's own
    scheduler, stochastic at every step, which is what the published 65.4 % used).

    `sampler` picks *whose* denoising loop runs for DDIM:

    - "stock": lerobot -> diffusers `DDIMScheduler.step` per step. The reference path, and
      the one this repo's E0 reproduction was measured with.
    - "fast": `policybench.models.fast_sampler`, which precomputes the DDIM scalars on the
      device and keeps the loop in tensor math. Numerically the same sampler (checked
      against diffusers in tests/unit/test_fast_sampler.py), but compilable and capturable
      because it has no host-side scalar math. This is what makes `torch.compile` and
      `torch.cuda.CUDAGraph` usable on this checkpoint at all; see docs/05 section 1.
    """

    def __init__(
        self,
        hf_id: str = DEFAULT_HF_ID,
        *,
        device: str = "cuda",
        scheduler: str = "ddim",
        sampler: str = "stock",
        dtype: str = "float32",
        pretrained_step: int = 175_000,
    ) -> None:
        super().__init__()
        DiffusionPolicy, DDIMScheduler = import_backend()
        self.hf_id = hf_id
        self.policy = DiffusionPolicy.from_pretrained(hf_id).eval()
        cfg = self.policy.config
        self.obs_horizon = int(cfg.n_obs_steps)
        self.pred_horizon = int(cfg.horizon)
        self.action_horizon = int(cfg.n_action_steps)
        self.action_dim = int(cfg.action_feature.shape[0])
        self.pretrained_step = pretrained_step
        self.scheduler = scheduler
        if sampler not in ("stock", "fast"):
            raise ValueError(f"unknown sampler {sampler!r}")
        if sampler == "fast" and scheduler != "ddim":
            raise ValueError("the fast sampler implements DDIM only (eta=0); use scheduler='ddim'")
        self.sampler = sampler
        self.torch_dtype = resolve_dtype(dtype)
        # The stock loop hands the tensors to `diffusers`, whose scheduler does host-side
        # scalar math on fp32 schedule tensors; casting it is not something this repo has
        # verified, so half precision is offered only on the path we own (docs/04 E8).
        if self.torch_dtype != torch.float32 and sampler != "fast":
            raise ValueError(
                "mixed-precision inference is implemented for sampler='fast' only; "
                "the stock lerobot/diffusers loop is fp32 (docs/04 E8)"
            )
        self._coeff_cache: dict[int, DDIMCoefficients] = {}

        stats = load_normalization(hf_id)
        # Annotated so mypy sees tensors, not `Tensor | Module` (register_buffer).
        self.image_mean: Tensor
        self.image_std: Tensor
        self.state_low: Tensor
        self.state_high: Tensor
        self.action_low: Tensor
        self.action_high: Tensor
        self.register_buffer("image_mean", stats["image_mean"].float())
        self.register_buffer("image_std", stats["image_std"].float())
        self.register_buffer("state_low", stats["state_low"].float())
        self.register_buffer("state_high", stats["state_high"].float())
        self.register_buffer("action_low", stats["action_low"].float())
        self.register_buffer("action_high", stats["action_high"].float())

        # `self.to` rather than `self.policy.to`: the normalization buffers are this
        # module's, and they have to travel with the network. Device only -- the
        # normalization statistics stay fp32 so half precision does not also move the
        # input/output scaling.
        self.to(device)
        if self.torch_dtype != torch.float32:
            self.policy.to(dtype=self.torch_dtype)
            # The sinusoidal timestep embedding constructs its own fp32 tensor
            # (`torch.arange(...)` inside `SinusoidalPosEmb`), so casting the network leaves
            # one fp32 producer feeding a half-precision linear and the unet raises
            # "mat1 and mat2 must have the same dtype". Keep that encoder in fp32 -- it is
            # a 320-wide MLP, so its cost is noise -- and cast its output where it enters
            # the rest of the network.
            encoder = self.policy.diffusion.unet.diffusion_step_encoder
            encoder.to(dtype=torch.float32)
            target = self.torch_dtype

            def _cast_step_embedding(
                _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
            ) -> Tensor:
                return output.to(target)

            encoder.register_forward_hook(_cast_step_embedding)
        if scheduler == "ddim":
            base = self.policy.diffusion.noise_scheduler
            self.policy.diffusion.noise_scheduler = DDIMScheduler(
                num_train_timesteps=int(base.config.num_train_timesteps),
                beta_start=float(base.config.beta_start),
                beta_end=float(base.config.beta_end),
                beta_schedule=str(base.config.beta_schedule),
                clip_sample=bool(base.config.clip_sample),
                clip_sample_range=float(base.config.clip_sample_range),
                prediction_type=str(base.config.prediction_type),
                set_alpha_to_one=False,
                steps_offset=0,
            )
        elif scheduler != "ddpm":
            raise ValueError(f"unknown scheduler {scheduler!r}")

    def _coefficients(self, steps: int) -> DDIMCoefficients:
        """DDIM scalars for `steps`, built once and reused (device-resident)."""
        cached = self._coeff_cache.get(steps)
        if cached is None:
            diffusion = self.policy.diffusion
            cfg = diffusion.noise_scheduler.config
            cached = ddim_coefficients(
                diffusion.noise_scheduler.alphas_cumprod,
                int(cfg.num_train_timesteps),
                steps,
                device=next(diffusion.unet.parameters()).device,
                dtype=self.torch_dtype,
                set_alpha_to_one=bool(cfg.set_alpha_to_one),
            )
            self._coeff_cache[steps] = cached
        return cached

    def warm(self, steps: int) -> None:
        """Build the DDIM coefficients for `steps` before tracing or capturing.

        `ddim_coefficients` reads scalars off tensors, which is a graph break. Doing it
        ahead of `torch.compile` means the traced function contains only the tensor loop.
        """
        if self.sampler == "fast":
            self._coefficients(int(steps))

    def _fast_chunk(self, image: Tensor, state: Tensor, steps: int, noise: Tensor | None) -> Tensor:
        diffusion = self.policy.diffusion
        image = image.to(self.torch_dtype)
        state = state.to(self.torch_dtype)
        batch = {
            "observation.state": state,
            "observation.images": image.unsqueeze(2),  # (B, To, 1 camera, C, H, W)
        }
        global_cond = diffusion._prepare_global_conditioning(batch)
        if noise is None:
            noise = torch.randn(
                batch["observation.state"].shape[0],
                int(diffusion.config.horizon),
                int(diffusion.config.action_feature.shape[0]),
                device=image.device,
                dtype=image.dtype,
            )
        else:
            # The latency protocol passes a static fp32 buffer so CUDA graph capture never
            # touches the RNG; casting it here keeps the sampler dtype-pure.
            noise = noise.to(image.device, self.torch_dtype)
        coeffs = self._coefficients(steps)

        def predict_epsilon(sample: Tensor, timesteps: Tensor) -> Tensor:
            # `unet` is an untyped third-party module, so name the return type here.
            predicted: Tensor = diffusion.unet(sample, timesteps, global_cond=global_cond)
            return predicted

        sample = ddim_sample(
            predict_epsilon,
            noise,
            coeffs,
            clip_sample=bool(diffusion.noise_scheduler.config.clip_sample),
            clip_range=float(diffusion.noise_scheduler.config.clip_sample_range),
        )
        start = self.obs_horizon - 1
        # Back to fp32 before de-normalization: the output is in environment pixels and the
        # statistics were fitted in fp32, so the last step should not lose mantissa bits.
        return sample[:, start : start + self.action_horizon].float()

    @torch.no_grad()
    def predict_chunk(
        self,
        obs: dict[str, Tensor],
        steps: int = 10,
        noise: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        image = normalize_image(obs["image"].float(), self.image_mean, self.image_std)
        state = normalize_state(obs["state"].float(), self.state_low, self.state_high)
        if self.sampler == "fast":
            actions = self._fast_chunk(image, state, int(steps), noise)
        else:
            self.policy.diffusion.num_inference_steps = int(steps)
            actions = self.policy.predict_action_chunk(
                {"observation.image": image, "observation.state": state}, noise=noise
            )
        chunk = denormalize_action(actions, self.action_low, self.action_high)
        return chunk.clamp_(0.0, 512.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())
