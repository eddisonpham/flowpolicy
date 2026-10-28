"""The fast DDIM sampler must be the same policy as the library's, not merely a fast one.

`tests/unit/test_fast_sampler.py` proves the arithmetic equals `diffusers` on a toy denoiser.
This test proves it on the checkpoint itself: same observation, same starting noise, two loops,
and the resulting action chunk must agree to floating-point rounding.

It also pins down what "equal" can and cannot mean here. The loops differ in summation order, so
they agree to ~1e-5 of the action range (hundreds of pixels wide) and no closer. PushT is
contact-rich and chaotic, so that rounding eventually changes individual episode outcomes: on 20
paired episodes the two loops agree on 9 of them even though mean coverage differs by 0.05. That
is a fact about 20-episode resolution in a chaotic task, not a defect in the sampler -- which is
why the bound asserted below is on *actions*, not on success counts.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch

from flowpolicy.envs.pusht import PushTEnv
from flowpolicy.eval.rollout import _obs_batch, load_policy

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

CHECKPOINT = "hf:lerobot/diffusion_pusht"
# 512 px is the action box; 0.1 px is 2e-4 of it, several times the observed 0.02 px and far
# below one pixel of either the action or the observation.
MAX_ABS_DIFF_PX = 0.1


@pytest.fixture(scope="module")
def chunks() -> dict[str, dict[int, np.ndarray]]:
    env = PushTEnv()
    out: dict[str, dict[int, np.ndarray]] = {}
    for sampler in ("stock", "fast"):
        policy, _ = load_policy(CHECKPOINT, scheduler="ddim", sampler=sampler)
        obs = env.reset(seed=10000)
        frames = deque([obs["image"]] * policy.obs_horizon, maxlen=policy.obs_horizon)
        states = deque([obs["state"]] * policy.obs_horizon, maxlen=policy.obs_horizon)
        batch = _obs_batch(frames, states, "cuda")
        per_steps: dict[int, np.ndarray] = {}
        for steps in (4, 10):
            torch.manual_seed(1234)
            noise = torch.randn(1, 16, 2, device="cuda")
            with torch.no_grad():
                chunk = policy.predict_chunk(batch, steps=steps, noise=noise)
            per_steps[steps] = chunk.float().cpu().numpy()
        out[sampler] = per_steps
        del policy
        torch.cuda.empty_cache()
    env.close()
    return out


@pytest.mark.parametrize("steps", [4, 10])
def test_fast_sampler_matches_library_on_the_checkpoint(
    chunks: dict[str, dict[int, np.ndarray]], steps: int
) -> None:
    stock = chunks["stock"][steps]
    fast = chunks["fast"][steps]
    assert stock.shape == fast.shape == (1, 8, 2)
    diff = float(np.abs(stock - fast).max())
    assert diff < MAX_ABS_DIFF_PX, f"N={steps}: max action difference {diff:.4f} px"


def test_both_samplers_respect_the_action_box(
    chunks: dict[str, dict[int, np.ndarray]],
) -> None:
    for sampler, per_steps in chunks.items():
        for steps, chunk in per_steps.items():
            assert chunk.min() >= 0.0 and chunk.max() <= 512.0, (sampler, steps)
