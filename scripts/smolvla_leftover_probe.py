"""Step 0 for the VLA adapter (docs/11 section 4.4): is a leftover in the space RTC guides in?

RTC guides a flow sampler toward the previous chunk's unexecuted tail, and the guidance acts on
`x_t` inside the loop -- the checkpoint's action space, normalized over its 7 dims and padded to
the flow head's 32. Nothing upstream hands this repo that space: `_get_action_chunk` slices the
pad off and `lerobot_eval` unnormalizes after the call, while `deploy/control_loop.py` holds raw
env units. So the adapter has to rebuild it, and if it rebuilds it wrongly the guidance pulls
toward the wrong region and returns plausible actions with a meaningless success rate.

This decides that before any GPU hour, deterministically and on CPU: no checkpoint, no episode,
no LIBERO, no `transformers` (only `lerobot`'s RTC processor, which is torch arithmetic).

It checks two things:

1. **The map is lossless.** Raw -> model -> raw and model -> raw -> model are identities at 1e-5;
   the pad is written as exact zeros and ignored on the way back, so the denoiser's garbage in
   dims 7-31 cannot reach an actuator.
2. **The guidance contract is what we think it is.** Against the real `RTCProcessor`, the guided
   step equals its closed form at 1e-5; a leftover equal to `x1_t` produces *exactly* zero
   guidance; the guidance vanishes on the final Euler step (`t=0`) and saturates on the first
   (`t=1`, clamped to `max_guidance_weight`), which is where a fresh leftover does most of its
   work. Because `denoise_step` detaches `x_t` before taking the autograd handle, the gradient is
   the identity, so the correction *is* the weighted residual -- measured here, not assumed.

Then it reports what makes the gate worth having: the same leftover supplied in raw units breaks
the fixed point that the correct space satisfies exactly, so a unit error is visible from one
synthetic step rather than only from a success rate. `pass` requires both the identities and that
separation.

    uv run python scripts/smolvla_leftover_probe.py

Reading: every identity holds and the separation is large -> the adapter's leftover path can be
written against `PaddedActionSpace` and validated at the end of the first episode on the GPU.
Any identity failing -> that is a located defect in the map, and it would have been invisible in a
success rate. `stats` in the report says `synthetic`, which is the one thing this cannot check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import Tensor

from policybench.models.leftover import PaddedActionSpace

# `HuggingFaceVLA/smolvla_libero`'s config.json, read 2026-11-01: `chunk_size` 50, `max_action_dim`
# 32, action 7. Constants rather than an import: the probe must run without the 2 GB download, and
# these are the numbers the adapter will have to agree with anyway.
ACTION_DIM = 7
MODEL_DIM = 32
CHUNK = 50
BATCH = 1

# `RTCConfig` defaults, which are what the library would use in-process.
EXECUTION_HORIZON = 10
MAX_GUIDANCE_WEIGHT = 10.0

BAR = 1e-5


def _space() -> PaddedActionSpace:
    """A `MEAN_STD` table of the right shape with the right anisotropy -- not the checkpoint's.

    Synthetic on purpose: the real statistics need the download. What that cannot catch is a wrong
    *statistic*; what it does catch is wrong arithmetic or plumbing, which is what an adapter gets
    wrong. Spreads differ per dim so a transposed or permuted map cannot pass by symmetry.
    """
    generator = torch.Generator().manual_seed(11)
    center = (torch.rand(ACTION_DIM, generator=generator) - 0.5) * 0.8
    spread = 0.3 + 0.7 * torch.rand(ACTION_DIM, generator=generator)
    return PaddedActionSpace(center=center, spread=spread, model_dim=MODEL_DIM)


def _raw_chunk(space: PaddedActionSpace, seed: int) -> Tensor:
    """`[B, CHUNK, ACTION_DIM]` in env units: near the mean, within a few dataset stds."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(BATCH, CHUNK, ACTION_DIM, generator=generator) * 2.0 - 1.0
    return space.center + space.spread * z


def _velocity(x_t: Tensor) -> Tensor:
    """A nonlinear stand-in for the 450 M action expert: deterministic, no weights, CPU."""
    return torch.tanh(0.5 * x_t) + 0.1


def _guidance_weight(time: float, cap: float = MAX_GUIDANCE_WEIGHT) -> Tensor:
    """`denoise_step`'s clamped guidance weight, replicated from its own lines to be compared."""
    tau = torch.as_tensor(1.0 - time, dtype=torch.float64)
    squared_one_minus_tau = (1 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    c = torch.nan_to_num((1 - tau) / tau, posinf=cap)
    weight = torch.nan_to_num(c * inv_r2, posinf=cap)
    return torch.minimum(weight, torch.as_tensor(cap, dtype=torch.float64))


def _processor() -> object:
    """`lerobot`'s RTC processor, configured as the library would configure it in-process."""
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    return RTCProcessor(
        RTCConfig(
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            max_guidance_weight=MAX_GUIDANCE_WEIGHT,
            execution_horizon=EXECUTION_HORIZON,
        )
    )


def _maxabs(tensor: Tensor) -> float:
    return float(tensor.detach().double().abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/smolvla_leftover_probe.json"))
    args = parser.parse_args()

    space = _space()
    checks: dict[str, float] = {}
    notes: dict[str, float] = {}
    verdicts: dict[str, bool] = {}

    # ---- 1. The map -------------------------------------------------------------------------
    raw = _raw_chunk(space, 21)
    model = space.to_model(raw)
    checks["roundtrip_raw"] = _maxabs(space.to_raw(model) - raw)

    # A *valid* model-space chunk (zeros in the pad), as the sampler would hand one back.
    generator = torch.Generator().manual_seed(7)
    valid = torch.randn(BATCH, CHUNK, MODEL_DIM, generator=generator)
    valid[..., ACTION_DIM:] = 0.0
    checks["roundtrip_model"] = _maxabs(space.to_model(space.to_raw(valid)) - valid)

    checks["pad_written"] = _maxabs(model[..., ACTION_DIM:])

    polluted = valid.clone()
    polluted[..., ACTION_DIM:] = 1e3  # what a denoiser is free to emit in the pad
    checks["pad_ignored"] = _maxabs(space.to_raw(polluted) - space.to_raw(valid))

    # ---- 2. The guidance contract -----------------------------------------------------------
    processor = _processor()
    time = 0.8  # not 1.0: the first step is unguided by construction, checked separately
    leftover = space.to_model(_raw_chunk(space, 22))
    velocity = _velocity(model)
    weights = processor.get_prefix_weights(0, EXECUTION_HORIZON, CHUNK).unsqueeze(0).unsqueeze(-1)

    def guide(prev: Tensor, at: float = time) -> Tensor:
        return processor.denoise_step(
            x_t=model,
            prev_chunk_left_over=prev,
            inference_delay=0,
            time=at,
            original_denoise_step_partial=_velocity,
            execution_horizon=EXECUTION_HORIZON,
        )

    guided = guide(leftover)
    x1_t = model - time * velocity
    analytic = velocity - _guidance_weight(time).to(velocity.dtype) * ((leftover - x1_t) * weights)
    checks["rtc_closed_form"] = _maxabs(guided - analytic)

    # Leftover identical to x1_t: the residual is zero, so guidance must be *exactly* zero. This is
    # the property a wrong unit convention destroys, and it needs no episode to observe.
    checks["rtc_fixed_point"] = _maxabs(guide(x1_t.detach()) - velocity)

    # The guidance weight is 0 at the action end and clamped to `max_guidance_weight` at the noise
    # end, so the *last* Euler step is the unguided one. Worth pinning: the opposite is the natural
    # (wrong) guess, and at t=1 the leftover already does its full pull.
    checks["rtc_final_step_unguided"] = _maxabs(guide(leftover, at=0.0) - velocity)

    checks["schedule_tail_zero"] = _maxabs(weights[0, EXECUTION_HORIZON:, 0])

    # `denoise_step` zero-pads a narrower leftover. That is the convention `to_model` writes, so a
    # 7-dim leftover and the same 7 dims widened with our own pad must guide identically.
    narrow = _raw_chunk(space, 22)
    widened = torch.cat(
        [narrow, narrow.new_zeros(*narrow.shape[:-1], MODEL_DIM - ACTION_DIM)], dim=-1
    )
    checks["library_pads_narrow_leftover"] = _maxabs(guide(narrow) - guide(widened))

    # ---- 3. Is the failure mode loud? -------------------------------------------------------
    # The same *information*, in the wrong space: `x1_t`'s own raw-unit twin. Correct units give a
    # fixed point of exactly zero, so any guidance here is the unit error, not the policy.
    notes["fixed_point_correct_units"] = _maxabs(guide(x1_t.detach()) - velocity)
    notes["fixed_point_raw_units"] = _maxabs(guide(space.to_raw(x1_t.detach())) - velocity)
    notes["step_right_units"] = _maxabs(guided - velocity)
    notes["guidance_weight_noise_end"] = float(_guidance_weight(1.0))
    notes["guidance_weight_action_end"] = float(_guidance_weight(0.0))
    verdicts["wrong_units_are_detectable"] = notes["fixed_point_raw_units"] > 1e-3

    worst = max(checks.values(), default=0.0)
    passed = worst <= BAR and all(verdicts.values())
    report = {
        "bar": BAR,
        "stats": "synthetic",
        "action_dim": ACTION_DIM,
        "model_dim": MODEL_DIM,
        "chunk_size": CHUNK,
        "execution_horizon": EXECUTION_HORIZON,
        "schedule_shape": list(weights.shape),
        "checks": checks,
        "verdicts": verdicts,
        "notes": notes,
        "worst": worst,
        "pass": passed,
        "verdict": (
            "the leftover maps into model space losslessly and a unit error breaks the fixed "
            "point it otherwise satisfies exactly; the adapter may be written against "
            "PaddedActionSpace"
            if passed
            else "one or more identities failed: the leftover map is defective, see the table"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'check':<32}{'max abs diff':>14}  {'bar':>9}  result")
    for name, value in checks.items():
        print(f"{name:<32}{value:>14.3e}  {'abs 1e-05':>9}  {'ok' if value <= BAR else 'DIFF'}")
    print("\nreported, not gated (model-space units):")
    print(f"  fixed point, correct units {notes['fixed_point_correct_units']:.3e}")
    print(f"  fixed point, raw units     {notes['fixed_point_raw_units']:.3e}")
    print(f"  a normal guided step moves {notes['step_right_units']:.3e}")
    print(
        f"  guidance weight at t=1.0   {notes['guidance_weight_noise_end']:.1f} (clamped),"
        f" at t=0.0 {notes['guidance_weight_action_end']:.1f}"
    )
    print(
        f"  prefix weights             shape {tuple(weights.shape)}, "
        f"horizon {EXECUTION_HORIZON} of {CHUNK} chunk steps"
    )
    for name, ok in verdicts.items():
        print(f"  {name:<25} {'yes' if ok else 'NO'}")
    print(f"\nbar {BAR:g} -> {'PASS' if passed else 'FAIL'}: {report['verdict']}")
    print(f"written to {args.out}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
