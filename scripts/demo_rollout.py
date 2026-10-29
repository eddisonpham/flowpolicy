"""Record a real rollout to a GIF, with the loop's own numbers printed on the frames.

The tables say the policy works and that a chunk costs tens of milliseconds; this is the
same episode as pixels, so a reader can see that the numbers describe a task being solved
rather than describing a harness. Three panels per frame:

``policy``     the observation the policy sees (96x96, nearest-neighbour upscale)
``chunk queue`` how many of the 8 queued actions are left, and whether a replan is due
``timing``     the measured chunk latency, the control period, and the deadline count

Nothing here feeds a table. It is packaging, and it is generated the same way everything
else is: one command, no hand-editing.

Usage:
    python scripts/demo_rollout.py --steps 4 --seed 10000 --out results/figures/demo_rollout.gif
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from policybench.envs.pusht import MAX_STEPS, PushTEnv
from policybench.eval.rollout import _obs_batch, load_policy

PANEL = 288  # 3x the 96 px observation, integer scale keeps the pixels honest
SIDEBAR = 260


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 9.2 has no sized default font
        return ImageFont.load_default()


FONT = _font()


def _frame(
    image: np.ndarray,
    *,
    step: int,
    coverage: float,
    queue_left: int,
    replans: int,
    chunk_ms: float,
    period_ms: float,
    misses: int,
) -> Image.Image:
    obs_img = Image.fromarray(image).resize((PANEL, PANEL), Image.NEAREST)
    canvas = Image.new("RGB", (PANEL + SIDEBAR, PANEL), (18, 18, 18))
    canvas.paste(obs_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = PANEL + 12
    lines = [
        (f"step {step:3d}/{MAX_STEPS}", "#e6e6e6"),
        (f"coverage {coverage:.2f}", "#9fe8a0" if coverage >= 0.95 else "#e6e6e6"),
        ("", "#000"),
        (f"queue {queue_left}/8", "#e6e6e6"),
        (f"replans {replans}", "#e6e6e6"),
        ("", "#000"),
        (f"chunk {chunk_ms:6.1f} ms", "#ffb454" if chunk_ms > period_ms else "#7fd1ff"),
        (f"period {period_ms:.0f} ms", "#e6e6e6"),
        (f"deadlines {misses}", "#ff6b6b" if misses else "#7fd1ff"),
        ("", "#000"),
        ("PushT, sim only", "#8a8a8a"),
        ("lerobot/diffusion_pusht", "#8a8a8a"),
    ]
    for i, (text, colour) in enumerate(lines):
        draw.text((x, 14 + i * 21), text, fill=colour, font=FONT)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="hf:lerobot/diffusion_pusht")
    parser.add_argument("--steps", type=int, default=4, help="DDIM sampler steps")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--period-ms", type=float, default=100.0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2, help="record every Nth control step")
    parser.add_argument("--out", type=Path, default=Path("results/figures/demo_rollout.gif"))
    parser.add_argument("--sampler", default="fast", choices=["stock", "fast"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy, meta = load_policy(args.ckpt, scheduler="ddim", sampler=args.sampler)
    warm = getattr(policy, "warm", None)
    if callable(warm):
        warm(args.steps)
    env = PushTEnv()
    obs = env.reset(seed=args.seed)
    # Bounded windows: an unbounded list would grow the observation stack with every step
    # and the U-Net's conditioning dimension would drift until it mismatches.
    frames = deque([obs["image"]] * policy.obs_horizon, maxlen=policy.obs_horizon)
    states = deque([obs["state"]] * policy.obs_horizon, maxlen=policy.obs_horizon)
    queue: list[np.ndarray] = []
    # Prime CUDA and the DDIM coefficients once, untimed: the first call in a process costs
    # ~700 ms (kernel autotuning, coefficient build) and would be drawn on frame 1 as if it
    # were the policy's steady-state cost. Anyone reading the GIF would believe that number.
    with torch.no_grad():
        policy.predict_chunk(_obs_batch(frames, states, "cuda"), steps=args.steps)
    torch.cuda.synchronize()
    frames_out: list[Image.Image] = []
    coverage = 0.0
    success = 0.0
    replans = 0
    misses = 0
    chunk_ms = 0.0
    step = 0
    try:
        while step < MAX_STEPS:
            if not queue:
                batch = _obs_batch(frames, states, "cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                chunk = policy.predict_chunk(batch, steps=args.steps)
                torch.cuda.synchronize()
                chunk_ms = (time.perf_counter() - t0) * 1000.0
                replans += 1
                if chunk_ms > args.period_ms:
                    misses += 1
                queue = list(chunk[0, : args.action_horizon].float().cpu().numpy())
            action = np.clip(queue.pop(0), 0.0, 512.0)
            obs, _reward, done, info = env.step(action)
            frames.append(obs["image"])
            states.append(obs["state"])
            step += 1
            coverage = max(coverage, info["coverage"])
            success = max(success, info["success"])
            if step % args.stride == 0 or done:
                frames_out.append(
                    _frame(
                        obs["image"],
                        step=step,
                        coverage=coverage,
                        queue_left=len(queue),
                        replans=replans,
                        chunk_ms=chunk_ms,
                        period_ms=args.period_ms,
                        misses=misses,
                    )
                )
            if done:
                break
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 10 control steps per second is the task's own rate; one recorded frame per `stride`
    # steps, so the GIF plays at roughly real time.
    frames_out[0].save(
        args.out,
        save_all=True,
        append_images=frames_out[1:],
        duration=int(1000 * args.stride * 0.1),
        loop=0,
        optimize=True,
    )
    print(
        f"wrote {args.out} ({len(frames_out)} frames)  seed={args.seed} N={args.steps} "
        f"success={success:.0f} coverage={coverage:.3f} replans={replans} "
        f"deadline_misses={misses} mean_chunk_ms={chunk_ms:.1f} params={meta['params']:,}"
    )


if __name__ == "__main__":
    main()
