"""Pre-flight for docs/10 option H: do the two implementations of one mapping agree?

The checkpoint declares MEAN_STD for images and MIN_MAX for state and actions.
`models/pretrained.py` applies that by hand from the eight buffers inside `model.safetensors`;
`lerobot`, after `scripts/lerobot_migrate.py`, applies it from `policy_preprocessor.json`. Two
hand-written implementations of one declared mapping is the shape of a silent defect (channel
order, eps, 0-1 vs 0-255, `max-min` vs `max-min+eps`), and nothing has ever compared them.

This script does, deterministically: no sampler, no episodes, no GPU block. It prints a table
and writes `results/processor_equiv.json`; it exits non-zero if any difference exceeds the bar,
so `just processor-equiv` fails loudly rather than reporting "close enough".

Run it with the isolated interpreter -- only that env has `lerobot`:

    .venv-lerobot/Scripts/python.exe scripts/processor_equiv.py

Reading (docs/10 section 3):

* every difference <= 1e-6 -> the implementations agree; H cannot explain the residual and is
  declined, and the numbers here are the evidence.
* any difference > 1e-6 -> that difference is a located defect; H stops being an adoption and
  becomes a fix, and docs/10 section 5 sizes what it was worth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from safetensors.torch import load_file

from policybench.envs.pusht import PushTEnv
from policybench.models.normalization import BuffersNorm, load_processor_norm
from policybench.models.pretrained import DEFAULT_HF_ID, load_normalization

BAR = 1e-6
NORM_KEYS = (
    "normalize_inputs.buffer_",
    "normalize_targets.buffer_",
    "unnormalize_outputs.buffer_",
)


def _frames(seeds: list[int]) -> list[dict[str, torch.Tensor]]:
    """One real first frame per env seed, in the harness's own conventions.

    Uses this repo's env wrapper so the tensors are exactly what `rollout._obs_batch` produces:
    image `float [3, 96, 96]` in [0, 1], state `float [2]` in pixels.
    """
    env = PushTEnv()
    out: list[dict[str, torch.Tensor]] = []
    try:
        for seed in seeds:
            obs = env.reset(seed=seed)
            image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 255.0
            out.append({"image": image, "state": torch.from_numpy(obs["state"]).float()})
    finally:
        env.close()
    return out


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute difference, compared on CPU.

    The pipeline's device step follows the checkpoint's JSON (`cuda`), while the buffers this
    repo reads are plain CPU tensors; the question here is the *arithmetic*, so devices are
    normalised away rather than aligned. Float64 on both sides so the comparison is not limited
    by the storage dtype.
    """
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a.detach().double().cpu() - b.detach().double().cpu()).abs().max())


def _weights_check(migrated: Path) -> float:
    """The migrated weights must be the published ones minus the eight normalization buffers."""
    from huggingface_hub import hf_hub_download

    original = load_file(hf_hub_download(DEFAULT_HF_ID, "model.safetensors"))
    kept = {k: v for k, v in original.items() if not any(p in k for p in NORM_KEYS)}
    migrated_state = load_file(str(migrated / "model.safetensors"))
    if set(kept) != set(migrated_state):
        missing = sorted(set(kept) - set(migrated_state))[:5]
        extra = sorted(set(migrated_state) - set(kept))[:5]
        raise SystemExit(f"migrated weights differ in keys: missing {missing}, extra {extra}")
    return max((_max_diff(kept[k], migrated_state[k]) for k in kept), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/diffusion_pusht_migrated")
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("results/processor_equiv.json"))
    args = parser.parse_args()

    migrated = Path(args.checkpoint)
    seeds = list(range(args.seed_start, args.seed_start + args.observations))
    frames = _frames(seeds)

    stats = load_normalization()
    buffers = BuffersNorm(**stats)
    processors = load_processor_norm(migrated, device="cpu")

    checks: dict[str, float] = {}
    checks["weights"] = _weights_check(migrated)

    # Two *different* history frames per observation, so a transposed reshape cannot pass.
    def pair(sample: str, i: int) -> torch.Tensor:
        return torch.stack([frames[i][sample], frames[(i + 1) % len(frames)][sample]])

    image = torch.stack([pair("image", i) for i in range(len(frames))])  # [N, T, 3, H, W]
    state = torch.stack([pair("state", i) for i in range(len(frames))])  # [N, T, 2]

    checks["image"] = _max_diff(buffers.image(image), processors.image(image))
    checks["state"] = _max_diff(buffers.state(state), processors.state(state))

    # Actions: a deterministic chunk in network units, [N, H, 2].
    generator = torch.Generator().manual_seed(0)
    chunk = torch.rand((len(frames), 16, 2), generator=generator) * 2.0 - 1.0
    checks["actions"] = _max_diff(buffers.actions(chunk), processors.actions(chunk))

    # Ordering: the same frames passed one at a time, by hand, must equal the batched call.
    # `[B, T]` and `[T, B]` hold the same count, so this is the check a shape assertion misses.
    manual = torch.stack(
        [
            torch.stack(
                [
                    processors.image(image[i, t].view(1, 1, *image.shape[2:]))[0, 0]
                    for t in range(image.shape[1])
                ]
            )
            for i in range(image.shape[0])
        ]
    )
    checks["history_order"] = _max_diff(manual, processors.image(image))

    worst = max(checks.values())
    passed = worst <= BAR
    report = {
        "bar": BAR,
        "n_observations": len(seeds),
        "seed_start": args.seed_start,
        "checkpoint": str(migrated),
        "checks": checks,
        "worst": worst,
        "pass": passed,
        "verdict": (
            "agree: the two implementations are the same arithmetic, so H is not the explanation"
            if passed
            else "DIFFER: this is a located defect, see docs/10 section 5"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'check':<16}{'max abs diff':>14}  {'bar':>9}  result")
    for name, value in checks.items():
        ok = value <= BAR
        print(f"{name:<16}{value:>14.3e}  {'abs 1e-06':>9}  {'ok' if ok else 'DIFF'}")
    print(f"\nbar {BAR:g} -> {'PASS' if passed else 'FAIL'}: {report['verdict']}")
    print(f"written to {args.out}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
