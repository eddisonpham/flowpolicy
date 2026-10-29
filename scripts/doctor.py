"""Environment check: `just doctor`. Read-only, no model load, no GPU work.

Exists so a fresh clone can tell the difference between "the environment is wrong" and
"the experiment is wrong" before spending an hour of GPU time. Prints PASS / FAIL / NOT RUN
per item, in the same honest style as the week-1 gates in `scripts/gates/`, and exits
non-zero only when something required for the eval path is broken.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

OK = "PASS"
BAD = "FAIL"
SKIP = "NOT RUN"


def _check(name: str, fn: object) -> tuple[str, str, bool]:
    try:
        detail = fn()  # type: ignore[operator]
        return name, str(detail), True
    except Exception as exc:
        return name, f"{type(exc).__name__}: {exc}", False


def _python() -> str:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required by pyproject, found {sys.version.split()[0]}")
    return f"{sys.version.split()[0]} ({sys.executable})"


def _torch() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; the eval path needs a GPU")
    free, total = torch.cuda.mem_get_info()
    return (
        f"torch {torch.__version__}, cuda {torch.version.cuda}, "
        f"{torch.cuda.get_device_name()} {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free"
    )


def _sim() -> str:
    import gym_pusht  # noqa: F401

    return "gym_pusht imports"


def _backend() -> str:
    for module in ("lerobot", "diffusers"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(
                f"{module} missing: run `just setup-pretrained` (docs/02 section 3.4)"
            )
    import lerobot

    return (
        f"lerobot {lerobot.__version__} + diffusers "
        "(lerobot installed outside the lockfile; diffusers is locked in dev)"
    )


def _lock() -> str:
    if not Path("uv.lock").exists():
        raise RuntimeError("uv.lock missing: run `just setup`")
    return "uv.lock present"


def _data() -> str:
    path = Path("data/pusht/pusht.zarr")
    if not path.exists():
        raise RuntimeError("Zarr missing: run `just data` (needed only for ablation training)")
    return f"{path} present (ablation training only)"


def _checkpoint() -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info("lerobot/diffusion_pusht", files_metadata=True)
    size = sum(f.size or 0 for f in (info.siblings or [])) / 2**20
    return f"HF repo reachable, {size:.0f} MiB of files (downloads on first eval)"


def _gpu_toolchain() -> str:
    """Optional since 2026-10-28: only the *stock* sampler needs MSVC.

    `lerobot` -> `diffusers` keeps DDIM's schedule on the host, so compiling that path wants a
    C++ compiler for the CPU partition. Our fast sampler (`--sampler fast`) has no such
    partition and compiles without MSVC, which is why this check is informational rather than
    required (docs/05 section 1).
    """
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("torch missing")
    try:
        subprocess.run(["cl"], shell=True, capture_output=True, check=True, timeout=20)
    except Exception as exc:
        raise RuntimeError(
            "MSVC cl.exe not found: only the stock sampler's compile path needs it; "
            f"`--sampler fast` does not (docs/05 section 1) [{type(exc).__name__}]"
        ) from exc
    return "cl.exe present"


REQUIRED = {"python", "torch"}


def main() -> None:
    checks = [
        ("python", _python),
        ("torch + cuda", _torch),
        ("simulator", _sim),
        ("lockfile", _lock),
        ("pretrained backend", _backend),
        ("checkpoint", _checkpoint),
        ("dataset (ablation)", _data),
        ("MSVC (stock-sampler compile, optional)", _gpu_toolchain),
    ]
    failed = False
    for name, fn in checks:
        label, detail, passed = _check(name, fn)
        if passed:
            status = OK
        elif name in REQUIRED:
            status, failed = BAD, True
        else:
            status = SKIP
        print(f"[{status:>7s}] {label}: {detail}")
    print("\nEval path needs: python, torch + cuda, simulator, checkpoint.")
    print(
        "Ablation training also needs: lockfile, dataset. Compiling the stock sampler needs "
        "MSVC; the fast sampler does not (docs/05 section 1)."
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
