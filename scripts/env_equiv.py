"""Do this repo's env wrapper and `lerobot`'s compute the same coverage trace?

R5 (docs/06) has been open since 2026-10-19: gym-pusht is not verified equivalent to the original
PushT env. The adapter-free control raises the sharper version of that question, because it runs
inside `lerobot`'s env stack -- so any difference between the two loops mixes the policy path
with the env path, and H's 8 pp residual inherits that ambiguity.

This compares the environments alone: one seed, one fixed action sequence, both wrappers, no
policy and no sampler. If the traces agree, the env term is zero and the residual belongs to the
policy path. If they diverge, every loop-to-loop comparison in `docs/04` carries an env term and
the divergence is the finding.

    .venv-lerobot/Scripts/python.exe scripts/env_equiv.py --seed 1000

Run it with the isolated interpreter, because `lerobot` is only installed there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from policybench.envs.pusht import PushTEnv


# Same PRNG, same sequence, both wrappers: the only variable is the wrapper.
def _actions(seed: int, steps: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 512.0, size=(steps, 2)).astype(np.float32)


def _trace_ours(seed: int, actions: np.ndarray) -> dict[str, Any]:
    env = PushTEnv()
    trace: dict[str, Any] = {"coverage": [], "success": [], "done_at": None}
    try:
        obs = env.reset(seed=seed)
        trace["image0"] = np.asarray(obs["image"]).copy()
        trace["state0"] = np.asarray(obs["state"]).copy()
        for index, action in enumerate(actions):
            _obs, _reward, done, info = env.step(action)
            trace["coverage"].append(float(info["coverage"]))
            trace["success"].append(float(info["success"]))
            if done and trace["done_at"] is None:
                trace["done_at"] = index + 1
    finally:
        env.close()
    return trace


def _info_value(infos: Any, key: str) -> float | None:
    if isinstance(infos, dict) and key in infos:
        value = infos[key]
        array = np.asarray(value).ravel()
        return float(array[0]) if array.size else None
    if isinstance(infos, list) and infos and key in infos[0]:
        return float(infos[0][key])
    return None


def _trace_theirs(seed: int, actions: np.ndarray) -> dict[str, Any]:
    from lerobot.envs.configs import PushtEnv as PushtEnvConfig
    from lerobot.envs.factory import make_env

    envs = make_env(PushtEnvConfig(), n_envs=1, use_async_envs=False)
    suite = next(iter(envs.values()))
    vec = next(iter(suite.values()))
    trace: dict[str, Any] = {"coverage": [], "success": [], "done_at": None, "keys": None}
    try:
        obs, _info = vec.reset(seed=seed)
        trace["keys"] = sorted(obs)
        trace["image0"] = np.asarray(obs["pixels"][0]).copy()
        trace["state0"] = np.asarray(obs["agent_pos"][0]).copy()
        for index, action in enumerate(actions):
            obs, _reward, terminated, truncated, infos = vec.step(
                np.asarray([action], dtype=np.float32)
            )
            coverage = _info_value(infos, "coverage")
            success = _info_value(infos, "is_success")
            trace["coverage"].append(np.nan if coverage is None else coverage)
            trace["success"].append(np.nan if success is None else success)
            done = bool(np.asarray(terminated).ravel()[0] or np.asarray(truncated).ravel()[0])
            if done and trace["done_at"] is None:
                trace["done_at"] = index + 1
    finally:
        vec.close()
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("results/env_equiv.json"))
    args = parser.parse_args()

    actions = _actions(args.seed, args.steps)
    ours = _trace_ours(args.seed, actions)
    theirs = _trace_theirs(args.seed, actions)

    n = min(len(ours["coverage"]), len(theirs["coverage"]))
    cov_ours = np.asarray(ours["coverage"][:n], dtype=float)
    cov_theirs = np.asarray(theirs["coverage"][:n], dtype=float)
    both = ~np.isnan(cov_theirs)
    cov_diff = (
        float(np.abs(cov_ours[both] - cov_theirs[both]).max()) if both.any() else float("nan")
    )

    suc_ours = np.asarray(ours["success"][:n], dtype=float)
    suc_theirs = np.asarray(theirs["success"][:n], dtype=float)
    suc_diff = (
        float(np.abs(suc_ours[both] - suc_theirs[both]).max()) if both.any() else float("nan")
    )

    report = {
        "seed": args.seed,
        "steps": args.steps,
        "ours": {"coverage_final": max(ours["coverage"]), "done_at": ours["done_at"]},
        "theirs": {
            "coverage_final": float(np.nanmax(cov_theirs)) if both.any() else None,
            "done_at": theirs["done_at"],
            "obs_keys": theirs["keys"],
        },
        "coverage_max_abs_diff": cov_diff,
        "success_max_abs_diff": suc_diff,
        "image0_identical": bool(np.array_equal(ours["image0"], theirs["image0"])),
        "state0_max_abs_diff": float(np.abs(ours["state0"] - theirs["state0"]).max()),
        "same_done_at": ours["done_at"] == theirs["done_at"],
        "verdict": (
            "same env: coverage and success traces agree step for step"
            if (not np.isnan(cov_diff) and cov_diff <= 1e-9 and suc_diff == 0.0)
            else "DIFFERENT env: this is an env term in every loop-to-loop comparison"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")
    if "DIFFERENT" in report["verdict"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
