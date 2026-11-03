"""Pilot: one LIBERO-Spatial episode that replaces `docs/11` 5.1's per-chunk band.

The band in 5.1 (150 / 250 / 700 ms per chunk) is the only extrapolated term in the whole
GPU-hours estimate, and everything downstream of it is sized from it: the 500-episode block, then
RTC's three-arm comparison. This measures the term instead, with one episode of LIBERO-Spatial
task 0 at initial state 0 under the 280-step cap, at the 10-action cadence the block would use --
28 chunks, one card, the checkpoint the block would evaluate.

It is a cost calibration, not an eval. n = 1 cannot measure success and nothing here reports a
rate. What it reports is the per-chunk cost split into the three terms the block pays for:
the policy call (synchronized around it), the observation build, and the batch of `env.step`
calls -- kept separate so the policy share and the environment share can be told apart rather
than summed into one number that hides which half a change would move. The `projected_*` fields
multiply that measurement by the block's real episode count; they are labelled projected because
they inherit the cadence assumption instead of measuring it.

Preconditions, in order, because each one invalidates the *measurement* rather than the report:

1. the env constructs on the pinned stack -- `robosuite==1.4.0` on Linux, per docs/11 4.4/4.5;
2. the checkpoint loads;
3. the postprocessor reproduces the checkpoint's own declared statistics at 1e-5 -- step 1 of
   4.4. If this fails the numbers would describe a different model, so the pilot stops on it.

Exit codes: 0 measured, 1 a precondition failed, 2 not run in this interpreter (no LIBERO, no
`transformers`) -- which is the expected outcome on Windows, and says so instead of pretending.

    uv run python scripts/smolvla_pilot.py               # the measurement
    uv run python scripts/smolvla_pilot.py --check-only  # the three preconditions, then stop
    uv run python scripts/smolvla_pilot.py --stats-only --device cpu   # 4.4 step 1 alone

The last form needs neither LIBERO nor the policy weights nor a GPU: it builds the postprocessor
pipeline straight from the checkpoint's `policy_postprocessor.json` and runs precondition 3, so
the one check that does not depend on the hardware can be settled on any host.

What it cannot decide, so nobody reads it as if it did: the block's discordance (a property of
episodes, not chunks), RTC's benefit (a three-arm comparison), SmolVLA's success rate, or anything
about another card -- the same 28 chunks have to be run there too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

# The checkpoint 5.1 costs the block with: the only VLA under 8 GB, per docs/11 section 1.
POLICY_ID = "HuggingFaceVLA/smolvla_libero"
SUITE = "libero_spatial"
TASK_ID = 0
# `lerobot.envs.libero.TASK_SUITE_MAX_STEPS["libero_spatial"]`; LIBERO's usual eval cap.
EPISODE_LENGTH = 280
ACTIONS_PER_CHUNK = 10
ACTION_DIM = 7
# What did the checkpoint's declared statistics have to reproduce? 4.4's bar, reused.
BAR = 1e-5
# The block 5.1 sizes, so the projection answers the question that motivated the pilot.
BLOCK_EPISODES = 500
# 5.2's stability rule: a p95 that drifts more than this from the first five chunks means the
# episode sat below thermal steady state, and one repeat is nearly free.
DRIFT_BAR = 0.20


def _import_backend() -> dict[str, Any]:
    """Import lerobot's pieces lazily, so `[NOT RUN]` is a message rather than a traceback.

    On Windows this is where the script stops: `lerobot.envs.libero` imports LIBERO, which needs
    the `robosuite==1.4.0` pin that cannot load there (docs/11 4.4), and `smolvla` needs
    `transformers`, which the project env does not carry. Both are expected, not defects.
    """
    try:
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[NOT RUN] {exc}")
        print(
            "          this needs the LIBERO-pinned interpreter (scripts/build_lerobot_env.sh,\n"
            "          docs/11 5.1) plus `transformers`; on Windows the pin cannot load at all,\n"
            "          so `not run` is the correct outcome here (docs/11 4.4)."
        )
        raise SystemExit(2) from exc
    return {
        "LiberoEnvConfig": LiberoEnvConfig,
        "make_env": make_env,
        "make_env_pre_post_processors": make_env_pre_post_processors,
        "preprocess_observation": preprocess_observation,
        "make_pre_post_processors": make_pre_post_processors,
        "SmolVLAPolicy": SmolVLAPolicy,
    }


def _single_env(envs: dict[str, Any]) -> Any:
    """The one vector env `make_env` built, or a loud failure.

    `make_env` returns suite -> task_id -> vector env. Taking "the first" would let a suite typo
    silently time a different task, which is a measurement about the wrong thing; anything but
    exactly one env is an error here.
    """
    flat: list[tuple[str, Any]] = []
    for key, value in envs.items():
        if isinstance(value, dict):
            flat.extend((f"{key}[{task_id}]", env) for task_id, env in value.items())
        else:
            flat.append((str(key), value))
    if len(flat) != 1:
        raise ValueError(f"expected exactly one env, built {[name for name, _ in flat]}")
    return flat[0][1]


def _reset(env: Any, init_state: int, seed: int | None) -> Any:
    """Reset, pinning the initial state so a repeat is the same episode.

    `LiberoEnv.reset` increments `init_state_id` on every call, so a second episode without this
    would start from initial state 1 and be reported as a repeat of state 0. The repeat rule in
    5.2 exists to average out thermal drift, not to change the episode.
    """
    for sub in getattr(env, "envs", ()) or ():
        if hasattr(sub, "init_state_id"):
            sub.init_state_id = init_state
    observation, _info = env.reset(seed=seed)
    return observation


# Where each normalization mode keeps its two numbers inside a step's safetensors sidecar. The
# keys are lerobot 0.6.1's own: `policy_postprocessor.json` carries the mode and the feature width
# and none of the statistics, so a probe that scans the JSON for a mean/std pair finds nothing on
# any checkpoint that library writes. Verified against the real SmolVLA pair 2026-11-02.
_STAT_KEYS: dict[str, tuple[str, str]] = {
    "mean_std": ("action.mean", "action.std"),
    "min_max": ("action.min", "action.max"),
    "quantiles": ("action.q01", "action.q99"),
    "quantile10": ("action.q10", "action.q90"),
}
_RANGE_MODES = ("min_max", "quantiles", "quantile10")
# The points each closed form pins. MIN_MAX's inverse is `(z + 1) / 2 * (high - low) + low`, so 0
# is the *midpoint* of the range, never the minimum -- probing 0 against `min` would reject a
# correct pipeline, and it did.
_PROBES_MEAN_STD = (0.0, 1.0)
_PROBES_RANGE = (-1.0, 0.0, 1.0)


def _postprocessor_from_checkpoint(policy_id: str, device: str) -> Any:
    """The postprocessor pipeline alone, accepting a raw action tensor.

    `make_pre_post_processors` builds its postprocessor with the action-tensor converters
    (`policy_action_to_transition` / `transition_to_policy_action`), which is why the pilot can
    call `postprocessor(chunk)` on a tensor; `from_pretrained` alone defaults to the batch
    converters and refuses one. Setting the same two here gives the same arithmetic without
    loading SmolVLA's config -- which needs `transformers` and, for a real episode, LIBERO.

    Used only by `--stats-only`: precondition 3 is arithmetic on the checkpoint's own numbers and
    has no reason to wait for a GPU.
    """
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    pipeline = PolicyProcessorPipeline.from_pretrained(
        policy_id, config_filename="policy_postprocessor.json"
    )
    pipeline.to_transition = policy_action_to_transition
    pipeline.to_output = transition_to_policy_action
    for step in getattr(pipeline, "steps", ()):  # the JSON names `cpu`; the probe may want cuda
        config = getattr(step, "config", None)
        if config is not None and hasattr(config, "device"):
            config.device = device
    return pipeline


def _declared_statistics(config: dict[str, Any]) -> tuple[str, str, int] | None:
    """(norm mode, state file, action width) from a postprocessor JSON, or None.

    Only a step that declares an ACTION feature *and* a sidecar can be probed: the mode selects the
    closed form, the `state_file` holds its numbers, and the declared width is what the probe has
    to feed the pipeline. `max_action_dim` (32 on SmolVLA) is the padded model space, and a 32-wide
    input is one a 7-wide unnormalizer refuses, so the width has to come from here.
    """
    steps = config.get("steps")
    if not isinstance(steps, list):
        return None
    found: tuple[str, str, int] | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        body, state_file = step.get("config"), step.get("state_file")
        if not isinstance(body, dict) or not isinstance(state_file, str):
            continue
        features, norm_map = body.get("features"), body.get("norm_map")
        if not isinstance(features, dict) or not isinstance(norm_map, dict):
            continue
        action = features.get("action")
        if not isinstance(action, dict):
            continue
        shape = action.get("shape")
        if not isinstance(shape, list) or not shape:
            continue
        found = (str(norm_map.get("ACTION", "identity")).lower(), state_file, int(shape[-1]))
    return found


def _probe_postprocessor(
    postprocessor: Any, action_dim: int, device: str, values: tuple[float, ...]
) -> tuple[dict[float, torch.Tensor], str] | None:
    """Run the postprocessor at fixed points in normalized space, returning outputs and the shape.

    The pipeline accepts a chunk `[B, T, D]`; a 2-D `[B, D]` is the fallback for a version that
    expects one action at a time. Both are tried so a shape change in the library is a message
    here rather than a crash mid-episode.
    """
    for shape in ((1, 1, action_dim), (1, action_dim)):
        outputs: dict[float, torch.Tensor] = {}
        try:
            with torch.inference_mode():
                for value in values:
                    outputs[value] = postprocessor(torch.full(shape, value, device=device))
        except Exception:  # the shape is the thing being discovered, not an error to report
            continue
        return outputs, str(shape)
    return None


def _statistics_check(policy_id: str, postprocessor: Any, device: str) -> dict[str, Any]:
    """4.4's step 1: does the postprocessor compute the statistics the checkpoint declares?

    Read from the checkpoint rather than reimplemented: the JSON names the mode and the width, the
    step's safetensors sidecar carries the numbers, and the closed forms come from
    `lerobot/processor/normalize_processor.py` -- MEAN_STD's inverse is `z * std + mean`, every
    range mode's is `(z + 1) / 2 * (high - low) + low`. Probing the points the mode actually pins
    tests the mapping the pilot depends on without this file owning a second copy of it.

    A sidecar that cannot be read, a mode this probe does not model, or a width that disagrees with
    the declared feature all return `not run` rather than `ok`: "could not check" is not a pass.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    config_path = Path(hf_hub_download(policy_id, "policy_postprocessor.json"))
    declared = _declared_statistics(json.loads(config_path.read_text(encoding="utf-8")))
    if declared is None:
        return {
            "verdict": "not run",
            "note": "no postprocessor step with an ACTION feature and a state_file",
        }
    mode, state_file, width = declared
    keys = _STAT_KEYS.get(mode)
    if keys is None:
        return {"verdict": "not run", "note": f"norm mode {mode!r} is not modelled here"}
    low_key, high_key = keys

    sidecar = Path(hf_hub_download(policy_id, state_file))
    with safe_open(sidecar, framework="pt") as handle:
        present = set(handle.keys())
        if not {low_key, high_key} <= present:
            return {
                "verdict": "not run",
                "note": f"{state_file} has no {low_key}/{high_key}; keys are {sorted(present)}",
            }
        low = handle.get_tensor(low_key).double()
        high = handle.get_tensor(high_key).double()
    if low.numel() != width or high.numel() != width:
        return {
            "verdict": "not run",
            "note": f"{low_key}/{high_key} hold {low.numel()}/{high.numel()} values, the ACTION "
            f"feature declares {width}",
        }

    family = "mean_std" if mode == "mean_std" else "range"
    values = _PROBES_MEAN_STD if family == "mean_std" else _PROBES_RANGE
    probed = _probe_postprocessor(postprocessor, width, device, values)
    if probed is None:
        return {"verdict": "not run", "note": f"the postprocessor rejected a {width}-wide probe"}
    outputs, shape = probed

    expected = (
        {0.0: low, 1.0: low + high}
        if family == "mean_std"
        else {-1.0: low, 0.0: (low + high) / 2.0, 1.0: high}
    )
    worst = 0.0
    for value, want in expected.items():
        got = outputs[value].reshape(-1).double().cpu()
        if got.numel() != width:
            return {
                "verdict": "not run",
                "note": f"the postprocessor returned {got.numel()} values for {width} declared "
                "dims",
            }
        worst = max(worst, float((got - want).abs().max().item()))
    return {
        "verdict": "ok" if worst <= BAR else "mismatch",
        "mode": mode,
        "dims": width,
        "probe_shape": shape,
        "max_abs_diff": worst,
        "bar": BAR,
    }


def _report_statistics(statistics: dict[str, Any]) -> bool:
    """Print 4.4 step 1's verdict; True only for `ok`.

    `not run` is deliberately not a pass: a probe that could not read the checkpoint's numbers
    says nothing about the postprocessor, and treating that as green is how a model nobody checked
    gets its cost measured under another model's name.
    """
    verdict = statistics["verdict"]
    if verdict == "ok":
        print(
            f"[ok] precondition 3: postprocessor matches the declared {statistics['mode']} "
            f"statistics over {statistics['dims']} dims (max |diff| "
            f"{statistics['max_abs_diff']:.2e}, bar {BAR:g})"
        )
        return True
    if verdict == "mismatch":
        print(
            f"[FAIL] precondition 3: the postprocessor differs from the checkpoint's own "
            f"statistics by {statistics['max_abs_diff']:.2e} (bar {BAR:g}). The pilot would "
            "calibrate a different model; stopping (docs/11 4.4 step 1)."
        )
        return False
    print(
        f"[STOP] precondition 3 could not be established: {statistics.get('note')}. "
        "Verify the statistics by hand, then re-run with --skip-stats."
    )
    return False


def _build_batch(
    observation: Any,
    env: Any,
    preprocess: Any,
    env_preprocessor: Any,
    preprocessor: Any,
) -> dict[str, Any]:
    """The batch lerobot's own eval loop hands the policy, at one chunk's cadence.

    Same order as `lerobot_eval`: env observation -> LeRobot keys -> the env's processor (the
    LIBERO step that names the views and carries the instruction) -> the policy's preprocessor
    (device, padding, tokenizer). Reproducing the order matters because the tokenizer is what
    turns `task` into the language slot.
    """
    batch = preprocess(observation)
    batch["task"] = list(env.call("task_description"))
    batch = env_preprocessor(batch)
    return preprocessor(batch)


def _to_env_actions(action: Any, actions_per_chunk: int) -> np.ndarray:
    """Postprocessed chunk -> `[T, 7]` numpy actions, with the failures that look like data.

    Two of them are worth a hard error rather than a warning: a single action means the policy
    did not hand back a chunk and the measured cadence would describe a different loop, and a
    value far outside LIBERO's `[-1, 1]` box means the chunk was never unnormalized -- which
    produces a plausible-looking timing row for a policy that never moved the arm.
    """
    raw = action.detach().to("cpu").float().numpy() if torch.is_tensor(action) else action
    array = np.asarray(raw)
    array = np.squeeze(array) if array.ndim == 3 and array.shape[0] == 1 else array
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2:
        raise ValueError(f"postprocessor returned {array.shape}; expected [T, D] or [B, T, D]")
    if array.shape[1] > ACTION_DIM:
        array = array[:, :ACTION_DIM]  # the 32-wide padded model space -> the env's 7
    if array.shape[1] != ACTION_DIM:
        raise ValueError(f"actions have width {array.shape[1]}, the env wants {ACTION_DIM}")
    if array.shape[0] < actions_per_chunk:
        raise ValueError(
            f"the policy returned {array.shape[0]} actions, fewer than the {actions_per_chunk} "
            "this cadence executes; that is a single action, not a chunk, and timing it would "
            "measure a different loop"
        )
    worst = float(np.abs(array).max())
    if worst > 1.5:
        raise ValueError(
            f"actions reach |a| = {worst:.2f}, outside LIBERO's [-1, 1] box; the chunk was "
            "probably never unnormalized (compare the postprocessor against the checkpoint)"
        )
    return array


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0, "n": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "min": float(array.min()),
        "n": float(array.size),
    }


def _run_episode(
    *,
    backend: dict[str, Any],
    env: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    args: argparse.Namespace,
    statistics: dict[str, Any],
    load_s: float,
) -> dict[str, Any]:
    """One episode, one row. No success rate: this is a cost measurement, not an eval."""
    dtypes = args.device.startswith("cuda")
    start = time.perf_counter()
    observation = _reset(env, args.init_state, args.seed)
    reset_s = time.perf_counter() - start
    if dtypes:
        torch.cuda.reset_peak_memory_stats()

    chunks = args.episode_length // args.actions_per_chunk
    policy_ms: list[float] = []
    build_ms: list[float] = []
    env_ms: list[float] = []
    total_ms: list[float] = []
    steps = 0
    first_call_ms = 0.0
    finished_early = False

    for index in range(chunks):
        t0 = time.perf_counter()
        batch = _build_batch(
            observation, env, backend["preprocess_observation"], env_preprocessor, preprocessor
        )
        build_ms.append((time.perf_counter() - t0) * 1000.0)

        if dtypes:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
            chunk = postprocessor(chunk)
        if dtypes:
            torch.cuda.synchronize()
        policy_ms.append((time.perf_counter() - t1) * 1000.0)
        if index == 0:
            first_call_ms = policy_ms[0]

        actions = _to_env_actions(chunk, args.actions_per_chunk)
        t2 = time.perf_counter()
        for action in actions[: args.actions_per_chunk]:
            observation, _reward, terminated, truncated, _info = env.step(
                np.asarray(action, dtype=np.float32)[None]
            )
            steps += 1
            if bool(np.any(terminated)) or bool(np.any(truncated)):
                finished_early = True
                break
        env_ms.append((time.perf_counter() - t2) * 1000.0)
        total_ms.append(build_ms[-1] + policy_ms[-1] + env_ms[-1])
        if finished_early:
            break

    episode_s = time.perf_counter() - start
    first_five = _quantiles(total_ms[:5])
    overall = _quantiles(total_ms)
    drift = (
        (overall["p95"] - first_five["p95"]) / first_five["p95"] if first_five["p95"] > 0 else 0.0
    )
    chunk_hours = overall["mean"] / 1000.0 / 3600.0
    projected = chunk_hours * chunks * BLOCK_EPISODES

    return {
        "kind": "pilot",
        "config_id": args.config_id,
        "policy": "smolvla",
        "checkpoint": args.policy_id,
        "task": "libero",
        "task_suite": SUITE,
        "task_id": TASK_ID,
        "init_state": args.init_state,
        "episode_length": args.episode_length,
        "actions_per_chunk": args.actions_per_chunk,
        "chunks_elapsed": len(total_ms),
        "steps": steps,
        "chunk_size": int(getattr(policy.config, "chunk_size", 0)),
        "n_action_steps": int(getattr(policy.config, "n_action_steps", 0)),
        "policy_ms": _quantiles(policy_ms),
        "build_ms": _quantiles(build_ms),
        "env_ms": _quantiles(env_ms),
        "chunk_ms": overall,
        "first_call_ms": first_call_ms,
        "drift_pct": drift * 100.0,
        "reset_s": reset_s,
        "episode_s": episode_s,
        "load_s": load_s,
        "total_s": load_s + reset_s + episode_s,
        "finished_early": finished_early,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 2**20) if dtypes else 0.0,
        "gpu": torch.cuda.get_device_name() if dtypes else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_hours": (load_s + reset_s + episode_s) / 3600.0,
        "projected_gpu_hours_500": projected,
        "projected_gpu_hours_three_arms": projected * 3.0,
        "preconditions": statistics,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _print_row(row: dict[str, Any]) -> None:
    chunk = row["chunk_ms"]
    print(
        f"\n[{row['config_id']}] chunks {row['chunks_elapsed']}, steps {row['steps']}, "
        f"episode {row['episode_s']:.1f} s, load {row['load_s']:.1f} s"
    )
    print(f"  {'term':<14}{'mean':>10}{'median':>10}{'p95':>10}{'max':>10}   (ms)")
    for name in ("policy_ms", "build_ms", "env_ms", "chunk_ms"):
        q = row[name]
        print(
            f"  {name.removesuffix('_ms'):<14}{q['mean']:>10.1f}{q['median']:>10.1f}"
            f"{q['p95']:>10.1f}{q['max']:>10.1f}"
        )
    print(
        f"  first call {row['first_call_ms']:.1f} ms   p95 drift vs first 5 chunks "
        f"{row['drift_pct']:+.1f} %   peak VRAM {row['peak_vram_mb']:.0f} MiB   {row['gpu']}"
    )
    print(
        f"  projected: {row['projected_gpu_hours_500']:.2f} GPU-h for the "
        f"{BLOCK_EPISODES}-episode arm, {row['projected_gpu_hours_three_arms']:.2f} GPU-h for "
        "three arms (projected: inherits the cadence, not measured per episode)"
    )
    if chunk["mean"] >= 2000.0:
        print(
            "  WARNING: over ~2 s per chunk. 5.2 says the plan changes shape here -- a 10-action "
            "cadence stops being defensible and RTC's premise goes with it."
        )
    if row["finished_early"]:
        print("  NOTE: the episode ended before the cap, so the chunk count is short.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One LIBERO-Spatial episode measuring SmolVLA's per-chunk cost (docs/11 5.2)"
    )
    parser.add_argument("--policy-id", default=POLICY_ID)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--actions-per-chunk", type=int, default=ACTIONS_PER_CHUNK)
    parser.add_argument("--seed", type=int, default=None, help="env seed; None leaves LIBERO's")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-id", default="smolvla_pilot")
    parser.add_argument("--out", type=Path, default=Path("results/smolvla_pilot.jsonl"))
    parser.add_argument("--check-only", action="store_true", help="run the preconditions and stop")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="run precondition 3 alone: no LIBERO, no policy weights, no GPU",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="skip precondition 3 (only if the step-0 statistics were verified by hand)",
    )
    args = parser.parse_args()

    if args.actions_per_chunk <= 0:
        raise SystemExit("--actions-per-chunk must be positive")

    if args.stats_only:
        # Before `_import_backend`, which is the whole point: this is the one precondition that
        # needs nothing but the checkpoint's files and a CPU.
        try:
            postprocessor = _postprocessor_from_checkpoint(args.policy_id, args.device)
            statistics = _statistics_check(args.policy_id, postprocessor, args.device)
        except Exception as exc:  # a download or a parse failure is a "not run", not a pass
            statistics = {"verdict": "not run", "note": f"{type(exc).__name__}: {exc}"}
        passed = _report_statistics(statistics)
        print(json.dumps(statistics, indent=2))
        raise SystemExit(0 if passed else 1)

    backend = _import_backend()

    # ---- precondition 1: the env constructs on the pinned stack ----------------------------
    env_config = backend["LiberoEnvConfig"](
        task=SUITE, task_ids=[TASK_ID], episode_length=args.episode_length
    )
    try:
        env = _single_env(backend["make_env"](env_config, n_envs=1, use_async_envs=False))
        env.reset()
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[NOT RUN] {exc}")
        print("          precondition 1 failed at import: LIBERO is not installed here.")
        raise SystemExit(2) from exc
    except Exception as exc:  # the failure, not a traceback, is the measurement
        print(f"[FAIL] precondition 1 (env constructs): {type(exc).__name__}: {exc}")
        print("       docs/11 4.5's ladder says which wall this is; `just port-walk` re-checks it.")
        raise SystemExit(1) from exc
    print(f"[ok] precondition 1: {SUITE} task {TASK_ID} constructs on this stack")

    # ---- precondition 2: the checkpoint loads ---------------------------------------------
    load_start = time.perf_counter()
    try:
        policy = backend["SmolVLAPolicy"].from_pretrained(args.policy_id)
        policy = policy.to(args.device).eval()
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[NOT RUN] {exc}")
        print("          precondition 2 failed at import: `transformers` is not installed here.")
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"[FAIL] precondition 2 (checkpoint loads): {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    load_s = time.perf_counter() - load_start
    preprocessor, postprocessor = backend["make_pre_post_processors"](
        policy_cfg=policy.config,
        pretrained_path=args.policy_id,
        preprocessor_overrides={"device_processor": {"device": str(args.device)}},
    )
    env_preprocessor, _env_postprocessor = backend["make_env_pre_post_processors"](
        env_cfg=env_config, policy_cfg=policy.config
    )
    print(f"[ok] precondition 2: {args.policy_id} loads in {load_s:.1f} s")

    # ---- precondition 3: the postprocessor reproduces the declared statistics --------------
    statistics: dict[str, Any] = {"verdict": "skipped", "note": "--skip-stats"}
    if not args.skip_stats:
        try:
            statistics = _statistics_check(args.policy_id, postprocessor, args.device)
        except Exception as exc:  # a download or a parse failure here is a "not run", not a pass
            statistics = {"verdict": "not run", "note": f"{type(exc).__name__}: {exc}"}
        if not _report_statistics(statistics):
            raise SystemExit(1)

    if args.check_only:
        print("\npreconditions green; the pilot was not run (--check-only).")
        return

    # ---- the measurement ------------------------------------------------------------------
    rows = []
    for attempt in range(2):
        row = _run_episode(
            backend=backend,
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            args=args,
            statistics=statistics,
            load_s=load_s,
        )
        rows.append(row)
        _print_row(row)
        if abs(row["drift_pct"]) <= DRIFT_BAR * 100.0 or attempt == 1:
            break
        print(
            f"  p95 drifted {row['drift_pct']:+.1f} % from the first five chunks (bar "
            f"{DRIFT_BAR * 100:.0f} %), so the repeat 5.2 asks for is taken; same initial state."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    chosen = min(rows, key=lambda item: abs(item["drift_pct"]))
    print(
        f"\nrow for the block's sizing: {chosen['config_id']} (|drift| "
        f"{abs(chosen['drift_pct']):.1f} %), projected "
        f"{chosen['projected_gpu_hours_500']:.2f} GPU-h per 500-episode arm. Written to {args.out}"
    )


if __name__ == "__main__":
    main()
