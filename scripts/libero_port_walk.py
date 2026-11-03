"""Does robosuite still break LIBERO's env layer the way docs/11 section 4.5 recorded?

`docs/11` 4.4 measured that V4 cannot run on Windows: `libero.libero.envs` imports
`robosuite.environments.manipulation.single_arm_env`, which 1.5.2 removed. Section 4.5
then walked the consequences and concluded the port is **2-4 days**, because the removed
`SingleArm` class was a full robot implementation rather than a dispatch stub. That conclusion is
only as good as the wall list, and a wall list is a claim about one specific robosuite release.

So this script re-checks the claim instead of trusting it, and answers exactly one question: *if the
port were attempted today, is the ladder still the five walls in 4.5?* Nothing third-party is edited
and no shim outlives the process.

Two halves, because the walk had two kinds of evidence:

1. **Static walls.** For each of the five, does the release still have the property 4.5 describes --
   module missing, kwarg absent, factory class missing, loader missing, `lite_physics` default
   flipped? Read from `importlib.util.find_spec` and `inspect.signature`, which is also what 4.5
   claimed, so a matching verdict means the *claim* still holds rather than that a shim still works.
2. **The terminal state.** The walk did not stop at wall 1 but at wall 3: with the other walls
   shimmed, a naive `SingleArm = FixedBaseRobot` clears every import and dies constructing an env
   at `KeyError: 'MountedMountedPanda'`. Stage 0 reproduces the first failure, stage 1 that stopping
   point. If stage 1 ever *constructs* an env, the port is cosmetic and 4.5's estimate is stale --
   which is worth knowing before someone spends the days.

Verdicts:

* ``pinned`` -- robosuite 1.4.x is installed, i.e. LIBERO's own pin. The ladder is inapplicable and
  the honest answer is "run the pinned stack, do not port" (``docs/11`` 5.1).
* ``ladder intact`` -- all five walls present and both stages failed where the record says.
* ``drift`` -- a wall is gone or a stage failed differently. Re-read 4.5 and re-price the port.
* ``probe incomplete`` -- a stage raised `ModuleNotFoundError` for something other than the wall
  being measured, i.e. the probe environment lacks a package. That is not evidence about the port,
  so it names the module instead of claiming drift. (The gate's first run did it, on `gymnasium`.)
* ``not run`` -- robosuite (or, for the dynamic half, libero) is missing from this interpreter.

Exit 0 covers every verdict except ``drift``: an optional third-party probe is not a gate on this
repo's own measurement path. ``drift`` exits 1: that is the case that silently costs days.

    .venv-libero/bin/python scripts/libero_port_walk.py     # Scripts/python.exe on Windows

`just port-walk` locates the interpreter. No GPU, no checkpoint, no episode: seconds. LIBERO may try
to fetch its assets from the Hub when a local copy is absent, and falls back when it cannot.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

# The record from docs/11 section 4.5, wall by wall: the claim, and what counts as still true.
# `expect` is the substring a matching observation must contain; short on purpose, because a full
# traceback changes with line numbers and Python patch releases while the type and symbol do not.
WALLS: list[dict[str, str]] = [
    {
        "id": "1",
        "claim": "`single_arm_env` module removed",
        "recorded": "ModuleNotFoundError at LIBERO import",
        "expect": "single_arm_env",
    },
    {
        "id": "2",
        "claim": "`mount_types=` kwarg gone (1.5.2 takes `base_types`)",
        "recorded": "TypeError: unexpected keyword 'mount_types'",
        "expect": "base_types",
    },
    {
        "id": "3",
        "claim": "`robosuite.robots.single_arm.SingleArm` removed",
        "recorded": "KeyError: 'MountedMountedPanda' under a naive alias",
        "expect": "MountedMountedPanda",
    },
    {
        "id": "4",
        "claim": "`robosuite.load_controller_config` removed",
        "recorded": "AttributeError from env_wrapper.py:47, before any sim",
        "expect": "load_controller_config",
    },
    {
        "id": "5",
        "claim": "`lite_physics` default flipped (absent in 1.4.0, True in 1.5.2)",
        "recorded": "no error -- a different integrator against 1.4.0 data",
        "expect": "lite_physics",
    },
]


def _module_missing(name: str) -> tuple[bool, str]:
    """True when `name` cannot be found, with the located path when it can."""
    spec = importlib.util.find_spec(name)
    if spec is None:
        return True, f"find_spec({name!r}) -> None"
    return False, f"find_spec({name!r}) -> {spec.origin}"


def _signature_has(fn: Any, param: str) -> tuple[bool, str]:
    params = inspect.signature(fn).parameters
    return param in params, ", ".join(list(params)[:8]) + (" ..." if len(params) > 8 else "")


def _lite_physics_default(env_cls: type) -> tuple[Any, str]:
    params = inspect.signature(env_cls.__init__).parameters
    if "lite_physics" not in params:
        return None, "absent from the signature (1.4.0 behaviour)"
    return params["lite_physics"].default, f"defaults to {params['lite_physics'].default!r}"


def _libero_config() -> str:
    """Write a `config.yaml` so importing LIBERO cannot block on `input()`.

    `libero.libero.__init__` prompts interactively when `~/.libero/config.yaml` is missing, which
    turns any unattended run into `EOFError` (docs/11 4.4). The paths are derived from the installed
    package rather than hardcoded, so this works on any host that can import `libero` at all.
    """
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("libero is not importable in this interpreter")
    root = Path(spec.origin).parent / "libero"
    cfg_dir = Path(tempfile.mkdtemp(prefix="port_walk_cfg_"))
    (cfg_dir / "config.yaml").write_text(
        f"benchmark_root: {root.as_posix()}\n"
        f"bddl_files: {(root / 'bddl_files').as_posix()}\n"
        f"init_states: {(root / 'init_files').as_posix()}\n"
        f"datasets: {(cfg_dir / 'datasets').as_posix()}\n"
        f"assets: {(root / 'assets').as_posix()}\n",
        encoding="utf-8",
    )
    return str(cfg_dir)


def _install_clearing_shims() -> None:
    """Shim walls 1, 2 and 4 and alias wall 3, then stop -- this is where the walk stopped.

    Deliberately *not* a working robot class: the point is to reproduce the failure that proved the
    port is architectural, so the naive alias 4.5 describes is what gets installed.
    """
    import robosuite
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.robots import FixedBaseRobot

    module = types.ModuleType("robosuite.environments.manipulation.single_arm_env")

    class SingleArmEnv(ManipulationEnv):
        """1.4.0's class re-expressed on `ManipulationEnv`: an assert plus three properties."""

        def __init__(self, *args: Any, mount_types: str = "default", **kwargs: Any) -> None:
            kwargs.pop("mount_types", None)
            super().__init__(*args, base_types=mount_types, **kwargs)

    module.SingleArmEnv = SingleArmEnv
    sys.modules[module.__name__] = module

    robots_module = types.ModuleType("robosuite.robots.single_arm")
    robots_module.SingleArm = FixedBaseRobot  # naive: clears imports, breaks the factory
    sys.modules[robots_module.__name__] = robots_module

    if not hasattr(robosuite, "load_controller_config"):
        robosuite.load_controller_config = robosuite.load_part_controller_config  # type: ignore[attr-defined]


def _attempt_env() -> str:
    """Import LIBERO's env layer and construct one env. Returns 'constructed' or the failure."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(0),
        robots=["MountedPanda"],
        camera_heights=128,
        camera_widths=128,
    )
    env.reset()
    return "constructed"


def _first_failure() -> str:
    try:
        _attempt_env()
    except Exception as exc:  # the type and message are the measurement, not an error to swallow
        return f"{type(exc).__name__}: {exc}"
    return "constructed"


def _static_rows() -> list[dict[str, str]]:
    import robosuite
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

    rows: list[dict[str, str]] = []

    missing, detail = _module_missing("robosuite.environments.manipulation.single_arm_env")
    rows.append(
        {
            "id": "1",
            "present": str(missing).lower(),
            "observed": "module absent" if missing else detail,
        }
    )

    has_kwarg, params = _signature_has(ManipulationEnv.__init__, "mount_types")
    rows.append(
        {
            "id": "2",
            "present": str(not has_kwarg).lower(),
            "observed": "`mount_types` in signature"
            if has_kwarg
            else f"no `mount_types`; got {params}",
        }
    )

    missing_robot, detail_robot = _module_missing("robosuite.robots.single_arm")
    rows.append(
        {
            "id": "3",
            "present": str(missing_robot).lower(),
            "observed": "robot module absent" if missing_robot else detail_robot,
        }
    )

    has_loader = hasattr(robosuite, "load_controller_config")
    recent_note = "present" if hasattr(robosuite, "load_part_controller_config") else "also absent"
    rows.append(
        {
            "id": "4",
            "present": str(not has_loader).lower(),
            "observed": (
                "`load_controller_config` present"
                if has_loader
                else f"absent; `load_part_controller_config` {recent_note}"
            ),
        }
    )

    default, detail_lite = _lite_physics_default(ManipulationEnv)
    rows.append({"id": "5", "present": str(default is True).lower(), "observed": detail_lite})
    return rows


def _robosuite_version() -> str:
    import robosuite

    return str(robosuite.__version__)


def main() -> None:
    if importlib.util.find_spec("robosuite") is None:
        print("[NOT RUN] robosuite is not installed in this interpreter -- see docs/11 4.4")
        raise SystemExit(0)

    version = _robosuite_version()
    major_minor = tuple(int(part) for part in version.split(".")[:2])

    if major_minor < (1, 5):
        print(f"[PINNED ] robosuite {version} is LIBERO's own pin: the ladder does not apply.")
        print("          Porting is not needed on this stack -- run it (docs/11 5.1).")
        raise SystemExit(0)

    print(f"robosuite {version}; recorded ladder from docs/11 4.5\n")
    rows = _static_rows()
    drifted: list[str] = []

    print(f"{'#':<2} {'wall':<52} {'present':<8} observed")
    for row, wall in zip(rows, WALLS, strict=True):
        print(f"{row['id']:<2} {wall['claim']:<52} {row['present']:<8} {row['observed']}")
        if row["present"] != "true":
            drifted.append(f"wall {row['id']} ({wall['claim']}) is gone")

    if importlib.util.find_spec("libero") is None:
        print("\n[NOT RUN] libero is not installed: static walls only, ladder stages skipped.")
        raise SystemExit(1 if drifted else 0)

    os.environ["LIBERO_CONFIG_PATH"] = _libero_config()

    stage0 = _first_failure()
    print(f"\nstage 0 (no shims)      -> {stage0}")
    _install_clearing_shims()
    stage1 = _first_failure()
    print(f"stage 1 (walls 1/2/4)   -> {stage1}")

    # A missing package in the probe environment is not evidence about robosuite's class layout: the
    # first run of this gate reported drift only because `gymnasium` was absent from `.venv-libero`.
    # Only a constructed env, a different exception type, or a different symbol moves the ladder.
    incomplete: list[str] = []
    for label, result, expected in (
        ("stage 0", stage0, "single_arm_env"),
        ("stage 1", stage1, "MountedMountedPanda"),
    ):
        if expected in result:
            continue
        if result == "constructed":
            drifted.append(f"{label} constructed an env -- the port is cosmetic, not architectural")
        elif result.startswith("ModuleNotFoundError"):
            incomplete.append(f"{label}: {result}")
        else:
            drifted.append(f"{label} fails differently: {result}")

    verdict = "drift" if drifted else ("probe incomplete" if incomplete else "ladder intact")
    report = {
        "robosuite": version,
        "verdict": verdict,
        "walls": rows,
        "stage_0": stage0,
        "stage_1": stage1,
        "drift": drifted,
        "probe_incomplete": incomplete,
    }
    out = Path("results/libero_port_walk.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if drifted:
        print("\n[DRIFT  ] the record in docs/11 4.5 needs revisiting:")
        for line in drifted:
            print(f"          - {line}")
        print(f"          full report: {out}")
        raise SystemExit(1)

    if incomplete:
        print("\n[NOT RUN] the probe environment is incomplete, so the ladder cannot be read:")
        for line in incomplete:
            print(f"          - {line}")
        print(
            "          install the missing package in the probe env and re-run; this is not a wall."
        )
        raise SystemExit(0)

    print(f"\n[INTACT ] all five walls present; the walk stops where it is recorded to. {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
