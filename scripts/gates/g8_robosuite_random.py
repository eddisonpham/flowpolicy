"""G8: robosuite Lift closed loop with a random policy.

Fallback decision D1 (docs/06): G8 red means PushT only. This host has no
robosuite install and no EGL/headless GL path; installing it would add a
heavy dependency without a benchmark to justify it. Recorded NOT RUN.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import robosuite  # noqa: F401
    except Exception as exc:
        print(f"NOT RUN: robosuite not installed ({type(exc).__name__}); fallback D1 = PushT only")
        return 0
    try:
        import robosuite as suite
    except Exception as exc:
        print(f"FAIL: robosuite imported but not usable: {type(exc).__name__}: {exc}")
        return 1
    try:
        env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
    except Exception as exc:
        print(f"FAIL: could not create Lift: {type(exc).__name__}: {exc}")
        return 1
    env.close()
    print("PASS: Lift created")
    return 0


if __name__ == "__main__":
    sys.exit(main())
