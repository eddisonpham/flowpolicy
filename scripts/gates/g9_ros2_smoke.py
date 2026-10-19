"""G9: ROS 2 Jazzy smoke test in Docker.

No Docker on this host (docs/02 section 2), so D9/D7 fall back to the plain
Python loop. Recorded NOT RUN, not PASS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    docker = shutil.which("docker")
    if docker is None:
        print("NOT RUN: docker not installed; deployment fallback = plain Python loop (D7)")
        return 0
    try:
        version = subprocess.run(
            [docker, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: docker present but not usable: {exc}")
        return 1
    if version.returncode != 0:
        print(f"FAIL: docker daemon unavailable: {version.stderr.strip()[:200]}")
        return 1
    print(f"NOT RUN: docker {version.stdout.strip()} present, but no ROS 2 image is built yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
