"""What produced a row: the commit and the lockfile it was written from.

Two callers need this and they need the same answer: `training/trainer.py` stamps a checkpoint so
a run can be traced to the code that produced it, and `eval/rollout.py` stamps every episode row
for the same reason. A `results/*.jsonl` line outlives the branch it was written on, and after a
rebase there is no way to re-derive which commit produced it.

One implementation, because two copies of "the git hash" would eventually disagree about what
"unknown" looks like -- and a row that cannot be traced is only honest when it says so rather than
carrying a plausible-looking hex string.
"""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path

__all__ = ["UNKNOWN", "git_hash", "lock_hash", "provenance"]

UNKNOWN = "unknown"


def git_hash() -> str:
    """`HEAD` at the time of the call, or `"unknown"` outside a work tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN


def lock_hash(path: str | Path = "uv.lock") -> str:
    """SHA-256 of the lockfile, or `"unknown"` when it is not next to the caller."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return UNKNOWN


@functools.lru_cache(maxsize=1)
def provenance() -> dict[str, str]:
    """Both stamps, computed once per process.

    `evaluate` stamps every row of a block and a 200-episode run must not fork `git` 200 times.
    The cache is per process, so a row always describes the tree it was written from; a caller
    that needs a fresh reading (a test changing directory, say) calls `provenance.cache_clear()`.
    """
    return {"git": git_hash(), "lock_sha256": lock_hash()}
