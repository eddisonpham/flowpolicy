"""Provenance on a row: the commit and the lockfile, under one nested key.

Every number in `results/` is supposed to be traceable to the code that produced it. Checkpoints
have carried `git` and `lock_sha256` since M3; eval rows did not, so a row re-read after a rebase
could not be tied to a tree. `provenance.py` is now the single implementation for both.

The shape is part of the contract, not a detail: the stamps are nested under `provenance`, so no
aggregator can mistake a hash for a measurement, and an old row with no key reads as "unknown"
rather than as a mismatch.
"""

from __future__ import annotations

import hashlib
import re
import subprocess

import pytest

from policybench.eval.rollout import _stamp
from policybench.provenance import UNKNOWN, git_hash, lock_hash, provenance

pytestmark = pytest.mark.unit


def test_git_hash_is_a_commit_in_a_work_tree() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert git_hash() == head
    assert re.fullmatch(r"[0-9a-f]{40}", git_hash())


def test_lock_hash_matches_the_file(tmp_path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"version = 1\n")
    assert lock_hash(lock) == hashlib.sha256(b"version = 1\n").hexdigest()


def test_a_missing_file_is_unknown_not_an_error(tmp_path) -> None:
    assert lock_hash(tmp_path / "absent.lock") == UNKNOWN


def test_the_stamps_are_nested_under_one_key() -> None:
    row: dict = {"eval_seed": 1000, "success": 1.0}
    _stamp(row, config_id="x", sampler_steps=10, action_horizon=8, noise_seed_offset=0)
    assert set(row["provenance"]) == {"git", "lock_sha256"}
    # No measurement key was shadowed by a stamp.
    assert row["eval_seed"] == 1000
    assert row["success"] == 1.0


def test_provenance_is_cached_per_process() -> None:
    assert provenance() is provenance()
