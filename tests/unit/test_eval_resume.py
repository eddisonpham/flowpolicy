"""Chunking and resume for the rollout harness (docs/09 option C, cheap version).

Two properties are pinned here, both of them about a multi-hour block surviving a crash:

* a block is cut into more than one chunk per worker, so a single-worker run flushes rows as it
  goes instead of holding all of them in memory until the last episode;
* a re-run of the same `config_id` skips the seeds already in `.partial`, and never stitches in
  rows from a different run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from policybench.eval.rollout import CHUNKS_PER_WORKER, _chunks, completed_seeds

pytestmark = pytest.mark.unit


def test_chunks_cover_every_seed_exactly_once() -> None:
    seeds = list(range(1000, 1200))
    chunks = _chunks(seeds, workers=1)
    flat = [seed for chunk in chunks for seed in chunk]
    assert sorted(flat) == seeds
    assert len(chunks) == CHUNKS_PER_WORKER


def test_chunks_scale_with_workers_and_stay_interleaved() -> None:
    seeds = list(range(40))
    chunks = _chunks(seeds, workers=2, per_worker=5)
    assert len(chunks) == 10
    # Interleaved, not sliced: the first chunk holds every 10th seed, not a contiguous run.
    assert chunks[0] == [0, 10, 20, 30]


def test_short_blocks_do_not_produce_empty_chunks() -> None:
    chunks = _chunks([7, 8, 9], workers=1, per_worker=8)
    assert sorted(seed for chunk in chunks for seed in chunk) == [7, 8, 9]
    assert all(chunks)


def test_completed_seeds_reads_only_matching_config(tmp_path: Path) -> None:
    partial = tmp_path / "out.jsonl.partial"
    partial.write_text(
        json.dumps({"eval_seed": 1, "config_id": "mine"})
        + "\n"
        + json.dumps({"eval_seed": 2, "config_id": "other"})
        + "\n"
        + json.dumps({"eval_seed": 3, "config_id": "mine"})
        + "\n",
        encoding="utf-8",
    )
    assert completed_seeds(partial, "mine") == {1, 3}
    assert completed_seeds(partial, "nothing") == set()


def test_missing_partial_means_nothing_completed(tmp_path: Path) -> None:
    assert completed_seeds(tmp_path / "absent.partial", "mine") == set()
