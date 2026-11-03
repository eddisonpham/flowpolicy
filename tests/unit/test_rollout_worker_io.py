"""What a worker writes, and what it says when an episode raises.

The harness used to hand rows back through `future.result()`, so a chunk that died on its 12th
episode lost all 25 and the parent reported `BrokenProcessPool` -- no seed, no traceback, nothing
to debug. These tests pin the replacement: rows hit disk as each episode finishes, and the failure
names the seed that caused it.

`rollout_episode` is stubbed, so no simulator, policy or GPU is involved. `_stamp` runs for real
(it only reads `git`), because the row schema is part of what is being checked.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from policybench.eval import rollout
from policybench.logging import read_events

pytestmark = pytest.mark.unit


def _fake_episode(_policy: object, seed: int, **_kwargs: object) -> dict[str, float]:
    return {
        "success": float(seed % 2),
        "coverage": 0.5,
        "steps": 300.0,
        "latency_mean_ms": 100.0,
        "latency_p95_ms": 120.0,
        "wall_s": 12.0,
        "chunks": 38.0,
    }


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "policy": None,
        "meta": {"policy": "stub", "train_seed": -1, "checkpoint": "stub", "step": -1},
        "device": "cpu",
        "steps": 10,
        "horizon": 8,
        "noise_offset": 0,
        "instruction": "",
    }.items():
        monkeypatch.setitem(rollout._WORKER, key, value)
    monkeypatch.setattr(rollout, "rollout_episode", _fake_episode)


def test_rows_are_written_as_each_episode_finishes(tmp_path: Path, worker: None) -> None:
    rows_path = tmp_path / "out.jsonl"
    log_path = tmp_path / "run.jsonl"
    written = rollout._run_episodes([1000, 1001], str(rows_path), str(log_path), "cfg")

    assert written == 2
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert [row["eval_seed"] for row in rows] == [1000, 1001]
    assert rows[0]["config_id"] == "cfg"
    assert rows[0]["noise_seed"] == 1000
    assert set(rows[0]["provenance"]) == {"git", "lock_sha256"}


def test_the_log_records_one_event_per_episode(tmp_path: Path, worker: None) -> None:
    log_path = tmp_path / "run.jsonl"
    rollout._run_episodes([1000, 1001], str(tmp_path / "out.jsonl"), str(log_path), "cfg")
    events = [event for event in read_events(log_path) if event["event"] == "episode_done"]
    assert [event["eval_seed"] for event in events] == [1000, 1001]
    assert {event["run_id"] for event in events} == {"cfg"}


def test_a_failing_episode_keeps_the_earlier_rows_and_names_its_seed(
    tmp_path: Path, worker: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(policy: object, seed: int, **kwargs: object) -> dict[str, float]:
        if seed == 1001:
            raise ValueError("env step failed")
        return _fake_episode(policy, seed, **kwargs)

    monkeypatch.setattr(rollout, "rollout_episode", fake)
    rows_path = tmp_path / "out.jsonl"
    log_path = tmp_path / "run.jsonl"

    with pytest.raises(rollout.RolloutError) as caught:
        rollout._run_episodes([1000, 1001, 1002], str(rows_path), str(log_path), "cfg")

    assert caught.value.eval_seed == 1001
    assert "1001" in str(caught.value)
    assert "ValueError" in str(caught.value)
    # What finished is kept: the block resumes instead of restarting.
    rows = rows_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    # And the worker that saw it wrote the traceback.
    errors = [event for event in read_events(log_path) if event["event"] == "error"]
    assert errors and errors[0]["eval_seed"] == 1001
    assert "ValueError: env step failed" in errors[0]["error"]


def test_a_block_cannot_spawn_a_main_module_that_is_not_on_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `python -c` / a notebook / a REPL: `__file__` is `<string>` or `<stdin>`.
    fake_main = types.SimpleNamespace(__file__="<stdin>", __spec__=None)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert rollout._can_spawn() is False


def test_a_module_invoked_with_dash_m_is_spawnable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_main = types.SimpleNamespace(
        __file__="../pytest/__main__.py", __spec__=types.SimpleNamespace(name="pytest")
    )
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert rollout._can_spawn() is True


def test_a_script_on_disk_is_spawnable(monkeypatch: pytest.MonkeyPatch) -> None:
    # The normal case, and the one every runner in `scripts/` takes: a real entry-point file.
    fake_main = types.SimpleNamespace(__file__=__file__, __spec__=None)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert rollout._can_spawn() is True
