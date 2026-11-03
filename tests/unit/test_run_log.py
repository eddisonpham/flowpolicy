"""Run logs and progress: the observability layer's own contract.

Two failures motivated this module, both invisible in the row files and both fatal to supervising a
long block: an eval printed nothing until its last episode, and a worker crash surfaced as
`BrokenProcessPool` with no seed. So the tests are about the *shape* of what gets written -- one
JSON object per line, attributable to a run, with a traceback on failure -- and about the ETA
arithmetic, which is the number a reader uses to decide whether to wait or to go to bed.
"""

from __future__ import annotations

import io
import json
import re

import pytest

from policybench.logging import Progress, RunLog, default_log_path, read_events

pytestmark = pytest.mark.unit


def test_an_event_is_one_json_object_per_line_with_its_run_id(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    with RunLog(path, run_id="demo", echo=False) as log:
        log.event("run_start", episodes=200, checkpoint="hf:x")
        log.event("episode_done", eval_seed=7, success=1.0)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["run_id"] == "demo"
    assert first["event"] == "run_start"
    assert first["episodes"] == 200
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d.*", first["ts"])


def test_two_handles_on_one_path_append_without_clobbering(tmp_path) -> None:
    # This is how workers log: each opens the same file in append mode.
    path = tmp_path / "run.jsonl"
    with RunLog(path, run_id="worker-0", echo=False) as first:
        first.event("episode_done", eval_seed=1)
    with RunLog(path, run_id="worker-1", echo=False) as second:
        second.event("episode_done", eval_seed=2)
    assert [event["run_id"] for event in read_events(path)] == ["worker-0", "worker-1"]


def test_an_error_event_carries_the_traceback_and_the_fields(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    with RunLog(path, run_id="demo", echo=False) as log:
        try:
            raise ValueError("chunk was the wrong width")
        except ValueError as exc:
            log.error(exc, stage="episode", eval_seed=1003)

    event = next(event for event in read_events(path) if event["event"] == "error")
    assert event["eval_seed"] == 1003
    assert event["stage"] == "episode"
    assert "ValueError: chunk was the wrong width" in event["error"]
    assert "Traceback (most recent call last)" in event["traceback"]


def test_progress_echoes_one_line_and_still_writes_the_event(tmp_path) -> None:
    stream = io.StringIO()
    path = tmp_path / "run.jsonl"
    with RunLog(path, run_id="demo", stream=stream) as log:
        log.progress("chunk_done", done=8, total=32)
    assert "chunk_done" in stream.getvalue()
    assert "done=8" in stream.getvalue()
    assert json.loads(path.read_text(encoding="utf-8").strip())["done"] == 8


def test_a_worker_does_not_echo(tmp_path) -> None:
    stream = io.StringIO()
    with RunLog(tmp_path / "run.jsonl", run_id="w", echo=False, stream=stream) as log:
        log.progress("chunk_done", done=8)
    assert stream.getvalue() == ""


def test_close_is_idempotent_and_writes_the_summary_once(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    log = RunLog(path, run_id="demo", echo=False)
    log.close(rows=3)
    log.close(rows=3)
    assert [event["event"] for event in read_events(path)] == ["run_end"]


def test_a_torn_line_does_not_hide_the_rest_of_the_log(tmp_path) -> None:
    # A killed process leaves half a line; every event before it is still readable.
    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"event": "run_start", "run_id": "x"}\n{"event": "episode_do\n',
        encoding="utf-8",
    )
    assert [event["event"] for event in read_events(path)] == ["run_start"]


def test_eta_is_unknown_before_anything_finishes() -> None:
    progress = Progress(total=100)
    first = progress.advance()
    assert first["done"] == 1
    assert first["eta_s"] is not None  # one unit is enough to have a rate
    empty = Progress(total=100)
    assert empty.done == 0
    assert empty.total == 100


def test_eta_never_reports_zero_for_an_unstarted_block() -> None:
    # A fabricated 0 would read as "done" on a block that has not started.
    progress = Progress(total=0)
    fields = progress.advance()
    assert fields["pct"] == 100.0
    assert fields["eta_s"] is None


def test_progress_tracks_a_rate_and_a_remaining_count() -> None:
    progress = Progress(total=10)
    progress.advance(4)
    fields = progress.advance(2)
    assert fields["done"] == 6
    assert fields["total"] == 10
    assert fields["pct"] == 60.0
    assert fields["rate_per_s"] > 0


def test_the_default_log_path_is_a_slugged_file_in_the_ignored_log_dir() -> None:
    path = default_log_path("eval dp_published n200/!!")
    assert path.parent.name == "logs"
    assert path.suffix == ".jsonl"
    assert "/" not in path.name
    assert "!!" not in path.name


def test_the_reader_states_episodes_rate_and_the_failure(tmp_path) -> None:
    # `just logs` is the screen a long run is supervised from, so what it states is the point.
    from scripts.logs import summarize

    path = tmp_path / "run.jsonl"
    with RunLog(path, run_id="cfg", echo=False) as log:
        log.event("run_start", episodes=2)
        log.event("episode_done", eval_seed=1, success=1.0, wall_s=5.0)
        log.event("episode_done", eval_seed=2, success=0.0, wall_s=7.0)
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            log.error(exc, stage="episode", eval_seed=3)

    text = summarize(path, errors_only=False, tail=2)
    assert "episodes: 2" in text
    assert "50.0%" in text
    assert "RuntimeError: boom" in text


def test_errors_only_omits_the_progress_lines(tmp_path) -> None:
    from scripts.logs import summarize

    path = tmp_path / "run.jsonl"
    with RunLog(path, run_id="cfg") as log:
        log.progress("chunk_done", done=1, total=8)
        log.event("episode_done", eval_seed=1, success=1.0, wall_s=5.0)
    text = summarize(path, errors_only=True, tail=5)
    # The counts line still names every event type; what `--errors` drops is the progress dump
    # and the per-episode summary, not the inventory of what is in the file.
    assert "progress:" not in text
    assert "episodes: 1" not in text
    assert "errors: none" in text
