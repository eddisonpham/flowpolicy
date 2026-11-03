"""The pipeline definition and its reproducibility comparison.

The pipeline is data (a list of `Stage`s), which is exactly the kind of thing that rots silently:
a command keeps its shape while the script it names is renamed, or a `--out` path quietly starts
pointing at the committed evidence. These tests check the three properties that would matter if it
went wrong -- rows land in the run directory, every script a stage names exists, and the two
profiles stay in the documented relationship -- plus the pairing rule that turns two blocks into an
agreement count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.pipeline import REPO, build_stages, compare_stage, render

pytestmark = pytest.mark.unit


def _stages(profile: str, tmp_path: Path) -> list:
    return build_stages(tmp_path, profile)


def test_quick_is_a_subset_of_full_and_never_the_other_way(tmp_path: Path) -> None:
    quick = {stage.name for stage in _stages("quick", tmp_path)}
    full = {stage.name for stage in _stages("full", tmp_path)}
    assert quick <= full
    # The blocks that exist only at full size are the evidence-generating ones.
    assert {"e0-published", "e3", "latency", "deploy"} <= full - quick


def test_every_stage_names_a_script_that_exists(tmp_path: Path) -> None:
    for profile in ("quick", "full"):
        for stage in _stages(profile, tmp_path):
            for command in stage.commands:
                scripts = [arg for arg in command if arg.endswith(".py")]
                for arg in scripts:
                    if arg.startswith("scripts/"):
                        assert (REPO / arg).is_file(), f"{stage.name}: {arg} is missing"


def test_rows_go_to_the_run_directory_not_to_the_committed_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for stage in _stages("full", run):
        for command in stage.commands:
            for index, arg in enumerate(command):
                if arg == "--out":
                    target = command[index + 1]
                    assert target.startswith(str(run)), f"{stage.name} writes {target}"
                    assert "results/e0.jsonl" not in target


def test_only_the_tables_stage_touches_the_committed_artifacts(tmp_path: Path) -> None:
    # `git diff --exit-code` on results/tables and README is the tables stage's whole job; any
    # other stage naming those paths would mean a run can edit the evidence.
    for stage in _stages("full", tmp_path):
        text = " ".join(" ".join(command) for command in stage.commands)
        if stage.name != "tables":
            assert "results/tables" not in text
            assert "README" not in text


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _rows(seeds: list[int], successes: list[int], steps: int = 10) -> list[dict]:
    return [
        {"eval_seed": seed, "sampler_steps": steps, "success": success, "coverage": 0.5}
        for seed, success in zip(seeds, successes, strict=True)
    ]


def test_identical_blocks_agree_completely(tmp_path: Path) -> None:
    rows = _rows([1, 2, 3, 4], [1, 0, 1, 0])
    result = compare_stage(
        _write(tmp_path / "e0.jsonl", rows), _write(tmp_path / "old.jsonl", rows)
    )
    assert result["n"] == 4
    assert result["agree"] == 4
    assert result["fresh_only"] == result["committed_only"] == 0
    assert result["p"] == 1.0


def test_a_disagreement_is_reported_as_discordant_not_as_a_rate(tmp_path: Path) -> None:
    fresh = _rows([1, 2, 3], [1, 1, 0])
    old = _rows([1, 2, 3], [1, 0, 0])
    result = compare_stage(
        _write(tmp_path / "e0.jsonl", fresh), _write(tmp_path / "old.jsonl", old)
    )
    assert result["n"] == 3
    assert result["agree"] == 2
    assert result["fresh_only"] == 1
    assert result["committed_only"] == 0
    assert result["fresh_success"] > result["committed_success"]


def test_pairing_is_on_shared_seeds_only(tmp_path: Path) -> None:
    # The quick profile runs a subset of the committed block; that must not read as a rate gap.
    fresh = _rows([1, 2], [1, 1])
    old = _rows([1, 2, 3, 4], [1, 0, 0, 0])
    result = compare_stage(
        _write(tmp_path / "e0.jsonl", fresh), _write(tmp_path / "old.jsonl", old)
    )
    assert result["n"] == 2
    assert result["fresh_success"] == 1.0
    assert result["committed_success"] == 0.5  # over the two shared seeds, not all four
    assert result["agree"] == 1


def test_step_counts_are_not_paired_with_each_other(tmp_path: Path) -> None:
    # E3 holds six N values over the same seeds; pairing across N would compare two policies.
    fresh = _rows([1, 2], [1, 1], steps=4)
    old = _rows([1, 2], [0, 0], steps=10)
    result = compare_stage(
        _write(tmp_path / "e3.jsonl", fresh), _write(tmp_path / "old.jsonl", old)
    )
    assert result["n"] == 0
    assert result["status"] == "no shared seeds"


def test_an_absent_block_is_not_compared_rather_than_reported_as_zero(tmp_path: Path) -> None:
    result = compare_stage(tmp_path / "missing.jsonl", tmp_path / "also.jsonl")
    assert result["n"] == 0
    assert result["status"] == "not compared"


def test_the_rendered_table_states_n_and_discordance(tmp_path: Path) -> None:
    rows = _rows([1, 2, 3], [1, 0, 1])
    result = compare_stage(
        _write(tmp_path / "e0.jsonl", rows), _write(tmp_path / "old.jsonl", rows)
    )
    text = render([result])
    assert "stage" in text.splitlines()[0]
    assert "3" in text.splitlines()[1]
