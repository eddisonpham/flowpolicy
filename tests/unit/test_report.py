"""The headline summary must compute, never invent.

Two failure modes matter here: a missing results file must render as "not measured yet"
rather than as a zero, and a row that is not part of the claim (the checkpoint's own DDPM
setup) must not leak into the number. Both are covered.
"""

from __future__ import annotations

from typing import Any

import pytest

from flowpolicy.eval.report import headline_rows, headline_table, inventory

pytestmark = pytest.mark.unit


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scheduler": "ddim",
        "sampler_steps": 10,
        "train_seed": -1,
        "success": 1.0,
        "coverage": 0.9,
        "mean_ms": 100.0,
        "deadline_misses": 3.0,
        "strategy": "naive",
        "latency_ms": 0.0,
        "replan_every": 0,
    }
    row.update(overrides)
    return row


def test_empty_results_are_reported_as_not_measured() -> None:
    rows = headline_rows({})
    assert len(rows) == 5
    assert {source for _, _, source in rows} == {"E0", "E3", "E9", "E10", "E12"}
    assert all("not measured yet" in answer for _, answer, _ in rows)
    assert "not measured" in headline_table({})


def test_e0_answer_uses_only_the_ddim_row() -> None:
    rows = {
        "e0": [_row() for _ in range(50)]
        + [_row(scheduler="ddpm", sampler_steps=100, success=0.0, coverage=0.1) for _ in range(50)]
    }
    _, answer, source = headline_rows(rows)[0]
    assert source == "E0"
    assert "100.0%" in answer
    assert "50 paired episodes" in answer
    assert "0.900" in answer


def test_e3_lists_every_step_count() -> None:
    rows = {
        "e3": [
            _row(sampler_steps=1, success=0.0, coverage=0.12),
            _row(sampler_steps=4, success=0.4, coverage=0.88),
        ]
    }
    _, answer, _ = headline_rows(rows)[1]
    assert "N=1" in answer and "N=4" in answer
    assert "0.120" in answer and "0.880" in answer


def test_e9_reports_eager_against_graph_and_the_missing_compile() -> None:
    rows = {
        "e9": [
            _row(mode="eager", steps=10, mean_ms=200.0),
            _row(mode="cudagraph", steps=10, mean_ms=100.0),
        ]
    }
    _, answer, _ = headline_rows(rows)[2]
    assert "200.0 ms eager / 100.0 ms graph" in answer
    assert "2.00x" in answer
    assert "torch.compile" in answer and "NOT RUN" in answer


def test_e10_reports_success_delta_and_deadline_misses() -> None:
    rows = {
        "e10": [
            *[_row(latency_ms=0.0, success=0.6) for _ in range(30)],
            *[_row(latency_ms=200.0, success=0.4) for _ in range(30)],
        ]
    }
    _, answer, _ = headline_rows(rows)[3]
    assert "60% at 0 ms injected vs 40% at 200 ms injected" in answer
    assert "3 deadline misses" in answer


def test_e12_separates_naive_from_ensembling_at_the_same_replan_rate() -> None:
    rows = {
        "e10": [
            *[_row(latency_ms=100.0, strategy="naive", success=0.6) for _ in range(30)],
            *[
                _row(latency_ms=100.0, strategy="naive-r2", success=0.0, replan_every=2)
                for _ in range(30)
            ],
            *[
                _row(
                    latency_ms=100.0,
                    strategy="temporal-ensemble-r2",
                    success=0.5,
                    replan_every=2,
                )
                for _ in range(30)
            ],
        ]
    }
    _, answer, source = headline_rows(rows)[4]
    assert source == "E12"
    assert "0% at 100 ms" in answer  # naive replacement, replanning every 2 steps
    assert "60%" in answer  # the 8-step naive baseline at the same injected latency
    assert "replanning every 8 steps" in answer
    assert "recovers 50%" in answer  # temporal ensembling over the same chunks
    assert "4x the chunk count" in answer


def test_inventory_names_every_file_it_looked_for() -> None:
    lines = inventory({})
    assert len(lines) == 8
    assert any("results/ablation/e1.jsonl" in line.replace("\\", "/") for line in lines)
    assert all("0 rows" in line for line in lines)
