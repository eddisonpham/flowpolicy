"""The headline summary must compute, never invent.

Two failure modes matter here: a missing results file must render as "not measured yet"
rather than as a zero, and a row that is not part of the claim (the checkpoint's own DDPM
setup) must not leak into the number. Both are covered.
"""

from __future__ import annotations

from typing import Any

import pytest

from policybench.eval.report import headline_rows, headline_table, inventory

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


def answer_for(rows: dict[str, list[dict[str, Any]]], source: str) -> str:
    """The computed answer for one experiment, independent of row order."""
    matches = [answer for _, answer, name in headline_rows(rows) if name == source]
    assert len(matches) == 1, (source, matches)
    return matches[0]


def test_empty_results_are_reported_as_not_measured() -> None:
    rows = headline_rows({})
    assert len(rows) == 7
    assert {source for _, _, source in rows} == {
        "E0",
        "E3",
        "E9",
        "E8",
        "E10",
        "E12",
        "E14",
    }
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
    assert "50 episodes" in answer
    assert "0.900" in answer
    # The model card's 0.955 is `max_reward` (coverage / 0.95), so the answer may not print it
    # as though it were comparable to our coverage column.
    assert "0.955 column is `max_reward`" in answer


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
    # Without E14 rows there is nothing to say about compile, and it must not be invented.
    assert "torch.compile" not in answer


def test_e9_quotes_the_measured_compile_row_when_it_exists() -> None:
    rows = {
        "e9": [
            _row(mode="eager", steps=4, mean_ms=80.0),
            _row(mode="cudagraph", steps=4, mean_ms=70.0),
        ],
        "e14": [
            _row(kind="latency", arm="fast-compile-default", steps=4, mean_ms=64.0),
            _row(kind="latency", arm="fast-compile-reduce-overhead", steps=4, mean_ms=60.0),
        ],
    }
    _, answer, _ = headline_rows(rows)[2]
    assert "torch.compile" in answer
    assert "60.0 ms at N=4" in answer  # the best compile mode, not the first one
    assert "1.33x" in answer  # 80 ms stock eager / 60 ms compiled


def test_e14_reports_the_cost_ratio_and_carries_quality() -> None:
    rows = {
        "e14": [
            _row(
                kind="throughput",
                arm="published",
                steps=100,
                s_per_100_steps=55.7,
                success_rate=0.5,
                coverage_mean=0.94,
                episodes=4,
            ),
            _row(
                kind="throughput",
                arm="fast-ddim4",
                steps=4,
                s_per_100_steps=0.81,
                success_rate=0.75,
                coverage_mean=0.96,
                episodes=4,
            ),
        ]
    }
    _, answer, source = headline_rows(rows)[-1]
    assert source == "E14"
    assert "55.7 s" in answer and "0.81 s" in answer
    assert "69x cheaper" in answer
    assert "75%" in answer and "0.960" in answer


def test_e14_flags_parallel_workers_that_do_not_help() -> None:
    rows = {
        "e14": [
            _row(
                kind="throughput",
                arm="published",
                steps=100,
                s_per_100_steps=50.0,
                success_rate=0.5,
                coverage_mean=0.94,
                episodes=4,
            ),
            _row(
                kind="throughput",
                arm="fast-ddim4",
                steps=4,
                s_per_100_steps=1.0,
                success_rate=0.5,
                coverage_mean=0.94,
                episodes=4,
            ),
            _row(
                kind="throughput",
                arm="fast-ddim4-2w",
                steps=4,
                s_per_100_steps=1.8,
                success_rate=0.5,
                coverage_mean=0.94,
                episodes=4,
                workers=2,
            ),
        ]
    }
    _, answer, _ = headline_rows(rows)[-1]
    assert "two rollout workers cost 1.8x per control step" in answer


def test_e10_reports_success_delta_and_deadline_misses() -> None:
    rows = {
        "e10": [
            *[_row(latency_ms=0.0, success=0.6) for _ in range(30)],
            *[_row(latency_ms=200.0, success=0.4) for _ in range(30)],
        ]
    }
    answer = answer_for(rows, "E10")
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
    answer = answer_for(rows, "E12")
    assert "0% at 100 ms" in answer  # naive replacement, replanning every 2 steps
    assert "60%" in answer  # the 8-step naive baseline at the same injected latency
    assert "replanning every 8 steps" in answer
    assert "recovers 50%" in answer  # temporal ensembling over the same chunks
    assert "4x the chunk count" in answer


def test_e8_reports_the_speedup_and_that_it_does_not_fit_the_period() -> None:
    rows = {
        "e8": [
            _row(kind="latency", dtype="float32", mode="cudagraph", steps=4, mean_ms=80.0),
            _row(kind="latency", dtype="bfloat16", mode="cudagraph", steps=4, mean_ms=64.0),
            _row(kind="latency", dtype="float32", mode="cudagraph", steps=10, mean_ms=200.0),
            _row(kind="latency", dtype="bfloat16", mode="cudagraph", steps=10, mean_ms=140.0),
            _row(
                kind="episode",
                dtype="bfloat16",
                steps=4,
                episodes=40,
                success_rate=0.5,
                success_lo=0.35,
                success_hi=0.65,
                coverage_mean=0.88,
            ),
            _row(
                kind="episode",
                dtype="float32",
                steps=4,
                episodes=40,
                success_rate=0.35,
                success_lo=0.22,
                success_hi=0.5,
                coverage_mean=0.9,
            ),
        ]
    }
    answer = answer_for(rows, "E8")
    assert "64.0 ms at N=4 from 80.0 ms" in answer
    assert "140.0 ms at N=10 from 200.0 ms" in answer
    assert "1.25x" in answer and "1.43x" in answer
    assert "stays N <= 4" in answer
    assert "50% [35%, 65%]" in answer


def test_inventory_names_every_file_it_looked_for() -> None:
    lines = inventory({})
    assert len(lines) == 12
    assert any("results/ablation/e1.jsonl" in line.replace("\\", "/") for line in lines)
    assert all("0 rows" in line for line in lines)
