"""The tables are computed from results/*.jsonl, so the statistics they print need tests.

The paired tests use McNemar (same episodes, both runs). The replication block sees different
episodes, so it uses Fisher exact; getting that test wrong would turn "replicates" into a claim
the data does not support.
"""

from __future__ import annotations

from typing import Any

import pytest
from scripts.make_tables import _fisher_exact, coverage_bin_table, replication_note

pytestmark = pytest.mark.unit


def _row(seed: int, success: bool, coverage: float = 0.85) -> dict[str, Any]:
    return {"eval_seed": seed, "success": int(success), "coverage": coverage}


def test_fisher_matches_a_hand_computed_table() -> None:
    # [[9, 1], [1, 9]]: two-sided p = 2 * (C(10,0)C(10,10) + C(10,1)C(10,9)) / C(20,10).
    assert _fisher_exact(9, 1, 1, 9) == pytest.approx(2 * (1 + 100) / 184756, rel=1e-3)


def test_fisher_is_symmetric_and_one_for_identical_rates() -> None:
    assert _fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert _fisher_exact(3, 7, 8, 2) == pytest.approx(_fisher_exact(8, 2, 3, 7))


def test_replication_note_reports_a_repeat_without_overclaiming() -> None:
    primary = [_row(100000 + i, i < 108) for i in range(200)]
    second = [_row(100200 + i, i < 104) for i in range(200)]
    note = replication_note(primary, second)
    assert "108/200 (54.0%)" in note
    assert "104/200 (52.0%)" in note
    assert "seeds 100000-100199" in note and "seeds 100200-100399" in note
    assert "no episode is in both blocks" in note
    assert "the rate repeats within sampling error" in note


def test_replication_note_refuses_to_call_a_collapse_a_repeat() -> None:
    primary = [_row(100000 + i, i < 108) for i in range(200)]
    second = [_row(100200 + i, i < 20) for i in range(200)]
    note = replication_note(primary, second)
    assert "does not reproduce the first" in note
    assert "repeats within sampling error" not in note


def test_replication_note_is_empty_without_a_second_block() -> None:
    assert replication_note([_row(1, True)], []) == ""


def test_coverage_bins_are_counted_per_bin_and_paired_by_seed() -> None:
    ours = [_row(1, False, 0.85), _row(2, False, 0.93), _row(3, True, 0.96)]
    published = [
        {"eval_seed": 1, "success": 1, "coverage": 0.96},
        {"eval_seed": 2, "success": 0, "coverage": 0.91},
        {"eval_seed": 3, "success": 1, "coverage": 0.97},
        # Not in our block, so it must not be counted anywhere.
        {"eval_seed": 99, "success": 1, "coverage": 0.99},
    ]
    table = coverage_bin_table(ours, published)
    # Their coverage is censored at 0.95 for successes, so only their failures carry a mean.
    assert "| 0.80-0.90 | 1 | 0 | 1 | 0.850 | - |" in table
    assert "| 0.90-0.95 | 1 | 0 | 0 | 0.930 | 0.910 |" in table
    assert "| 0.95-1.00 | 1 | 1 | 1 | 0.960 | - |" in table
    assert "| 0.00-0.50 |" not in table  # empty bins are dropped, not printed as zero
    assert "| 0.50-0.80 |" not in table


def test_coverage_bin_table_is_empty_without_a_published_block() -> None:
    assert coverage_bin_table([_row(1, True, 0.99)], []) == ""
