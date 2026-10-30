"""The adapter-free control's row conversion (docs/09 option B).

The point of the conversion is that the control becomes *paired* with this repo's rows, so the
tests pin the two things that could silently break the pairing: the coverage inversion (gym-pusht
reports coverage/0.95, clipped) and the seed join.
"""

from __future__ import annotations

import pytest

from policybench.eval.lerobot_rows import control_row, load_success, paired_counts

pytestmark = pytest.mark.unit


def test_success_row_is_censored_and_reports_lower_bound() -> None:
    row = control_row(
        {"seed": 1000, "success": True, "sum_reward": 12.0, "max_reward": 1.0},
        config_id="c",
        checkpoint="ckpt",
    )
    # max_reward 1.0 is the clipped reward for any coverage >= 0.95.
    assert row["coverage"] == pytest.approx(0.95)
    assert row["coverage_censored"] is True
    assert row["success"] == 1.0


def test_failure_row_is_exact_coverage() -> None:
    row = control_row(
        {"seed": 1001, "success": False, "sum_reward": 3.0, "max_reward": 0.9},
        config_id="c",
        checkpoint="ckpt",
    )
    assert row["coverage"] == pytest.approx(0.855)
    assert row["coverage_censored"] is False
    assert row["success"] == 0.0
    assert row["eval_seed"] == 1001


def test_paired_counts_joins_only_shared_seeds() -> None:
    ours = {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0}
    theirs = {1: 1.0, 2: 1.0, 3: 0.0, 9: 1.0}
    both, ours_only, theirs_only, neither = paired_counts(ours, theirs)
    # Seed 4 (ours only) and seed 9 (theirs only) are absent from one run: dropped, not counted.
    assert (both, ours_only, theirs_only, neither) == (1, 1, 1, 0)


def test_paired_counts_is_empty_without_overlap() -> None:
    assert paired_counts({1: 1.0}, {2: 1.0}) == (0, 0, 0, 0)


def test_load_success_reads_the_row_schema(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"eval_seed": 7, "success": 1.0, "coverage": 0.96}\n'
        '{"eval_seed": 8, "success": 0.0, "coverage": 0.5}\n',
        encoding="utf-8",
    )
    assert load_success(path) == {7: 1.0, 8: 0.0}
