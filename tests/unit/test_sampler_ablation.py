"""What the sampler-sensitivity block must and must not say.

The claim it supports is narrow and easy to over-read: a numerically equivalent sampler changes
which deploy episodes succeed, because PushT amplifies 1e-5 of action rounding. What the code has
to get right is the pairing (same episode, same injected latency), the direction of the gap, and
quoting the resolution of the block the number came from -- otherwise "23/45 changed hands" is read
as a policy verdict instead of a measurement of sensitivity.
"""

from __future__ import annotations

import pytest

from policybench.eval.sampler_ablation import pairs, render, summarize

pytestmark = pytest.mark.unit


def _rows(
    successes: list[int],
    *,
    latency: float = 0.0,
    seed_start: int = 1,
    steps: list[int] | None = None,
) -> list[dict]:
    return [
        {
            "eval_seed": seed_start + index,
            "episode": 0,
            "latency_ms": latency,
            "success": success,
            "steps": None if steps is None else steps[index],
            "coverage": 0.9,
        }
        for index, success in enumerate(successes)
    ]


def test_an_episode_missing_from_one_block_is_dropped() -> None:
    fast = [*_rows([1, 1, 1]), {"eval_seed": 99, "episode": 0, "latency_ms": 0.0, "success": 1}]
    stock = _rows([1, 0, 0])
    assert len(pairs(fast, stock)) == 3


def test_a_row_without_a_latency_is_not_an_episode() -> None:
    # The latency is a condition, not a label: without it there is no episode to pair.
    fast = [{"eval_seed": 1, "episode": 0, "success": 1}]
    stock = [{"eval_seed": 1, "episode": 0, "success": 0}]
    assert pairs(fast, stock) == []


def test_identical_blocks_report_no_discordance_and_no_resolution() -> None:
    summary = summarize(pairs(_rows([1, 0, 1]), _rows([1, 0, 1])), n_boot=50)
    overall = summary["overall"]
    assert overall["discordance"] == 0.0
    assert overall["resolves_pp"] is None  # nothing to resolve when nothing changed hands
    assert overall["mcnemar_p"] == 1.0
    assert overall["agree"] == 3


def test_a_one_sided_disagreement_carries_its_direction() -> None:
    # Fast fails every episode the stock loop succeeds, so the gap is positive *towards* stock:
    # the sign is (stock - fast), and reading it backwards would invert every sentence built on it.
    summary = summarize(pairs(_rows([0] * 6), _rows([1] * 6)), n_boot=50)
    overall = summary["overall"]
    assert overall["stock_only"] == 6
    assert overall["fast_only"] == 0
    assert overall["delta_pp"] == 100.0
    assert overall["mcnemar_p"] < 0.05


def test_the_interval_brackets_the_point_estimate() -> None:
    summary = summarize(pairs(_rows([1, 1, 0, 1, 0, 1]), _rows([0, 1, 0, 0, 0, 1])), n_boot=400)
    overall = summary["overall"]
    low, high = overall["ci_pp"]
    assert low <= overall["delta_pp"] <= high


def test_latency_levels_stay_separate_cells() -> None:
    fast = _rows([1], latency=0.0) + _rows([0], latency=200.0, seed_start=1)
    stock = _rows([0], latency=0.0) + _rows([0], latency=200.0, seed_start=1)
    summary = summarize(pairs(fast, stock), n_boot=50)
    assert [cell["latency_ms"] for cell in summary["cells"]] == [0.0, 200.0]
    # 0 ms changed hands and 200 ms agreed, so the pooled discordance is half of the 0 ms one and
    # the cells stay separate rather than one of them standing in for the other.
    assert summary["cells"][0]["fast_only"] == 1
    assert summary["cells"][1]["agree"] == 1
    assert summary["overall"]["discordance"] == 0.5


def test_discordant_episodes_report_whether_they_ended_elsewhere() -> None:
    fast = _rows([1, 1], steps=[300, 233])
    stock = _rows([0, 1], steps=[233, 233])
    summary = summarize(pairs(fast, stock), n_boot=50)
    assert summary["discordant_with_step_data"] == 1
    assert summary["discordant_ending_elsewhere"] == 1
    assert summary["mean_abs_step_change"] == 67.0


def test_the_rendered_table_states_psi_and_the_resolution() -> None:
    text = render(summarize(pairs(_rows([1] * 6), _rows([0] * 6)), n_boot=50))
    assert "Resolves" in text.splitlines()[0]
    # ASCII in the rendered table: the CLI prints to a cp1252 console, where a glyph costs a
    # UnicodeEncodeError rather than a nicer column name.
    assert "psi" in text
    assert "stock loop" in text


def test_a_gap_below_the_bar_is_a_null_not_an_equality() -> None:
    # 105 episodes at psi = 0.38 resolve ~16.7 pp; a 7.6 pp gap on that block is a null at that
    # resolution. The first version of this sentence assumed the gap would always clear the bar,
    # which was true of the 45-episode pilot and false of the 105-episode extension.
    fast = _rows([1] * 16 + [0] * 24 + [1] * 40 + [0] * 25)
    stock = _rows([0] * 16 + [1] * 24 + [1] * 40 + [0] * 25)
    text = render(summarize(pairs(fast, stock), n_boot=50))
    assert "null at that resolution" in text


def test_a_gap_above_the_bar_is_read_as_a_difference() -> None:
    text = render(summarize(pairs(_rows([0] * 8), _rows([1] * 8)), n_boot=50))
    assert "read as a difference" in text


def test_the_e10_table_states_its_resolution() -> None:
    # The table prints one rate per cell, so a reader has no way to tell a null from a block that
    # could not see the effect. The generated note supplies the bar.
    from scripts.make_tables import deploy_table

    rows = []
    for strategy, successes in (("naive", [1, 1, 1, 0, 1, 1]), ("naive-r2", [0, 0, 1, 0, 1, 1])):
        for index, success in enumerate(successes):
            rows.append(
                {
                    "latency_ms": 100.0,
                    "strategy": strategy,
                    "eval_seed": 10000 + index,
                    "episode": 0,
                    "train_seed": 0,
                    "success": success,
                    "coverage": 0.9,
                    "deadline_misses": 5,
                }
            )
    text = deploy_table(rows)
    assert "Resolution:" in text
    note = text.split("Resolution:", 1)[1]
    assert "naive-r2" in note and "80 % power" in note
