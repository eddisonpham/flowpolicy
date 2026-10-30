"""The repeat-aware interval for the like-for-like gap (docs/04 section 4).

`paired_delta_ci` exists because the adapter-free control is not deterministic between runs, so
an interval over episodes alone would be too narrow by the term the null depends on. These tests
pin the three properties that matters for: a zero gap is reported as zero, disagreement between
runs widens the interval instead of being averaged away, and seeds one run is missing are dropped
rather than counted as failures.
"""

from __future__ import annotations

import pytest

from policybench.eval.stats import paired_delta_ci

pytestmark = pytest.mark.unit


def test_identical_arms_give_a_zero_interval() -> None:
    ours = {seed: 0.0 for seed in range(50)}
    lo, hi = paired_delta_ci(ours, [ours, ours], n_boot=200, seed=1)
    assert (lo, hi) == (0.0, 0.0)


def test_runs_that_disagree_widen_the_interval() -> None:
    seeds = list(range(50))
    ours = {seed: 0.0 for seed in seeds}
    all_hit = {seed: 1.0 for seed in seeds}
    all_miss = {seed: 0.0 for seed in seeds}

    tight_lo, tight_hi = paired_delta_ci(ours, [all_hit], n_boot=400, seed=2)
    wide_lo, wide_hi = paired_delta_ci(ours, [all_hit, all_miss], n_boot=400, seed=2)

    # One run that always succeeds: the gap is exactly 1.0 whatever episodes are drawn.
    assert (tight_lo, tight_hi) == (1.0, 1.0)
    # Two runs that disagree completely: the interval must span both, not average them away.
    assert wide_lo < 0.5 < wide_hi
    assert wide_hi - wide_lo > tight_hi - tight_lo


def test_seeds_missing_from_any_run_are_dropped() -> None:
    ours = {1: 0.0, 2: 0.0, 3: 0.0}
    run_a = {1: 1.0, 2: 1.0}  # no seed 3
    run_b = {1: 1.0, 2: 1.0, 3: 1.0}
    lo, hi = paired_delta_ci(ours, [run_a, run_b], n_boot=200, seed=3)
    assert (lo, hi) == (1.0, 1.0)


def test_same_seed_reproduces_the_interval() -> None:
    seeds = list(range(40))
    ours = {seed: float(seed % 2) for seed in seeds}
    run_a = {seed: float((seed + 1) % 2) for seed in seeds}
    run_b = {seed: float(seed % 3 == 0) for seed in seeds}
    assert paired_delta_ci(ours, [run_a, run_b], n_boot=300, seed=7) == paired_delta_ci(
        ours, [run_a, run_b], n_boot=300, seed=7
    )


def test_no_runs_is_nan_not_a_number() -> None:
    lo, hi = paired_delta_ci({1: 1.0}, [], n_boot=10, seed=0)
    assert lo != lo and hi != hi  # nan, so a missing measurement cannot print as 0.0
