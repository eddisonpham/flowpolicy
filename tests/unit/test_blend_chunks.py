"""The E12 third arm's blending math (deploy/control_loop.py).

Only the arithmetic is testable on CPU without a simulator; the loop itself is exercised by
`just eval-e10`. What matters here is the property the strategy claims: the first action
after a replan still leans on the old chunk, the ramp reaches pure new within `blend_steps`,
and a short queue never raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowpolicy.deploy.control_loop import blend_chunks

pytestmark = pytest.mark.unit


def test_first_action_is_halfway_when_blending_over_two_steps() -> None:
    old = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    new = [np.array([10.0, 10.0]), np.array([10.0, 10.0])]
    out = blend_chunks(old, new, 2)
    assert np.allclose(out[0], [5.0, 5.0])
    assert np.allclose(out[1], [10.0, 10.0])  # weight reaches 1 at the last blend step


def test_blend_of_one_step_is_naive_replacement() -> None:
    old = [np.array([0.0, 0.0])]
    new = [np.array([7.0, -3.0])]
    out = blend_chunks(old, new, 1)
    assert np.allclose(out[0], [7.0, -3.0])


def test_identical_chunks_are_unchanged() -> None:
    chunk = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    out = blend_chunks(list(chunk), list(chunk), 3)
    assert np.allclose(np.stack(out), np.stack(chunk))


def test_actions_beyond_the_old_queue_are_untouched() -> None:
    old = [np.array([0.0, 0.0])]  # shorter than the new chunk
    new = [np.array([10.0, 10.0]), np.array([20.0, 20.0]), np.array([30.0, 30.0])]
    out = blend_chunks(old, new, 3)
    assert np.allclose(out[0], [10.0 / 3.0, 10.0 / 3.0])  # weight 1/3, only index 0 overlaps
    assert np.allclose(out[1], [20.0, 20.0])
    assert np.allclose(out[2], [30.0, 30.0])


def test_output_is_float32_and_a_copy() -> None:
    old = [np.array([0.0, 0.0], dtype=np.float64)]
    new = [np.array([2.0, 2.0], dtype=np.float64)]
    out = blend_chunks(old, new, 2)
    assert out[0].dtype == np.float32
    assert not np.allclose(out[0], new[0])  # new was not modified in place
