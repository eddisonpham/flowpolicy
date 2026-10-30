"""The migration's one transformation: which weights survive it.

`scripts/lerobot_migrate.py` repairs a 2024 checkpoint that keeps its normalization inside
`model.safetensors`. Everything it does is copying files plus this key filter, so the filter can
be tested without lerobot, a GPU or the Hub -- and it is worth testing, because dropping a real
weight (or keeping a buffer the processors now own) would silently change the policy the
adapter-free control measures.
"""

from __future__ import annotations

from typing import Any

import pytest
from scripts.lerobot_migrate import NORM_KEYS, strip_normalization

pytestmark = pytest.mark.unit


def _state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "model.unet.down_blocks.0.weight": "weight",
        "model.unet.up_blocks.0.weight": "weight",
        "normalize_inputs.buffer_observation_state_mean": "buffer",
        "normalize_inputs.buffer_observation_state_std": "buffer",
        "normalize_targets.buffer_action_min": "buffer",
        "normalize_targets.buffer_action_max": "buffer",
        "unnormalize_outputs.buffer_action_min": "buffer",
        "unnormalize_outputs.buffer_action_max": "buffer",
    }
    return state


def test_normalization_buffers_are_removed_and_every_weight_is_kept() -> None:
    clean = strip_normalization(_state())
    assert set(clean) == {"model.unet.down_blocks.0.weight", "model.unet.up_blocks.0.weight"}


def test_the_filter_names_the_buffers_the_processors_take_over() -> None:
    # The migration's premise: these three prefixes are exactly what it moves out of the
    # weights. A new prefix here without a processor to hold it would lose statistics.
    assert set(NORM_KEYS) == {
        "normalize_inputs.buffer_",
        "normalize_targets.buffer_",
        "unnormalize_outputs.buffer_",
    }


def test_a_key_that_merely_contains_a_prefix_is_dropped() -> None:
    # The migration matches substrings, so this documents the blast radius rather than
    # assuming it: any future weight whose name holds one of those prefixes disappears.
    clean = strip_normalization({"backbone.normalize_inputs.buffer_x": "buffer", "other": "w"})
    assert clean == {"other": "w"}


def test_an_already_clean_state_dict_passes_through_unchanged() -> None:
    state = {"model.a": "w", "model.b": "w"}
    assert strip_normalization(state) == state
