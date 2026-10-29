"""The VQ-BeT adapter's pure glue: config filtering, no checkpoint and no GPU.

`models/pretrained_vqbet.py` needs the network to load weights, so the part that can be
checked in CI is the translation between a 2024 checkpoint config and the installed
library's dataclass. That translation is where a silent failure would live: a dropped
required field produces an error, but keeping a stale field produces a `draccus` parse
failure that looks like a missing dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from policybench.models.pretrained_vqbet import filter_config

pytestmark = pytest.mark.unit


@dataclass
class _Config:
    n_obs_steps: int = 5
    action_chunk_size: int = 5
    input_features: dict[str, Any] = field(default_factory=dict)


def test_unknown_field_is_dropped() -> None:
    """`mlp_hidden_dim` is the field that actually broke the load on 2026-10-28."""
    raw = {
        "type": "vqbet",
        "n_obs_steps": 5,
        "action_chunk_size": 5,
        "mlp_hidden_dim": 256,
    }
    filtered = filter_config(raw, _Config)
    assert filtered == {"type": "vqbet", "n_obs_steps": 5, "action_chunk_size": 5}


def test_type_survives_even_though_it_is_not_a_field() -> None:
    """`from_pretrained` refuses a config without `type`; filtering must not remove it."""
    assert "type" in filter_config({"type": "vqbet", "n_obs_steps": 5}, _Config)


def test_nested_feature_dicts_pass_through_untouched() -> None:
    raw = {
        "type": "vqbet",
        "input_features": {"observation.image": {"type": "VISUAL", "shape": [3, 96, 96]}},
    }
    filtered = filter_config(raw, _Config)
    assert filtered["input_features"] == raw["input_features"]


def test_required_field_is_kept_so_a_missing_one_still_fails() -> None:
    """The filter is a subset operation, not a defaults factory: absence stays absent."""
    filtered = filter_config({"type": "vqbet", "n_obs_steps": 5}, _Config)
    assert "action_chunk_size" not in filtered
