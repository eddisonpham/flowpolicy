"""Normalization math of the pretrained adapter (docs/04, models/pretrained.py).

These three transforms are the only thing standing between the checkpoint and a
success rate near zero, and they are the part that `lerobot` 0.6.1 cannot do for us
(it drops the checkpoint's normalization buffers). They are tested against the
definitions the checkpoint's `normalization_mapping` names: MEAN_STD for images,
MIN_MAX for state and actions. The model itself is not loaded here, so this runs in CI.
"""

from __future__ import annotations

import pytest
import torch

from flowpolicy.models.pretrained import (
    denormalize_action,
    normalize_image,
    normalize_state,
    resolve_dtype,
)

pytestmark = pytest.mark.unit


def test_image_mean_std_matches_imagenet_constants() -> None:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = torch.full((1, 2, 3, 4, 5), 0.5)
    out = normalize_image(image, mean, std)
    expected = ((0.5 - mean) / std).flatten()
    assert torch.allclose(out[0, 0, :, 0, 0], expected, atol=1e-6)


def test_state_min_max_round_trips_through_action_min_max() -> None:
    low = torch.tensor([12.0, 25.0])
    high = torch.tensor([511.0, 511.0])
    raw = torch.tensor([[[100.0, 300.0], [500.0, 12.0]]])
    normalized = normalize_state(raw, low, high)
    # MIN_MAX maps the dataset extremes onto [-1, 1] exactly.
    assert torch.allclose(normalize_state(low, low, high), torch.full((2,), -1.0))
    assert torch.allclose(normalize_state(high, low, high), torch.full((2,), 1.0))
    assert torch.allclose(denormalize_action(normalized, low, high), raw, atol=1e-4)


def test_denormalize_action_inverts_normalize_state() -> None:
    low = torch.tensor([0.0, 5.0])
    high = torch.tensor([10.0, 15.0])
    values = torch.linspace(0.0, 1.0, 11).view(1, 11, 1).expand(1, 11, 2).contiguous()
    back = denormalize_action(normalize_state(values, low, high), low, high)
    assert torch.allclose(back, values, atol=1e-5)


def test_resolve_dtype_accepts_both_spellings() -> None:
    assert resolve_dtype("float32") == resolve_dtype("fp32") == torch.float32
    assert resolve_dtype("bfloat16") == resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("float16") == resolve_dtype("fp16") == torch.float16


def test_resolve_dtype_rejects_unknown_names_with_the_options() -> None:
    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype("float8")


def test_normalize_state_is_not_in_place() -> None:
    low = torch.zeros(2)
    high = torch.full((2,), 512.0)
    state = torch.full((1, 2, 2), 256.0)
    before = state.clone()
    normalize_state(state, low, high)
    assert pytest.approx(0.0) == (state - before).abs().max().item()
