"""The leftover map's invariants, CPU-only and without `lerobot` (docs/11 section 4.4).

`scripts/smolvla_leftover_probe.py` runs the same identities against the real `RTCProcessor`, but
it needs `lerobot` and so cannot run in CI. These are the parts of that gate that must never
regress silently: the round trip, the zero pad, and the refusal of a non-invertible table.
"""

from __future__ import annotations

import pytest
import torch

from policybench.models.leftover import PaddedActionSpace

pytestmark = pytest.mark.unit

ACTION_DIM = 7
MODEL_DIM = 32
BATCH, HORIZON = 2, 6
BAR = 1e-6


def _space() -> PaddedActionSpace:
    """Unequal spreads: an isotropic table would let a permuted map pass."""
    return PaddedActionSpace(
        center=torch.linspace(-0.4, 0.4, ACTION_DIM),
        spread=torch.linspace(0.3, 1.0, ACTION_DIM),
        model_dim=MODEL_DIM,
    )


def _raw(space: PaddedActionSpace) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    z = torch.rand(BATCH, HORIZON, ACTION_DIM, generator=generator) * 2.0 - 1.0
    return space.center + space.spread * z


def test_raw_survives_the_round_trip() -> None:
    space = _space()
    raw = _raw(space)
    assert torch.allclose(space.to_raw(space.to_model(raw)), raw, atol=BAR)


def test_model_survives_the_round_trip_on_valid_dims() -> None:
    space = _space()
    generator = torch.Generator().manual_seed(5)
    model = torch.randn(BATCH, HORIZON, MODEL_DIM, generator=generator)
    model[..., ACTION_DIM:] = 0.0
    assert torch.allclose(space.to_model(space.to_raw(model)), model, atol=BAR)


def test_the_pad_is_written_as_exact_zeros() -> None:
    space = _space()
    model = space.to_model(_raw(space))
    assert model.shape == (BATCH, HORIZON, MODEL_DIM)
    assert torch.count_nonzero(model[..., ACTION_DIM:]) == 0


def test_garbage_in_the_pad_never_reaches_raw_units() -> None:
    """The denoiser emits whatever it likes in dims 7-31; none of it is an action."""
    space = _space()
    model = space.to_model(_raw(space))
    polluted = model.clone()
    polluted[..., ACTION_DIM:] = 1e3
    assert torch.equal(space.to_raw(polluted), space.to_raw(model))


def test_min_max_is_the_same_affine() -> None:
    """One form, both declared normalizations: min_max is mean_std about (low+high)/2."""
    low = torch.tensor([-1.0, 0.0, -2.0])
    high = torch.tensor([1.0, 4.0, 2.0])
    space = PaddedActionSpace(center=(low + high) / 2, spread=(high - low) / 2, model_dim=8)
    raw = torch.stack([low, high, (low + high) / 2])
    model = space.to_model(raw)
    assert torch.allclose(
        model[..., :3], torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    )
    assert torch.allclose(space.to_raw(model), raw, atol=BAR)


def test_a_chunk_of_the_wrong_width_is_refused_in_both_spaces() -> None:
    space = _space()
    with pytest.raises(ValueError, match="raw-space dims"):
        space.to_model(torch.zeros(BATCH, HORIZON, MODEL_DIM))
    with pytest.raises(ValueError, match="model-space dims"):
        space.to_raw(torch.zeros(BATCH, HORIZON, ACTION_DIM))


def test_a_flat_chunk_without_a_time_axis_is_refused() -> None:
    space = _space()
    with pytest.raises(ValueError, match=r"\[T, 7\]"):
        space.to_model(torch.zeros(ACTION_DIM))


def test_a_non_invertible_table_is_refused_rather_than_clamped() -> None:
    """A constant training dimension makes the map non-invertible: it must not round-trip."""
    with pytest.raises(ValueError, match="non-invertible"):
        PaddedActionSpace(center=torch.zeros(3), spread=torch.tensor([1.0, 0.0, 1.0]), model_dim=8)


def test_mismatched_statistic_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="one pair per action dimension"):
        PaddedActionSpace(center=torch.zeros(3), spread=torch.ones(4), model_dim=8)


def test_a_model_narrower_than_the_action_is_refused() -> None:
    with pytest.raises(ValueError, match="narrower than"):
        PaddedActionSpace(center=torch.zeros(7), spread=torch.ones(7), model_dim=2)
