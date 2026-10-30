"""The normalization seam: one declared mapping, two implementations (docs/10).

`lerobot` is not installed in CI (it caps torch < 2.12), so the library-facing part of this
module is never imported here: `ProcessorsNorm` takes injected callables and the tensor
plumbing is what these tests pin. The one thing that cannot be caught by shape is a transposed
history reshape, because `[B, T]` and `[T, B]` hold the same number of elements -- that is what
`test_restore_history_keeps_batch_then_time` asserts.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from policybench.models.normalization import (
    BuffersNorm,
    ProcessorsNorm,
    config_id_marks_path,
    denormalize_action,
    flatten_history,
    normalize_image,
    normalize_state,
    override_processor_device,
    restore_history,
)

pytestmark = pytest.mark.unit


def _buffers() -> BuffersNorm:
    return BuffersNorm(
        image_mean=torch.zeros(3, 1, 1),
        image_std=torch.ones(3, 1, 1),
        state_low=torch.tensor([10.0, 20.0]),
        state_high=torch.tensor([510.0, 510.0]),
        action_low=torch.tensor([10.0, 20.0]),
        action_high=torch.tensor([510.0, 510.0]),
    )


def test_buffers_norm_is_the_module_math() -> None:
    norm = _buffers()
    image = torch.full((1, 2, 3, 4, 4), 0.5)
    state = torch.full((1, 2, 2), 260.0)
    chunk = torch.zeros(1, 16, 2)
    assert torch.equal(norm.image(image), normalize_image(image, norm.image_mean, norm.image_std))
    assert torch.equal(norm.state(state), normalize_state(state, norm.state_low, norm.state_high))
    assert torch.equal(
        norm.actions(chunk), denormalize_action(chunk, norm.action_low, norm.action_high)
    )


def test_flatten_and_restore_round_trip() -> None:
    tensor = torch.arange(2 * 3 * 2).float().view(2, 3, 2)
    assert torch.equal(restore_history(flatten_history(tensor), 2, 3), tensor)


def test_restore_history_keeps_batch_then_time() -> None:
    # Element (b, t) must be flat[b * horizon + t]; a transposed reshape would pass a shape check.
    horizon, batch = 3, 2
    flat = torch.arange(batch * horizon).float().view(batch * horizon, 1)
    restored = restore_history(flat, batch, horizon)
    for b in range(batch):
        for t in range(horizon):
            assert restored[b, t, 0].item() == b * horizon + t


def _identity_pre(transition: dict[str, Any]) -> dict[str, Any]:
    return transition


def test_processors_norm_calls_the_pipeline_once_per_sample() -> None:
    calls: list[tuple[int, ...]] = []

    def pre(transition: dict[str, Any]) -> dict[str, Any]:
        (value,) = transition.values()
        calls.append(tuple(value.shape))
        return {"observation.image": value + 1.0}

    def post(transition: dict[str, Any]) -> dict[str, Any]:
        return {"action": transition["action"] * 2.0}

    norm = ProcessorsNorm(pre, post)
    image = torch.zeros(2, 3, 3, 4, 4)
    out = norm.image(image)
    assert out.shape == image.shape
    # Batched on the outside, one unbatched frame per call: 2 observations x 3 history frames,
    # each a single [3, 4, 4] frame -- not a batch of 2 that the pipeline would batch again.
    assert calls == [(3, 4, 4)] * 6
    assert torch.allclose(out, torch.ones_like(out))

    chunk = torch.ones(2, 5, 2)
    assert torch.allclose(norm.actions(chunk), torch.full_like(chunk, 2.0))


def test_processors_norm_preserves_history_order() -> None:
    counter = {"n": 0}

    def pre(transition: dict[str, Any]) -> dict[str, Any]:
        (value,) = transition.values()
        counter["n"] += 1
        return {"observation.state": value + float(counter["n"])}

    norm = ProcessorsNorm(pre, _identity_pre)
    state = torch.zeros(1, 2, 2)
    out = norm.state(state)
    # Frame 0 is the first call, frame 1 the second: [1, 2] not [2, 1].
    assert out[0, 0, 0].item() == 1.0
    assert out[0, 1, 0].item() == 2.0


def test_processors_norm_squeezes_the_pipelines_batch_dim() -> None:
    # `to_batch_processor` is part of every real pipeline: one frame in, a leading 1 out.
    def pre(transition: dict[str, Any]) -> dict[str, Any]:
        (value,) = transition.values()
        return {"observation.state": value.unsqueeze(0)}

    norm = ProcessorsNorm(pre, _identity_pre)
    state = torch.zeros(2, 3, 2)
    assert norm.state(state).shape == state.shape


def test_processors_norm_names_the_key_it_wanted() -> None:
    norm = ProcessorsNorm(lambda transition: {}, _identity_pre)
    with pytest.raises(KeyError, match=r"observation\.image"):
        norm.image(torch.zeros(1, 1, 3, 4, 4))


def test_override_processor_device_moves_only_the_device_step() -> None:
    config = {
        "name": "policy_postprocessor",
        "steps": [
            {"registry_name": "unnormalizer_processor", "config": {"eps": 1e-8}},
            {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}},
        ],
    }
    out = override_processor_device(config, "cuda")
    assert out["steps"][1]["config"]["device"] == "cuda"
    assert out["steps"][0]["config"] == {"eps": 1e-8}
    # The input is the checkpoint's own JSON, and it is not touched (docs/10 condition 4).
    assert config["steps"][1]["config"]["device"] == "cpu"
    assert out["name"] == "policy_postprocessor"


def test_non_default_path_must_be_visible_in_config_id() -> None:
    assert config_id_marks_path("dp_published_n200", "buffers")
    assert config_id_marks_path("dp_proc_n600", "processors")
    assert not config_id_marks_path("dp_published_n200", "processors")
