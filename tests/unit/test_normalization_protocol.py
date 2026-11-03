"""The normalization seam: one declared mapping, two implementations (docs/10).

`lerobot` is not installed in CI (it caps torch < 2.12), so the library-facing part of this
module is never imported here: `ProcessorsNorm` takes injected callables and the tensor
plumbing is what these tests pin. The one thing that cannot be caught by shape is a transposed
history reshape, because `[B, T]` and `[T, B]` hold the same number of elements -- that is what
`test_restore_history_keeps_batch_then_time` asserts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from policybench.models.normalization import (
    BuffersNorm,
    ProcessorsNorm,
    check_processor_files,
    config_id_marks_path,
    denormalize_action,
    flatten_history,
    load_processor_norm,
    named_sidecars,
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


# The migrated checkpoint's processor JSONs, field for field from
# `checkpoints/diffusion_pusht_migrated/` (that directory is gitignored, so the layout is kept
# here and read from no file). Both carry their statistics the same way: the step names a
# `.safetensors` sibling and the JSON holds only the mode and the feature widths.
_FEATURES: dict[str, Any] = {
    "observation.image": {"type": "VISUAL", "shape": [3, 96, 96]},
    "observation.state": {"type": "STATE", "shape": [2]},
    "action": {"type": "ACTION", "shape": [2]},
}
_NORM_MAP = {"ACTION": "MIN_MAX", "STATE": "MIN_MAX", "VISUAL": "MEAN_STD"}
_REAL_PREPROCESSOR: dict[str, Any] = {
    "name": "policy_preprocessor",
    "steps": [
        {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
        {"registry_name": "to_batch_processor", "config": {}},
        {"registry_name": "device_processor", "config": {"device": "cuda", "float_dtype": None}},
        {
            "registry_name": "normalizer_processor",
            "config": {"eps": 1e-08, "features": _FEATURES, "norm_map": _NORM_MAP},
            "state_file": "policy_preprocessor_step_3_normalizer_processor.safetensors",
        },
    ],
}
_REAL_POSTPROCESSOR: dict[str, Any] = {
    "name": "policy_postprocessor",
    "steps": [
        {
            "registry_name": "unnormalizer_processor",
            "config": {"eps": 1e-08, "features": _FEATURES, "norm_map": _NORM_MAP},
            "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        },
        {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}},
    ],
}


def test_the_real_processor_json_carries_no_statistics_of_its_own() -> None:
    # A reader that scans the JSON for mean/min finds nothing on either file and concludes "no
    # normalization" -- a different model, loaded without complaint. The numbers are in the
    # sidecar the step names, and the two sidecars sit at different step indices (`step_3_` on
    # the preprocessor, `step_0_` on the postprocessor).
    for config in (_REAL_PREPROCESSOR, _REAL_POSTPROCESSOR):
        text = json.dumps(config)
        assert '"stats"' not in text
        assert "mean" not in text and "min" not in text  # norm_map's MIN_MAX is uppercase
    assert named_sidecars(_REAL_PREPROCESSOR) == [
        "policy_preprocessor_step_3_normalizer_processor.safetensors"
    ]
    assert named_sidecars(_REAL_POSTPROCESSOR) == [
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    ]


def test_named_sidecars_reads_the_name_instead_of_deriving_it() -> None:
    config = {
        "steps": [
            {
                "registry_name": "normalizer_processor",
                "config": {},
                "state_file": "a_step_7_x.safetensors",
            },
            {"registry_name": "device_processor", "config": {"device": "cpu"}},
        ]
    }
    # `step_7_` at index 0: a name assembled from the step's position would miss the real file.
    assert named_sidecars(config) == ["a_step_7_x.safetensors"]


def test_named_sidecars_is_empty_for_a_config_without_steps() -> None:
    assert named_sidecars({"name": "policy_preprocessor"}) == []
    assert named_sidecars({"steps": "not-a-list"}) == []


def _write_processors(root: Path, *, with_sidecars: bool) -> None:
    for name, config in (
        ("policy_preprocessor.json", _REAL_PREPROCESSOR),
        ("policy_postprocessor.json", _REAL_POSTPROCESSOR),
    ):
        (root / name).write_text(json.dumps(config), encoding="utf-8")
        if with_sidecars:
            for sidecar in named_sidecars(config):
                (root / sidecar).write_text("", encoding="utf-8")  # existence is the check


def test_a_missing_sidecar_is_named_before_lerobot_is_imported(tmp_path: Path) -> None:
    # Both JSONs present, the file they point at absent: the state the preflight exists for. It
    # must be this error and not `ImportError` -- CI has no lerobot, and a checkpoint problem that
    # gets reported as a missing library is a wasted afternoon.
    _write_processors(tmp_path, with_sidecars=False)
    with pytest.raises(FileNotFoundError, match=r"normalizer_processor\.safetensors"):
        load_processor_norm(tmp_path)


def test_a_directory_without_the_processor_json_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"policy_preprocessor\.json"):
        load_processor_norm(tmp_path)


def test_check_processor_files_returns_what_it_proved(tmp_path: Path) -> None:
    # The migration, the control's row writer and `lerobot_control.sh` now gate on this same
    # function, so its contract is the directory being usable *and* which files said so.
    _write_processors(tmp_path, with_sidecars=True)
    assert check_processor_files(tmp_path) == [
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]


def test_check_processor_files_follows_each_json_not_just_the_first(tmp_path: Path) -> None:
    # Every other name present -- `config.json`, the weights, the preprocessor's statistics -- and
    # only the postprocessor's sidecar gone. That is the state the migration's four-name list
    # accepted, and the first JSON's own check passes, so this is what proves both are followed.
    _write_processors(tmp_path, with_sidecars=True)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_text("", encoding="utf-8")
    (tmp_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match=r"step_0_unnormalizer_processor\.safetensors"):
        check_processor_files(tmp_path)


def test_override_processor_device_keeps_the_sidecar_and_the_declared_mapping() -> None:
    # docs/10 condition 4: the override may move the device step and nothing else. Dropping
    # `state_file` while rebuilding a step would load the pipeline with no statistics, and
    # rewriting `norm_map` would change the declared mapping the gate is meant to compare.
    out = override_processor_device(_REAL_PREPROCESSOR, "cpu")
    assert named_sidecars(out) == named_sidecars(_REAL_PREPROCESSOR)
    assert out["steps"][2]["config"]["device"] == "cpu"
    assert out["steps"][3]["config"]["norm_map"] == _NORM_MAP
    assert out["steps"][3]["state_file"] == (
        "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
