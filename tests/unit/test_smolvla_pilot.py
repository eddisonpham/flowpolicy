"""The pilot's action decoding: model-space chunk in, env actions out (docs/11 section 5.2).

`scripts/smolvla_pilot.py` is the one measurement that replaces 5.1's extrapolated per-chunk
band, so the numbers it prints are load-bearing. This file pins the two refusals that keep them
honest, because both failure modes otherwise produce a *plausible timing row* for a policy that
did something else:

- **a single action**: the policy returned one step instead of a chunk, so the reported
  per-chunk cost describes a different loop (`select_action`, not `predict_action_chunk`);
- **an unnormalized chunk**: actions outside LIBERO's `[-1, 1]` box, so the episode is timed
  while the arm barely moves -- and neither number says anything about the real run.

Neither is visible in milliseconds, which is why they are errors rather than warnings.

The guard on magnitudes is a *gross-value* trap, not a scale check: a chunk scaled by 0.5 stays
inside the box and passes. Catching that is precondition 3's job -- the postprocessor has to
reproduce the checkpoint's declared statistics -- and this file does not pretend otherwise.

No lerobot, no torch CUDA, no checkpoint: the module imports its backend lazily, so everything
here runs in CI.

The second half covers precondition 3, the statistics probe, and its fixtures are shaped like the
real checkpoint rather than like the code's first guess: `HuggingFaceVLA/smolvla_libero`'s
`policy_postprocessor.json` carries the norm mode and the action width and **no numbers**, with the
statistics living as `action.mean` / `action.std` in a `state_file` safetensors sidecar (read
2026-11-02). A fixture with inline stats would have kept the JSON-scanning version green while it
could never pass a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from scripts.smolvla_pilot import ACTION_DIM, _statistics_check, _to_env_actions

pytestmark = pytest.mark.unit


def _chunk(rows: int = 10, width: int = ACTION_DIM, scale: float = 0.5) -> np.ndarray:
    """A chunk of legal-looking actions, spread across the box so truncation is visible."""
    return np.linspace(-scale, scale, rows * width, dtype=np.float32).reshape(rows, width)


def test_a_full_chunk_decodes_to_float32_env_rows() -> None:
    actions = _to_env_actions(_chunk(), actions_per_chunk=10)
    assert actions.shape == (10, ACTION_DIM)
    assert actions.dtype == np.float32


def test_the_padded_model_width_is_cut_to_the_env_width() -> None:
    # SmolVLA samples in `max_action_dim`-wide space; LIBERO takes 7. Handing 32 to `env.step`
    # is a shape error mid-episode, so the cut has to happen here.
    actions = _to_env_actions(_chunk(width=32), actions_per_chunk=10)
    assert actions.shape == (10, ACTION_DIM)


def test_the_single_batch_axis_is_dropped() -> None:
    tensor = torch.tensor(_chunk()[None], dtype=torch.float32)
    actions = _to_env_actions(tensor, actions_per_chunk=10)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (10, ACTION_DIM)


def test_a_single_action_is_refused_as_not_a_chunk() -> None:
    # The `select_action` shape: one action, or a chunk of exactly one. Timing this would measure
    # a loop the block does not run.
    with pytest.raises(ValueError, match="fewer than the 10"):
        _to_env_actions(np.zeros((1, ACTION_DIM), dtype=np.float32), actions_per_chunk=10)
    with pytest.raises(ValueError, match="fewer than the 10"):
        _to_env_actions(torch.zeros(1, 1, 32), actions_per_chunk=10)


def test_an_unnormalized_chunk_is_refused_rather_than_timed() -> None:
    # Model-space values that never went through the postprocessor: legally shaped, wildly out of
    # LIBERO's box. The row would look fine and describe an episode that never moved the arm.
    unnormalized = _chunk() * 20.0
    assert np.abs(unnormalized).max() > 1.5
    with pytest.raises(ValueError, match=r"outside LIBERO's \[-1, 1\] box"):
        _to_env_actions(unnormalized, actions_per_chunk=10)


def test_a_chunk_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(ValueError, match="the env wants 7"):
        _to_env_actions(_chunk(width=4), actions_per_chunk=10)


def test_more_than_one_batch_entry_is_refused() -> None:
    # Two batch entries means the caller is decoding a batch, and keeping the first row would
    # silently mix two episodes' actions into one trajectory.
    with pytest.raises(ValueError, match="expected \\[T, D\\]"):
        _to_env_actions(np.zeros((2, 10, ACTION_DIM), dtype=np.float32), actions_per_chunk=10)


# ---- precondition 3: the postprocessor against the checkpoint's declared statistics ----------
#
# The statistics in these fixtures are real numbers read off the SmolVLA checkpoint's sidecar on
# 2026-11-02, truncated to seven dims; the layout is that checkpoint's, unchanged.
_SIDECAR = "policy_postprocessor_step_1_unnormalizer_processor.safetensors"
_MEAN = [0.0628, 0.0868, -0.0904, 0.0005, 0.0056, -0.0052, -0.0496]
_STD = [0.3355, 0.3784, 0.4447, 0.0392, 0.0634, 0.0780, 0.9988]


def _write_checkpoint(
    directory: Path, *, mode: str = "MEAN_STD", width: int = 7, low: list[float] | None = None
) -> tuple[list[float], list[float]]:
    """Write the postprocessor pair in the layout lerobot 0.6.1 actually saves.

    Returns the two vectors the pipeline should reproduce, so a test can hand them to a fake
    postprocessor and control the only thing it is testing: the arithmetic.
    """
    if low is None:
        low = _MEAN[:width] if mode == "MEAN_STD" else [-0.9 + 0.1 * i for i in range(width)]
    high = _STD[:width] if mode == "MEAN_STD" else [0.8 + 0.1 * i for i in range(width)]
    suffix = ("mean", "std") if mode == "MEAN_STD" else ("min", "max")
    save_file(
        {
            f"action.{suffix[0]}": torch.tensor(low, dtype=torch.float32),
            f"action.{suffix[1]}": torch.tensor(high, dtype=torch.float32),
            # The state entry is wider (8 vs 7) and must not be mistaken for the action one.
            "observation.state.mean": torch.zeros(8),
            "observation.state.std": torch.ones(8),
        },
        str(directory / _SIDECAR),
    )
    (directory / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {
                        "registry_name": "device_processor",
                        "config": {"device": "cpu", "float_dtype": None},
                    },
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {
                            "eps": 1e-08,
                            "features": {"action": {"type": "ACTION", "shape": [width]}},
                            "norm_map": {
                                "VISUAL": "IDENTITY",
                                "STATE": "MEAN_STD",
                                "ACTION": mode,
                            },
                        },
                        "state_file": _SIDECAR,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return low, high


def _serve(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Answer `hf_hub_download` from the fixture directory, so no test touches the network."""

    def fake_download(repo_id: str, filename: str | None = None, **kwargs: object) -> str:
        path = directory / str(filename)
        if not path.exists():
            raise FileNotFoundError(f"{repo_id}: {filename}")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)


class _Unnormalizer:
    """lerobot's inverse, with one knob to break on purpose and a shape guard like the real one."""

    def __init__(
        self,
        low: list[float],
        high: list[float],
        mode: str = "MEAN_STD",
        *,
        scale: float = 1.0,
        rank: int = 3,
        width: int = 7,
        zero_maps_to_low: bool = False,
    ) -> None:
        self.low = torch.tensor(low, dtype=torch.float32)
        self.high = torch.tensor(high, dtype=torch.float32)
        self.mode, self.scale, self.rank, self.width = mode, scale, rank, width
        self.zero_maps_to_low = zero_maps_to_low

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        # The real pipeline is configured for one width and one call shape; refusing anything else
        # is what makes the probe's shape discovery observable.
        if z.ndim != self.rank or z.shape[-1] != self.width:
            raise ValueError(f"unsupported probe shape {tuple(z.shape)}")
        if self.mode == "MEAN_STD":
            return z * (self.high * self.scale) + self.low
        if self.zero_maps_to_low:  # the plausible-but-wrong inverse this probe used to assume
            return z * (self.high - self.low) / 2 + self.low
        return (z + 1) / 2 * (self.high - self.low) + self.low


def test_the_statistics_come_from_the_sidecar_and_a_matching_pipeline_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    low, high = _write_checkpoint(tmp_path)
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("HuggingFaceVLA/smolvla_libero", _Unnormalizer(low, high), "cpu")
    assert result["verdict"] == "ok"
    assert result["mode"] == "mean_std"
    assert result["dims"] == ACTION_DIM
    assert result["probe_shape"] == "(1, 1, 7)"
    # Not exactly 0: the fake does its arithmetic in float32, so `mean + std` at z = 1 lands ~7e-9
    # away in float64. That is five orders under the bar, which is the point of the bar.
    assert result["max_abs_diff"] < 1e-6


def test_the_declared_width_drives_the_probe_not_the_padded_model_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SmolVLA declares `max_action_dim` 32 and unnormalizes 7. The fake here refuses any width but
    # 7, like the real processor, so this is green only if the width came from the JSON and not
    # from the config's padded action space.
    low, high = _write_checkpoint(tmp_path)
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("x", _Unnormalizer(low, high, width=ACTION_DIM), "cpu")
    assert result["verdict"] == "ok"


def test_a_wrong_std_is_a_mismatch_rather_than_a_rounding_quibble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 0.01 % scale error on the std vector: the mean still matches at z = 0, so only the z = 1
    # probe can catch it, and the difference is ~1e-4 -- small enough that a looser bar would call
    # it agreement.
    low, high = _write_checkpoint(tmp_path)
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("x", _Unnormalizer(low, high, scale=1.0001), "cpu")
    assert result["verdict"] == "mismatch"
    assert 0.0 < result["max_abs_diff"] < 1e-2
    assert result["max_abs_diff"] > result["bar"]


def test_min_max_is_probed_at_its_endpoints_and_midpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `(z + 1) / 2 * (max - min) + min`: -1 is the minimum, 1 the maximum, 0 the midpoint.
    low, high = _write_checkpoint(tmp_path, mode="MIN_MAX")
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("x", _Unnormalizer(low, high, mode="MIN_MAX"), "cpu")
    assert result["verdict"] == "ok"
    assert result["mode"] == "min_max"


def test_the_plausible_but_wrong_min_max_inverse_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Treating normalized 0 as the minimum is the natural guess and it is wrong: it is the midpoint.
    # Launching a block on this would unnormalize every action into the lower half of its range.
    low, high = _write_checkpoint(tmp_path, mode="MIN_MAX")
    _serve(monkeypatch, tmp_path)
    result = _statistics_check(
        "x", _Unnormalizer(low, high, mode="MIN_MAX", zero_maps_to_low=True), "cpu"
    )
    assert result["verdict"] == "mismatch"


def test_a_sidecar_without_action_statistics_is_not_run_rather_than_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    low, high = _write_checkpoint(tmp_path)
    save_file(
        {"observation.state.mean": torch.zeros(8), "observation.state.std": torch.ones(8)},
        str(tmp_path / _SIDECAR),
    )
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("x", _Unnormalizer(low, high), "cpu")
    assert result["verdict"] == "not run"
    assert "action.mean" in result["note"]


def test_a_norm_mode_this_probe_does_not_model_is_not_run_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_checkpoint(tmp_path, mode="IDENTITY")
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("x", _Unnormalizer(_MEAN, _STD), "cpu")
    assert result["verdict"] == "not run"
    assert "identity" in result["note"]


def test_the_two_dimensional_probe_shape_is_used_when_the_chunk_shape_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    low, high = _write_checkpoint(tmp_path)
    _serve(monkeypatch, tmp_path)
    result = _statistics_check("x", _Unnormalizer(low, high, rank=2), "cpu")
    assert result["verdict"] == "ok"
    assert result["probe_shape"] == "(1, 7)"
