"""The observation contract: named views and the language slot (docs/11 section 3).

CPU-only and torch-only, no checkpoint. The point of these tests is the *refusals*: a policy
that declares two views and is handed one must raise instead of being fed the same frame twice,
because that mistake produces plausible actions and a success rate that means nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from policybench.models.observation import (
    DEFAULT_CAMERA,
    ObservationSpec,
    check_cameras,
    check_frames,
    check_instruction,
    spec_for,
)

pytestmark = pytest.mark.unit


class SingleViewPolicy:
    """The historical contract: one view, no instruction, nothing declared."""

    obs_horizon = 2

    def predict_chunk(self, obs: dict[str, torch.Tensor], **_: Any) -> torch.Tensor:
        return torch.zeros(1, 4, 2)


class TwoViewPolicy:
    """A VLA-shaped policy: two named views and a required instruction."""

    obs_horizon = 2
    camera_names = ("image", "wrist")
    requires_instruction = True


def _obs(cameras: tuple[str, ...] = (DEFAULT_CAMERA,), horizon: int = 2) -> dict[str, torch.Tensor]:
    obs = {"state": torch.zeros(1, horizon, 2)}
    for name in cameras:
        obs[name] = torch.zeros(1, horizon, 3, 8, 8)
    return obs


def test_a_policy_that_declares_nothing_gets_the_historical_contract() -> None:
    spec = spec_for(SingleViewPolicy())
    assert spec.cameras == (DEFAULT_CAMERA,)
    assert spec.requires_instruction is False
    assert spec.single_view is True


def test_a_declared_policy_reports_its_views_in_order() -> None:
    spec = spec_for(TwoViewPolicy())
    assert spec.cameras == ("image", "wrist")
    assert spec.requires_instruction is True
    assert spec.single_view is False


def test_repeated_camera_names_are_rejected_at_source() -> None:
    class Duplicated:
        camera_names = ("image", "image")

    with pytest.raises(ValueError, match="repeated views"):
        spec_for(Duplicated())


def test_empty_camera_list_is_rejected() -> None:
    class Nothing:
        camera_names: tuple[str, ...] = ()

    with pytest.raises(ValueError, match="at least one view"):
        spec_for(Nothing())


def test_one_view_cannot_fill_a_two_view_policy() -> None:
    """The documented failure mode: duplicating a frame looks like it worked."""
    spec = spec_for(TwoViewPolicy())
    with pytest.raises(ValueError) as excinfo:
        check_cameras(spec, (DEFAULT_CAMERA,))
    message = str(excinfo.value)
    assert "wrist" in message
    assert "never duplicated" in message


def test_the_push_t_environment_satisfies_the_default_contract() -> None:
    check_cameras(spec_for(SingleViewPolicy()), (DEFAULT_CAMERA,))


def test_an_extra_available_view_is_not_an_error() -> None:
    """An env may expose more than the policy reads; only a shortfall is a mismatch."""
    check_cameras(ObservationSpec(), ("image", "image2"))


def test_a_batch_missing_a_declared_view_is_rejected() -> None:
    spec = spec_for(TwoViewPolicy())
    with pytest.raises(ValueError, match="wrist"):
        check_frames(spec, _obs((DEFAULT_CAMERA,)))


def test_a_batch_with_an_unconsumed_camera_is_rejected() -> None:
    spec = ObservationSpec()
    with pytest.raises(ValueError, match="does not consume"):
        check_frames(spec, _obs((DEFAULT_CAMERA, "wrist")))


def test_a_two_view_batch_validates() -> None:
    check_frames(spec_for(TwoViewPolicy()), _obs(("image", "wrist")))


def test_frame_shape_is_checked_against_the_state_horizon() -> None:
    spec = ObservationSpec()
    obs = _obs(horizon=2)
    obs[DEFAULT_CAMERA] = torch.zeros(1, 3, 3, 8, 8)  # history length disagrees with state
    with pytest.raises(ValueError, match="history lengths must agree"):
        check_frames(spec, obs)


def test_a_flat_image_is_rejected_rather_than_guessed() -> None:
    spec = ObservationSpec()
    obs = _obs()
    obs[DEFAULT_CAMERA] = torch.zeros(2, 3, 8, 8)  # no batch dimension
    with pytest.raises(ValueError, match=r"\[B, To, C, H, W\]"):
        check_frames(spec, obs)


def test_a_state_only_batch_is_rejected() -> None:
    """No views at all is the same shortfall as one view short: it names what is missing."""
    with pytest.raises(ValueError, match=r"missing \['image'\]"):
        check_frames(ObservationSpec(), {"state": torch.zeros(1, 2, 2)})


def test_an_unconditional_policy_accepts_an_empty_instruction() -> None:
    check_instruction(ObservationSpec(), "")


def test_a_language_conditioned_policy_refuses_an_empty_instruction() -> None:
    spec = spec_for(TwoViewPolicy())
    with pytest.raises(ValueError, match="requires an instruction"):
        check_instruction(spec, "   ")
    check_instruction(spec, "push the T-shaped block into the target")
