"""The local policy server's HTTP contract, exercised end to end on CPU.

A stub policy stands in for the checkpoint, so this runs in CI with no GPU, no network and
no download. It is the only test that goes through a real socket, which is the point: the
contract in docs/05 section 4 (0-255 images in, raw pixel actions out, errors as 400) is
what a client implementation will depend on.
"""

from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest
import torch
from torch import Tensor

from policybench.deploy.server import MAX_BODY_BYTES, PolicyServer

pytestmark = pytest.mark.unit


class StubPolicy(torch.nn.Module):
    """Minimal `PolicyLike`: echoes a fixed chunk and records what it was asked for."""

    obs_horizon = 2
    pred_horizon = 16
    action_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, Any] = {}

    def predict_chunk(
        self, obs: dict[str, Tensor], steps: int = 10, instruction: str = "", **_: Any
    ) -> Tensor:
        self.seen = {
            "image_shape": tuple(obs["image"].shape),
            "image_max": float(obs["image"].max()),
            "state": obs["state"].tolist(),
            "steps": steps,
            "instruction": instruction,
        }
        return torch.tensor([[[i, i + 1] for i in range(16)]], dtype=torch.float32)

    def num_parameters(self) -> int:
        return 0


class TwoViewStubPolicy(StubPolicy):
    """VLA-shaped: two named views and a required instruction (models/observation.py)."""

    camera_names = ("image", "wrist")
    requires_instruction = True

    def predict_chunk(
        self, obs: dict[str, Tensor], steps: int = 10, instruction: str = "", **_: Any
    ) -> Tensor:
        self.seen = {
            "image_max": float(obs["image"].max()),
            "wrist_max": float(obs["wrist"].max()),
            "instruction": instruction,
            "steps": steps,
        }
        return torch.tensor([[[i, i + 1] for i in range(16)]], dtype=torch.float32)


@pytest.fixture
def served() -> Any:
    policy = StubPolicy()
    server = PolicyServer(("127.0.0.1", 0), policy, name="stub", device="cpu", action_horizon=8)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.port}", policy
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def served_two_view() -> Any:
    policy = TwoViewStubPolicy()
    server = PolicyServer(("127.0.0.1", 0), policy, name="vla", device="cpu", action_horizon=8)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.port}", policy
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _raw_post(url: str, *, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """Send headers and no body, so the server's own length check is what is under test."""
    host, port = url.removeprefix("http://").rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        connection.putrequest("POST", "/chunk")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _payload(steps: int = 10) -> dict[str, Any]:
    rng = [[[[128, 128, 128] for _ in range(96)] for _ in range(96)] for _ in range(2)]
    return {"image": rng, "state": [[100.0, 200.0], [101.0, 201.0]], "steps": steps}


def test_health_reports_the_interface(served: Any) -> None:
    url, _ = served
    with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
        body = json.loads(response.read())
    assert response.status == 200
    assert body["status"] == "ok"
    assert body["policy"] == "stub"
    assert (body["obs_horizon"], body["pred_horizon"], body["action_dim"]) == (2, 16, 2)
    # The health payload is where a client learns the observation contract.
    assert body["cameras"] == ["image"]
    assert body["instruction_required"] is False


def test_chunk_returns_actions_and_normalizes_images(served: Any) -> None:
    url, policy = served
    status, body = _post(f"{url}/chunk", _payload(steps=4))
    assert status == 200
    assert len(body["actions"]) == 8  # action_horizon, not the full 16-step prediction
    assert body["steps"] == 4
    assert body["latency_ms"] > 0.0
    assert policy.seen["image_shape"] == (1, 2, 3, 96, 96)
    assert policy.seen["image_max"] == pytest.approx(128 / 255)


def test_state_units_are_passed_through_untouched(served: Any) -> None:
    url, policy = served
    _post(f"{url}/chunk", _payload())
    assert policy.seen["state"][0][0] == [100.0, 200.0]


def test_unknown_path_is_404(served: Any) -> None:
    url, _ = served
    status, body = _post(f"{url}/nope", _payload())
    assert status == 404
    assert "unknown path" in body["error"]


def test_bad_shape_is_a_400_with_a_message(served: Any) -> None:
    url, _ = served
    payload = _payload()
    payload["image"] = payload["image"][:1]  # one frame instead of obs_horizon=2
    status, body = _post(f"{url}/chunk", payload)
    assert status == 400
    assert "image must be" in body["error"]


def test_already_normalized_images_are_rejected(served: Any) -> None:
    """A client sending [0, 1] floats would otherwise produce plausible-looking garbage."""
    url, _ = served
    payload = _payload()
    payload["image"] = [[[[0.5, 0.5, 0.5]] * 96] * 96] * 2
    status, body = _post(f"{url}/chunk", payload)
    assert status == 400
    assert "0-255" in body["error"]


def _two_view_payload(instruction: str = "push the T into the goal") -> dict[str, Any]:
    def frames(value: int) -> list[Any]:
        return [[[value, value, value] for _ in range(96)] for _ in range(96)] * 1

    return {
        "cameras": {"image": [frames(128)] * 2, "wrist": [frames(200)] * 2},
        "state": [[100.0, 200.0], [101.0, 201.0]],
        "instruction": instruction,
    }


def test_two_view_health_advertises_its_contract(served_two_view: Any) -> None:
    url, _ = served_two_view
    with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
        body = json.loads(response.read())
    assert body["cameras"] == ["image", "wrist"]
    assert body["instruction_required"] is True


def test_two_views_and_an_instruction_reach_the_policy_bound_to_their_names(
    served_two_view: Any,
) -> None:
    """Distinct pixel values per view: a swapped or duplicated view cannot pass this."""
    url, policy = served_two_view
    status, body = _post(f"{url}/chunk", _two_view_payload())
    assert status == 200, body
    assert policy.seen["image_max"] == pytest.approx(128 / 255)
    assert policy.seen["wrist_max"] == pytest.approx(200 / 255)
    assert policy.seen["instruction"] == "push the T into the goal"


def test_one_view_cannot_satisfy_a_two_view_policy(served_two_view: Any) -> None:
    url, _ = served_two_view
    payload = _payload()
    payload["instruction"] = "push the T into the goal"
    status, body = _post(f"{url}/chunk", payload)
    assert status == 400
    assert "wrist" in body["error"]


def test_a_two_view_policy_refuses_an_empty_instruction(served_two_view: Any) -> None:
    url, _ = served_two_view
    payload = _two_view_payload(instruction="")
    status, body = _post(f"{url}/chunk", payload)
    assert status == 400
    assert "instruction" in body["error"]


def test_a_payload_with_no_camera_is_a_400(served: Any) -> None:
    url, _ = served
    payload = _payload()
    payload.pop("image")
    status, body = _post(f"{url}/chunk", payload)
    assert status == 400
    assert "no camera in the payload" in body["error"]


def test_an_oversized_body_is_refused_from_its_length_header(served: Any) -> None:
    # The header is not trusted into `rfile.read`: a declared length above the cap is answered
    # before a byte of it is read, so a client cannot make the server allocate what it names.
    url, _ = served
    status, body = _raw_post(url, headers={"Content-Length": str(MAX_BODY_BYTES + 1)})
    assert status == 413
    assert str(MAX_BODY_BYTES) in body["error"]


def test_a_non_integer_content_length_is_a_400(served: Any) -> None:
    url, _ = served
    status, body = _raw_post(url, headers={"Content-Length": "not-a-number"})
    assert status == 400
    assert "Content-Length" in body["error"]
