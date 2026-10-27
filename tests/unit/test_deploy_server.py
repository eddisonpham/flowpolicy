"""The local policy server's HTTP contract, exercised end to end on CPU.

A stub policy stands in for the checkpoint, so this runs in CI with no GPU, no network and
no download. It is the only test that goes through a real socket, which is the point: the
contract in docs/05 section 4 (0-255 images in, raw pixel actions out, errors as 400) is
what a client implementation will depend on.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest
import torch
from torch import Tensor

from flowpolicy.deploy.server import PolicyServer

pytestmark = pytest.mark.unit


class StubPolicy(torch.nn.Module):
    """Minimal `PolicyLike`: echoes a fixed chunk and records what it was asked for."""

    obs_horizon = 2
    pred_horizon = 16
    action_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, Any] = {}

    def predict_chunk(self, obs: dict[str, Tensor], steps: int = 10, **_: Any) -> Tensor:
        self.seen = {
            "image_shape": tuple(obs["image"].shape),
            "image_max": float(obs["image"].max()),
            "state": obs["state"].tolist(),
            "steps": steps,
        }
        return torch.tensor([[[i, i + 1] for i in range(16)]], dtype=torch.float32)

    def num_parameters(self) -> int:
        return 0


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


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


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
