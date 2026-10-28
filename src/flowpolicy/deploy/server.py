"""A local HTTP wrapper around any `PolicyLike`: the deployment path without ROS 2.

`docs/05` section 2 describes the two-node ROS 2 design and the documented plain-Python
fallback (D7) for hosts without Docker. This module is the serving half of that fallback:
a real process boundary around the policy, so a client can be written in any language and
the policy's cost is measured where it actually runs.

Standard library only, on purpose. Adding a web framework to serve one endpoint would be a
dependency this project does not need, and the repo's invariant is that dependencies are
named before they land.

Contract (documented in `docs/05` section 4):

    GET  /health -> {"status": "ok", "policy": "<name>", "obs_horizon": 2, "action_dim": 2,
                     "pred_horizon": 16}
    POST /chunk  <- {"image": [[[[r, g, b], ...]]],   # [To, H, W, 3] uint8, 0-255
                     "state": [[x, y], ...],          # [To, 2] raw pixels
                     "steps": 10}                     # optional, default 10
                 -> {"actions": [[x, y], ...],        # [action_horizon, 2] raw pixels
                     "latency_ms": 183.4}

Errors are reported as HTTP 400 with `{"error": "<message>"}`. The server never rescales
actions: the contract is raw pixels in, raw pixels out, the same units the environment
takes, so a client cannot silently disagree with the harness about units.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from torch import Tensor

from flowpolicy.models.policies import PolicyLike

__all__ = ["PolicyServer", "build_obs", "serve"]


def build_obs(payload: Mapping[str, Any], obs_horizon: int, device: str) -> dict[str, Tensor]:
    """Turn the JSON observation into the batch the policy interface expects.

    Images arrive as 0-255 integers and are scaled to [0, 1] here; the policy applies its
    own normalization afterwards. Shapes are validated eagerly, because a silently
    transposed image produces plausible actions and a useless success rate.
    """
    image = np.asarray(payload["image"], dtype=np.float32)
    state = np.asarray(payload["state"], dtype=np.float32)
    if image.ndim != 4 or image.shape[0] != obs_horizon or image.shape[-1] != 3:
        raise ValueError(f"image must be [To={obs_horizon}, H, W, 3], got {image.shape}")
    if state.ndim != 2 or state.shape[0] != obs_horizon:
        raise ValueError(f"state must be [To={obs_horizon}, Ds], got {state.shape}")
    if float(image.max(initial=0.0)) <= 1.0:
        raise ValueError("image values look normalized; the contract is 0-255 integers")
    image = np.ascontiguousarray(image.transpose(0, 3, 1, 2)) / 255.0
    return {
        "image": torch.from_numpy(image).unsqueeze(0).to(device),
        "state": torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).to(device),
    }


class PolicyServer(ThreadingHTTPServer):
    """HTTP server holding one policy. Bind port 0 to let the OS pick one."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        policy: PolicyLike,
        *,
        name: str = "policy",
        device: str = "cuda",
        steps: int = 10,
        action_horizon: int = 8,
    ) -> None:
        self.policy = policy
        self.policy_name = name
        self.device = device
        self.steps = steps
        self.action_horizon = action_horizon
        super().__init__(address, _Handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])


class _Handler(BaseHTTPRequestHandler):
    server: PolicyServer

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Keep stdout clean: the caller's own logging is the record."""

    def _send(self, code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in ("", "/health"):
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        policy = self.server.policy
        self._send(
            200,
            {
                "status": "ok",
                "policy": self.server.policy_name,
                "obs_horizon": policy.obs_horizon,
                "pred_horizon": policy.pred_horizon,
                "action_dim": policy.action_dim,
                "steps": self.server.steps,
                "action_horizon": self.server.action_horizon,
            },
        )

    def do_POST(self) -> None:
        # Read the body *before* deciding the path, even for an unknown one. HTTP/1.1
        # keep-alive means an unread body is parsed as the next request line, so a client that
        # posts to the wrong path used to see a connection reset instead of the 404 (which is
        # how `test_unknown_path_is_404` failed intermittently). Discarding it keeps the error
        # the contract promises.
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        if self.path.rstrip("/") != "/chunk":
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            payload = json.loads(raw or b"{}")
            policy = self.server.policy
            obs = build_obs(payload, policy.obs_horizon, self.server.device)
            steps = int(payload.get("steps", self.server.steps))
            if self.server.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            chunk = policy.predict_chunk(obs, steps=steps)
            if self.server.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            actions = chunk[0, : self.server.action_horizon].float().cpu().numpy()
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(
            200,
            {"actions": actions.tolist(), "latency_ms": latency_ms, "steps": steps},
        )


def serve(
    policy: PolicyLike,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    name: str = "policy",
    device: str = "cuda",
    steps: int = 10,
    action_horizon: int = 8,
) -> None:
    """Serve until interrupted. `port=0` picks a free port (used by the test)."""
    server = PolicyServer(
        (host, port),
        policy,
        name=name,
        device=device,
        steps=steps,
        action_horizon=action_horizon,
    )
    print(f"serving {name} on http://{host}:{server.port} (POST /chunk, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
