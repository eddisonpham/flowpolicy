"""A local HTTP wrapper around any `PolicyLike`: the deployment path without ROS 2.

`docs/05` section 2 describes the two-node ROS 2 design and the documented plain-Python
fallback (D7) for hosts without Docker. This module is the serving half of that fallback:
a real process boundary around the policy, so a client can be written in any language and
the policy's cost is measured where it actually runs.

Standard library only, on purpose. Adding a web framework to serve one endpoint would be a
dependency this project does not need, and the repo's invariant is that dependencies are
named before they land.

Contract (documented in `docs/05` section 3):

    GET  /health -> {"status": "ok", "policy": "<name>", "obs_horizon": 2, "action_dim": 2,
                     "pred_horizon": 16, "cameras": ["image"], "instruction_required": false}
    POST /chunk  <- {"image": [[[[r, g, b], ...]]],   # [To, H, W, 3] uint8, 0-255
                     "state": [[x, y], ...],          # [To, 2] raw pixels
                     "cameras": {"image": ..., "wrist": ...},   # optional, instead of "image"
                     "instruction": "<task string>",  # optional; required if the policy says so
                     "steps": 10,                     # optional, default 10, in [1, max_steps]
                     "deadline_ms": 250.0}            # optional; a label, never a refusal
                 -> {"actions": [[x, y], ...],        # [action_horizon, 2] raw pixels
                     "latency_ms": 183.4, "steps": 10,
                     "deadline_ms": 250.0,             # echoed only when the client sent one
                     "deadline_missed": false}

Cameras are named, not positional (models/observation.py): a payload may carry one view as
`image` or any number under `cameras`, and the names are checked against the policy's own
declaration. A client that sends one view to a policy expecting two gets HTTP 400, never the
same frame twice.

Errors are reported as HTTP 400 with `{"error": "<message>"}` (413 for a body above the size
limit), and a request is only ever executed under the server's inference lock, so two clients
queue on the policy instead of interleaving inside it. The server never rescales
actions: the contract is raw pixels in, raw pixels out, the same units the environment
takes, so a client cannot silently disagree with the harness about units.

Three edges are enforced rather than trusted, because each one is a client bug that would
otherwise be measured as a policy result:

- `steps` is bounded by the sampler's own schedule (`max_steps`, 100 for the diffusion
  checkpoint). `steps: 0` is not a fast policy, it is an invalid request, and a value above the
  schedule is not a better policy either; both are 400s naming the bound.
- `deadline_ms` is a *label*: the action is still returned, with `deadline_missed` set from the
  measured `latency_ms`. Refusing a late chunk would make lateness unobservable at the one
  point it matters.
- `shutdown()` drains: it stops accepting, waits for the requests already inside the policy, then
  closes. `ThreadingHTTPServer` would otherwise drop a response on Ctrl-C.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from torch import Tensor

from policybench.models.observation import (
    DEFAULT_CAMERA,
    ObservationSpec,
    check_cameras,
    check_instruction,
    spec_for,
)
from policybench.models.policies import PolicyLike

__all__ = ["MAX_BODY_BYTES", "PolicyServer", "build_obs", "serve"]

# Two 256x256 RGB views over two observation steps are ~1.5 MB as JSON integers, so this is
# headroom rather than a policy: it exists to stop an unbounded `Content-Length` from being
# trusted into `rfile.read`.
MAX_BODY_BYTES = 32 * 2**20

# The sampler's schedule length for the diffusion checkpoint this server was built around
# (`num_train_timesteps=100`). A policy with a longer schedule passes its own bound to the
# constructor; a request above the bound is answered 400 rather than run, because `steps=10**6`
# is not a policy setting, it is a client that computed a step count in the wrong unit.
DEFAULT_MAX_STEPS = 100


def _frames(payload: Mapping[str, Any], spec: ObservationSpec) -> dict[str, np.ndarray]:
    """Pull the named views out of the payload, accepting `image` or `cameras`.

    Both spellings are supported so the single-view rows and clients written before the
    contract existed keep working; `cameras` is what a multi-view policy needs.
    """
    if "cameras" in payload:
        raw = payload["cameras"]
        if not isinstance(raw, Mapping):
            raise ValueError("cameras must be an object of {name: [To, H, W, 3]} arrays")
        frames = {str(name): np.asarray(value, dtype=np.float32) for name, value in raw.items()}
    elif DEFAULT_CAMERA in payload:
        frames = {DEFAULT_CAMERA: np.asarray(payload[DEFAULT_CAMERA], dtype=np.float32)}
    else:
        raise ValueError(
            "no camera in the payload: send 'image' for the single-view contract, or "
            "'cameras': {name: [[[[r, g, b], ...]]]}"
        )
    check_cameras(spec, frames)
    return frames


def build_obs(
    payload: Mapping[str, Any],
    obs_horizon: int,
    device: str,
    spec: ObservationSpec | None = None,
) -> dict[str, Tensor]:
    """Turn the JSON observation into the batch the policy interface expects.

    Images arrive as 0-255 integers and are scaled to [0, 1] here; the policy applies its
    own normalization afterwards. Shapes are validated eagerly, because a silently
    transposed image produces plausible actions and a useless success rate.
    """
    spec = spec or ObservationSpec()
    frames = _frames(payload, spec)
    for name, frame in frames.items():
        if frame.ndim != 4 or frame.shape[0] != obs_horizon or frame.shape[-1] != 3:
            raise ValueError(f"{name} must be [To={obs_horizon}, H, W, 3], got {frame.shape}")
        if float(frame.max(initial=0.0)) <= 1.0:
            raise ValueError(f"{name} values look normalized; the contract is 0-255 integers")
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.ndim != 2 or state.shape[0] != obs_horizon:
        raise ValueError(f"state must be [To={obs_horizon}, Ds], got {state.shape}")
    batch: dict[str, Tensor] = {
        "state": torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).to(device)
    }
    # Built in the policy's declared order, so a view cannot be bound to the wrong slot.
    for name in spec.cameras:
        image = np.ascontiguousarray(frames[name].transpose(0, 3, 1, 2)) / 255.0
        batch[name] = torch.from_numpy(image).unsqueeze(0).to(device)
    return batch


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
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.policy = policy
        self.policy_name = name
        self.device = device
        self.max_steps = max_steps
        self.steps = self.check_steps(steps)
        self.action_horizon = action_horizon
        # What this policy needs from the observation path, read once at construction so
        # `/health` advertises it and every request is checked against it.
        self.spec = spec_for(policy)
        # Serializes inference across handler threads. `ThreadingHTTPServer` is the right choice
        # for a UI-style endpoint, but a single policy on one CUDA context cannot answer two
        # requests at once: they interleave inside the model and the second request's queueing
        # delay is reported as the policy's latency, which is the one number `/chunk` exists to
        # report honestly.
        self.inference_lock = threading.Lock()
        # In-flight request count, for the drain in `shutdown`. `ThreadingHTTPServer` answers each
        # request on its own thread and `daemon_threads` means `server_close` does not wait for
        # them, so without a count a Ctrl-C during inference loses that client's response.
        self._inflight = 0
        self._drained = threading.Condition()
        super().__init__(address, _Handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def check_steps(self, raw: Any) -> int:
        """Validate a sampler step count, naming the bound when it is out of range."""
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"steps must be an integer, got {type(raw).__name__}")
        if not 1 <= raw <= self.max_steps:
            raise ValueError(
                f"steps must be in [1, {self.max_steps}] (the sampler's schedule length), got {raw}"
            )
        return raw

    def check_deadline(self, raw: Any) -> float | None:
        """Validate `deadline_ms`, or return None when the client did not send one."""
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError("deadline_ms must be a number of milliseconds")
        if not raw > 0:
            raise ValueError(f"deadline_ms must be > 0, got {raw}")
        return float(raw)

    def _finished(self) -> None:
        with self._drained:
            self._inflight -= 1
            if self._inflight <= 0:
                self._drained.notify_all()

    def process_request(self, request: Any, client_address: Any) -> None:
        with self._drained:
            self._inflight += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            # The request never reached a handler thread, so nothing else will release it.
            self._finished()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._finished()

    def shutdown(self, drain_timeout_s: float = 30.0) -> None:
        """Stop accepting, then wait for the requests already inside the policy.

        The order is the point: stopping first means no new request can be counted after the
        wait begins, so `_inflight == 0` is an end state rather than a snapshot. Blocking
        requests inside the policy are exactly what a `daemon_threads` shutdown would truncate.
        """
        super().shutdown()
        deadline = time.monotonic() + drain_timeout_s
        with self._drained:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        f"shutdown: {self._inflight} request(s) still running after "
                        f"{drain_timeout_s:.0f}s; closing anyway",
                        flush=True,
                    )
                    return
                self._drained.wait(remaining)


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
                "max_steps": self.server.max_steps,
                "action_horizon": self.server.action_horizon,
                "cameras": list(self.server.spec.cameras),
                "instruction_required": self.server.spec.requires_instruction,
            },
        )

    def do_POST(self) -> None:
        # Read the body *before* deciding the path, even for an unknown one. HTTP/1.1
        # keep-alive means an unread body is parsed as the next request line, so a client that
        # posts to the wrong path used to see a connection reset instead of the 404 (which is
        # how `test_unknown_path_is_404` failed intermittently). Discarding it keeps the error
        # the contract promises.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Content-Length must be an integer"})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            # The body is unread, so keep-alive would parse it as the next request line; closing
            # the connection is the only honest answer to a length we refuse to read.
            self.close_connection = True
            self._send(413, {"error": f"body is {length} bytes; the limit is {MAX_BODY_BYTES}"})
            return
        raw = self.rfile.read(length) if length else b""
        if self.path.rstrip("/") != "/chunk":
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            payload = json.loads(raw or b"{}")
            policy = self.server.policy
            spec = self.server.spec
            instruction = str(payload.get("instruction", ""))
            check_instruction(spec, instruction)
            obs = build_obs(payload, policy.obs_horizon, self.server.device, spec)
            steps = self.server.check_steps(payload.get("steps", self.server.steps))
            deadline_ms = self.server.check_deadline(payload.get("deadline_ms"))
            with self.server.inference_lock:
                if self.server.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                chunk = policy.predict_chunk(obs, steps=steps, instruction=instruction)
                if self.server.device.startswith("cuda"):
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0
                actions = chunk[0, : self.server.action_horizon].float().cpu().numpy()
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            return
        body: dict[str, Any] = {
            "actions": actions.tolist(),
            "latency_ms": latency_ms,
            "steps": steps,
        }
        # A missed deadline labels the measurement; it never withholds the action (see the
        # module docstring). Absent when the client did not ask, so a reply cannot imply a
        # deadline it was never given.
        if deadline_ms is not None:
            body["deadline_ms"] = deadline_ms
            body["deadline_missed"] = latency_ms > deadline_ms
        self._send(200, body)


def serve(
    policy: PolicyLike,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    name: str = "policy",
    device: str = "cuda",
    steps: int = 10,
    action_horizon: int = 8,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    """Serve until interrupted. `port=0` picks a free port (used by the test)."""
    server = PolicyServer(
        (host, port),
        policy,
        name=name,
        device=device,
        steps=steps,
        action_horizon=action_horizon,
        max_steps=max_steps,
    )
    print(f"serving {name} on http://{host}:{server.port} (POST /chunk, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("interrupted: draining in-flight requests", flush=True)
    finally:
        # Ctrl-C is the normal way to stop this, and a client mid-inference deserves its
        # answer: `shutdown` stops accepting and waits for what is already inside the policy.
        server.shutdown()
        server.server_close()
