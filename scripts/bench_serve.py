"""Measure the serving boundary (docs/05 section 3): throughput, tail, and HTTP overhead.

`flowpolicy.deploy.server` is a real process boundary with a real contract, and until now it
had only been *correctness*-tested (six socket tests on CPU) and never load-tested. This
script answers the three questions a serving boundary has to answer:

1. **How much does the HTTP round trip add?** The server reports its own in-process
   `latency_ms` (same protocol as E9: `torch.cuda.synchronize()` on both sides); the client
   measures the wall clock. The difference is the JSON encode/decode, the socket and the
   base64-free payload copy -- i.e. what a client actually pays.
2. **What does the tail look like?** Mean hides the case that matters for a control loop,
   so p50/p95/max are reported per concurrency level.
3. **Does it survive two clients?** `ThreadingHTTPServer` will accept both and serialize
   them on one GPU; measuring it says whether the second client sees queueing (it should).

The server is the real `scripts/serve.py`, spawned as a subprocess and stopped afterwards, so
this measures the artifact a user runs, not an in-process stand-in.

Usage:
    python scripts/bench_serve.py                     # 50 sequential + 2 x 25 concurrent
    python scripts/bench_serve.py --steps 4 --requests 100
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _wait_ready(port: int, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            body = json.loads(response.read())
            conn.close()
            if response.status == 200 and body.get("status") == "ok":
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"server on port {port} never became ready")


def _payload(obs_horizon: int, size: int = 96) -> dict[str, Any]:
    """A fixed, contract-valid observation: [To, H, W, 3] uint8 and [To, 2] raw pixels."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(obs_horizon, size, size, 3), dtype=np.uint8)
    state = rng.random((obs_horizon, 2)) * 512.0
    return {"image": image.tolist(), "state": state.tolist()}


def _one_request(port: int, body: bytes, headers: dict[str, str]) -> tuple[float, float]:
    """One POST /chunk. Returns (client round trip ms, server-reported ms)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        start = time.perf_counter()
        conn.request("POST", "/chunk", body=body, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read())
        elapsed = (time.perf_counter() - start) * 1000.0
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {payload}")
        return elapsed, float(payload["latency_ms"])
    finally:
        conn.close()


def _measure(
    port: int, body: bytes, headers: dict[str, str], requests: int, workers: int
) -> dict[str, Any]:
    client_ms: list[float] = []
    server_ms: list[float] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_request, port, body, headers) for _ in range(requests)]
        for future in futures:
            client, server = future.result()
            client_ms.append(client)
            server_ms.append(server)
    client_arr = np.asarray(client_ms)
    server_arr = np.asarray(server_ms)
    return {
        "requests": requests,
        "concurrency": workers,
        "client_mean_ms": float(client_arr.mean()),
        "client_p50_ms": float(np.percentile(client_arr, 50)),
        "client_p95_ms": float(np.percentile(client_arr, 95)),
        "client_max_ms": float(client_arr.max()),
        "server_mean_ms": float(server_arr.mean()),
        "server_p50_ms": float(np.percentile(server_arr, 50)),
        "server_p95_ms": float(np.percentile(server_arr, 95)),
        "http_overhead_p50_ms": float(np.percentile(client_arr - server_arr, 50)),
        "requests_per_s": float(1000.0 * len(client_ms) / client_arr.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="hf:lerobot/diffusion_pusht")
    parser.add_argument(
        "--steps", type=int, default=4, help="DDIM steps per chunk (N=4 fits 100 ms)"
    )
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrent-requests", type=int, default=25)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", type=Path, default=Path("results/serve.jsonl"))
    args = parser.parse_args()

    server = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "scripts/serve.py",
            "--ckpt",
            args.ckpt,
            "--port",
            str(args.port),
            "--steps",
            str(args.steps),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    try:
        assert server.stdout is not None
        line = server.stdout.readline()
        match = re.search(r"http://[^:]+:(\d+)", line)
        if match:
            args.port = int(match.group(1))
        _wait_ready(args.port)
        print(f"[serve] ready on port {args.port}: {line.strip()}", flush=True)

        body = json.dumps(_payload(2)).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        # Warm up the way the latency protocol does (docs/05 section 1): the first request to a
        # fresh server pays CUDA context creation and kernel loading, which is a startup cost,
        # not a serving cost. Without this the p95 hides a ~1 s first call and the max looks
        # like a tail when it is really a cold start.
        cold_ms, _ = _one_request(args.port, body, headers)
        print(f"[serve] cold start (warm-up request): {cold_ms:.1f} ms", flush=True)
        for _ in range(2):
            _one_request(args.port, body, headers)
        for workers in args.concurrency:
            requests = args.requests if workers == 1 else args.concurrent_requests
            result = _measure(args.port, body, headers, requests, workers)
            rows.append(
                {
                    "kind": "load",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    "steps": args.steps,
                    "payload_kb": len(body) / 1024.0,
                    **result,
                }
            )
            print(
                f"[serve] c={workers} n={requests}: client p50 {result['client_p50_ms']:6.1f} ms "
                f"p95 {result['client_p95_ms']:6.1f} ms | server p50 "
                f"{result['server_p50_ms']:6.1f} ms | http overhead "
                f"{result['http_overhead_p50_ms']:5.2f} ms | {result['requests_per_s']:5.2f} req/s",
                flush=True,
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\n[serve] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
