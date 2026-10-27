# 05. Inference engineering and deployment

Tags: [V] verified, [S] secondary, [U] unverified. Links: [02](02_environment_setup.md) [04](04_eval_and_experiments.md)

## 1. Inference optimization (E3, E8, E9)

**Since 2026-10-27 (D11) the ladder runs on the pretrained checkpoint**, not on a
from-scratch model: 262.7 M parameters in fp32, DDIM at N=10 as the operating point, batch 1.
That changes what the ladder is for. The old rationale -- launch overhead dominating at batch 1
with a 10 M model -- was true but academic, because the policy was not good enough to deploy.
A 65 % policy of this size spends real time in the network itself, so eager inference is the
number that decides whether a 100 ms control period is even reachable, and the compile/graph
rows answer whether the remaining gap is worth a 25-60 s compile and a CUDA graph capture.
Measure to confirm or refute; do not assume the from-scratch ratios carry over.

Benchmark protocol (all rows): batch 1, 50 warmup calls, 500 timed calls, `torch.cuda.synchronize()` around timing, report mean and p95, same input every call, report GPU clocks state (`nvidia-smi -q -d CLOCK`), single process, nothing else on the GPU.

Order of work, each with before/after in `results/latency.jsonl`:
1. Eager baseline, split into encoder, per-step network, solver loop, host-device sync (`torch.profiler`, then Nsight Systems; Nsight availability [U]).
2. N sweep (E3).
3. `torch.compile` modes `default`, `reduce-overhead`, `max-autotune`; record compile time and first-call cost.
4. CUDA graph of the whole N-step loop. Requirements: static shapes, static input buffers, warmup before capture, RNG handled outside the graph for the initial noise (pass noise in as a static buffer).
5. One Triton kernel, only after profiling. Candidates: (a) fused FiLM/AdaLN modulation plus norm, (b) fused Euler update plus time-embedding plus denormalization. Pass criteria: equivalence test vs PyTorch reference (fp32 `atol=1e-5`, bf16 `atol=1e-2`, [proposed]), `triton.testing.do_bench` vs eager and vs `torch.compile`, honest write-up if it loses. Known risk: assembler availability on cu13 builds ([02](02_environment_setup.md) pitfall 2).
6. Precision (E8): bf16 vs fp16; fp8/int8 only if the hardware path exists [U], and always report success-rate cost.
7. Optional (D6): same kernel as CUDA C++ extension (gate G6 must pass first).

**Compile modes are NOT RUN for the pretrained path (2026-10-27).** `torch.compile` fails with
`InvalidCxxCompiler: Compiler: cl is not found`: the checkpoint's DDIM scheduler creates host
timesteps inside the sampled function, Inductor sees a CPU partition and asks MSVC to codegen
it. This host has no MSVC build tools, and the failure is reproducible outside this project
(`torch.compile(lambda x: x * 2)` fails on a CPU tensor, works on a CUDA tensor), so it is an
environment limitation rather than a property of the policy. `scripts/latency_ladder.sh` tries
each mode, prints NOT RUN when it fails, and never writes a row for it. Two fixes to try before
calling it closed: pre-compute the DDIM schedule once per N so the hot path is CUDA-only (this
is also the right real-time-engineering move, and it needs a hand-rolled sampler loop whose
correctness must then be checked against the library sampler), or install MSVC and re-run.

Expected gains are not stated here; they are results. Measured 2026-10-27 on the pretrained
checkpoint (262.7 M parameters fp32, batch 1, single process, idle GPU, DDIM):

| N | eager mean | cudagraph mean | graph gain |
|---|---|---|---|
| 1 | 20.2 ms | 17.5 ms | 1.16x |
| 4 | 73.5 ms | 66.8 ms | 1.10x |
| 10 | 190.5 ms | 181.0 ms | 1.05x |
| 16 | 309.4 ms | 304.0 ms | 1.02x |

The graph gain shrinks as the model gets bigger, which is the opposite of the ablation result
(4.7x at N=10 on an 11 M model) and the honest conclusion of this section: **with a policy that
is actually good, CUDA graphs are a few percent, not a multiple.** What remains true is that
latency is linear in N, so the step count -- not the launch path -- is the dial that decides
whether a 100 ms period is reachable. At N=10 the pretrained policy is 1.9x over a 100 ms
budget, and E3 shows only N <= 4 fits, at the point where coverage has just plateaued.

## 2. Deployment (E10 to E12)

Architecture: two nodes. `sim_bridge` steps the env at a fixed rate (default 10 Hz for PushT, 20 Hz for robosuite, matching dataset control freq [U confirm]) and publishes observations; `policy_node` subscribes and publishes action chunks. Distro and container plan: [02](02_environment_setup.md) section 6.

Messages (keep simple): observation as `sensor_msgs/Image` plus `std_msgs/Float32MultiArray` for state, with `header.stamp` set by the bridge; action chunk as `std_msgs/Float32MultiArray` with shape metadata in a `MultiArrayLayout` and a `chunk_id` and `obs_stamp` carried in a small custom message if layout is awkward [decide at M7]. QoS: reliable, keep-last depth 1 for observations (stale data is worse than dropped data), reliable depth 2 for action chunks. Justify any change in an ADR note.

Fixed-rate loop: a timer at the control rate; each tick pops the next action from a queue; if the queue is empty, hold the last action and log a deadline miss. Observability: per-tick timestamps, queue depth, deadline misses, inference time, written to JSONL; one latency histogram figure.

Async chunking (E12): policy computes chunk k+1 while executing chunk k, triggered when remaining queued actions fall below a threshold. Strategies: (1) naive replace, (2) temporal ensembling with exponential weights, (3) inpainting-style blending where the already-committed actions condition the new chunk (the "real-time chunking" line of work; current literature not yet reviewed, [U], do a short survey at M7 and pick one primary plus naive as baseline).

Latency injection: wrapper delays observation delivery or action application by a configured mean plus jitter (0, 50, 100, 150, 200 ms; jitter distribution recorded). Disturbances: apply a pose perturbation to the object (PushT) or end-effector (robosuite) at a random step in a fixed window, with seeded magnitude.

Fallback (record as ADR if used): plain Python fixed-rate loop with the same `Policy`, env, wrapper, and logging interfaces, no ROS.

## 3. Serving the policy locally (2026-10-27)

ROS 2 is unavailable on this host (D7), so the plain-Python fallback now includes the serving
half of the two-node design: `flowpolicy.deploy.server`, started with `just serve`.

```bash
just serve                                     # hf:lerobot/diffusion_pusht, DDIM N=10, port 8000
just serve port=8100 steps=4                   # the only step count that fits a 100 ms period
curl -s localhost:8000/health
```

Contract (standard library only; adding a web framework for one endpoint would be a
dependency this project does not need):

```
GET  /health -> {"status": "ok", "policy": "pretrained-dp", "obs_horizon": 2,
                 "pred_horizon": 16, "action_dim": 2, "steps": 10, "action_horizon": 8}
POST /chunk  <- {"image": [[[[r, g, b], ...]]],   # [To, H, W, 3] uint8, 0-255
                 "state": [[x, y], ...],          # [To, 2] raw pixels
                 "steps": 10}                     # optional
             -> {"actions": [[x, y], ...],        # [action_horizon, 2] raw pixels
                 "latency_ms": 183.4, "steps": 10}
```

Rules the implementation enforces, each one because getting it wrong produces
plausible-looking actions:

- Images are 0-255 integers in, [0, 1] floats internally; a client that sends normalized
  floats gets HTTP 400 instead of a wrong answer.
- Actions come back in raw environment pixels: no rescaling on the way out, so the client
  cannot silently disagree with the harness about units.
- Shapes are validated against `To` (2) and the `3` channels eagerly.
- Errors are HTTP 400 with `{"error": "<ExceptionType>: <message>"}`.
- `latency_ms` is measured in-process around `predict_chunk`, with a `torch.cuda.synchronize()`
  on both sides, so it is comparable to the E9 protocol rather than to a network round trip.

Honest limits: one process, one policy, no batching, no auth, no TLS, and no queue -- it is a
serving boundary for measurement and for clients on the same machine, not a production
scheduler. The test (`tests/unit/test_deploy_server.py`) runs the contract over a real socket
with a stub policy on CPU, so it is in CI; the endpoint itself has never been load-tested.

## 4. Acceptance criteria

- [x] `results/latency.jsonl` has a row per ladder step with mean, p95, N and mode
  (16 rows: eager and cudagraph N sweeps, three compile modes at N=10), re-run
  2026-10-27 on the pretrained checkpoint. The from-scratch ladder is kept under
  `results/ablation/latency.jsonl`.
- [x] Triton kernel passes equivalence test and has a benchmark vs compile, win or
  lose. It wins vs eager (2.2-2.4x) and **loses vs compile** (0.76-0.83x);
  `results/ablation/kernel.jsonl`. It was written for this repo's own Euler loop and
  was not ported to the pretrained model: that path is third-party code, and fusing it
  would mean owning it.
- [x] Control loop logs deadline misses per episode against the 100 ms period;
  histogram regenerates from the log (`results/figures/e10_latency_hist.png`).
- [x] E10 runs 3 eval seeds x 10 episodes per latency level on the pretrained
  checkpoint (one policy, so no training seed); measured results are in section 3 of
  [04](04_eval_and_experiments.md).

