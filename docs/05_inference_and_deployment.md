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

**Compile modes: resolved 2026-10-28, five days after they were first blocked.** On 2026-10-27
every compile mode failed with `InvalidCxxCompiler: Compiler: cl is not found`, which was
recorded as an environment limitation (no MSVC). The diagnosis was right and the conclusion was
wrong: the missing compiler was a *symptom*. `diffusers` keeps `alphas_cumprod` on the host and
indexes it with a Python int per step, so the denoising loop contained 10 host scalars mixed
into CUDA tensor math. That produces (a) one host/device synchronization per step and (b) a CPU
partition in the Inductor graph, which is why a C++ compiler was wanted at all.

`src/policybench/models/fast_sampler.py` fixes the cause: the DDIM coefficients are precomputed
once on the sampling device and the loop is tensor-only. It is the same sampler -- equality with
`diffusers` is checked at five step counts and two beta schedules in
`tests/unit/test_fast_sampler.py` -- and it is now a single graph, so `torch.compile` and
`CUDAGraph` both apply without MSVC. Two consequences worth keeping:

- **The `torch.compile` rows exist and are reported** (`results/tables/e14_stack.md`). They are
  worth roughly a tenth, not a multiple, and so are CUDA graphs. Launch overhead was never the
  bottleneck at 262.7 M parameters; the network is. The from-scratch model's 4.7x graph / 5.8x
  compile result was a property of an 11 M model, not of the technique.
- **The real fix for a deadline is fewer sampler steps.** E3 plus E9/E14 say the only knob that
  moves a chunk from missing a 100 ms period to fitting inside it is N.

Measured 2026-10-27 on the pretrained checkpoint (262.7 M parameters fp32, batch 1, single
process, idle GPU, DDIM) through the stock sampler; the fast sampler with graphs and compile is
in E14:

| N | eager mean | cudagraph mean | graph gain |
|---|---|---|---|
| 1 | 20.2 ms | 17.5 ms | 1.16x |
| 4 | 73.5 ms | 66.8 ms | 1.10x |
| 10 | 190.5 ms | 181.0 ms | 1.05x |
| 16 | 309.4 ms | 304.0 ms | 1.02x |

**Precision (E8), measured 2026-10-28 on the same checkpoint.** Half precision is real but not
decisive: the best bf16 row is 55.6 ms at N=4 against 73.4 ms in fp32 (1.32x, CUDA graph on
both) and 138.4 ms against 194.4 ms at N=10 (1.41x), and peak VRAM drops from 1189 MiB to
710 MiB. fp16 matches bf16 within noise. At N=10 the best half-precision row is still 1.4x
over a 100 ms period, so precision does **not** move the deployment envelope -- only N does.
On quality the three arms are indistinguishable over 40 paired episodes each (success 35.0 %
fp32, 50.0 % bf16, 45.0 % fp16, intervals overlapping; coverage 0.900 / 0.885 / 0.901), which
is a no-gross-regression result, not an equality proof: 40 episodes cannot resolve the 2 pp
the pre-registered hypothesis named. Implementation notes that cost real debugging time:
`--dtype` is threaded through the adapter and every runner, it is allowed only on the fast
sampler (the stock `diffusers` loop does its arithmetic on host fp32 schedule tensors, which
this repo has not verified under a cast), and lerobot's sinusoidal timestep embedding builds
its own fp32 tensor, so it is kept in fp32 and its output is cast at the module boundary --
without that hook the unet raises `mat1 and mat2 must have the same dtype`.

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

**Third strategy, implemented 2026-10-28 (M9.7) as a cross-fade, and named accordingly.**
The library version of "inpainting-style" conditions the *sampler* on the actions already
committed, which needs a policy that accepts action conditioning; this borrowed checkpoint is
unconditional at inference, so there is no honest way to do that without retraining. What is
implementable is a receding-horizon cross-fade: on every replan the queue keeps its length and
the first `blend_steps` actions ramp from the old chunk into the new one (weights 1/k, 2/k, ...
1), so a large disagreement between chunks costs a smooth few ticks instead of the jump that
naive replacement takes. `LoopConfig(strategy="blend", blend_steps=4)`, `--strategy blend[-r2]`
on the runner, arithmetic pinned by `tests/unit/test_blend_chunks.py`. It is reported as
"blend", never as "inpainting", and the literature survey that would justify a stronger claim
is still `[U]`.

Measured 2026-10-28 (replan every 2 steps, 30 episodes per row, `results/deploy.jsonl`): the
cross-fade holds coverage near the temporal-ensembling rows (0.875 at 100 ms, 0.864 at 200 ms,
against naive replacement's 0.749 / 0.553) but does not recover success (26.7 % at 100 ms and
10.0 % at 200 ms, against ensembling's 53.3 % / 43.3 %). Smoothing the splice therefore helps
the policy reach the goal without converting, which is the useful separation: what buys
success is averaging several overlapping chunks, not hiding the discontinuity between two.

**Wall-clock pacing (M9.6).** E10/E12 run in virtual time: inference is timed for real, but
the loop never sleeps, so "deadline miss" is a counter. `--paced` sleeps to the next tick and
records, per tick, how far behind the wall clock the loop is (`late_ticks`, `lateness_mean_ms`,
`lateness_p95_ms`, `lateness_max_ms`). This is the difference between "a chunk would have
missed" and "the loop fell behind and stayed behind".

Measured 2026-10-28, 12 episodes per step count, 100 ms period, no injected latency: at N=4
the loop keeps time (0.2 late ticks per episode, mean lateness 1.1 ms) and at N=10 it does not
(52.5 late ticks, late on every episode, 65.8 ms mean and 102.9 ms p95 lateness). Pacing also
lets the miss reach the policy outcome -- 3/12 successes at N=10 against 10/12 at N=4, coverage
unchanged -- which virtual time could not show; the intervals only just overlap at 12 episodes,
so it is a direction, not a separated result (`results/tables/paced_loop.md`).

Latency injection: wrapper delays observation delivery or action application by a configured mean plus jitter (0, 50, 100, 150, 200 ms; jitter distribution recorded). Disturbances: apply a pose perturbation to the object (PushT) or end-effector (robosuite) at a random step in a fixed window, with seeded magnitude.

Fallback (record as ADR if used): plain Python fixed-rate loop with the same `Policy`, env, wrapper, and logging interfaces, no ROS.

## 3. Serving the policy locally (2026-10-27)

ROS 2 is unavailable on this host (D7), so the plain-Python fallback now includes the serving
half of the two-node design: `policybench.deploy.server`, started with `just serve`.

```bash
just serve                                     # hf:lerobot/diffusion_pusht, DDIM N=10, port 8000
just serve hf:lerobot/diffusion_pusht 8100 4   # ckpt port steps: the only N that fits 100 ms
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

Measured 2026-10-28 (`just bench-serve`, N=4, one process, `scripts/bench_serve.py` spawning
the real `scripts/serve.py`):

| Clients | Requests | Client p50 | Client p95 | Server p50 | HTTP overhead p50 | Req/s |
|---|---|---|---|---|---|---|
| 1 | 50 | 91.5 ms | 93.1 ms | 74.0 ms | 17.8 ms | 11.0 |
| 2 | 25 | 171.6 ms | 196.1 ms | 154.2 ms | 18.1 ms | 5.8 |

The first request to a freshly started server costs 655 ms (CUDA context and kernel load);
the script sends three untimed warm-up requests before timing anything, the same rule the
latency protocol uses.

Two things follow. The **JSON/socket layer costs about 17 ms per request** -- roughly a quarter
of the GPU time at this step count, not a rounding error, because the observation is 2 x 96 x
96 x 3 integers and `json` spends that long turning it into text. And **two clients do not run
in parallel**: the second client's p50 nearly doubles because `ThreadingHTTPServer` accepts
both and they serialize on one GPU. Both are properties of the artifact a user runs rather
than of the policy, which is why they are measured here.

Honest limits: one process, one policy, no batching, no auth, no TLS, and no queue -- it is a
serving boundary for measurement and for clients on the same machine, not a production
scheduler. The contract test (`tests/unit/test_deploy_server.py`) runs over a real socket with
a stub policy on CPU, so it is in CI; the load numbers above are from one quiet run on this
laptop, not a benchmark under contention.

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

