# 05. Inference engineering and deployment

Tags: [V] verified, [S] secondary, [U] unverified. Links: [02](02_environment_setup.md) [04](04_eval_and_experiments.md)

## 1. Inference optimization (E3, E8, E9)

Rationale to test, not assume: at batch 1 with a 20 to 50 M model, latency is dominated by kernel launch overhead and Python dispatch, which `torch.compile` and CUDA graphs mostly remove. Measure to confirm or refute.

Benchmark protocol (all rows): batch 1, 50 warmup calls, 500 timed calls, `torch.cuda.synchronize()` around timing, report mean and p95, same input every call, report GPU clocks state (`nvidia-smi -q -d CLOCK`), single process, nothing else on the GPU.

Order of work, each with before/after in `results/latency.jsonl`:
1. Eager baseline, split into encoder, per-step network, solver loop, host-device sync (`torch.profiler`, then Nsight Systems; Nsight availability [U]).
2. N sweep (E3).
3. `torch.compile` modes `default`, `reduce-overhead`, `max-autotune`; record compile time and first-call cost.
4. CUDA graph of the whole N-step loop. Requirements: static shapes, static input buffers, warmup before capture, RNG handled outside the graph for the initial noise (pass noise in as a static buffer).
5. One Triton kernel, only after profiling. Candidates: (a) fused FiLM/AdaLN modulation plus norm, (b) fused Euler update plus time-embedding plus denormalization. Pass criteria: equivalence test vs PyTorch reference (fp32 `atol=1e-5`, bf16 `atol=1e-2`, [proposed]), `triton.testing.do_bench` vs eager and vs `torch.compile`, honest write-up if it loses. Known risk: assembler availability on cu13 builds ([02](02_environment_setup.md) pitfall 2).
6. Precision (E8): bf16 vs fp16; fp8/int8 only if the hardware path exists [U], and always report success-rate cost.
7. Optional (D6): same kernel as CUDA C++ extension (gate G6 must pass first).

Expected gains are not stated here; they are results.

## 2. Deployment (E10 to E12)

Architecture: two nodes. `sim_bridge` steps the env at a fixed rate (default 10 Hz for PushT, 20 Hz for robosuite, matching dataset control freq [U confirm]) and publishes observations; `policy_node` subscribes and publishes action chunks. Distro and container plan: [02](02_environment_setup.md) section 6.

Messages (keep simple): observation as `sensor_msgs/Image` plus `std_msgs/Float32MultiArray` for state, with `header.stamp` set by the bridge; action chunk as `std_msgs/Float32MultiArray` with shape metadata in a `MultiArrayLayout` and a `chunk_id` and `obs_stamp` carried in a small custom message if layout is awkward [decide at M7]. QoS: reliable, keep-last depth 1 for observations (stale data is worse than dropped data), reliable depth 2 for action chunks. Justify any change in an ADR note.

Fixed-rate loop: a timer at the control rate; each tick pops the next action from a queue; if the queue is empty, hold the last action and log a deadline miss. Observability: per-tick timestamps, queue depth, deadline misses, inference time, written to JSONL; one latency histogram figure.

Async chunking (E12): policy computes chunk k+1 while executing chunk k, triggered when remaining queued actions fall below a threshold. Strategies: (1) naive replace, (2) temporal ensembling with exponential weights, (3) inpainting-style blending where the already-committed actions condition the new chunk (the "real-time chunking" line of work; current literature not yet reviewed, [U], do a short survey at M7 and pick one primary plus naive as baseline).

Latency injection: wrapper delays observation delivery or action application by a configured mean plus jitter (0, 50, 100, 150, 200 ms; jitter distribution recorded). Disturbances: apply a pose perturbation to the object (PushT) or end-effector (robosuite) at a random step in a fixed window, with seeded magnitude.

Fallback (record as ADR if used): plain Python fixed-rate loop with the same `Policy`, env, wrapper, and logging interfaces, no ROS.

## 3. Acceptance criteria

- [ ] `results/latency.jsonl` has a row per ladder step with mean, p95, N, mode.
- [ ] Triton kernel passes equivalence test and has a benchmark vs compile, win or lose.
- [ ] Control loop logs deadline misses; histogram figure regenerates from the log.
- [ ] E10 runs with at least 3 seeds per latency level.
