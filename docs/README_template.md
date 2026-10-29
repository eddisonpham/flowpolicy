# PolicyBench

**A deployment-cost measurement rig for a visuomotor policy: what one action chunk costs,
whether that fits a control period, and what breaks when it does not.** Every number here is
computed from raw per-episode files, printed with an interval, and paired with the condition
that would make it false.

## Why this exists

Most policies are published with one number, a success rate. That number says nothing about
whether the policy can drive a robot's control loop, and the four questions that decide it are
usually answered by guesswork, or by a latency figure measured once, on other hardware, with a
harness nobody can re-run. This repository answers all four for one concrete policy, on one
consumer GPU, with every number generated from raw per-episode files:

1. **How few sampler steps still work?** (E3)
2. **What does one action chunk cost at batch 1?** (E9)
3. **Does that cost fit a control period, and what breaks when observations arrive late?** (E10)
4. **Does a faster replan rate help or destroy it?** (E12)

**What it is not.** Not a new policy, method or hardware result. The policy is borrowed --
`lerobot/diffusion_pusht` (Apache-2.0, 262.7 M parameters, Diffusion Policy trained on
gym-pusht) -- and we claim no credit for it. What is ours is the harness, the sampler
engineering that makes it measurable, the control loop, the serving boundary and the numbers.
That division is deliberate: a stack that measures a policy scoring 5 % cannot tell you anything
about latency, because the policy is failing for reasons unrelated to time. This repo learned
that the hard way -- the from-scratch policies it started with reach 4-7 % success (kept as the
ablation below), which is a floor effect, not a result.

All work ran on one Windows 11 laptop (RTX 5060 Laptop, 8 GB, driver 617.42), Python 3.12,
torch 2.14.1+cu130. GPU, simulator and deployment runs are local only; CI is CPU-only.
Pretrained rows: 2026-10-27/28. Ablation rows: 2026-10-19 to 2026-10-24.

## What this software is, concretely

| Component | What it does | Where |
|---|---|---|
| Policy interface | `PolicyLike` protocol (`obs_horizon`, `pred_horizon`, `action_dim`, `predict_chunk`, `num_parameters`); the HF checkpoint and the from-scratch models both satisfy it | `src/policybench/models/` |
| Adapter | loads `lerobot/diffusion_pusht`, restores the normalization statistics the current `lerobot` release drops, guesses nothing | `src/policybench/models/pretrained.py` |
| Fast sampler | DDIM with coefficients precomputed on-device, so the denoising loop is one graph that compiles and captures; verified equal to `diffusers` | `src/policybench/models/fast_sampler.py` |
| Rollout harness | parallel episode workers, fixed eval seeds, per-episode JSONL, bootstrap intervals over episodes/seeds | `src/policybench/eval/` |
| Latency protocol | batch-1, 50 warmup + 200 timed calls, eager / CUDA-graph / `torch.compile` | `src/policybench/eval/latency.py` |
| Control loop | fixed period, action queue, injected observation latency with jitter, deadline-miss accounting, naive vs temporal ensembling | `src/policybench/deploy/control_loop.py` |
| Serving boundary | stdlib-only HTTP endpoint with a documented JSON contract, so the measured latency is something a client can actually use | `src/policybench/deploy/server.py` |
| Generators | every table, figure and this README are rebuilt from `results/*.jsonl`; CI fails if they go stale | `scripts/` |

## Stack

Python 3.12, PyTorch 2.14.1+cu130, torchvision 0.29.1, lerobot 0.6.1 (inference only),
diffusers 0.42.1, gym-pusht 0.1.8 (PushT), Hydra + dataclasses for config, `uv` + `just` for
environment and recipes, ruff + mypy `--strict` + pytest for the gate, GitHub Actions
(CPU-only) for CI, matplotlib for figures, Triton (ablation only, where it lost to
`torch.compile`), and `torch.cuda.CUDAGraph` / `torch.compile` for the latency ladder.

## What it buys you, measured against the obvious baseline

"Faster" means nothing without a reference. The reference here is not a straw man: it is the
checkpoint's own published configuration (DDPM, 100 steps) and the stock `lerobot` ->
`diffusers` sampling path, on the same GPU, the same episodes and the same success metric.

{{E14_TABLE}}

![E14 throughput](results/figures/e14_stack_throughput.png)

Read it as four concrete decisions the stack makes possible, each of which a user of this
checkpoint would otherwise have to guess:

- **The published configuration is not deployable, and the stack says by how much.** 100 DDPM
  steps per chunk costs seconds, not milliseconds: an episode takes two orders of magnitude
  longer than at the operating point this repo selects, which is why nobody can evaluate this
  checkpoint by brute force.
- **The step count, not the framework, is the dial.** Quality plateaus well below the
  checkpoint's training step count, and the plateau is an order of magnitude cheaper per chunk
  than the published setup.
- **Launch overhead is not the bottleneck, so neither compile nor graphs rescue a deadline.**
  Both help by a few percent, not a multiple, because the 262.7 M-parameter U-Net is
  compute-bound; the honest conclusion is that the only way into a 100 ms period is fewer
  sampler steps.
- **Parallelism does not help either.** Two rollout workers make each episode slower, not
  faster, because one process already saturates an 8 GB laptop GPU. The harness stays
  multi-worker for wall-clock isolation and per-worker seeding, not for speed.

The equivalence block is the honesty check on the sampler swap: action chunks from the two loops
agree to floating-point rounding (0.02 px on a 512 px action space, pinned by
`tests/gpu/test_pretrained_sampler_equivalence.py`), but over 20 paired episodes in a
contact-rich simulator that rounding moves individual outcomes. Both columns are printed so the
difference in success counts is visible rather than averaged away.

A rendered episode with the loop's own timings on the frame is in
`results/figures/demo_rollout.gif` (`just demo` regenerates it).

## The results in four lines

{{HEADLINE_TABLE}}

Each row is expanded below with its table, figure, interval and the condition under which
it would be false (`docs/04_eval_and_experiments.md` section 3).

## Start here

```bash
uv sync --python 3.12       # or: just setup      -> the locked environment
just setup-pretrained       # optional backend for the borrowed policy (lerobot + diffusers)
just test                   # ruff, mypy strict, CPU unit tests (what CI runs)
just eval-e0                # the acceptance gate: does the harness reproduce the model card?
just tables && just figures && just readme
```

`just eval-e0` is the one to run first. It puts the pretrained checkpoint through our
gym-pusht wrapper, our harness and our success definition (max coverage >= 95 %). If it
lands near zero, the bug is ours -- the checkpoint ships its own normalization statistics
and the current `lerobot` release drops them on load -- and everything downstream would be
measuring our plumbing rather than the policy.

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just).
`just doctor` checks the environment before you spend GPU time.

## Results

### E0: the pretrained checkpoint through our harness

This is a validation, not a result we own. It is also the project's reproduction check, and
after 2026-10-29 a partial one: the model card reports 65.4 % success and 0.955 mean
`max_reward` over 500 episodes -- which gym-pusht defines as `coverage / 0.95`, so the coverage
it implies is 0.907.

{{E0_TABLE}}

![E0 validation](results/figures/e0_validation.png)

`ddim@10` is the operating point every later experiment uses; `ddpm@100` is the checkpoint's
own scheduler and step count, i.e. the published setup. Two caveats: the intervals are over
the 50 episodes, because PushT has exactly one pretrained checkpoint and there is no training
seed to resample; and these two rows ran with two rollout workers sharing the GPU, so their
latency column is roughly 2x the clean single-process numbers in E9.

On the checkpoint's own seed block the comparison can be paired, and at 200 episodes it stops
flattering us: 104/200 against the checkpoint's 138/200, exact McNemar p < 0.001. Coverage
reproduces (0.895 [0.872, 0.917] against 0.915), so what separates the runs is a ~2 pp coverage
difference meeting a hard 0.95 success threshold -- 64 episodes land in [0.90, 0.95), where this
repo crosses it zero times and the checkpoint 42. That is recorded as open, not explained
(`docs/04` section 4).

Reading it: the checkpoint's own 100-step DDPM sampling is over ten times slower per chunk
than 10-step DDIM and no better here (compare the two rows), which is the first hint that this
policy's cost profile is not where a small model's is.

### E3: how far can the sampler be truncated?

Same checkpoint, same 50 paired eval seeds per N, only the number of DDIM steps changes.
DDIM rather than the checkpoint's DDPM because a DDPM step injects noise: truncating that
scheduler would measure the scheduler, not the policy.

{{E3_TABLE}}

![E3 success](results/figures/e3_success_vs_steps.png)
![E3 latency](results/figures/e3_latency_vs_steps.png)

Reading it: one step is catastrophic -- its coverage falls below what the from-scratch ablation
models reach -- two is still poor, four reaches the plateau, and past four every interval
overlaps. Chunk latency grows linearly in N, so the step count is the dial that decides whether
a control period is reachable at all. Measured under two-worker contention; the clean
single-process numbers are E9.

An independent block of 50 fresh eval seeds re-runs the ladder, because "the plateau is real"
is a claim about the policy and not about one sample of episodes:

{{E3_REPLICATION_TABLE}}

Reading it: coverage replicates to within a couple of hundredths at every step count, so the
plateau and the two-step collapse are properties of the policy. Success does not replicate --
the block-to-block swing is larger than any difference between step counts above two -- which
is what the intervals in the table above were already saying, now demonstrated rather than
argued.

### E9: batch-1 latency ladder

{{E9_TABLE}}

Reading it: the CUDA-graph column is a few percent off eager here, where the same comparison
on the 11 M ablation model was a multiple -- this policy is compute-bound rather than
launch-bound, so launch-overhead work does not transfer. This ladder runs on the stock
`lerobot` -> `diffusers` sampler, the same path E0 and E3 were measured with. The
`torch.compile` column lives in E14 rather than here because compiling the stock path needed
a C++ CPU partition (MSVC) that the from-scratch models never touched: this repo's DDIM
sampler in `src/policybench/models/fast_sampler.py` removes that partition, which is what makes
`torch.compile` and `CUDAGraph` measurable on this checkpoint at all. `docs/05` section 1 has
the mechanism.

### E8: does half precision buy a control period?

{{E8_TABLE}}

Reading it: half precision is cheaper per chunk and much smaller in VRAM, and at N=10 it is
still over the period, so the envelope does not move -- fewer sampler steps remain the only
knob that changes what fits. On the policy side all three arms land inside each other's
intervals at 40 paired episodes, which is a no-gross-regression check, not an equality proof.

### E10: injected observation latency in the fixed-rate loop

{{E10_TABLE}}

`naive` replans when the 8-step action queue empties; the `-r2` rows replan every 2 steps so
four chunks overlap, which is the only regime where temporal ensembling differs from naive
replacement. The loop runs in virtual time; chunk inference is timed for real, and every
chunk that exceeds the 100 ms period is logged as a deadline miss. `--paced` instead sleeps
to the tick, which turns that counter into measured wall-clock lateness; `--strategy blend`
adds the third arm (cross-fading the queue into a new chunk) next to naive replacement and
temporal ensembling.

![E10 success](results/figures/e10_success_vs_latency.png)
![E10 latency histogram](results/figures/e10_latency_hist.png)

Reading it: injected latency costs success and the naive intervals overlap at 30 episodes per
row, so read it as a trend; coverage barely moves, so the policy still reaches the goal and
just converts the last few percent less often. The loud column is the deadline-miss count: at
N=10 one chunk needs about twice the control period, so the loop misses on most chunks. The
bottleneck is the policy's own compute, not the network, which is what E9 and E3 say from two
other directions.

The `-r2` rows are the replay-rate experiment: replanning every 2 steps overlaps four chunks,
and at that rate naive replacement collapses -- its intervals do not come near the naive rows
at the same injected latency -- while uniform temporal ensembling over the same chunks recovers
to roughly the naive level, at four times the chunk count and a much heavier deadline-miss
penalty. Replanning fast is not free: without ensembling it is actively harmful.

The third arm (`blend`) cross-fades the queue into each new chunk instead of replacing or
averaging it. It holds coverage near the ensembling rows but not success, which separates the
two effects: smoothing the splice helps the policy reach the goal, while averaging several
chunks is what converts that into successes.

### Real-time pacing (does the loop actually keep the period?)

E10/E12 run in virtual time, so a deadline miss is a counter. `just bench-paced` makes the loop
sleep to the tick and reports how far behind the wall clock it falls:

{{PACED_TABLE}}

Reading it: at the operating point the loop holds the 100 ms period and the wall clock is
exactly the step count times the period. At N=10 it does not: once a chunk overruns, the loop
never catches up, so the lateness is not a spike that recovers but a floor it carries for the
rest of the episode.

## Serving the policy over a process boundary

`just serve` puts the same policy behind a stdlib-only HTTP endpoint with a documented JSON
contract (`docs/05` section 3), so the measured latency is something a client can actually
consume. `just bench-serve` loads that endpoint and reports what a client pays:

{{SERVE_TABLE}}

Reading it: the GPU chunk time is the bulk of the round trip, and the JSON/socket overhead is
a fixed slice of it rather than a rounding error -- which is why the serving path is measured
rather than assumed. Two clients do not run in parallel; they queue on the GPU.

### E15: the same harness, a second policy family

Every measurement above is about one checkpoint, and a harness that only ever held one
policy proves nothing about the harness. So the same interface was pointed at
`lerobot/vqbet_pusht` (Apache-2.0): a behaviour transformer over discretized action tokens
with 5 observation steps and a 5-action chunk -- no denoising sampler, no step count to
truncate, 37.5 M parameters instead of 262.7 M. It needed one adapter
(`models/pretrained_vqbet.py`, which restores that checkpoint's normalization the same way
the diffusion adapter does) and a dispatch on the checkpoint's own `config.json` `type`.
The rollout harness, the latency protocol, the control loop and every generator were not
touched.

{{GENERALITY_TABLE}}

Reading it: the harness reproduces the second checkpoint's published band (54 % against
57 %) and prices it at a fifth of the diffusion policy's cost per control step, without any
code knowing which family it was holding. Both checkpoints ship a per-episode eval block, so
both comparisons are paired rather than hand-waved, and the bullets under the table are those
tests, each on 200 shared seeds. VQ-BeT is not distinguished from its own block (108/200
against 113/200, p = 0.68), and it repeats exactly on a disjoint block of 200 fresh seeds
(108/200 again, Fisher p = 1.00). Diffusion gives the opposite answer at 200 seeds: 104/200
against the checkpoint's 138/200, p < 0.001. Fifty seeds had said "not distinguished"; that
was a power problem, not a result. Coverage is the column that separates them (0.895 against
0.915) and also the likely link: 64 episodes land in [0.90, 0.95), where this repo crosses the
0.95 success threshold 0 times and the checkpoint 42 times. The point of the experiment is the
seam, not the checkpoint -- and the seam currently carries one unresolved gap.

## Ablation: what we tried first

Before the pivot this repository trained three policies from scratch on PushT -- a
conditional flow-matching 1-D U-Net, an epsilon-prediction DDPM, and an ACT-style regression
chunk head -- for 12000 steps each at batch 64. They reach 4-7 % success, which is a floor
effect: 6-11 successes in 150 episodes cannot order two policies, and a deployment
experiment cannot detect a policy degrading from there. Those rows are kept because that
negative result is the reason this stack now runs a borrowed policy. Raw rows:
`results/ablation/`.

{{E1_ABLATION_TABLE}}

![Ablation success](results/figures/e1_ablation_success.png)

Training cost, overfit sanity checks and the Triton kernel that this repo wrote for its own
Euler loop (it loses to `torch.compile`, 0.76-0.83x, and is reported as such):

{{M2_TABLE}}

{{OVERFIT_TABLE}}

{{KERNEL_TABLE}}

## Run anything

| Recipe | What it does | Cost |
|---|---|---|
| `just setup` | locked env from `uv.lock` | minutes |
| `just setup-pretrained` | adds `lerobot` 0.6.1 outside the lock (`diffusers` is already locked in `dev`) | ~1 min, ~2 GB |
| `just doctor` | environment check: versions, CUDA, backend, checkpoint availability | seconds |
| `just test` | ruff + mypy strict + CPU unit tests (what CI runs) | ~10 s |
| `just smoke` | 20 steps of real training on CPU + resume equivalence (the CI smoke job) | ~5 s |
| `just eval-e0` | the acceptance gate: 100 episodes, two samplers | ~50 min |
| `just eval-e3` | DDIM step sweep, 300 episodes | ~25 min |
| `just eval-vqbet` | E15: a second published policy family through the same harness, 50 episodes | ~2 min |
| `just latency-ladder` | batch-1 latency ladder, 12 rows | ~15 min |
| `just eval-e10` | latency injection + async chunking, 270 episodes | ~75 min |
| `just bench-stack` | E14: published vs stock vs fast sampler, latency + episodes/hour | ~25 min |
| `just bench-precision` | E8: fp32 vs bf16 vs fp16, latency ladder + 40 paired episodes each | ~20 min |
| `just eval-vqbet` | E15: 200 episodes of a second policy family through the same harness | ~7 min |
| `just bench-serve` | load-test the HTTP boundary (client vs server latency) | ~2 min |
| `just bench-paced 4` | wall-clock paced loop at one step count (12 episodes) | ~7 min |
| `just demo` | render one episode to an annotated GIF (`results/figures/demo_rollout.gif`) | ~1 min |
| `just report` | one-screen summary of everything measured so far | seconds |
| `just serve` | serve the policy over local HTTP (`docs/05` section 4) | seconds to start |
| `just tables` / `just figures` / `just readme` | regenerate outputs from `results/*.jsonl` | seconds |
| `just train-e1` / `just eval-e1` | ablation only: retrain and re-measure the from-scratch policies | ~3 GPU-h |
| `just gates` | week-1 host gates G1-G9, reporting NOT RUN where the host lacks the prerequisite | minutes |

Costs are wall-clock on this laptop GPU with two workers; the eval recipes are resumable at
row granularity (each invocation prunes and rewrites only its own rows). `just --list` shows
the rest: `lint`, `types`, `format`, `hooks`, `test-local`, `data`, `memory`, `kernel`,
`overfit`.

## What to read, in what order

1. `docs/04_eval_and_experiments.md` -- protocol, experiment matrix, claims with their
   "false if" conditions, and the dated deviations log (the honest part).
2. `docs/05_inference_and_deployment.md` -- the latency ladder, the control loop, async
   chunking, and why the sampler had to lose its host-side scalars before compile or CUDA
   graphs would apply at all.
3. `docs/02_environment_setup.md` -- versions and the two-environment story for the backend
   (section 3.4).
4. `docs/01_architecture.md` -- the policy interface, and why the substrate swap touched
   almost nothing.
5. `docs/06_decisions_and_risks.md`, `docs/STATE.md` -- decisions D1-D11 and the live state.
6. `docs/MODEL_CARD.md` -- what the borrowed policy is, what the stack provides, and the
   metrics with intervals that a user of either needs.

## Project layout

```
conf/                  Hydra configs + experiment files (ablation training)
src/policybench/
  models/pretrained.py adapter: HF checkpoint behind the Policy interface
  models/fast_sampler.py DDIM loop with on-device coefficients (compiles, captures)
  models/policies.py   PolicyLike protocol + the from-scratch policies
  eval/                parallel rollout harness, bootstrap CIs, latency protocol
  eval/report.py       the computed headline summary shared by `just report` and the README
  deploy/control_loop.py fixed-rate loop, latency injection, temporal ensembling
  deploy/server.py     local HTTP serving boundary (stdlib only)
scripts/               eval runners (E0/E3/E9/E10), bench_stack (E14), demo, tables, figures,
                       doctor, report, serve, gates
tests/                 unit (CI), gpu/integration (local)
docs/                  the spec (00-07), STATE, CHANGELOG, MODEL_CARD
results/               raw JSONL, generated tables and figures, ablation/
```

## Limitations

- **The policy is not ours.** `lerobot/diffusion_pusht` was trained by the LeRobot team
  (175k steps, Apache-2.0). Nothing here is a result about Diffusion Policy itself.
- **One checkpoint, so episode-level intervals.** E0, E3, E9 and E10 cannot report
  seed-to-seed variance, and two 50-episode samples of the same configuration (E0's eval seeds
  against E3's N=10 row) differ by about ten percentage points. That is the resolution limit of
  every table here, and it is why they print intervals rather than single numbers.
- **`lerobot` is installed outside the lockfile.** lerobot 0.6.1 requires `torch < 2.12`;
  this project pins `torch 2.14.1+cu130`. `just setup-pretrained` installs it `--no-deps` on
  top of the sync and `just setup` prunes it. Verified working, not supported. `diffusers`,
  which the sampler-equivalence unit test imports, is in the locked `dev` group instead, so the
  unit gate does not depend on this step.
- **The checkpoint's normalization is applied by our code**, because current lerobot moved
  normalization into a processor pipeline this 2024 checkpoint does not ship. E0 is the gate
  that says the mapping is right.
- **Simulation only, virtual-time loop.** E10 injects latency as observation staleness in
  control ticks and does not wall-clock pace the loop; inference is timed for real and
  deadline misses are logged, but this is not a real-time scheduler.
- Robomimic, ROS 2, Docker and `nvcc` are unavailable on this host, so the documented
  fallbacks are in force (PushT only, plain-Python loop, no CUDA C++ stretch). G8 and G9 are
  NOT RUN and never reported as passes. E2, E4-E8 and E11 were not run. No W&B; JSONL only.
- The demo artifact is a rendered GIF of one episode, not a narrated video.
- E14's throughput arms use a handful of episodes per arm to keep the comparison affordable;
  they are there to size the cost of an experiment, and the success/coverage columns are
  printed so a fast arm cannot be mistaken for an equally good policy.

The full deviation log, including everything cut and why, is `docs/04` section 4.

## License

Code: not decided by the owner yet. The pretrained checkpoint `lerobot/diffusion_pusht` is
Apache-2.0 (model card, accessed 2026-10-27). Dataset `lerobot/pusht` is MIT and derives from
the Diffusion Policy PushT demonstrations.
