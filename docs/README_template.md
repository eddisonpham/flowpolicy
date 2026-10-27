# FlowPolicy

An inference and deployment stack for a **pretrained** visuomotor policy on PushT. One
known-good checkpoint behind a single policy interface, a parallel rollout harness, a
batch-1 latency ladder, and a fixed-rate control loop with injected observation latency.
Every number in this README is generated from `results/*.jsonl` by the scripts in
`scripts/`; nothing is hand-typed.

The policy is `lerobot/diffusion_pusht` (Apache-2.0, 262.7 M parameters, Diffusion Policy
trained on gym-pusht). We did not train it and claim no credit for it. What is ours is the
measurement and deployment machinery around it, plus the honest answer to "what does a
policy that actually works cost to run?" -- which turns out to differ from what a model
small enough to train on a laptop suggests.

All work ran on one Windows 11 laptop (RTX 5060 Laptop, 8 GB, driver 617.42), Python 3.12,
torch 2.14.1+cu130. GPU, simulator and deployment runs are local only; CI is CPU-only.
Pretrained rows were measured 2026-10-27; ablation rows 2026-10-19 to 2026-10-24.

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

This is a validation, not a result we own. It is also the project's reproduction check: the
model card reports 65.4 % success and 0.955 mean max overlap over 500 episodes.

{{E0_TABLE}}

![E0 validation](results/figures/e0_validation.png)

`ddim@10` is the operating point every later experiment uses; `ddpm@100` is the checkpoint's
own scheduler and step count, i.e. the published setup. Two caveats: the intervals are over
the 50 episodes, because PushT has exactly one pretrained checkpoint and there is no training
seed to resample; and these two rows ran with two rollout workers sharing the GPU, so their
latency column is roughly 2x the clean single-process numbers in E9.

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

### E9: batch-1 latency ladder

{{E9_TABLE}}

Reading it: the CUDA-graph column is a few percent off eager here, where the same comparison
on the 11 M ablation model was a multiple -- this policy is compute-bound rather than
launch-bound, so launch-overhead work does not transfer. `torch.compile`, the mode with the
largest gain on the small model, is NOT RUN: Inductor needs MSVC to codegen the CPU partition
that the checkpoint's DDIM scheduler creates, and this host has no C++ build tools.
`scripts/latency_ladder.sh` reports it as NOT RUN and writes no row; two candidate fixes are in
`docs/05` section 1.

### E10: injected observation latency in the fixed-rate loop

{{E10_TABLE}}

`naive` replans when the 8-step action queue empties; the `-r2` rows replan every 2 steps so
four chunks overlap, which is the only regime where temporal ensembling differs from naive
replacement. The loop runs in virtual time; chunk inference is timed for real, and every
chunk that exceeds the 100 ms period is logged as a deadline miss.

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
| `just setup-pretrained` | adds the optional backend (lerobot 0.6.1 + diffusers) outside the lock | ~1 min, ~2 GB |
| `just doctor` | environment check: versions, CUDA, backend, checkpoint availability | seconds |
| `just test` | ruff + mypy strict + CPU unit tests (what CI runs) | ~10 s |
| `just eval-e0` | the acceptance gate: 100 episodes, two samplers | ~50 min |
| `just eval-e3` | DDIM step sweep, 300 episodes | ~25 min |
| `just latency-ladder` | batch-1 latency ladder, 12 rows | ~15 min |
| `just eval-e10` | latency injection + async chunking, 270 episodes | ~75 min |
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
   chunking, and why compile is NOT RUN.
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
src/flowpolicy/
  models/pretrained.py adapter: HF checkpoint behind the Policy interface
  models/policies.py   PolicyLike protocol + the from-scratch policies
  eval/                parallel rollout harness, bootstrap CIs, latency protocol
  eval/report.py       the computed headline summary shared by `just report` and the README
  deploy/control_loop.py fixed-rate loop, latency injection, temporal ensembling
  deploy/server.py     local HTTP serving boundary (stdlib only)
scripts/               eval runners (E0/E3/E9/E10), tables, figures, doctor, report, serve, gates
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
- **The backend is installed outside the lockfile.** lerobot 0.6.1 requires `torch < 2.12`;
  this project pins `torch 2.14.1+cu130`. `just setup-pretrained` installs it `--no-deps` on
  top of the sync and `just setup` prunes it. Verified working, not supported.
- **The checkpoint's normalization is applied by our code**, because current lerobot moved
  normalization into a processor pipeline this 2024 checkpoint does not ship. E0 is the gate
  that says the mapping is right.
- **Simulation only, virtual-time loop.** E10 injects latency as observation staleness in
  control ticks and does not wall-clock pace the loop; inference is timed for real and
  deadline misses are logged, but this is not a real-time scheduler.
- Robomimic, ROS 2, Docker and `nvcc` are unavailable on this host, so the documented
  fallbacks are in force (PushT only, plain-Python loop, no CUDA C++ stretch). G8 and G9 are
  NOT RUN and never reported as passes. E2, E4-E8 and E11 were not run. No W&B; JSONL only.
- No demo video: there is no capture path on this host.

The full deviation log, including everything cut and why, is `docs/04` section 4.

## License

Code: not decided by the owner yet. The pretrained checkpoint `lerobot/diffusion_pusht` is
Apache-2.0 (model card, accessed 2026-10-27). Dataset `lerobot/pusht` is MIT and derives from
the Diffusion Policy PushT demonstrations.
