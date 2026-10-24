# 04. Evaluation protocol and experiment matrix

PRE-REGISTRATION: commit this file before running E1 onward. Any change after that goes in the "Deviations" section with date and reason. Links: [03](03_data_models_training.md) [05](05_inference_and_deployment.md)

## 1. Protocol

- **Metrics:** success rate (PushT: max coverage at least 95%, also log mean max coverage); robomimic success flag; latency mean and p95 (batch 1, warmed up); peak VRAM; wall-clock to convergence.
- **Seeds:** at least 3 training seeds per configuration (5 if time allows). Eval seeds are a fixed list disjoint from training seeds; the harness asserts this.
- **Rollouts:** 50 episodes per training seed per checkpoint [proposed; confirm against DP and robomimic protocols, U].
- **Checkpoint rule (fixed in advance):** report the LAST checkpoint, and separately the mean of the last K=5 evaluated checkpoints. Best-checkpoint numbers may be shown only if labelled "oracle, not a result".
- **Statistics:** per-config mean over seeds with 95% bootstrap CI (resample seeds, then episodes). Pairwise policy comparisons use the same eval seeds (paired). Meaningful difference, set in advance: 5 percentage points absolute [proposed default; owner may change before E1]. Wilson intervals for single-seed success rates.
- **Harness:** 20 worker processes, per-worker seeding from (eval_seed, episode_idx), output one JSON line per episode: `{config_id, train_seed, eval_seed, task, ckpt, success, coverage, steps, latency_mean_ms, latency_p95_ms}`. All tables and figures are generated from `results/*.jsonl` by scripts; no hand-edited numbers.
- **Null results:** a null or negative result is reported with the same prominence as a positive one.

## 2. Experiment matrix

The Compute column below is filled with measured GPU time from the 1800-step runs and the eval logs, not guesses. "Not run" means exactly that; the fallback is recorded in §4.

| ID | Experiment | Hypothesis (falsifiable) | Config | Seeds | Compute | Output |
|---|---|---|---|---|---|---|
| E0 | Reproduce DP on PushT (strict success) | Our DDPM policy is within the CI of about 64% [S] | ddpm, image | 3 | not run as a reproduction; DDPM rows in E1 used as the M4 fallback explanation | written explanation, §4 |
| E1 | Flow vs DDPM vs ACT-style, PushT | Flow within 5 pp of DDPM at N=10 | 3 policies, same encoder/data | 3 (1 at 12000 steps, see §4) | 3.1 GPU-h train (3 x 12000 steps, 62-64 min each) + 0.03 h eval (150 episodes) | main table, `results/e1.jsonl` |
| E2 | Same, robomimic Lift (+Can) | Same ordering as E1 or not (report either) | 3 policies | 3 | not run (G8 NOT RUN, D1) | — |
| E3 | Integration steps N in {1,2,4,8,10,16}; DDIM steps matched | Flow degrades less than DDIM as N falls | E1 checkpoints, no retrain | 3 (1 at 12000 steps, see §4) | 0.24 GPU-h (600 episodes) | HEADLINE fig: N vs success vs latency |
| E4 | Chunk lengths (Ta in {1,4,8,16}) | Longer Ta hurts under disturbance | flow, DDPM | 3 | not run | — |
| E5 | Timestep sampling: uniform vs Beta | No difference at this scale (null is fine) | flow | 3 | not run | — |
| E6 | Head: U-Net vs transformer | U-Net at least as good at equal params | flow | 3 | not run | — |
| E7 | Encoder: ResNet-18 GN vs cached DINOv2 (D3) | Cached features trade augmentation for speed | flow | 3 | not run | — |
| E8 | Precision: bf16 vs fp16 (fp8/int8 if supported) | Success drop under 2 pp [proposed] | E1 checkpoints | 3 | not run | — |
| E9 | Latency/compile ladder (eager, compile modes, CUDA graph, Triton) | Compile + graphs remove most launch overhead at batch 1 | inference only | n/a | 0.1 GPU-h | `results/latency.jsonl`, `results/kernel.jsonl` |
| E10 | Injected latency 0/50/100/150/200 ms + jitter | Success degrades monotonically; async chunking recovers part | deployment loop | 3 (1 at 12000 steps, see §4) | 1.1 GPU-h (540 episodes) | `results/deploy.jsonl`, fig |
| E11 | Disturbance recovery (pose perturbation mid-episode) | Shorter Ta and flow recover better | deployment loop | 3 | not run | — |
| E12 | Async chunk strategies: naive replace vs temporal ensemble vs inpainting-style (D-level) | Inpainting-style best under latency | flow | 3 (1 at 12000 steps, see §4) | partial: naive replace and temporal ensemble at 100/200 ms; no inpainting-style | `results/deploy.jsonl` |

Priority order if time shrinks: E0, E1, E3, E9, E10, E2, E4, E11, E8, E5, E6, E12, E7.

## 3. Claims table (fill only from results)

| Claim | Evidence (file) | Condition under which it is false |
|---|---|---|
| At 12000 steps flow is the only policy with non-zero E1 success: 8.0% (4/50) [2.0%, 16.0%], vs regression 4.0% (2/50) [0.0%, 10.0%] and DDPM 0.0% (0/50) [0.0%, 0.0%]. Mean coverage ranks flow 0.471, regression 0.303, DDPM 0.284. This is one training seed per policy, so the intervals are episode-level. | `results/e1.jsonl`, `results/tables/e1_main.md` | A 3-seed rerun at the same budget puts DDPM at or above flow, or flow's success CI includes 0. |
| The 1800-step null was a budget artifact, not a broken pipeline: same code, data, eval seeds and checkpoint rule gave flow 0% success / 0.252 coverage at 1800 steps and 8% / 0.471 at 12000 steps. DDPM stayed at 0% but its coverage rose 0.132 to 0.284. | `logs/archive/` (1800-step rows), `results/e1.jsonl` (12000-step rows) | The archived 1800-step rows were not produced by the same pipeline, or a rerun at 1800 steps reproduces the 12000-step numbers. |
| Injected observation latency of 0-200 ms (20 ms jitter) does not degrade this policy beyond noise: coverage 0.439 at 0 ms and 0.385-0.417 at 50-200 ms, success 3.3% at 0 ms and 0-3.3% otherwise, every CI overlapping. Deadline misses are rare (3-8 per ~2280 chunk inferences, 0.1-0.3%) because chunk inference is usually under the 100 ms period (mean 45-53 ms per row, worst per-episode p95 98.7 ms); the few misses are individual slow chunks, not a systematic overrun. | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | Coverage at 100 or 200 ms falls outside the 0 ms interval, or misses appear at a rate the loop cannot absorb. |
| E12 (partial): when the loop replans every 2 steps so chunks overlap, naive action replacement collapses (coverage 0.212 at 100 ms and 0.154 at 200 ms, 0/120 successes) while uniform temporal ensembling over the four overlapping chunks recovers most of it (0.386 / 0.323, 4/120 successes, 95% CIs overlapping). Neither beats the 8-step naive baseline (0.411 / 0.413, 2/120), and ensembling costs 4x chunk inference (150 vs 38 calls per episode). | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | More episodes put naive-r2 at or above temporal-ensemble-r2, or ensembling beats the 8-step baseline outside the CI. |
| CUDA graphs cut batch-1, N=10 chunk latency by 4.7x vs eager (33.46 to 7.09 ms mean); `torch.compile` `reduce-overhead` cuts it 5.8x (5.80 ms) at a 25.8 s compile cost. | `results/latency.jsonl`, `results/tables/e9_latency.md` | Re-measured on an idle GPU with the same protocol and the ratio falls below 2x, or the compile time exceeds the measurement budget. |
| The fused Triton kernel is equivalent (max abs diff 1.2e-4, fp32) but slower than compiled PyTorch: 0.83x at batch 1 and 0.76x at batch 64, while beating eager 2.2-2.4x. | `results/kernel.jsonl`, `results/tables/kernel.md` | A re-run on the same GPU shows the kernel beating compile, or the equivalence check fails at `atol=1e-5`. |
| E3's coverage result supports "flow degrades less than DDIM as N falls": at N=1 flow keeps 0.401 coverage against 0.468 at N=10 (-14%), while DDPM drops from 0.239 to 0.172 (-28%). Success rates are too sparse to order N values: flow reaches 14% at N=4 and N=16 but 0% at N=10, which is non-monotonic and inside the +-6 pp resolution of 50 episodes. | `results/e3.jsonl`, `results/tables/e3_steps.md` | More episodes or seeds show DDPM retaining coverage as well as flow at N=1, or show a monotonic success curve in N. |

## 4. Deviations log

- **2026-10-19: PushT only.** G8 (robosuite) is NOT RUN and D1 chose the documented fallback. E2, E4, E5, E6, E7, E8, E11 and E12 were not run. E0 is therefore also a PushT-only check.
- **2026-10-19: training budget 1800 steps per run**, batch 64, 11.0 M parameters, 96x96 images, 3 seeds per policy. This is a scaled MVP run, not the ~200k-step reference recipe (docs/03 section 2). E0 is reported as a written explanation, not as a reproduction (M4 fallback), and E1 is a relative comparison only.
- **2026-10-19: rollout count reduced** from the proposed 50 to 30 episodes per seed in E1, 20 in E3 and 5 in E10. All configurations still use 3 training seeds and the same paired eval seeds; CIs are bootstrap over seeds and episodes.
- **2026-10-19: harness workers = 2, not 20.** The owner reported the laptop becoming unresponsive with four CUDA worker processes, so E1 and E3 were run with two workers. The harness still supports any worker count. E1/E3 latency numbers are therefore under two-process GPU contention; the clean single-process batch-1 numbers are the E9 rows in `results/latency.jsonl`.
- **2026-10-19: E10 runs the loop in virtual time** and injects latency as observation staleness in control ticks (`ceil(latency_ms / period_ms)`, plus uniform jitter of up to one tick for a 20 ms jitter setting). Chunk inference is timed for real on the GPU. Deadline misses are logged when inference exceeds the 100 ms period. The rows carry `virtual_time=true`.
- **2026-10-23: training budget raised to 12000 steps, one training seed per policy.** The 1800-step MVP budget produced a null result (0% flow/DDPM, 1.1% regression), so the owner asked for full results and the budget was tripled in epochs at the same batch size (12000 steps = 7.6 epochs of PushT, about 62 min per run on this GPU at 3.4 steps/s). A 3-seed sweep at this budget would be about 6 GPU-h and could not be supervised unattended, so E1, E3 and E10 run **one training seed (seed 0) per policy**; CIs are bootstrap over the 50 episodes per checkpoint and are therefore episode-level, not seed-level. The 3-seed 1800-step logs are kept under `logs/archive/`.
- **2026-10-23: rollouts per checkpoint raised to 50** in E1 and E3 (from 30 and 20) and to 10 per eval seed per latency row in E10, to compensate for the single training seed with a larger episode sample. Eval seeds stay disjoint from training seeds and are the same list for every policy (paired comparison).
- **2026-10-23: E12 needed a replan-rate change to mean anything.** The temporal-ensemble path only replanned when the action queue emptied, so every control step was covered by exactly one chunk and "ensembling" reduced to naive replacement: its rows reproduced naive exactly (coverage 0.411 at 100 ms, 0.413 at 200 ms, same success). Added `replan_every` to `LoopConfig`; the `-r2` rows replan every 2 steps so four chunks overlap, and the no-overlap ensemble rows were removed from `results/deploy.jsonl`. The naive `-r2` rows are the matched baseline for the same replan rate.
- **2026-10-23: about 19% of eval episodes end at exactly 0.0 coverage regardless of policy** (e.g. env seed 10001, where the block starts far from the goal). It is uniform across policies and configurations so paired comparisons are unaffected, but it lowers the effective sample size and widens the episode-level intervals.
- **2026-10-19: W&B omitted** (D8 deviation); local JSONL under `results/train/` is the only scalar log.
- **2026-10-19: E0 is not a reproduction (M4 fallback).** The DDPM baseline is trained for 1800 steps at batch 64, about 1.5 epochs of PushT, versus roughly 200k steps in the reference recipe. The measured DDPM success is 0% (95% CI [0%, 0%], 90 episodes), so the "within the CI of about 64%" hypothesis is rejected for this budget. No E0 reproduction is claimed; E1 is a relative comparison only.
- **2026-10-19: E10 warmup correction.** The first E10 pass timed the first chunk cold (CUDA context and kernel load), which produced a 871 ms first sample and a mean above p95. The loop now performs one untimed warmup chunk per episode, matching the "warmed up" protocol in [05](05_inference_and_deployment.md); `results/deploy.jsonl` was regenerated from scratch afterwards. No cold-start samples remain.
- **2026-10-19: resume-path fix found by the reboot.** Resuming loaded the RNG tensors with `map_location=cuda`, and `torch.Generator.set_state` / `torch.set_rng_state` require CPU ByteTensors. `Trainer.resume` now moves them to CPU. The interrupted `e1_ddpm_s2` run resumed from its step-1200 checkpoint (the 1200-1625 segment that had no checkpoint was dropped from its JSONL, not silently kept).

## 5. Acceptance criteria

- [ ] Harness test: eval/train seed overlap raises an error.
- [ ] A figure script exists for every figure in E3, E10, E11 and regenerates from raw JSONL with one command.
- [ ] This file is committed before the first experiment result.
