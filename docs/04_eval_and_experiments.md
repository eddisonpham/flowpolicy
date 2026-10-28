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
| E0 | Pretrained DP checkpoint through our harness (strict success) | The model card's 65.4% success / 0.955 overlap reproduces within the CI of 50 paired episodes | `lerobot/diffusion_pusht`, image | 1 checkpoint | 0.7 GPU-h (100 episodes) | validation table, `results/e0.jsonl` |
| E1 | Ablation: from-scratch flow vs DDPM vs regression, PushT | Flow within 5 pp of DDPM at N=10 | 3 policies, same encoder/data | 3 at 12000 steps | 3.1 GPU-h train (3 x 12000 steps, 62-64 min each) + 0.03 h eval (450 episodes) | ablation table, `results/ablation/e1.jsonl` |
| E2 | Same, robomimic Lift (+Can) | Same ordering as E1 or not (report either) | 3 policies | 3 | not run (G8 NOT RUN, D1) | — |
| E3 | Integration steps N in {1,2,4,8,10,16} on the pretrained checkpoint, DDIM | Success degrades as N falls, and the DDIM ladder is usable where the DDPM one is not | pretrained, no retrain | 1 checkpoint | 0.4 GPU-h (300 episodes) | HEADLINE fig: N vs success vs latency, `results/e3.jsonl` |
| E4 | Chunk lengths (Ta in {1,4,8,16}) | Longer Ta hurts under disturbance | flow, DDPM | 3 | not run | — |
| E5 | Timestep sampling: uniform vs Beta | No difference at this scale (null is fine) | flow | 3 | not run | — |
| E6 | Head: U-Net vs transformer | U-Net at least as good at equal params | flow | 3 | not run | — |
| E7 | Encoder: ResNet-18 GN vs cached DINOv2 (D3) | Cached features trade augmentation for speed | flow | 3 | not run | — |
| E8 | Precision: bf16 vs fp16 (fp8/int8 if supported) | Success drop under 2 pp [proposed] | E1 checkpoints | 3 | not run | — |
| E9 | Latency/compile ladder (eager, compile modes, CUDA graph) on the pretrained checkpoint | Compile + graphs cut batch-1 chunk latency enough to matter at a 100 ms period | pretrained, inference only | 1 checkpoint | 0.3 GPU-h | `results/latency.jsonl` |
| E10 | Injected latency 0/50/100/150/200 ms + jitter | Success degrades monotonically; async chunking recovers part | deployment loop | 1 checkpoint | about 1.5 GPU-h (270 episodes) | `results/deploy.jsonl`, fig |
| E11 | Disturbance recovery (pose perturbation mid-episode) | Shorter Ta and flow recover better | deployment loop | 3 | not run | — |
| E12 | Async chunk strategies: naive replace vs temporal ensemble vs inpainting-style (D-level) | Inpainting-style best under latency | pretrained | 1 checkpoint | partial: naive replace and temporal ensemble at 100/200 ms; no inpainting-style | `results/deploy.jsonl` |
E13 (3x training budget plus random-crop augmentation on the from-scratch policies)
was scaffolded and then removed unrun on 2026-10-27: it only served the training
thesis that the pivot abandoned (see section 4).
Priority order if time shrinks: E0, E3, E9, E10, E12, E1, E2, E4, E11, E8, E5, E6, E7.
## 3. Claims table (fill only from results)
| Claim | Evidence (file) | Condition under which it is false |
|---|---|---|
| **Ablation (from-scratch, 12000 steps, 3 seeds x 50 paired episodes).** Flow has the highest point estimate (7.3 % [3.3 %, 12.0 %], 11/150) and the highest coverage (0.519), ahead of DDPM (4.7 % [1.3 %, 8.7 %], 7/150, coverage 0.468) and regression (4.0 % [1.3 %, 7.3 %], 6/150, coverage 0.336). Every success CI overlaps, so this orders point estimates and nothing else; only coverage puts regression below the two generative policies, and even there the gap to DDPM is marginal. This is the floor-effect evidence for the pivot: 6-11 successes in 150 episodes cannot order three policies, and a deployment experiment cannot see a policy degrade from there. | `results/ablation/e1.jsonl`, `results/tables/e1_ablation.md` | A 4th seed or more episodes separate two success CIs, or the coverage ordering flips. |
| **E3 (DDIM steps, 50 paired episodes per N, seed 0 checkpointless row).** Coverage saturates at four steps and collapses below it: 0.121 [0.076, 0.165] at N=1, 0.716 [0.627, 0.804] at N=2, 0.885 [0.847, 0.919] at N=4, then a flat plateau (0.919 at N=8, 0.888 at N=10, 0.892 at N=16, every interval overlapping). N=1 and N=2 are disjoint from every N >= 4 row. Success counts agree on the collapse (0/50 at N=1, 10/50 at N=2) but cannot order anything above N=4: 42 % at N=4 and 60 % at N=8 have overlapping intervals, and N=10/N=16 sit at 54 %. Chunk latency is linear in N and therefore the real cost of safety margin: 31.5 ms at N=1 to 726.7 ms at N=16, measured under two-worker contention (the clean single-process numbers are ~2x lower, see E9). | `results/e3.jsonl`, `results/tables/e3_steps.md` | Coverage intervals separate two N >= 4 rows, or a rerun at N=1 succeeds at a rate comparable to N >= 4. |
| **E10 (injected observation latency, 30 episodes per row, DDIM N=10, one policy).** Success declines with staleness but stays inside noise: 56.7 % [40.0 %, 73.3 %] at 0 ms, 60.0 % [43.3 %, 76.7 %] at 50 ms, 56.7 % at 100 ms, 46.7 % [30.0 %, 63.3 %] at 150 ms and 43.3 % [26.7 %, 60.0 %] at 200 ms -- every interval overlapping. Mean coverage barely moves (0.897-0.932), so the policy still reaches the goal and converts slightly less often. The number that is *not* noise is the deadline-miss count: 26-33 misses per episode, because one chunk needs 1.9-2.2x the 100 ms period (E9). Injected network latency is not the binding constraint here; inference cost is. | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | Success at 200 ms falls outside the 0 ms interval, or a rerun shows the loop meeting most deadlines at N=10. |
| **E12 (pretrained checkpoint, replan every 2 steps so four chunks overlap).** Naive replacement collapses: 10.0 % [0.0 %, 20.1 %] at 100 ms and 10.0 % [0.0 %, 23.3 %] at 200 ms with coverage 0.749 / 0.553, both disjoint from the 8-step naive baseline (56.7 % / 43.3 %, coverage 0.917 / 0.915). Uniform temporal ensembling over the same overlapping chunks recovers it to 53.3 % [36.7 %, 70.0 %] and 43.3 % [26.7 %, 60.0 %] (coverage 0.926 / 0.880) -- i.e. back to the 8-step baseline -- at 4x the chunk count, with 120 deadline misses per episode against 26-33 for the 8-step rows (the naive-r2 rows spend 141-147 misses and still succeed only 10 % of the time). At a fast replan rate ensembling is therefore a requirement rather than an optimization: it buys equality with replanning four times less often, and skipping it is worse than replanning slowly. | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | More episodes put naive-r2 at or above temporal-ensemble-r2, or ensembling beats the 8-step baseline outside the CI. |
| **E9 (batch-1 latency, pretrained checkpoint, DDIM N=10).** eager is 190.5 ms mean (p95 204.3 ms) and a CUDA graph is 181.0 ms (p95 185.1 ms): the graph saves 5 %. On the 11 M from-scratch model the same comparison saved **4.7x** (33.46 -> 7.09 ms, `results/ablation/latency.jsonl`), and the mode that mattered most there, `torch.compile` (5.8x), is NOT RUN on the pretrained path at all (section 4: Inductor needs MSVC for the CPU partition the checkpoint's DDIM scheduler creates). So the launch-overhead result does not transfer: at 262.7 M parameters the policy is compute-bound and graphs are worth 5 %. Coupled with E3, that fixes the deployment envelope: a 100 ms control period admits only N <= 4 (66.8 ms graph, 73.5 ms eager), and N=4 is exactly where E3's coverage ladder reaches its plateau (0.885), with latency then growing linearly in N (see E3). | `results/latency.jsonl`, `results/tables/e9_latency.md` | A rerun on an idle GPU with the same protocol shows graphs cutting pretrained latency by more than 1.2x, or shows N=10 fitting inside a 100 ms period. |
| **Ablation (Triton kernel).** The fused Euler/de-normalization kernel is equivalent (max abs diff 1.2e-4, fp32) but loses to `torch.compile`: 0.83x at batch 1 and 0.76x at batch 64, while beating eager 2.2-2.4x. It was written for this repo's own Euler loop and was not ported to the pretrained model. | `results/ablation/kernel.jsonl`, `results/tables/kernel.md` | A re-run on the same GPU shows the kernel beating compile, or the equivalence check fails at `atol=1e-5`. |
## 4. Deviations log
- **2026-10-19: PushT only.** G8 (robosuite) is NOT RUN and D1 chose the documented fallback. E2, E4, E5, E6, E7, E8 and E11 were not run (E12 was added on 2026-10-23; this line is kept as written on the day). E0 is therefore also a PushT-only check.
- **2026-10-19: training budget 1800 steps per run**, batch 64, 11.0 M parameters, 96x96 images, 3 seeds per policy. This is a scaled MVP run, not the ~200k-step reference recipe (docs/03 section 2). E0 is reported as a written explanation, not as a reproduction (M4 fallback), and E1 is a relative comparison only.
- **2026-10-19: rollout count reduced** from the proposed 50 to 30 episodes per seed in E1, 20 in E3 and 5 in E10. All configurations still use 3 training seeds and the same paired eval seeds; CIs are bootstrap over seeds and episodes.
- **2026-10-19: harness workers = 2, not 20.** The owner reported the laptop becoming unresponsive with four CUDA worker processes, so E1 and E3 were run with two workers. The harness still supports any worker count. E1/E3 latency numbers are therefore under two-process GPU contention; the clean single-process batch-1 numbers are the E9 rows in `results/latency.jsonl`.
- **2026-10-19: E10 runs the loop in virtual time** and injects latency as observation staleness in control ticks (`ceil(latency_ms / period_ms)`, plus uniform jitter of up to one tick for a 20 ms jitter setting). Chunk inference is timed for real on the GPU. Deadline misses are logged when inference exceeds the 100 ms period. The rows carry `virtual_time=true`.
- **2026-10-23: training budget raised to 12000 steps, one training seed per policy.** The 1800-step MVP budget produced a null result (0% flow/DDPM, 1.1% regression), so the owner asked for full results and the budget was tripled in epochs at the same batch size (12000 steps = 7.6 epochs of PushT, about 62 min per run on this GPU at 3.4 steps/s). A 3-seed sweep at this budget would be about 6 GPU-h and could not be supervised unattended, so E1, E3 and E10 run **one training seed (seed 0) per policy**; CIs are bootstrap over the 50 episodes per checkpoint and are therefore episode-level, not seed-level. The 3-seed 1800-step logs are kept under `logs/archive/`.
- **2026-10-23: rollouts per checkpoint raised to 50** in E1 and E3 (from 30 and 20) and to 10 per eval seed per latency row in E10, to compensate for the single training seed with a larger episode sample. Eval seeds stay disjoint from training seeds and are the same list for every policy (paired comparison).
- **2026-10-23: E12 needed a replan-rate change to mean anything.** The temporal-ensemble path only replanned when the action queue emptied, so every control step was covered by exactly one chunk and "ensembling" reduced to naive replacement: its rows reproduced naive exactly (coverage 0.411 at 100 ms, 0.413 at 200 ms, same success). Added `replan_every` to `LoopConfig`; the `-r2` rows replan every 2 steps so four chunks overlap, and the no-overlap ensemble rows were removed from `results/deploy.jsonl`. The naive `-r2` rows are the matched baseline for the same replan rate.
- **2026-10-23: about 19% of eval episodes end at exactly 0.0 coverage regardless of policy** (e.g. env seed 10001, where the block starts far from the goal). It is uniform across policies and configurations so paired comparisons are unaffected, but it lowers the effective sample size and widens the episode-level intervals.
- **2026-10-24: E1 is now 3 training seeds at the 12000-step budget.** Seeds 1 and 2 were trained at the same budget as seed 0 (3.2 steps/s, 62-64 min per run), so the E1 intervals resample training seeds and then the 50 paired episodes per checkpoint instead of being episode-level. 450 episodes total. E3 and E10 still run seed 0 only, so their intervals remain episode-level. With 3 seeds the success CIs overlap (flow 7.3%, DDPM 4.7%, regression 4.0%); the previous "flow is the only policy with non-zero success" claim was a one-seed artifact and is retracted in §3.
- **2026-10-24: DDPM sampling ladder bug, found by the E3 rerun and fixed.** `DDPMPolicy._sample` laid its ladder from `train_steps-1` down to **t=0**, but `alphas_cumprod[0] == 1.0` exactly for the squared-cosine schedule, so the final step mapped x to x unchanged. Every measurement therefore used `steps-1` useful steps, and N=1 and N=2 produced bit-identical samples (coverage 0.2294 both, before the fix). The ladder now ends at t=1, so the final iteration uses `ab_prev = 1` and lands exactly on x0. Guarded by `tests/unit/test_policies.py::test_ddpm_step_count_is_effective` (fails on the old code). DDPM E1 (3 seeds) and E3 rows were re-evaluated after the fix; no retraining was needed because only the sampling path changed.
- **2026-10-24: E3's coverage column carries a 95% bootstrap interval.** Coverage moved by up to 0.05 between adjacent N on 50 episodes, which made the ladder read as ordered when it was noise; the interval is now printed next to it (`scripts/make_tables.py`).
- **2026-10-19: E0 is not a reproduction (M4 fallback).** The DDPM baseline is trained for 1800 steps at batch 64, about 1.5 epochs of PushT, versus roughly 200k steps in the reference recipe. The measured DDPM success is 0% (95% CI [0%, 0%], 90 episodes), so the "within the CI of about 64%" hypothesis is rejected for this budget. No E0 reproduction is claimed; E1 is a relative comparison only.
- **2026-10-19: E10 warmup correction.** The first E10 pass timed the first chunk cold (CUDA context and kernel load), which produced a 871 ms first sample and a mean above p95. The loop now performs one untimed warmup chunk per episode, matching the "warmed up" protocol in [05](05_inference_and_deployment.md); `results/deploy.jsonl` was regenerated from scratch afterwards. No cold-start samples remain.
- **2026-10-19: resume-path fix found by the reboot.** Resuming loaded the RNG tensors with `map_location=cuda`, and `torch.Generator.set_state` / `torch.set_rng_state` require CPU ByteTensors. `Trainer.resume` now moves them to CPU. The interrupted `e1_ddpm_s2` run resumed from its step-1200 checkpoint (the 1200-1625 segment that had no checkpoint was dropped from its JSONL, not silently kept).
- **2026-10-27: the project pivots to a pretrained policy (D11).** The from-scratch models
  reach 4-7 % success at 12000 steps, which is a floor effect: with 6-11 successful episodes
  in 150, no two policies can be ordered, and a deployment experiment cannot see a policy
  degrade when it is not succeeding to begin with. The owner chose to stop training and run
  the inference and deployment stack against a known-good checkpoint instead:
  `lerobot/diffusion_pusht` (Apache-2.0, Diffusion Policy trained on gym-pusht, 175k steps,
  model card reports 65.4 % success and 0.955 mean max overlap over 500 episodes). The
  from-scratch measurements are kept as an ablation under `results/ablation/`, and the
  training path stays runnable (`just train-e1`) but is no longer on the critical path.
- **2026-10-27: rule 4 of docs/07 is amended for the pretrained backend.** It said "do not
  use LeRobot or HF policy code as model code". Under D11 the policy *is* borrowed code, so
  the rule now excludes only the measurement machinery: the harness, the loop, the latency
  protocol, the normalization glue and the tables stay ours. The pretrained path imports
  `lerobot` at call time only, so the package still works without it installed.
- **2026-10-27: `lerobot` is installed outside the lockfile.** lerobot 0.6.1 requires
  `torch < 2.12`; this project pins `torch 2.14.1+cu130` for the RTX 5060 (docs/02). A normal
  resolve would downgrade torch and invalidate every measurement made on this GPU, so
  `just setup-pretrained` installs lerobot `--no-deps` plus the packages its inference path
  imports (diffusers, draccus, einops, safetensors, opencv-python-headless, termcolor). The
  combination is verified working, not supported by either project, and `just setup` prunes
  it again.
- **2026-10-27: the checkpoint's normalization is applied by our code.** lerobot 0.6.1 moved
  normalization out of the policy into a processor pipeline that this 2024 checkpoint does
  not ship, so `from_pretrained` loads its eight `normalize_inputs.*` / `unnormalize_outputs.*`
  tensors as *unexpected keys* and drops them. `models/pretrained.py` reads them from
  `model.safetensors` and applies the checkpoint's own mapping (MEAN_STD for images, MIN_MAX
  for state and actions) instead of this repo's train-time normalizer. E0 is the gate that
  proves the mapping: 64.0 % success and 0.924 mean coverage at ddim@10 against the published
  65.4 % / 0.955.
- **2026-10-27: E3 uses DDIM, not the checkpoint's DDPM scheduler.** The checkpoint ships a
  DDPM scheduler, whose every step injects noise; truncating it to N=1 would measure the
  scheduler rather than the policy. The E3 ladder therefore swaps in a DDIM scheduler built
  from the same beta schedule and reports the checkpoint's own DDPM@100 setup separately as
  E0's reference row (`--scheduler ddpm --steps 100`).
- **2026-10-27: E13 was removed unrun.** The 36000-step plus random-crop run was scaffolded
  before the pivot and only served the training thesis; the configs and runners were deleted
  rather than left half-finished.
- **2026-10-27: pretrained rows carry `train_seed = -1`.** There is exactly one pretrained
  PushT checkpoint, so E0, E3, E9 and E10 intervals are over episodes only. This is a real
  limitation of the substrate, not of the harness: it is stated on every table that has them.
- **2026-10-27: `torch.compile` was NOT RUN on the pretrained path; superseded 2026-10-28.**
  Every compile mode failed with
  `torch._inductor.exc.InductorError: InvalidCxxCompiler: Compiler: cl is not found`. The
  recorded cause (Inductor wanting MSVC for a CPU partition created by the checkpoint's DDIM
  scheduler) was correct, but the fix was not in the environment: `diffusers` keeps
  `alphas_cumprod` on the host, so the loop blended host scalars into CUDA math and needed a
  C++ compiler for it. `src/flowpolicy/models/fast_sampler.py` precomputes the coefficients
  on-device and keeps the loop tensor-only, which removes both the partition and the per-step
  host/device synchronization. That sampler is proven equal to `diffusers` at five step counts
  and two beta schedules (`tests/unit/test_fast_sampler.py`).
- **2026-10-28: E14 added (stack value), E9 compile rows now exist.** The question a
  deployment stack has to answer is "compared to what", so E14 measures the checkpoint three
  ways on the same GPU and the same episodes: the published configuration (DDPM@100), the stock
  `lerobot` -> `diffusers` DDIM path, and this repo's fast sampler with graphs/compile. Latency
  rows give per-chunk cost and control rate; throughput rows give seconds per episode with
  success and coverage attached, so a cheaper arm cannot be a quietly worse policy. The E9
  N-sweep stays on the stock sampler for comparability with the E0/E3 rows; the mode comparison
  moved to E14. `torch.compile` rows are now real measurements, and they are worth about a
  tenth rather than a multiple -- which is itself the finding: this model is compute-bound, so
  launch-overhead work does not buy a control period.
- **2026-10-28: the stock sampler is still what E0/E3/E10 rows were measured with.** The fast
  sampler is the one E14 measures for speed; the two agree numerically, and E0's 64.0 % is the
  anchor for both. Anything measured with `--sampler fast` is reported as such in its rows.

## 5. Acceptance criteria
- [x] Harness test: eval/train seed overlap raises an error (`tests/unit/test_eval_seeds.py`, asserted in `rollout.evaluate`).
- [x] A figure script exists for every figure that was run: E0, E3 (both panels), E10 (success curve and latency histogram) and the ablation bar regenerates from raw JSONL with `just figures`. E11 was never run, so it has no figure.
- [x] This file was committed before the first experiment result (the claims table and the deviations log have been updated in place ever since, each entry dated).