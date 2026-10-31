# 04. Evaluation protocol and experiment matrix
PRE-REGISTRATION: commit this file before running E1 onward. Any change after that goes in the "Deviations" section with date and reason. Links: [03](03_data_models_training.md) [05](05_inference_and_deployment.md)
## 1. Protocol
- **Metrics:** success rate (PushT: max coverage at least 95%, also log mean max coverage); robomimic success flag; latency mean and p95 (batch 1, warmed up); peak VRAM; wall-clock to convergence.
- **Seeds:** at least 3 training seeds per configuration (5 if time allows). Eval seeds are a fixed list disjoint from training seeds; the harness asserts this.
- **Rollouts:** 50 episodes per training seed per checkpoint [proposed; confirm against DP and robomimic protocols, U].
- **Checkpoint rule (fixed in advance):** report the LAST checkpoint, and separately the mean of the last K=5 evaluated checkpoints. Best-checkpoint numbers may be shown only if labelled "oracle, not a result".
- **Statistics:** per-config mean over seeds with 95% bootstrap CI (resample seeds, then episodes). Pairwise policy comparisons use the same eval seeds (paired). Meaningful difference, set in advance: 5 percentage points absolute [proposed default; owner may change before E1]. Wilson intervals for single-seed success rates.
- **Multiplicity and repeated looks (added 2026-10-30).** The README quotes six hypothesis tests: E0 paired McNemar, E15 paired McNemar, the VQ-BeT replication Fisher test, the adapter-free control Fisher test, E3's coverage ladder, and E12's three arms. Each answers a pre-registered question about its own seed block, so each is reported at alpha = 0.05 with no correction, and the README says so rather than implying a family-wise guarantee. For a reader who wants one: Bonferroni over six tests sets the bar at 0.008, and only E0's p < 0.001 clears it -- every other quoted p-value is read as "not distinguished" either way, so the correction would change no sentence in the README. Blocks that grew after a first look (E0 from 50 to 200 seeds, VQ-BeT from 50 to 200) are reported at their final size only; the earlier looks are in the CHANGELOG, are not counted as alpha spent, and no block was extended or stopped because of what its p-value was.
- **Resolution of each block (added 2026-10-30).** A null is only as informative as the smallest effect its block could have seen, so `just power-audit` states that number for every paired comparison in `results/` (`results/power_audit.json`, `stats.resolvable_effect`, 80 % power). **Every 200-episode block in this repo resolves 13.1-13.5 pp** and the one 500-episode block (VQ-BeT) resolves 8.6 pp, because the discordance between any two of these runs is 44-48 %. Four quoted nulls observe less than their block can see -- the sampler 7 pp on a block worth 13.1, the noise stream 6.5 on 13.3, the like-for-like control 4.0 on 13.1, and VQ-BeT 2.2 on 8.6 -- so they are "not distinguished" in the weak sense: none of them excludes effects up to its own resolution, and no sentence here or in the README may read them as "no difference". Where an experiment had to resolve a small effect (option H's arm), the block is sized from a pilot rather than fixed (`docs/10` §5.1).
- **Harness:** 20 worker processes, per-worker seeding from (eval_seed, episode_idx), output one JSON line per episode: `{config_id, train_seed, eval_seed, task, ckpt, success, coverage, steps, latency_mean_ms, latency_p95_ms}`. All tables and figures are generated from `results/*.jsonl` by scripts; no hand-edited numbers.
- **Null results:** a null or negative result is reported with the same prominence as a positive one.
## 2. Experiment matrix
The Compute column below is filled with measured GPU time from the 1800-step runs and the eval logs, not guesses. "Not run" means exactly that; the fallback is recorded in §4.
| ID | Experiment | Hypothesis (falsifiable) | Config | Seeds | Compute | Output |
|---|---|---|---|---|---|---|
| E0 | Pretrained DP checkpoint through our harness (strict success) | The model card's 65.4 % success / 0.907 mean coverage reproduces within the CI of 50 episodes paired on the block's own seeds (its 0.955 is coverage / 0.95) -- amended 2026-10-29 to 200 paired episodes, which reproduces the coverage but **not** the success rate (§4, entry of 2026-10-29) | `lerobot/diffusion_pusht`, image | 1 checkpoint | 0.7 GPU-h (100 episodes) + 200 paired episodes | validation table, `results/e0.jsonl`, `results/e0_published_seeds.jsonl` |
| E1 | Ablation: from-scratch flow vs DDPM vs regression, PushT | Flow within 5 pp of DDPM at N=10 | 3 policies, same encoder/data | 3 at 12000 steps | 3.1 GPU-h train (3 x 12000 steps, 62-64 min each) + 0.03 h eval (450 episodes) | ablation table, `results/ablation/e1.jsonl` |
| E2 | Same, robomimic Lift (+Can) | Same ordering as E1 or not (report either) | 3 policies | 3 | not run (G8 NOT RUN, D1) | — |
| E3 | Integration steps N in {1,2,4,8,10,16} on the pretrained checkpoint, DDIM | Success degrades as N falls, and the DDIM ladder is usable where the DDPM one is not | pretrained, no retrain | 1 checkpoint | 0.4 GPU-h (300 episodes) | HEADLINE fig: N vs success vs latency, `results/e3.jsonl` |
| E4 | Chunk lengths (Ta in {1,4,8,16}) | Longer Ta hurts under disturbance | flow, DDPM | 3 | not run | — |
| E5 | Timestep sampling: uniform vs Beta | No difference at this scale (null is fine) | flow | 3 | not run | — |
| E6 | Head: U-Net vs transformer | U-Net at least as good at equal params | flow | 3 | not run | — |
| E7 | Encoder: ResNet-18 GN vs cached DINOv2 (D3) | Cached features trade augmentation for speed | flow | 3 | not run | — |
| E8 | Precision: bf16 vs fp16 (fp8/int8 if supported) | Success drop under 2 pp [proposed] | pretrained checkpoint, fast sampler, 40 paired episodes per dtype at N=4; latency at N=4/10 (amended 2026-10-28, section 4) | 1 checkpoint | 0.3 GPU-h | `results/precision.jsonl`, `results/tables/e8_precision.md` |
| E9 | Latency/compile ladder (eager, compile modes, CUDA graph) on the pretrained checkpoint | Compile + graphs cut batch-1 chunk latency enough to matter at a 100 ms period | pretrained, inference only | 1 checkpoint | 0.3 GPU-h | `results/latency.jsonl` |
| E10 | Injected latency 0/50/100/150/200 ms + jitter | Success degrades monotonically; async chunking recovers part | deployment loop | 1 checkpoint | about 1.5 GPU-h (270 episodes) | `results/deploy.jsonl`, fig |
| E11 | Disturbance recovery (pose perturbation mid-episode) | Shorter Ta and flow recover better | deployment loop | 3 | not run | — |
| E12 | Async chunk strategies: naive replace vs temporal ensemble vs inpainting-style (D-level) | Inpainting-style best under latency | pretrained | 1 checkpoint | partial: naive replace and temporal ensemble at 100/200 ms; no inpainting-style | `results/deploy.jsonl` |
| E14 | Stack value baseline: the checkpoint run as published (DDPM@100), through the stock library DDIM path, and through this repo's sampler with graphs/compile | The operating point this repo selects is at least 10x cheaper per control step than the published configuration, without losing success or coverage | pretrained, inference + episodes | 1 checkpoint | 0.2 GPU-h (4 episodes per throughput arm, 20 paired equivalence) | `results/stack.jsonl`, `results/tables/e14_stack.md`, `results/figures/e14_stack_throughput.png` |
| E15 | Generality: a second published PushT checkpoint (VQ-BeT) through the same harness, loop and latency protocol | A different policy family plugs in behind `PolicyLike` with no change outside one adapter and a dispatch on the checkpoint's own `config.json` | `lerobot/vqbet_pusht`, 5 observation steps, 5-action chunk, 37.5 M parameters | 1 checkpoint | 0.06 GPU-h (200 episodes) + 3 s latency | `results/vqbet.jsonl`, `results/vqbet_latency.jsonl`, `results/vqbet_loop.jsonl`, `results/tables/generality.md` |
E13 (3x training budget plus random-crop augmentation on the from-scratch policies)
was scaffolded and then removed unrun on 2026-10-27: it only served the training
thesis that the pivot abandoned (see section 4).
Priority order if time shrinks: E0, E3, E9, E10, E12, E1, E2, E4, E11, E8, E5, E6, E7.
## 3. Claims table (fill only from results)
| Claim | Evidence (file) | Condition under which it is false |
|---|---|---|
| **Ablation (from-scratch, 12000 steps, 3 seeds x 50 paired episodes).** Flow has the highest point estimate (7.3 % [3.3 %, 12.0 %], 11/150) and the highest coverage (0.519), ahead of DDPM (4.7 % [1.3 %, 8.7 %], 7/150, coverage 0.468) and regression (4.0 % [1.3 %, 7.3 %], 6/150, coverage 0.336). Every success CI overlaps, so this orders point estimates and nothing else; only coverage puts regression below the two generative policies, and even there the gap to DDPM is marginal. This is the floor-effect evidence for the pivot: 6-11 successes in 150 episodes cannot order three policies, and a deployment experiment cannot see a policy degrade from there. | `results/ablation/e1.jsonl`, `results/tables/e1_ablation.md` | A 4th seed or more episodes separate two success CIs, or the coverage ordering flips. |
| **E3 (DDIM steps, 50 paired episodes per N, seed 0 checkpointless row).** Coverage saturates at four steps and collapses below it: 0.121 [0.076, 0.165] at N=1, 0.716 [0.627, 0.804] at N=2, 0.885 [0.847, 0.919] at N=4, then a flat plateau (0.919 at N=8, 0.888 at N=10, 0.892 at N=16, every interval overlapping). N=1 and N=2 are disjoint from every N >= 4 row. Success counts agree on the collapse (0/50 at N=1, 10/50 at N=2) but cannot order anything above N=4: 42 % at N=4 and 60 % at N=8 have overlapping intervals, and N=10/N=16 sit at 54 %. Chunk latency is linear in N and therefore the real cost of safety margin: 31.5 ms at N=1 to 726.7 ms at N=16, measured under two-worker contention (the clean single-process numbers are ~2x lower, see E9). | `results/e3.jsonl`, `results/tables/e3_steps.md` | Coverage intervals separate two N >= 4 rows, or a rerun at N=1 succeeds at a rate comparable to N >= 4. |
| **E3 replication (independent 50-episode block, seeds 30000-30049, DDIM, 2026-10-28).** The coverage ladder replicates; the success ladder does not. Coverage: 0.761 / 0.897 / 0.902 / 0.896 at N=2/4/8/10 against 0.716 / 0.885 / 0.919 / 0.888 in the first block, changes of +0.044 / +0.012 / -0.016 / +0.008 -- so "two steps are catastrophic, four reach the plateau" is a property of the policy, not of the seed block. Success: 24 % vs 20 % at N=2, 38 % vs 42 % at N=4, 48 % vs 60 % at N=8, 40 % vs 54 % at N=10; every pair of intervals overlaps, and the first block's apparent "N=8 is best" ordering does not survive. That is the 50-episode resolution limit, now measured twice rather than asserted once. | `results/e3_replication.jsonl`, `results/tables/e3_replication.md` | A third block puts N=4 coverage below 0.80 while N=8 stays above 0.90, or N=4 and N=8 success separate by more than 25 pp in both blocks. |
| **E10 (injected observation latency, 30 episodes per row, DDIM N=10, one policy).** Success declines with staleness but stays inside noise: 56.7 % [40.0 %, 73.3 %] at 0 ms, 60.0 % [43.3 %, 76.7 %] at 50 ms, 56.7 % at 100 ms, 46.7 % [30.0 %, 63.3 %] at 150 ms and 43.3 % [26.7 %, 60.0 %] at 200 ms -- every interval overlapping. Mean coverage barely moves (0.897-0.932), so the policy still reaches the goal and converts slightly less often. The number that is *not* noise is the deadline-miss count: 26-33 misses per episode, because one chunk needs 1.9-2.2x the 100 ms period (E9). Injected network latency is not the binding constraint here; inference cost is. | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | Success at 200 ms falls outside the 0 ms interval, or a rerun shows the loop meeting most deadlines at N=10. |
| **E12 (pretrained checkpoint, replan every 2 steps so four chunks overlap).** Naive replacement collapses: 10.0 % [0.0 %, 20.1 %] at 100 ms and 10.0 % [0.0 %, 23.3 %] at 200 ms with coverage 0.749 / 0.553, both disjoint from the 8-step naive baseline (56.7 % / 43.3 %, coverage 0.917 / 0.915). Uniform temporal ensembling over the same overlapping chunks recovers it to 53.3 % [36.7 %, 70.0 %] and 43.3 % [26.7 %, 60.0 %] (coverage 0.926 / 0.880) -- i.e. back to the 8-step baseline -- at 4x the chunk count, with 120 deadline misses per episode against 26-33 for the 8-step rows (the naive-r2 rows spend 141-147 misses and still succeed only 10 % of the time). At a fast replan rate ensembling is therefore a requirement rather than an optimization: it buys equality with replanning four times less often, and skipping it is worse than replanning slowly. | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | More episodes put naive-r2 at or above temporal-ensemble-r2, or ensembling beats the 8-step baseline outside the CI. |
| **E12 third arm (cross-fade "blend", replan every 2 steps, 30 episodes per row, 2026-10-28).** Smoothing the splice is not the same as averaging the predictions. The cross-fade keeps coverage near the ensembling rows (0.875 at 100 ms, 0.864 at 200 ms, against naive replacement's 0.749 / 0.553) but does not recover success: 26.7 % [13.3 %, 43.3 %] at 100 ms and 10.0 % [0.0 %, 20.0 %] at 200 ms, against temporal ensembling's 53.3 % / 43.3 % and naive's 10.0 % / 10.0 %. Its deadline-miss count (132 / 143) sits nearer naive's than the ensemble's (120 / 121) because it replans just as often. The pre-registered hypothesis named an inpainting-style strategy as the expected best; the implementable stand-in is not, so the hypothesis is refuted for this strategy and the earlier conclusion stands: with a fast replan rate, ensembling is a requirement rather than an optimization. | `results/deploy.jsonl`, `results/tables/e10_deploy.md` | A rerun puts the blend success interval above the ensemble's at 200 ms, or its coverage below naive's at 100 ms. |
| **M9.6 pacing (12 episodes per step count, 100 ms period, 0 ms injected latency, 2026-10-28).** With real sleeps the loop's overruns become wall-clock lateness instead of a counter. At N=4 it holds the period: 0.2 late ticks per episode (2 of 12 episodes slipped once), mean lateness 1.1 ms, wall clock 22.0 s for 220 mean steps. At N=10 it never holds it: 52.5 late ticks per episode, late on all 12, mean lateness 65.8 ms and p95 102.9 ms -- and the part virtual time could not show, success falls to 3/12 (25.0 % [0.0 %, 50.0 %]) against 10/12 at N=4 (83.3 % [58.3 %, 100.0 %]), with mean coverage unchanged (0.943 vs 0.947). The success intervals overlap at their edges, so this is directional rather than separated; it is the first measurement where missing the deadline appears in the policy's outcome and not only in a counter. | `results/paced.jsonl`, `results/tables/paced_loop.md` | A 50-episode paced rerun puts N=10's success interval inside N=4's, or shows N=10 meeting most ticks. |
| **E9 (batch-1 latency, pretrained checkpoint, DDIM N=10).** eager is 190.5 ms mean (p95 204.3 ms) and a CUDA graph is 181.0 ms (p95 185.1 ms): the graph saves 5 %. On the 11 M from-scratch model the same comparison saved **4.7x** (33.46 -> 7.09 ms, `results/ablation/latency.jsonl`), and `torch.compile`, the mode that mattered most there (5.8x), only applies to this checkpoint after the sampler rewrite (section 4): on the fast sampler it is 65.6 ms against 72.4 ms stock eager at N=4, about a tenth, because a 262.7 M-parameter network at batch 1 is compute-bound (`results/stack.jsonl`, E14). So the launch-overhead result does not transfer: at 262.7 M parameters the policy is compute-bound and graphs are worth 5 %. Coupled with E3, that fixes the deployment envelope: a 100 ms control period admits only N <= 4 (66.8 ms graph, 73.5 ms eager), and N=4 is exactly where E3's coverage ladder reaches its plateau (0.885), with latency then growing linearly in N (see E3). | `results/latency.jsonl`, `results/tables/e9_latency.md` | A rerun on an idle GPU with the same protocol shows graphs cutting pretrained latency by more than 1.2x, or shows N=10 fitting inside a 100 ms period. |
| **E8 (half precision, pretrained checkpoint, 40 paired episodes per arm at N=4).** bf16 is 1.25x cheaper per chunk than fp32 at N=4 (62.7 ms vs 79.3 ms in the episode runs) and 1.37x at N=10 (152.3 vs 209.0 ms eager; 138.4 vs 194.4 ms with a CUDA graph), and it more than halves peak VRAM (710 vs 1189 MiB). **It does not buy a control period:** at N=10 the best half-precision number is 136.7 ms (fp16 + graph), still 1.4x over 100 ms, so the deployable envelope stays N <= 4 and the dial is still the step count, not the arithmetic. On quality the arms are indistinguishable at this sample size: success 35.0 % [22.5 %, 50.0 %] fp32, 50.0 % [35.0 %, 65.0 %] bf16, 45.0 % [30.0 %, 60.0 %] fp16, coverage 0.900 / 0.885 / 0.901. Every interval overlaps, and 40 episodes cannot resolve the 2 pp the pre-registered hypothesis named, so the result is "no gross regression", not "equal to fp32". | `results/precision.jsonl`, `results/tables/e8_precision.md` | A rerun at 100+ paired episodes separates two success intervals by more than 10 pp, or a half-precision N=10 row lands under 100 ms. |
| **E15 (generality, second published checkpoint, the checkpoint's whole 500-episode block on eval seeds 100000-100499, 2026-10-28, extended from 200 on 2026-10-30).** `lerobot/vqbet_pusht` -- a behaviour transformer over discretized action tokens, 5 observation steps, a 5-action chunk, 37.5 M parameters -- runs through the same `PolicyLike`, rollout harness, latency protocol and control loop. Nothing outside `models/pretrained_vqbet.py` and a `config.json`-type dispatch in `load_policy` changed. Measured 54.8 % [50.6 %, 59.2 %] success and 0.798 [0.773, 0.824] coverage, against the checkpoint's published 57.0 % over the same 500 episodes. Chunk latency 9.2 ms eager at batch 1 (50 warmup + 200 timed calls, 199 MiB peak VRAM) and 1.83 ms per control step, against the diffusion checkpoint's 73.5 ms and 9.19 ms -- a 5x cheaper policy that the harness measured without knowing either family. **The same-seed difference is measured, and it is not there.** The checkpoint ships `eval_info.json` with one row per episode, so the comparison is paired instead of asserted, and the block was grown to the checkpoint's whole published block -- 500 shared seeds, because the 200-seed block could only resolve ~13 pp (`just power-audit`): 274/500 (54.8 %) here against 285/500 (57.0 %) there, 261 episodes agreeing, 114 succeeding only here and 125 only there, exact McNemar p = 0.52 -- the two runs are not distinguished, and that null is worth exactly 8.6 pp, the smallest difference 500 paired episodes can see at 80 % power. **The 2.5 pp this adapter actually differs by cannot be resolved against this checkpoint at all:** it would take 5,838 paired episodes, i.e. an adapter-free VQ-BeT reference this repo runs itself (~1.75 GPU-h per arm), which does not exist. So the claim is stated at the resolution the reference allows -- 8.6 pp -- rather than at the size of the effect. Mean coverage is 0.798 here against 0.776 there (a lower bound), so this repo reaches the goal slightly further on the seeds it wins. (The earlier "10 pp gap, 4 pp coverage gap" reading was a units error, corrected 2026-10-28: gym-pusht reports `reward = clip(coverage / 0.95, 0, 1)`, so the published `max_reward` column is coverage divided by 0.95, and its `success` flag is the same `coverage > 0.95` criterion this repo uses. `scripts/published_blocks.py` pins both checkpoints' blocks so the tests regenerate offline, and the table prints them under the rows.) The adapter is validated by the paired test and by landing inside the published band, not by matching episode for episode. | `results/vqbet.jsonl`, `results/vqbet_published.jsonl`, `results/vqbet_latency.jsonl`, `results/tables/generality.md` | A block that resolves better than 8.6 pp puts this adapter outside the published checkpoint's interval (which needs an adapter-free VQ-BeT reference, not more of ours), or a third family needs changes outside `models/` plus the `load_policy` dispatch. |
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
- **2026-10-28: `diffusers` moved into the locked `dev` group.** It was left to
  `just setup-pretrained` on 2026-10-27, which meant `tests/unit/test_fast_sampler.py` -- a unit
  test that compares this repo's DDIM loop against the library's scheduler -- could not be
  collected after a clean `uv sync`. CI's `unit` job runs exactly that sequence, so it had been
  failing since `9e5be8c`; the fix is one dependency, not a skipped test. `lerobot` itself stays
  outside the lockfile for the torch reason above.
- **2026-10-27: the checkpoint's normalization is applied by our code.** lerobot 0.6.1 moved
  normalization out of the policy into a processor pipeline that this 2024 checkpoint does
  not ship, so `from_pretrained` loads its eight `normalize_inputs.*` / `unnormalize_outputs.*`
  tensors as *unexpected keys* and drops them. `models/pretrained.py` reads them from
  `model.safetensors` and applies the checkpoint's own mapping (MEAN_STD for images, MIN_MAX
  for state and actions) instead of this repo's train-time normalizer. E0 is the gate that
  proves the mapping: 64.0 % success and 0.924 mean coverage at ddim@10 against the published
  65.4 % / 0.907 coverage (the card's 0.955 is `max_reward`, i.e. coverage / 0.95).
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
  C++ compiler for it. `src/policybench/models/fast_sampler.py` precomputes the coefficients
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
- **2026-10-28: E8 moved off the ablation checkpoints and onto the pretrained one.** It was
  pre-registered as "E1 checkpoints, 3 seeds, success drop under 2 pp". The E1 model is 11 M
  parameters, so its precision trade says nothing about the 262.7 M-parameter policy that is
  actually deployed; and with one checkpoint there are no training seeds to resample. The
  amended row is 40 paired episodes per dtype at N=4 plus a latency ladder at N=4/10, which
  keeps the hypothesis and changes the resolution: episode-level intervals over 40 episodes
  cannot resolve 2 pp, and the claims table says "no gross regression" rather than equality.
  Half precision is implemented for the fast sampler only -- the stock `diffusers` loop keeps
  its schedule on the host in fp32, and casting third-party arithmetic this repo has not
  verified would be a silent risk.
- **2026-10-28: E12's third strategy is a cross-fade, not model-level inpainting.** The
  pre-registered third arm was "inpainting-style blending where the already-committed actions
  condition the new chunk". That requires a policy that accepts action conditioning at
  inference; this checkpoint does not, and retraining it is out of scope (D11). The arm that
  shipped is `strategy="blend"`: on each replan the queue keeps its length and the first four
  actions ramp from the old chunk into the new one. It is reported as "blend" and the
  hypothesis it was supposed to test is recorded as refuted for that implementation, not
  quietly re-scoped.
- **2026-10-28: E10's loop also has a wall-clock paced mode, measured separately.**
  `--paced` sleeps to the tick instead of running in virtual time, and rows carry `late_ticks`
  plus lateness mean/p95/max (fixed first-tick offset included: the first tick's planning is
  part of the first period). Those rows go to `results/paced.jsonl` and
  `results/tables/paced_loop.md` rather than into `results/deploy.jsonl`, because they answer
  a different question (does the loop keep time?) from E10 (does stale observation hurt?).
  The original E10/E12 rows stay virtual-time and keep `virtual_time=true`.
- **2026-10-28: deploy rows now record `sampler_steps`.** E10/E12 rows written before this date
  carry only `steps` (environment steps taken), so the sampler step count has to be read from
  `config_id`; new rows name it explicitly. The paced rows were re-run after the fix rather
  than patched.
- **2026-10-28: the serving boundary is load-tested, and E3 has a second seed block.**
  `scripts/bench_serve.py` spawns the real `scripts/serve.py` and reports client round trip,
  server-side chunk time and the difference (`results/serve.jsonl`); docs/05 section 3 carries
  the numbers and the two findings (about 17 ms of JSON/socket per request, and two clients
  serializing on one GPU). `results/e3_replication.jsonl` is an independent 50-episode block
  per step count on eval seeds 30000-30049, disjoint from the E3 block at 20000-20049 and from
  every training seed, run to test whether the earlier ladder reproduced rather than to
  enlarge it. The two blocks are reported side by side in `results/tables/e3_replication.md`;
  the original E3 table is left untouched so a reader can see what the first sample said.
- **2026-10-28: the stock sampler is still what E0/E3/E10 rows were measured with.** The fast
  sampler is the one E14 measures for speed; the two agree numerically, and E0's 64.0 % is the
  anchor for both. Anything measured with `--sampler fast` is reported as such in its rows.

- **2026-10-29: E0's paired block was extended to 200 seeds, and its success hypothesis failed.**
  The 50-episode block (seeds 1000-1049) reported 32/50 against the checkpoint's 37/50, p = 0.40,
  which was read as "not distinguished". Keeping those seeds and adding 150 more (1000-1199)
  gives 104/200 (52.0 % [45.0 %, 58.5 %]) against 138/200 (69.0 %), exact McNemar p < 0.001; the
  first 50 reproduce exactly, so the earlier reading was a power problem, not a different result.
  Coverage is 0.895 [0.872, 0.917]; the block's own 0.915 is a **censored lower bound** --
  gym-pusht clips reward at 1.0, so all 138 of its successes on these seeds record exactly 0.95.
  The outcome gap has two components,
  now visible in `results/tables/generality.md`'s coverage bins: 64 episodes land in
  [0.90, 0.95), where this repo succeeds 0 times and the checkpoint 42 (the hard threshold
  meeting a small coverage deficit), and 32 episodes score below 0.90 here, where the checkpoint
  still succeeds 22 times -- rollouts that fail outright rather than narrowly. Neither metric
  reproduces once that censoring is accounted for (on the 32 episodes both runs fail, coverage is
  0.850 against 0.832), and no root cause is established -- E0's criterion in `docs/00` is now
  PARTIAL.
- **2026-10-29: VQ-BeT's 54 % was replicated on a disjoint block.** `just eval-vqbet-replication`
  ran seeds 100200-100399 (`results/vqbet_replication.jsonl`), which share no episode with the
  first block: 108/200 against 108/200, two-sided Fisher exact p = 1.00, coverage 0.777 against
  0.818. Independent blocks get an independent test; McNemar applies only to the paired ones.

- **2026-10-29: the diffusion gap is not action scaling, and not a global coverage offset.**
  Two diagnostics. (i) Their per-episode coverage is censored: gym-pusht reports
  `reward = clip(coverage / 0.95, 0, 1)`, so all 138 of their successes on the 200 shared seeds
  record exactly 0.95 and their 0.915 mean is a lower bound. On the 32 episodes both runs fail --
  the only uncensored comparison -- coverage is 0.850 here against 0.832: parity, which rules out
  both a multiplicative action-scale error and a global input-normalization offset. (ii)
  `results/e0_published_seeds_ddpm.jsonl` (50 shared seeds, the checkpoint's own DDPM@100,
  `just eval-dp-published-ddpm`) is 30/50 (60.0 %) against their 37/50 (74.0 %), exact McNemar
  p = 0.14 -- no better than our DDIM@10 at 32/50 (p = 0.40) and ten times slower, so truncated
  sampling is not the cause either. What remains is an upper-tail difference: of the 64 episodes
  that land in [0.90, 0.95) here, the checkpoint crosses 0.95 in 42, and a +0.02 coverage shift
  would flip 35 of the 64. That is consistent with a small upper-tail shift whose source is not
  identified; note also that the initial noise draw is ours (seeded per episode), not the
  checkpoint's, so per-seed outcomes are not expected to align exactly (both-fail coverage
  correlation -0.02).

- **2026-10-30: the 200-seed DDPM@100 cell is filled, and it splits the gap.** `just
  eval-dp-published-ddpm` re-ran this repo's adapter at the checkpoint's own published sampler
  (DDPM, 100 steps, stock schedule) on the same 200 seeds: 118/200 (59.0 %) against DDIM@10's
  104/200 (52.0 %) on identical episodes, 37/51 discordant, exact McNemar p = 0.17 -- not separated
  in aggregate, +7.0 pp in point estimate, mean coverage unchanged (0.893 vs 0.895), at 4.4x the
  per-chunk cost (1819 ms vs 417 ms). Per 50-seed block (post-hoc, four looks): the deficit on
  1100-1149 shrinks from 18/50 (36.0 %) to 30/50 (60.0 %) against the checkpoint's 34/50 (68.0 %),
  discordant 5/17, p = 0.017 -- the largest effect in the split, but outside the 0.0125 Bonferroni
  bar for four looks, so it is directional; 1050-1099 gains (25 -> 35, p = 0.064) and 1150-1199
  loses (29 -> 23, p = 0.26). This amends the 2026-10-29 entry's "truncated sampling is not the
  cause": it is worth ~7 pp and it is not the whole cause, so the ~10 pp that survives at the
  published sampler is adapter or process -- the thing a processor adoption (docs/08 option H)
  would test. E0's criterion stays PARTIAL, because the claim is about the deployment sampler and
  the published sampler does not reproduce the published block here either. Counting this test the
  family would be seven and Bonferroni would set 0.007; A is not quoted in the README, so no quoted
  sentence moves.
- **2026-10-29: the published 69 % cannot be re-run on this stack, and that bounds the gap.**
  The planned discriminator was to run the checkpoint's *own* eval script
  (`lerobot-eval --env.type=pusht`) in this environment on the same seeds: a 69 % result would
  blame our adapter, a 52 % result would exculpate it. It does not run, for two separate
  reasons, both measured. (i) `av` 19.0.1 breaks the import
  (`module 'av' has no attribute 'option'`; lerobot's own extra caps `av<16`). (ii) After
  pinning `av` 15.1.0 the run reaches `make_policy`/`make_pre_post_processors` and raises
  `ProcessorMigrationError: ... Config file 'policy_preprocessor.json' not found` -- the 2024
  checkpoint predates the 2026 processor format, which is exactly the mismatch
  `models/pretrained.py` exists to bridge. So the checkpoint's own eval path cannot regenerate
  its published block in this library generation: "their 138/200" is a pinned artifact, not a
  run this host can repeat. The `av` pin was a local experiment only -- `uv.lock` is unchanged
  and `uv sync` restores 19.0.1. *(Superseded 2026-10-30 -- both obstacles turned out to be
  workable: rebuilt processors plus a pinned `av` in an isolated env give 134/200 (67.0 %) from
  lerobot's own loop, p = 0.75 against that block. See the control entry below.)*
- **2026-10-29: the remaining mechanisms were checked and excluded one at a time.** (a)
  *Normalization buffers*: all eight read from `model.safetensors` have the right shapes -- the
  means/stds are `(3,1,1)`, so the subtraction is per channel rather than over the last axis --
  and the mapping matches the checkpoint's own `normalization_mapping` (VISUAL MEAN_STD, STATE
  and ACTION MIN_MAX). (b) *The 84x84 crop*: `crop_shape=[84,84]` is applied inside
  `DiffusionRgbEncoder.forward`, so both the stock and the fast path get it; the adapter cannot
  be skipping it, and it also means a 384-px observation fed without a resize would be
  catastrophic rather than a 2 pp effect. (c) *Action slicing*: `generate_actions` returns
  `actions[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps]` (indices 1-8) and the rollout
  executes indices 0-7 of that window, so the action applied is the one predicted for the
  newest observation in both loops -- no one-step offset. (d) *Observation rendering*:
  gym-pusht draws at 512 and resizes to the requested size, so a 96-px frame and a resampled
  larger frame are the same picture, and the only size-dependent drawing (the action marker)
  is drawn for visualization only. (e) *Observation history*: `select_action` fills its 2-frame
  queue before predicting, padding by repeating the first frame, which is what the rollout
  deque does at reset. With (a)-(e) excluded, plus DDPM@100 at p = 0.14, nothing remains in
  the adapter or the loop to point at: the gap stays unexplained and is reported that way.

- **2026-10-30: the diffusion gap is not the noise stream this repo draws.** `just eval-dp-noise`
  re-ran the same 200 published seeds (1000-1199) with the same sampler and only the per-episode
  policy noise seed shifted (`--noise-seed-offset 500000`,
  `results/e0_published_seeds_noise2.jsonl`): 91/200 (45.5 %) against 104/200 (52.0 %) on the
  original stream, exact McNemar p = 0.21 (Fisher p = 0.23), coverage 0.884 against 0.895. The
  two blocks share every environment seed and differ only in which Gaussian the sampler starts
  from. They agree on 109/200 episodes (54.5 %), so the initial noise does decide individual
  episodes -- it is the loosest joint in the pipeline -- but it does not move the rate, and it
  certainly does not move it toward 69 %: against the checkpoint's block the second stream is
  23 vs 70 discordant, p < 0.001. Any explanation of the gap has to survive both streams.
  `rollout.evaluate(noise_seed_offset=...)` defaults to 0, so every earlier row is unchanged.
- **2026-10-30: the adapter-free control runs, after two migrations.** `lerobot-eval` cannot run
  on this host as shipped, for a reason the earlier attempt only half found: the checkpoint ships
  no processor files (its normalization lived inside `model.safetensors`), which is what
  `models/pretrained.py` exists to bridge. lerobot's own fix is
  `lerobot.processor.migrate_policy_normalization`; it rebuilds `policy_preprocessor.json` and
  `policy_postprocessor.json` correctly, then dies in its model save inside draccus' encoder
  (`Exception: Couldn't encode 84`, raised on this checkpoint's `crop_shape: [84, 84]`) leaving a
  0-byte `config.json` and no weights. `scripts/lerobot_migrate.py` runs lerobot's migration,
  finishes those two writes unchanged (the same weights minus the eight normalization keys, and
  the config minus `normalization_mapping`), and verifies the directory loads through
  `DiffusionPolicy.from_pretrained` -- 262.7 M parameters, `n_action_steps=8`, `horizon=16`.
  `scripts/lerobot_control.sh` then runs `lerobot-eval` on the migrated copy in an isolated
  `.venv-lerobot` (lerobot 0.6.1, av 15.1.0, torch 2.14.1+cu130), on seeds 1000-1199, with
  `use_async_envs=false` because the async vector env asks for a `forkserver` context that does
  not exist on Windows. `lerobot-eval` seeds episode *i* with `--seed + i`, so `--seed=1000` is
  the same 200 episodes this repo scores 52 % on. The migrated copy runs the checkpoint's own
  DDPM@100, i.e. the configuration its 65.4 % block was measured with.
- **2026-10-30: the control reproduces the checkpoint (67.0 % against 69.0 %), so the shortfall
  is in this repo's path.** `just lerobot-control` scored 134/200 (67.0 %) on seeds 1000-1199
  with mean coverage 0.885 (censored: `max_reward * 0.95`) in 2042 s of eval time
  (`results/lerobot_control/eval_info.json`). Against the checkpoint's own block on the same
  seeds -- 138/200, 69.0 % -- Fisher exact p = 0.75: **their published number is reproducible on
  this host**, which corrects the 2026-10-29 reading of the same block as a pinned artifact of
  another library generation. Against this repo's 104/200 (52.0 %) on the same episodes, Fisher
  p = 0.003: a 15 pp shortfall with the same weights, the same seeds and the same 96x96
  observations (checked: lerobot's `PushtEnv` declares 384 but gym-pusht renders 96, exactly as
  here).
  What this does *not* yet say is which stage is short. The control runs the checkpoint's own
  DDPM@100 while E0's 52 % is DDIM@10, so the 15 pp is split, not assigned. The 50-seed DDPM@100
  row (30/50 = 60.0 % against 37/50) sits between the two ends, consistent with the sampler
  carrying part of it, but p = 0.14 at that sample resolves nothing. The missing cell is this
  repo's adapter at DDPM@100 on all 200 seeds; until it is run, the supported statement is that
  the reference is reproducible and this repo's path is short of it by an amount not yet
  attributed. **Superseded later the same day** by the row-level paired control (two entries
  below): that cell was run, it does not separate from the control, and the 15 pp is a
  cross-sampler artefact rather than an adapter shortfall.

- **2026-10-30: rule 4 is amended for the processor load path (option H), on four conditions.**
  The D11 amendment left "the normalization glue" on the list of things that stay ours. H may
  replace it with `lerobot`'s own processor pipeline (`policy_preprocessor.json` /
  `policy_postprocessor.json`, produced by `scripts/lerobot_migrate.py`) for the diffusion load
  path only, under `docs/10` §1: (1) scope is that one mapping inside `models/`, with `eval/`,
  `deploy/`, the row schema and the generators untouched; (2) the buffer path stays the default
  and a caller opts in explicitly; (3) the adoption is one revertable commit on its own branch;
  (4) the library's JSON is never edited -- local deviations are constructed in this repo's code.
  **Nothing has been adopted, and no learning rate of GPU time has been spent on it.** The
  decision rule is deterministic: `scripts/processor_equiv.py` compares the two implementations
  on 20 real frames and 1e-6 is the bar. A pass ends the option, because a mapping that agrees to
  round-off cannot explain a success gap. Only a difference opens the episode arm, whose size is now
  set by a pilot rather than fixed at 600: at the ~4 pp residual the repeats measured, paired-binary
  power costs 243 episodes when the two arms disagree on 5 % of episodes and 2,157 when they
  disagree on 44 %, so the observed discordance decides (`just arm-power`, `docs/10` §5.1). Reason for the amendment: the
  residual after A is ~8 pp at the checkpoint's own sampler (118/200 against the control's
  134/200, and the control runs `lerobot`'s normalization), coverage is flat across every arm
  (0.884-0.895), and the two implementations of one declared mapping have never been compared
  numerically. Design, audit and runbook: `docs/10`.
- **2026-10-30: the paired control is null at the checkpoint's own sampler.**
  `just lerobot-control-rows` re-ran the adapter-free control on seeds 1000-1199 writing one row
  per episode (`results/lerobot_control/rows.jsonl`, 200 rows, unique seeds 1000-1199): **126/200
  (63.0 %)** for the control against this repo's DDPM@100 118/200 (59.0 %) -- 78 both, 40 ours
  only, 48 control only, 34 neither, exact McNemar **p = 0.46**. At the sampler the checkpoint's
  own numbers were produced with, the two loops are **not separated**. The control separates only
  from DDIM@10 (126 vs 104, 35 / 57 discordant, p = 0.03), and that is a statement about truncated
  sampling, not about the adapter. **Amended later the same day by the repeats entry below:**
  this entry read the day's 134/200 and 126/200 as one command disagreeing with itself by 4 pp.
  They came from two different entry points into the same loop (`just lerobot-control`, which
  writes only an aggregate, and `just lerobot-control-rows`, which writes rows), and five repeats
  of the row writer land in 61.5-64.0 %, so the 4 pp was never that path's spread. This supersedes
  the 2026-10-30 reading of the same control as a 15 pp shortfall: 15 pp compares DDIM@10 with a
  DDPM@100 control.
- **2026-10-30: the two normalization implementations are the same arithmetic (option H, null).**
  `just processor-equiv` compared the checkpoint's eight buffers against `lerobot`'s processor
  pipelines on 20 real frames from seeds 1000-1019 (`results/processor_equiv.json`): migrated
  weights identical (0.000e+00), images 2.384e-07 (one fp32 ulp), state 0.0, actions 0.0, and
  history ordering 0.0 between the batched and per-frame paths -- all inside the pre-registered
  1e-6 bar, so `pass: true`. The mapping declared by `normalization_mapping` is not where the
  residual lives, H is declined as an explanation, and `normalization="processors"` stays
  default-off and unmeasured (`docs/10` §3).
- **2026-10-30: the env wrapper is identical too, so R5's environment half is closed.**
  `just env-equiv` stepped seed 1000 through this repo's `PushTEnv` and `lerobot`'s env stack with
  one fixed 300-action sequence (`results/env_equiv.json`): per-step coverage and success traces
  agree exactly (max abs diff 0.0), the termination step matches (300), the first observation is
  bit-identical, and both report `pixels` / `agent_pos`. The environment term inside every
  loop-to-loop comparison in this file is therefore zero. What remains open under R5 is only the
  original PushT code path, which this repo does not run.
- **2026-10-30: the control is not deterministic, so its spread is now measured, not assumed.**
  Before repeating the 200-episode block, the identical `just lerobot-control-rows` command was run
  twice on 20 seeds (1000-1019): **14/20 and then 13/20**, seven seeds flipping and ten of the
  twenty coverage values differing, one by 0.28. Nothing here can configure that away --
  `lerobot-eval` calls `set_seed(cfg.seed)` once, so the policy's noise stream is fixed for the run;
  what is left is kernel-level float nondeterminism, amplified by PushT's contact dynamics over 300
  steps near the 0.95 success line into whole-episode flips. A verdict on the ~8 pp residual
  therefore cannot rest on a single rerun's p-value: the control is measured over K repeats of the
  identical command (`scripts/control_repeats.sh`, `just control-repeats`) and the gap is quoted
  with a two-stage bootstrap that resamples episodes *and* runs
  (`policybench.eval.stats.paired_delta_ci`, `just control-spread`,
  `results/lerobot_control/spread.json`). An episodes-only interval would be too narrow by exactly
  the term the null depends on.
- **2026-10-30: the control's spread is 1.0 pp, so the residual is ~4 pp, not 8.** Four repeats of
  the identical row-writing command (`scripts/control_repeats.sh`, r2-r5, one worker, ~38 min each)
  join the original row file: 126, 128, 125, 123 and 127 of 200, i.e. **62.9 % with an sd of
  1.0 pp** (range 61.5-64.0 %, t interval over runs [61.7 %, 64.1 %]). Against this repo's DDPM@100
  118/200 (59.0 %) that is **+3.9 pp**, and resampling episodes and runs together
  (`stats.paired_delta_ci`) gives **[-5.5, +13.0] pp** at 95 % -- a null that spans zero, now stated
  with an interval instead of two anecdotes (`results/lerobot_control/spread.json`).
  The repeats also retire the 4 pp "run-to-run spread" that the entry above recorded: the
  134/200 (67.0 %) run came from `just lerobot-control`, the *aggregate* entry point
  (`lerobot-eval` directly, 2042 s), not from the row writer (2312 s), and it sits ~4 sd outside
  the five-run band. It is therefore one observation of a path that has never been repeated
  against itself, not a draw from the row writer's distribution. **The aggregate path was then
  repeated and lands at 126/200 (63.5 %)** (`results/lerobot_control/eval_info_plain_200.json`),
  inside the row writer's band, so the 67.0 % was one high draw rather than a property of that
  entry point: seven runs of the loop span 61.5-67.0 % around a 63.5 % mean. The paired analysis
  stays on the five runs that wrote rows, because only those pair by seed. Consequence for the
  quoted numbers: the
  ~8 pp residual computed against the single 134/200 run is superseded, because the control's own
  repeated centre is 62.9 % against that run's 67.0 %. What survives is a ~4 pp residual this
  design does not resolve at 200 episodes, still not attributed to the adapter.

## 5. Acceptance criteria
- [x] Harness test: eval/train seed overlap raises an error (`tests/unit/test_eval_seeds.py`, asserted in `rollout.evaluate`).
- [x] A figure script exists for every figure that was run: E0, E3 (both panels), E10 (success curve and latency histogram) and the ablation bar regenerates from raw JSONL with `just figures`. E11 was never run, so it has no figure.
- [x] This file was committed before the first experiment result (the claims table and the deviations log have been updated in place ever since, each entry dated).