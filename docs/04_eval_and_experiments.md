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

Compute column is TBD until throughput is measured in M2 ([07](07_roadmap.md)); fill with measured hours, not guesses.

| ID | Experiment | Hypothesis (falsifiable) | Config | Seeds | Compute | Output |
|---|---|---|---|---|---|---|
| E0 | Reproduce DP on PushT (strict success) | Our DDPM policy is within the CI of about 64% [S] | ddpm, image | 3 | TBD | table row + discussion if off |
| E1 | Flow vs DDPM vs ACT-style, PushT | Flow within 5 pp of DDPM at N=10 | 3 policies, same encoder/data | 3 | TBD | main table |
| E2 | Same, robomimic Lift (+Can) | Same ordering as E1 or not (report either) | 3 policies | 3 | TBD | main table |
| E3 | Integration steps N in {1,2,4,8,10,16}; DDIM steps matched | Flow degrades less than DDIM as N falls | E1 checkpoints, no retrain | 3 | TBD | HEADLINE fig: N vs success vs latency |
| E4 | Chunk lengths (Ta in {1,4,8,16}) | Longer Ta hurts under disturbance | flow, DDPM | 3 | TBD | fig |
| E5 | Timestep sampling: uniform vs Beta | No difference at this scale (null is fine) | flow | 3 | TBD | table |
| E6 | Head: U-Net vs transformer | U-Net at least as good at equal params | flow | 3 | TBD | table |
| E7 | Encoder: ResNet-18 GN vs cached DINOv2 (D3) | Cached features trade augmentation for speed | flow | 3 | TBD | table, only if time |
| E8 | Precision: bf16 vs fp16 (fp8/int8 if supported) | Success drop under 2 pp [proposed] | E1 checkpoints | 3 | TBD | table |
| E9 | Latency/compile ladder (eager, compile modes, CUDA graph, Triton) | Compile + graphs remove most launch overhead at batch 1 | inference only | n/a | TBD | latency table |
| E10 | Injected latency 0/50/100/150/200 ms + jitter | Success degrades monotonically; async chunking recovers part | deployment loop | 3 | TBD | fig |
| E11 | Disturbance recovery (pose perturbation mid-episode) | Shorter Ta and flow recover better | deployment loop | 3 | TBD | fig |
| E12 | Async chunk strategies: naive replace vs temporal ensemble vs inpainting-style (D-level) | Inpainting-style best under latency | flow | 3 | TBD | table |

Priority order if time shrinks: E0, E1, E3, E9, E10, E2, E4, E11, E8, E5, E6, E12, E7.

## 3. Claims table (fill only from results)

| Claim template | Evidence (file) | Condition under which it is false |
|---|---|---|
| Flow matches DDPM success on PushT with N steps, X times faster (CI ...) | TBD | CIs overlap by less than meaningful difference rule fails, or different eval seeds |
| Success degrades by Y pp under 100 ms latency; async chunking recovers Z pp | TBD | single seed, or latency not measured on the control path |

## 4. Deviations log

(empty)

## 5. Acceptance criteria

- [ ] Harness test: eval/train seed overlap raises an error.
- [ ] A figure script exists for every figure in E3, E10, E11 and regenerates from raw JSONL with one command.
- [ ] This file is committed before the first experiment result.
