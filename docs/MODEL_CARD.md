# Model card: the policy this stack serves, and the stack itself

Written 2026-10-27. Two things are described here, because the distinction matters:
the **borrowed policy** (not ours) and the **serving stack** (ours).

**What the stack is for:** it is a deployment-cost rig -- what one action chunk costs, whether
that fits a control period, and what breaks when it does not. It is not a policy, a training
recipe or a benchmark leaderboard; the policy is third-party and the measurements are about
running it.

Every measured number below is copied from `results/tables/*.md`, which
`scripts/make_tables.py` generates from the raw per-episode rows in `results/*.jsonl`; nothing
here is typed by hand. Published numbers come from the checkpoint's own model card (accessed
2026-10-27) and are labelled as such.

## 1. The policy (third-party)

| Field | Value |
|---|---|
| Checkpoint | `lerobot/diffusion_pusht` |
| Architecture | Diffusion Policy (Chi et al., 2023): ResNet-18 vision encoder + conditional 1-D U-Net, epsilon-prediction |
| Parameters | 262.7 M, fp32 (measured 2026-10-27: `results/tables/e0_validation.md`, 1.05 GB on disk) |
| Trained by | Hugging Face LeRobot team, 175k steps, batch 64, on `lerobot/pusht` |
| License | Apache-2.0 (model card, accessed 2026-10-27) |
| Observations | two 96x96 RGB frames (`n_obs_steps=2`) + 2-D agent position |
| Outputs | 16-step action chunk, 8 executed per replan (`n_action_steps=8`), 2-D actions in pixels [0, 512] |
| Normalization | images MEAN_STD (ImageNet), state and actions MIN_MAX (dataset extremes), shipped inside the checkpoint |
| Reported performance | 65.4 % success, 0.955 mean max overlap over 500 gym-pusht episodes (`lerobot/diffusion_pusht` model card, accessed 2026-10-27) |

**Intended use.** Simulation research on PushT: evaluating inference optimizations,
deployment loops and evaluation harnesses. **Not** intended for real hardware, for
safety-relevant decisions, or as a claim about Diffusion Policy's capabilities at scale: it
is one checkpoint on one simulated task.

## 2. This repository's stack (ours)

| Field | Value |
|---|---|
| What it provides | parallel rollout harness, success/coverage metrics with bootstrap intervals, batch-1 latency protocol, fixed-rate control loop with latency injection, temporal ensembling, JSONL-to-table/figure/README generators |
| Policy interface | `policybench.models.policies.PolicyLike` (`obs_horizon`, `pred_horizon`, `action_dim`, `predict_chunk(obs, **sampler_kw)`, `num_parameters()`) |
| Adapters | `PretrainedDiffusionPolicy` (HF checkpoint, own normalization) and `BasePolicy` (trained checkpoints, from-scratch ablation) |
| Reproduced reference | E0: 64.0 % success (32/50, 95 % CI [52.0 %, 78.0 %]) and 0.924 mean coverage at DDIM N=10 on seeds 10000-10049, against the model card's 65.4 % / 0.907 coverage. On the checkpoint's own seed block at 200 paired episodes **neither success nor coverage is shown to reproduce**: 104/200 here against the checkpoint's 138/200 (exact McNemar p < 0.001), and 0.895 [0.872, 0.917] against a 0.915 that is only a lower bound -- reward is clipped at 1.0, so every success records exactly 0.95. On the 32 episodes both runs fail, coverage matches (0.850 against 0.832). The first 50 seeds of that run are 32/50 against 37/50 (p = 0.40), so the earlier "within interval" reading was a sample-size artefact. The gap is **not** an adapter shortfall: through `lerobot`'s own eval loop the checkpoint reproduces (126/200 against its block's 69.0 %, Fisher p = 0.25, five repeats spanning 61.5-64.0 %), and at its own DDPM@100 sampler this stack reaches 118/200 (59.0 %) against that control's 126/200 -- a five-run mean of 62.9 % with a 1.0 pp standard deviation (61.5-64.0 %, 200 episodes per run) -- a difference the paired test does not separate (exact McNemar p = 0.46 over 40 / 48 discordant episodes) and that is no larger than the control's own measured spread: +3.9 pp, with a bootstrap over episodes and runs of [-5.5, +13.0] pp that spans zero. The other entry point into that loop scored 134/200 (67.0 %) once and 126/200 (63.5 %) when repeated, so the high run was a draw rather than an entry-point effect; seven runs of the loop span 61.5-67.0 % around a 63.5 % mean. The 15-point figure is a cross-sampler artefact (DDIM@10 against the control's DDPM@100), not a measured adapter loss; like-for-like the residual is about 4 points and this design does not resolve it at 200 episodes, and the two mechanisms that could have produced a real adapter loss are excluded: the checkpoint's eight normalization buffers and `lerobot`'s processors agree to 2.4e-07 over 20 real frames, and the two environment wrappers produce identical coverage traces (`just processor-equiv`, `just env-equiv`) |
| Environment | gym-pusht 0.1.8, 96x96, horizon 300 steps, success = max coverage >= 95 % |
| Compute | one RTX 5060 Laptop (8 GB), torch 2.14.1+cu130, no training performed by this project for the served policy |
| Samplers | stock (`lerobot` -> `diffusers`) and `fast` (this repo's DDIM loop: coefficients precomputed on-device, numerically equal, compilable and capturable) |
| Serving cost, measured | E14: the published DDPM@100 configuration, the stock DDIM path and the fast sampler, as seconds per episode and episodes/hour with success and coverage attached |

### Metrics with intervals (the operating-relevant ones)

- **Success at the operating point**: 64.0 % [52.0 %, 78.0 %], 50 paired episodes, DDIM N=10,
  seeds 10000-10049. The same configuration on E3's seed set (20000-20049) scores 54.0 %
  [40.0 %, 68.0 %]: **two 50-episode samples of one policy differ by 10 percentage points**,
  which is the resolution limit of this harness at this episode count.
- **Sampler steps**: coverage 0.121 at N=1, 0.716 at N=2, 0.885 at N=4, flat afterwards
  (0.888-0.919 at N=8/10/16, intervals overlapping). One-step DDIM is not usable.
- **Cost per chunk, batch 1**: 73.5 ms eager at N=4, 190.5 ms eager / 181.0 ms with a CUDA
  graph at N=10. The 100 ms control period therefore admits only N <= 4, where the coverage
  ladder has just plateaued.
- **Under injected observation latency** (0-200 ms): success 56.7 % -> 43.3 % with
  overlapping intervals at 30 episodes per row; coverage stays flat at 0.897-0.932. The
  binding constraint is inference cost, not the network: ~26-33 deadline misses per episode
  at N=10 against a 100 ms period.
- **Replanning every 2 steps instead of 8** (four chunks overlapping) is where the loop design
  starts to matter: naive replacement scores 10.0 % at 100 ms and 10.0 % at 200 ms, disjoint
  from the 56.7 % / 43.3 % of the 8-step rows, while temporal ensembling over the same chunks
  recovers 53.3 % / 43.3 % at 4x the chunk count. Serving this policy at a fast replan rate
  requires ensembling, and both fast-rate rows need ~120 deadline misses per episode.

### Limitations

- One checkpoint exists for PushT, so all intervals are over episodes; no seed-to-seed
  variance can be reported for the pretrained rows.
- All measurements are on one laptop GPU with two rollout workers for the eval rows and a
  single idle process for the latency rows (the two differ by roughly 2x, stated per table).
- E10's loop runs in virtual time; deadline misses are real (measured inference) but there is
  no wall-clock pacing or real-time scheduler.
- `torch.compile` and CUDA graphs are measured on this repo's DDIM sampler (E14), not on the
  stock one: the stock loop keeps the diffusion schedule on the host and cannot be compiled
  here without MSVC. Both are worth about a tenth at this model size, so the gap does not
  change any conclusion.
- Simulation only; robomimic, ROS 2 and real hardware were out of scope (D1, D6, D7).

### Data

`lerobot/pusht` (MIT, accessed 2026-10-19): 206 human demonstrations, 96x96, 10 fps, from the
Diffusion Policy PushT dataset. It is used by this repository only for the from-scratch
ablation; the served checkpoint's own training used the same public dataset.

### Out of scope

Training a policy (the ablation path exists but is not the deliverable), language
conditioning, multi-task generalization, quantization beyond the measured ladder, real-robot
transfer.
