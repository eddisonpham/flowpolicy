# 03. Data, models, training

**Status 2026-10-27 (D11): this file documents the ablation path.** The stack now runs the
pretrained checkpoint `lerobot/diffusion_pusht`; nothing here is on the critical path, but the
trainer, the policies and this recipe stay runnable (`just train-e1`) because the ablation rows
in `results/ablation/` are the evidence for the pivot (4-7 % success at 12000 steps is a floor
effect). The recipe table in section 2 is still the reference these local runs fall short of.

Tags: [V] verified, [S] secondary source, [U] unverified. Access date for all sources: 2026-10-18. Links: [01](01_architecture.md) [04](04_eval_and_experiments.md)

## 1. Data

| Benchmark | Source | Format | Notes |
|---|---|---|---|
| PushT | Diffusion Policy public demos (about 200 human demos) [S]; also on Hugging Face as `lerobot/pusht` [S] | LeRobot parquet + AV1 video, converted to our Zarr | **Used source: `lerobot/pusht` v3.0, MIT licence (dataset card, accessed 2026-10-19), 206 episodes / 25 650 frames / 96x96 / 10 fps.** Conversion script: `scripts/data/prepare_pusht.py`; output `data/pusht/pusht.zarr`. |
| robomimic Lift, Can (Square if time) | robomimic datasets, moved to Hugging Face in robomimic v0.4 [S] | HDF5 | Convert to Zarr once. Repo names and license: [U]. |

Zarr schema (own, documented; PushT-DP layout assumed `data/{img,state,action}` plus `meta/episode_ends` [U], confirm on download):
`episode_ends[E]`, `image[N,H,W,3] uint8`, `state[N,Ds] f32`, `action[N,Da] f32`. Chunk along time with chunk size covering a few hundred steps; images stored uint8, converted on GPU.

Loader: sample windows of `To` past observations and `Tp` future actions, pad at episode edges by repeating the first/last frame (the Diffusion Policy convention, [U] confirm). Normalization: actions min-max to [-1, 1]; state min-max or standardize; images to [0,1] then ImageNet-style or none (pick one, record). Stats computed on the train split only and stored in the checkpoint. Augmentation for images: random crop (about 90% area) at train, center crop at eval [U confirm DP default].

Splits: demonstrations split by episode, never by window. Hold out a validation set of episodes for loss monitoring only; task success is measured by rollouts.

## 2. Reference recipe to reproduce (Diffusion Policy on PushT)

Sourced from papers/repos that replicate the official code and from the LeRobot port; none is the original paper table, so confirm against the paper and official repo first [U].

| Item | Value | Source |
|---|---|---|
| Obs horizon To / pred horizon Tp / action horizon Ta | 2 / 16 / 8 | [S] arXiv 2606.24403, 2606.10371 |
| Encoder | ResNet-18, GroupNorm | [S] arXiv 2606.10371 |
| Action head | ConditionalUnet1D, sinusoidal time embed | [S] arXiv 2606.24403 |
| Diffusion | DDPM 100 steps, squared-cosine beta, epsilon-prediction, clipped samples; 10-step DDIM at inference | [S] arXiv 2606.24403 |
| Training length | image: 3000 epochs, state: 4500 epochs (as replicated) | [S] arXiv 2606.24403 |
| LeRobot port | batch 64, lr 1e-4, cosine, 500 warmup, Adam betas (0.95, 0.999), eps 1e-8, wd 1e-6, grad clip 10, 200k steps, 84x84 images, 10 fps | [S] HF `osanseviero/diffusion_pusht` config |
| Reference success (gym-pusht, success = coverage at least 95%) | about 63.8% (LeRobot) vs 64.2% (original code), 500 episodes | [S] HF `lerobot/diffusion_pusht` card |

Important: the original paper reports a coverage-based score; strict success (at least 95% coverage) is lower. The reproduction target is therefore defined on the strict metric against the roughly 64% figure above, with the paper's number discussed separately. Agent: pull the paper's exact numbers and metric definition into this table before experiment E0.

## 3. Policies

Common: shared encoder (ResNet-18 GN, global avg pool or spatial softmax; state concatenated), cond vector `c` = flatten of encoded To frames. Parameter budget 20 to 50 M; head sizes chosen to hit this, final counts TBD (measured).

**Flow matching (headline).** Convention used here: `t=0` noise, `t=1` data. `x_t = (1-t) * eps + t * a`, `eps ~ N(0,I)`, target velocity `v* = a - eps`, loss `MSE(v_theta(x_t, t, c), v*)`. Sample: `x_0 = eps`, Euler `x_{k+1} = x_k + (1/N) * v_theta(x_k, k/N, c)`, optional Heun/midpoint. pi0 uses the reverse time convention (noise at t=1) and a Beta-shaped timestep distribution; exact form [U], read the pi0 paper before implementing the Beta ablation. Timestep sampling: uniform (default), Beta (ablation). Time embedding: sinusoidal, then FiLM (U-Net) or AdaLN (transformer).

**DDPM/DDIM (baseline).** Epsilon-prediction, squared-cosine schedule, 100 train steps, DDIM sampler with configurable steps; recipe in section 2. Same head and encoder as flow.

**ACT-style regression (baseline).** Transformer encoder-decoder predicting a chunk by L1 regression; CVAE latent optional; temporal ensembling optional. Original details (chunk size, KL weight, ensembling decay) [U], verify from the ACT paper; default here: deterministic, no CVAE, to keep the comparison clean, with CVAE as a flagged variant.

**Heads.** Primary: 1D U-Net with FiLM (D4 default). Ablation: small transformer with AdaLN.

## 4. Memory plan (to measure, not guess)

Estimate by component: parameters x bytes (bf16 weights plus fp32 master and Adam states, about 12 to 16 bytes per param trainable, [U] confirm), activations scale with batch x To x image size. Procedure: `torch.cuda.reset_peak_memory_stats()`, run 20 steps at batch sizes 32/64/128/256, record `max_memory_allocated`, fill the table below. Anything above the measured free VRAM minus 0.5 GiB is flagged "does not fit"; mitigations in order: smaller batch, activation checkpointing in the encoder, 84x84 images, gradient accumulation.

Measured 2026-10-19 on the RTX 5060 Laptop (7.93 GiB total; 6.98 GiB free before the run), 96x96, To=2, Tp=16, bf16, 10.95 M parameters. Raw rows: `results/ablation/memory.jsonl`; table: `results/tables/m2_memory.md`.

| Model + data | Batch | Peak VRAM | Steps/s | Fits |
|---|---|---|---|---|
| flow 96px | 32 | 2317 MiB | 7.80 | yes |
| flow 96px | 64 | 4414 MiB | 4.00 | yes |
| flow 96px | 128 | 8616 MiB | 0.05 | no (fits flag uses free VRAM minus 0.5 GiB) |

The 1800-step E1 runs used batch 64, which peaks at 4.41 GiB and leaves room for the desktop. Batch 128 is reported as measured (it did not OOM on an otherwise empty GPU) but is flagged `fits: false` by the pre-registered free-VRAM rule.

## 5. Training infrastructure

- bf16 autocast, grad clip, cosine LR with warmup, AdamW; EMA (decay and warmup schedule as in the DP recipe [U]); EMA weights used for all evaluation.
- Checkpoint contents: model, EMA, optimizer, scheduler, step, Python/NumPy/Torch/CUDA RNG states, sampler state (epoch, position, seed), normalizer stats, resolved Hydra config, git hash, `uv.lock` hash.
- Signals: SIGTERM/SIGINT write a final checkpoint then exit 0. Periodic checkpoints keep the last 3 plus every K evals.
- **Resume equivalence** (defined): interrupt at step k, resume, compare per-step losses for next M steps to an uninterrupted run. CPU tiny-model test in CI at tolerance `rtol=1e-5` [proposed]; the doc and README say "bit-exact" only if a `torch.equal` version of the test passes with deterministic algorithms and fixed worker seeding. Otherwise say "tolerance-based".
- Logging: W&B (offline-capable) plus local JSONL of every logged scalar.
- Hydra: structured configs from dataclasses; `conf/experiment/<id>.yaml` per experiment in [04](04_eval_and_experiments.md); multirun for seeds.

## 6. Common failure modes and tests

| Failure | Check (test) |
|---|---|
| Normalizer fit on val/test or not saved | unit: stats equal train-only stats; round-trip in checkpoint |
| Action de-normalization missing at rollout | unit: `predict_chunk` output within raw action range on a fixed batch |
| BatchNorm left in encoder | unit: assert no `BatchNorm` modules |
| EMA used for training eval or not used at all | integration: eval log records which weights were used |
| Obs/action horizon off-by-one | unit: window sampler returns expected indices on a toy episode |
| Execution horizon bigger than prediction horizon | config validation error |
| Eval seeds overlap train seeds | eval harness asserts disjoint seed sets |
| Time convention flipped between train and sample | unit: sampling a trained 1-D toy distribution recovers its mean/variance |
| Overfit sanity | integration: each policy overfits 8 episodes to near-zero loss |

## 7. Acceptance criteria

- [ ] Recipe table in section 2 has paper-sourced rows and metric definition. **Not met, and
  left open deliberately:** the recipe is sourced from replicating papers and the LeRobot port
  and tagged `[S]`/`[U]`, not read off the Diffusion Policy paper. Under D11 the from-scratch
  recipe is the ablation path, so verifying it against the paper would not change any result.
  The metric definition (max overlap >= 95 %) *is* anchored: E0 reproduces the published
  64.0 % vs 65.4 % with it.
- [x] Each policy overfits a tiny set; loss curves committed in `results/`. Done 2026-10-19:
  flow 1.230 -> 0.065, DDPM 1.096 -> 0.091, regression 0.376 -> 0.029 on 8 episodes;
  `results/overfit/summary.json`, `results/tables/overfit.md`.
- [x] Memory table filled by measurement; no row left TBD at training time. Done 2026-10-19:
  `results/ablation/memory.jsonl`, `results/tables/m2_memory.md` (batch 32/64/128 measured, the
  128 row flagged as not fitting rather than omitted).
- [x] Resume-equivalence test in CI; wording on determinism matches test outcome. Done: the
  tolerance-based test runs in the `unit` job and the 20-step end-to-end run in the `smoke`
  job; docs, code and README say "tolerance-based (`rtol=1e-5`)", never "bit-exact".
