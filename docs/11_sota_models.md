# 11. SOTA policies, their setup, and what this repository should wrap

Tags: [V] verified against a primary source or this tree, [S] secondary (vendor blog, third-party
report), [U] unverified. Dated 2026-10-30. Companion to `docs/08` (menu) and `docs/10` (the
processor adoption that this page generalises).

## 0. The framing, stated once

This repository is a **deployment wrapper**, not a policy lab. That has been true since D11
(2026-10-27), when the measured stack stopped using the from-scratch models and started running
`lerobot/diffusion_pusht` through unchanged eval, latency, control-loop and serving code. This
page asks the obvious next question: *which published models should the wrapper wrap next, and
what does each one cost to adopt?*

Three rules follow from the precedent already in the tree, and they are the whole method here:

1. **Do not train what can be downloaded.** The training half of this repo stays subordinate.
   Adoption means a checkpoint plus an adapter, not a fine-tuning project.
2. **Do not reimplement a policy.** `models/pretrained.py` already owns glue only -- it imports
   `lerobot`'s `DiffusionPolicy` and `diffusers`' scheduler. Every candidate below is adopted the
   same way, or not at all.
3. **Do not reimplement a serving protocol either -- but keep the measurement.** Upstream ships
   policy servers now (openpi websocket, GR00T ZMQ, `lerobot` gRPC). What stays ours is the thing
   none of them have: paired, seed-exact, interval-bearing numbers for the same episodes.

The corollary matters for scoping: a model that cannot be run on this host's 8 GB GPU or a
rented 24 GB card, and has no published simulation cell to check against, is not adoptable at
any price this repo is willing to pay.

## 1. Landscape, 2026-10-30

Filtered to open weights with an action-chunk interface and a published sim result. Closed APIs
(Gemini Robotics, Figure Helix) are out by definition: they cannot be wrapped, only called.

| Family | Params | Licence | Action interface | Chunk | Inference memory | Fits 8 GB | Published sim cell to check against | Source |
|---|---|---|---|---|---|---|---|---|
| Diffusion Policy (`lerobot/diffusion_pusht`) | 262.7 M | Apache-2.0 | 2-D pixel actions, DDPM/DDIM | 16 pred / 8 exec | 1.19 GiB fp32 [V, E9] | yes | 65.4 % PushT [V, E0] | in tree |
| VQ-BeT (`lerobot/vqbet_pusht`) | 37.5 M | Apache-2.0 | discretized action tokens | 5 | small [V, E15] | yes | 54 % PushT block [V, E15] | in tree |
| ACT | ~50 M | Apache-2.0 | joint positions, L1 regression | 100 | small | yes | ALOHA / LIBERO suites | lerobot |
| SmolVLA base | 450 M | Apache-2.0 [S] | 32-padded continuous, flow matching, 10 steps | 50 | ~2 GB [V, lerobot docs] | yes | LIBERO ~81.5 % spatial reproduced [S] | lerobot |
| π0 | 3.3 B | Apache-2.0 [S] | 32-padded continuous, flow matching, 10 steps | 50 | 14 GB [V, lerobot docs]; openpi asks >8 GB [V] | **no** | LIBERO / ALOHA / DROID | openpi, lerobot `pi0` |
| π0.5 | 3.6 B | Apache-2.0 [S] | same as π0 (flow head only in openpi) | 50 | ~14 GB [S] | **no** | `pi05_libero` SOTA cell [V, openpi README] | openpi, lerobot `pi05` |
| π0-FAST | 3.3 B | Apache-2.0 [S] | autoregressive FAST tokens | 50 | ~14 GB+ (KV cache) | no | DROID 0-shot [V] | openpi `pi0_fast` |
| GR00T N1.7 | 3 B | NVIDIA Open Model Licence (commercial) [V] | flow-matching DiT, state-relative option | 40 (132-wide) | not stated; Jetson Thor-class [S] | no | LIBERO 95 / 100 / 98 / 93 [V, lerobot docs] | lerobot `groot` |
| OpenVLA / OFT | 7 B | Llama-2 community [S] | OFT: parallel decode + 8-action chunk, continuous | 8 | ~16 GB [S] | no | LIBERO 97.1 % [S] | third-party |
| X-VLA, Wall-X, MolmoAct2, LingBot-VA, VLA-JEPA, EVO1, FastWAM, multi_task_dit | varies | varies | varies | varies | varies | mostly no | varies | **shipped in lerobot 0.6.1** [V] |
| π*0.6 (RECAP) | 3.6 B | **not released** [U] | as π0.5 | 50 | — | no | laundry / espresso throughput [S] | — |

Two rows decide everything on this hardware: **SmolVLA is the only VLA that fits**, and
**π0/π0.5/GR00T are the ones worth renting a GPU for** because their published cells are the
only external checks strong enough to falsify an adapter.

## 2. The model zoo is already a dependency

Speculation is unnecessary: `pyproject.toml` pins `lerobot` 0.6.1, and its installed
`policies/` directory contains `act`, `diffusion`, `vqbet`, `pi0`, `pi05`, `pi0_fast`, `smolvla`,
`groot`, `wall_x`, `xvla`, `molmoact2`, `lingbot_va`, `fastwam`, `evo1`, `neo1`, `vla_jepa`,
`multi_task_dit`, `gaussian_actor`, `tdmpc`, plus `rtc/` [V, read 2026-10-30]. `envs/` contains
`libero.py`, `metaworld.py`, `robocasa.py`, `robotmme.py`, `robotwin.py`, `vlabench.py` [V].

So "use actual models" costs **zero new dependencies and zero new policy code**. It costs one
adapter and one protocol change (section 3).

Their contracts, read out of the installed config classes [V]:

| Config | `chunk_size` | `n_action_steps` | state / action width | inference steps | dtype |
|---|---|---|---|---|---|
| `SmolVLAConfig` | 50 | 50 | 32 / 32 (padded) | flow, 10 | fp32/bf16 |
| `PI0Config` | 50 | 50 | 32 / 32 (padded) | 10 | bf16 or fp32 only |
| `PI05Config` | 50 | 50 | 32 / 32 (padded) | 10 | bf16 or fp32 only |
| `GrootConfig` | 40 | 40 | 132 / 132 | flow | bf16 |

That is the same shape `PolicyLike` already carries -- horizon, action dim, a chunk of
`n_action_steps` executed per replan, a denoising loop with a step count -- on a wider padded
action vector. The adapter is a translation table, not new mathematics. The existing
`models/pretrained_vqbet.py` (config filtering + the checkpoint's own normalization) is the
template.

## 3. The two seams that did not exist yet (landed 2026-10-31, no model)

`PolicyLike` is `predict_chunk(obs: dict[str, Tensor])`, where `obs` has exactly `image`
(single camera, `[To, H, W, 3]`) and `state` (`[To, Da]`), and the server contract is the same
[V, `docs/05` section 3]. Every VLA above needs two things it does not have:

1. **A language slot.** π0.5, GR00T, SmolVLA and OpenVLA are all conditioned on a task string.
   Today there is nowhere to put one. This is a protocol amendment, not a shim: a `task: str`
   field on the observation contract, threaded through rollout, the server and the control loop,
   with an empty string meaning "unconditional" so every existing row stays valid.
2. **N camera views.** `image` is one camera. VLA checkpoints are trained with 2-3 named views
   (`observation.images.<name>`), and `lerobot` maps feature keys positionally. A single-image
   adapter would silently feed the same frame twice, which produces plausible actions and a
   broken success rate -- exactly the failure mode `docs/10` was written to prevent.

Both are small, both are recorded here rather than guessed at, and both must land **before** any
VLA row exists in `results/`, because a row measured across the wrong observation contract is
worse than no row.

**Landed 2026-10-31 as step 1 of section 7, with no model attached.** `models/observation.py`
now holds `ObservationSpec` (ordered camera names, whether an instruction is required),
`spec_for(policy)` (reads the optional `camera_names` / `requires_instruction` attributes and
defaults to the historical `("image",)` + unconditional), and three validators. The language
slot is the `instruction=` call argument to `predict_chunk`, not a key in `obs`: `obs` is
`dict[str, Tensor]` and an instruction is a string, so it travels where sampler settings already
travel -- and every existing implementation swallows it through `**sampler_kw` unchanged.
n
The name is `instruction`, not `task`, because a row already has `task: "pusht"` meaning the
task *identity*; a language instruction is a different thing and must not share the key. Rows
gain an `instruction` field only when one was actually used, so no existing row changes meaning.
Enforced at all three boundaries: the rollout harness and the control loop refuse a policy whose
views the PushT env cannot supply (one), and the HTTP server refuses a payload that is short a
view or missing a required instruction, with `GET /health` advertising both. Tests:
`tests/unit/test_observation_contract.py` (15 cases: defaults and refusals) and five new cases in
`tests/unit/test_deploy_server.py` over a real socket.

Deliberately still absent: any adapter, any second camera in the env, and any multi-view client.
This seam exists so that adopting one is a translation table rather than a protocol change made
by accident -- it is not evidence that a VLA has been wrapped, and nothing in `results/` uses it.

## 4. Three setup shapes, and what each one buys

### 4.1 In-process adapter (what this repo does today)

Upstream policy class + our sampler + our graphs. Precedent: `pretrained.py` (diffusion) and
`pretrained_vqbet.py`. For SmolVLA this is cheap: 450 M parameters, ~2 GB, and its denoiser is
the same 10-step flow loop `fast_sampler.py` already implements for DDIM -- a flow-Euler
coefficient set, not a new kernel.

- **Buys:** E9/E14-comparable latency (in-process, `torch.cuda.synchronize` on both sides, same
  protocol), `torch.compile`/CUDAGraph applicability, and step-count sweeps that mean the same
  thing they mean for the diffusion checkpoint.
- **Loses:** the honesty of "what a user actually experiences", because there is no transport.

### 4.2 Remote upstream server (`RemotePolicyLike`)

openpi ships `scripts/serve_policy.py` on port 8000 with websocket remote inference [V]; GR00T
ships a ZMQ server-client split explicitly for real deployment [V]; `lerobot` ships a gRPC
`PolicyServer` + `RobotClient` pair with `actions_per_chunk` (default 50) and
`chunk_size_threshold` (0.5-0.6) as its two tuning dials [V].

A thin client that satisfies `PolicyLike` lets the *existing* control loop talk to any of them,
which is the actual product story: the wrapper does not care who computes the chunk.

- **Buys:** a latency row that includes serialization and transport -- the number a deployment
  engineer needs -- and reuse of three servers we do not maintain.
- **Loses:** comparability with E9. Those numbers must live in their own table and never be
  averaged with in-process rows.
- **Security note [S]:** CVE-2026-25874 (CVSS 9.3) is an unauthenticated RCE through
  `pickle.loads()` in `lerobot`'s async inference path, affecting releases through 0.5.1. The
  pin here is 0.6.1, but the advisory must be re-read and the fix confirmed before this seam is
  exposed on anything but localhost. A pickle-over-socket protocol that was shipped as *the*
  recommended deployment path is the single best argument for keeping the wrapper's own
  stdlib-HTTP boundary.

### 4.3 Upstream RTC, in-process

`lerobot` 0.6.1 ships Real-Time Chunking: a guidance term added to the flow-matching denoising
loop so the new chunk's overlap tracks the already-executed tail, with
`execution_horizon` 8-12, `max_guidance_weight` 10.0 and an exponential prefix schedule [V].
It lives at `policies/rtc/` (there is no `lerobot/rtc/`) and applies to the flow/DiT families
that declare `rtc_config` -- `evo1`, `groot`, `molmoact2`, `pi0`, `pi0_fast`, `pi05`, `smolvla`
[V, grep of the installed package, 2026-11-01] -- and **not** to this repo's DDIM diffusion
checkpoint, whose `policies/diffusion/` module has no RTC reference at all.

This looked like the one adoption where our measurement machinery is uniquely valuable, because
`docs/05` section 2 records that the third chunking strategy was implemented as a *cross-fade*
(the cross-fade recovered coverage, 0.875 vs 0.749, without recovering success, 26.7 % vs 53.3 %
for temporal ensembling), and RTC is the published version of what was approximated by hand.

**Read in full 2026-11-01: the same-episodes comparison does not exist.** The earlier version of
this paragraph promised that measuring RTC "on the same episodes" would either upgrade that
paragraph or confirm that averaging is the mechanism. It cannot, for three verified reasons, and
they are recorded here so the option is not re-proposed as cheap:

1. **RTC needs a velocity field.** `RTCProcessor.denoise_step` wraps a velocity callable, derives
   `x1_t = x_t - time * v_t` and returns `v_t - min(1/tau, max_guidance_weight) * (prefix - x1_t)
   * weights`. The DDIM checkpoint behind every blend row is epsilon-prediction; there is no
   velocity for the term to act on. This is the same obstruction that shipped a cross-fade
   instead of "inpainting" in the first place.
2. **Its only entry point is a different loop.** `euler_integrate` in
   `policies/common/flow_matching.py` is the sole hook (imported by `eo1`, `multi_task_dit`,
   `pi0`, `pi05`, `smolvla`), and RTC wraps every Euler step with the leftover prefix and a
   gradient enabled inside it (`torch.enable_grad`, `torch.autograd.grad`). Our `predict_chunk`
   implementations are closed samplers under `@torch.no_grad()` with no per-step hook -- and the
   fast-sampler path is a captured CUDA graph -- so RTC cannot ride the adapter even on a flow
   checkpoint; it needs its own sampler.
3. **No flow checkpoint here has a meaningful success rate.** `checkpoints/e1_flow_s0..s2` are the
   from-scratch ablation models D11 retired at 4-7 %; there is no published flow-matching
   checkpoint for PushT. Coverage and chunk disagreement would be measurable, success would be
   a floor.

So a like-for-like RTC-vs-cross-fade row requires a **second policy**, which is V3 below (SmolVLA
on LIBERO, a rented GPU, and V4's environment work first). What remains true and useful: the
cross-fade's measured separation -- coverage up, success flat against temporal ensembling -- is
unchanged, and RTC is now correctly scoped as a property of flow policies rather than a gap in
this repo's chunking strategy.

## 5. Fit against this host, and the honest cost

This is one Windows 11 laptop with an RTX 5060 (8 GB) [V, README]. Consequences:

- **π0, π0.5, π0-FAST, GR00T N1.7, OpenVLA-OFT: not runnable here.** 14 GB at inference for π0
  is not a tuning problem. Any arm using them needs a rented 24-48 GB GPU, and the numbers stop
  being "one consumer GPU" -- which is currently the README's strongest framing.
- **SmolVLA: runnable**, but there is no SmolVLA checkpoint for PushT, and fine-tuning one is
  ~4 h on an A100 [V, lerobot docs] -- out of scope by rule 1, and out of reach on 8 GB.
- **So the VLA arm needs a different task.** `lerobot` 0.6.1 ships `envs/libero.py` and published
  LIBERO checkpoints exist for GR00T N1.7 (95 / 100 / 98 / 93 %), π0.5 (`pi05_libero`) and
  SmolVLA. LIBERO is robosuite/MuJoCo -- the same class of environment archaeology that cost
  this stage a day for `lerobot` itself, and the reason option D was declined on 2026-10-30.
- **SimplerEnv** is the other standard real-to-sim harness for these checkpoints [S]; it carries
  the same dependency risk.

The uncomfortable conclusion, stated plainly: **"use actual SOTA models" and "one 8 GB laptop"
are not simultaneously satisfiable for anything above SmolVLA.** Either the wrapper demonstrates
generality on SmolVLA plus the two PushT checkpoints already in `results/`, or the next stage
rents a GPU and the README's hardware claim changes. That choice is the user's, and it should
not be made implicitly by an adapter landing first.

## 6. Options, sized

| # | Option | Cost | Buys | False if |
|---|---|---|---|---|
| V1 | `task` + multi-camera observation amendment, then a generic `PretrainedLeRobotPolicy` keyed off the checkpoint's `config.json` `type` (covers smolvla / pi0 / pi05 / groot with one class) | 1-2 days, no GPU | The actual "wrap real models" claim, with one adapter instead of four | A published LIBERO cell cannot be reproduced within its own reported interval -- then the wrapper is not policy-agnostic and *that* is the result |
| V2 | `RemotePolicy` client against openpi websocket / GR00T ZMQ, with a separate transport-inclusive latency table | 1-2 days + rented GPU | Deployment-honest latency; three servers not maintained by us | The transport number moves by more than the queueing term the server itself reports, i.e. the measurement is about the network, not the policy |
| V3 | RTC in-process on a SmolVLA checkpoint, same episodes vs naive / temporal ensemble / our cross-fade. **Not available on the diffusion checkpoint (section 4.3): RTC is flow-only and needs a grad-enabled loop, so a second policy is a prerequisite, not a convenience** | 1 day + GPU, after V4 | Upgrades the cross-fade limitation into a measured comparison; converges with upstream | RTC does not beat temporal ensembling on success at the same latency -- then conditioning is not the mechanism and `docs/05` was right |
| V4 | LIBERO as the task arm (needed for V1 at all) | 3-5 days of environment work | The only place VLA success numbers can be reproduced against a published cell | The env cannot be pinned reproducibly on Windows, which is the current expectation given mujoco/robosuite history |
| V5 | Port π0's flow loop onto `fast_sampler` | weeks | Nothing for us | Would violate rule 2 by owning third-party denoiser code; declines itself |
| V6 | Train or fine-tune a VLA | weeks + GPUs | Not this project | Violates rule 1; the from-scratch ablation is the recorded reason |

## 7. What I would do, in order

1. **V1's protocol half only** (task string + named cameras, no model). It is pure interface,
   testable on CPU, breaks no existing row, and every later option needs it. Do this even if no
   VLA is ever adopted, because the current protocol cannot express one.
2. **Decide the hardware question before writing an adapter.** On 8 GB the only landable VLA is
   SmolVLA-on-LIBERO; if the answer is "stay on this laptop", the honest next stage is V3's
   sibling measured on the diffusion checkpoint's own blend rows and the VLA plan is shelved.
3. **V1's adapter, gated by a published cell.** The gate is the same discipline `docs/10` used:
   a deterministic or cheap check first (contract shapes, padding, normalization round-trip),
   GPU hours only after it passes.
4. **V3, because it is cheap once V1 exists** and it answers a question the repo has already
   asked in public.
5. **V2 last**, and only with a rented GPU: it is the option that most changes what the README
   promises.

**What would falsify the whole direction:** if V1 cannot reproduce any published VLA cell on any
task, then "a policy-agnostic deployment wrapper" is a claim about two Diffusion Policy
checkpoints and a VQ-BeT, not about policies in general. The README's E15 sentence should then
say "three checkpoints from two families", which is what it can defend.

## 8. Sources

- openpi README (models, memory table >8 GB inference / >22.5 GB LoRA, checkpoint ids,
  `serve_policy.py`, PyTorch support) -- github.com/Physical-Intelligence/openpi, read
  2026-10-30. [V]
- `lerobot` docs, Asynchronous Inference (gRPC `PolicyServer`/`RobotClient`,
  `actions_per_chunk`, `chunk_size_threshold`, "π0 occupies 14 GB, SmolVLA ~2 GB") -- [V].
- `lerobot` docs, Real-Time Chunking (guidance term, `execution_horizon` 8-12,
  `max_guidance_weight` 10.0, applies to π0/π0.5/SmolVLA) -- [V].
- `lerobot` docs, GR00T Policy (N1.7 only, N1.5 removed, ZMQ server-client, 30 FPS, chunk >= 32
  with RTC, state-relative actions, LIBERO 95/100/98/93) -- [V].
- `lerobot` docs, SmolVLA (450 M, `smolvla_base`, ~4 h for 20 k steps on one A100) -- [V].
- Installed package: `lerobot` 0.6.1 `policies/`, `envs/`, config classes -- read from this
  tree's `.venv` 2026-10-30. [V]
- Physical Intelligence, Real-Time Chunking (arxiv 2506.07339, pi.website/research/
  real_time_chunking) -- [S].
- π*0.6 / RECAP (pi.website/blog/pistar06; arxiv 2025): research release, no open weights found;
  `exla-ai/openpie-0.6` claims a reimplementation [U, not verified]. [S]
- OpenVLA-OFT (arxiv 2502.19645; 7 B, ~16 GB for LIBERO, parallel decoding 4.2 Hz -> 108.8 Hz on
  the paper's hardware) -- [S], figures not re-measured here and not comparable to E9's protocol.
- CVE-2026-25874, `lerobot` async inference pickle deserialization, CVSS 9.3, affects <= 0.5.1 --
  [S], re-read before enabling seam 4.2 off localhost.
- SimplerEnv (simpler-env.github.io) -- [S], not installed or tried on this host.
