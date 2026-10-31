# 00. Overview

Status: Stage 1 draft. Written 2026-10-18. **Pivoted 2026-10-27 (D11): the stack now runs
a pretrained checkpoint and the from-scratch models are an ablation.** See
[CHANGELOG](CHANGELOG.md) and the deviations log in [04](04_eval_and_experiments.md).
Source of truth for scope: the uploaded Agent Brief, still kept verbatim under its original name in
`flowpolicy_complete/`. This file restates it, then records what Stage 1 research changed.

---

## Document map

| File | Content |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | one page: the code as built, its seams, invariants and extension points (read this first after a pivot) |
| [01_architecture.md](01_architecture.md) | diagram, interfaces, repo layout, CI, Docker |
| [02_environment_setup.md](02_environment_setup.md) | verified versions, setup, gates G1-G9 |
| [03_data_models_training.md](03_data_models_training.md) | data, DP recipe, policies, trainer, failure tests |
| [04_eval_and_experiments.md](04_eval_and_experiments.md) | pre-registered protocol, experiment matrix E0-E15 |
| [05_inference_and_deployment.md](05_inference_and_deployment.md) | latency ladder, Triton, ROS 2, async chunking |
| [06_decisions_and_risks.md](06_decisions_and_risks.md) | D1-D11 and risk register |
| [07_roadmap.md](07_roadmap.md) | milestones M0-M9, agent rules, packaging |
| [MODEL_CARD.md](MODEL_CARD.md) | the borrowed policy and this stack: data, intended use, metrics with intervals, limitations |

Coding agent: read 07 first, then the file for the current milestone.

---

## START HERE (day one)

Do these in order. Each step is a gate; stop and fix before moving on. Details and scripts are in [02_environment_setup.md](02_environment_setup.md).

1. **Record your machine facts.** Run `nvidia-smi`, `lsb_release -a`, `python3 --version`, `docker --version`. Paste the output into `docs/02_environment_setup.md` section 2 (machine fact sheet). Two facts decide a lot: your Ubuntu version and your exact GPU VRAM.
2. **Check the driver.** The PyTorch wheels that support your Blackwell GPU use CUDA 13.0 or 13.2. CUDA 13.0 needs an NVIDIA driver of at least 580.65.06 (580.95.05 for CUDA 13.0 Update 2). If `nvidia-smi` shows something older, update the driver first.
3. **Create the project with uv** and install PyTorch from the `cu130` index (commands in 02, section 5).
4. **Run gate G1 to G5** (GPU visible, bf16, `torch.compile`, CUDA graphs, Triton). All scripts are in 02, section 7.
5. **Run gate G6** (CUDA C++ extension). This one needs `nvcc`, which the pip PyTorch wheel does not give you. Do not skip it; decide now whether the optional CUDA stretch is possible.
6. **Run gate G7 and G8** (PushT closed loop, robosuite Lift closed loop with a random policy).
7. **Run gate G9** (ROS 2 smoke test in Docker).
8. Fill in the gate results table in 02, section 8. Anything red triggers the week 1 fallbacks in the brief (drop robomimic, defer ROS 2).

Do not start model code until G1 to G5 pass.

---

## 1. Goals

Build and honestly evaluate a **measurement and deployment stack for a pretrained
visuomotor policy** on PushT: one known-good checkpoint behind a single policy interface,
validated against its published success rate, optimized at batch size 1, and deployed in a
fixed-rate loop with latency injection and async chunking experiments. All on one consumer
GPU with about 7 GB usable VRAM and 24 CPU cores.

Original goal, kept as the ablation: train a conditional flow-matching policy (pi0-style,
small scale) from scratch and compare it against DDPM/DDIM and ACT-style regression
baselines with identical data and encoder. At the budget this host allows (12000 steps, about
7.6 epochs of PushT) the three policies land at 4-7 % success, which cannot order them and
cannot express a deployment degradation. The pivot was the owner's call on 2026-10-27; the
ablation rows are kept under `results/ablation/` because that floor effect is the reason for
the pivot.

## 2. Scope and non-goals

In scope: the four pillars in the brief (model and training, rigorous comparison, inference engineering, deployment).

Non-goals: real-robot work, language-conditioned VLA (decision D5, revisit at week 5), beating published state of the art, custom raw-CUDA attention or GEMM kernels.

## 3. Success criteria

| Criterion | Measured how | Status |
|---|---|---|
| Resumable trainer with resume-equivalence test | CI test on tiny model; tolerance stated | PASS 2026-10-19: `tests/unit/test_trainer_resume.py`, `rtol=1e-5` (tolerance-based, not bit-exact) |
| Three-policy comparison, at least 3 training seeds, CIs | Raw per-episode JSONL, scripts regenerate tables | PASS 2026-10-19 as an **ablation**: `results/ablation/e1.jsonl` (450 episodes, 3 seeds x 50 paired), all success CIs overlapping |
| PushT reproduction sanity check against published Diffusion Policy numbers | Within a pre-stated tolerance, or written explanation | **PARTIAL 2026-10-29** via the pretrained checkpoint, down from the PASS recorded on 2026-10-27: with our env wrapper, harness and metric, neither metric is shown to reproduce on 200 paired episodes of its own seed block: success is 104/200 against the checkpoint's 138/200 (exact McNemar p < 0.001), and coverage is 0.895 against a **censored** lower bound of 0.915 -- gym-pusht clips reward at 1.0, so every success records exactly 0.95, and the card's 0.955 implies only 0.907. On the 32 episodes both runs fail, coverage matches (0.850 against 0.832). The earlier 50-seed reading (p = 0.40) was underpowered, not a result. The gap sits where coverage lands just under the 0.95 threshold. Since 2026-10-30 the checkpoint itself is known to reproduce: through lerobot's own eval loop on the same 200 seeds it scores 126/200 against its block's 69.0 % (Fisher p = 0.25, five repeats spanning 61.5-64.0 %), so the shortfall is in this repo's path rather than in the reference -- but the 15 pp figure is a cross-sampler artefact (DDIM@10 against the control's DDPM@100); like-for-like at DDPM@100 on the same 200 seeds the residual is 118/200 against the control's 126/200 (five repeats of that loop: 62.9 % with a 1.0 pp sd), +3.9 pp with a bootstrap over episodes and runs of [-5.5, +13.0] pp that spans zero and a paired test that does not separate them (p = 0.46), so it is not attributed to the adapter. Written up as open in `docs/04` §4. The from-scratch route also did not reproduce it (1800/12000 steps vs ~200k) |
| Integration-step vs success vs latency figure | Sweep N in {1,2,4,8,10,16} | Re-run 2026-10-27 on the pretrained checkpoint: `results/figures/e3_success_vs_steps.png`, `e3_latency_vs_steps.png` |
| A measured baseline for the stack's own usefulness | Same checkpoint, same GPU, same episodes: published config vs stock library path vs this stack | **PASS 2026-10-28** (E14): `results/tables/e14_stack.md`, `results/figures/e14_stack_throughput.png`; the published setup is two orders of magnitude more expensive per episode and the stack's sampler is what makes compile/capture possible |
| One Triton kernel with honest benchmark, including the case where it loses | `triton.testing.do_bench` plus equivalence test | PASS 2026-10-19 on the ablation model: equivalent (max diff 1.2e-4), 2.2-2.4x eager, **0.76-0.83x compiled** (it loses to compile); `results/ablation/kernel.jsonl`. Not attempted on the pretrained model |
| ROS 2 (or plain-Python fallback) fixed-rate loop with latency injection | Latency histogram, deadline-miss log | PASS with fallback D7 (plain Python), re-run 2026-10-27 on the pretrained checkpoint: `results/deploy.jsonl`, `results/figures/e10_success_vs_latency.png`, `e10_latency_hist.png` |
| Week 8 packaging complete | README, results tables, failure analysis, demo | README/tables/figures generated by scripts; `results/figures/demo_rollout.gif` renders one episode with the loop's own timings (`just demo`); no narrated video |

No number in this table is filled in until it is measured.

## 4. What Stage 1 research changed in the brief

The brief was written with some assumptions that are out of date as of 2026-10-18. These are the corrections. Evidence and URLs are in 02, section 10.

| Brief says | Verified finding | Consequence |
|---|---|---|
| CUDA build "likely 12.8+" | PyTorch dropped CUDA 12.8 and 12.9 wheels starting with 2.12. Current wheels are CUDA 12.6 (legacy GPUs), 13.0 (default), 13.2. | Install the `cu130` build. Do not use `cu126` on Blackwell. |
| PyTorch "2.x" | Latest stable is 2.14.1 (2026-09-30), with torchvision 0.29.1. 2.14 is the last release with Python 3.10 wheels. | Pin `torch==2.14.1`, `torchvision==0.29.1`. |
| ROS 2 "likely Jazzy" | Lyrical Luth (LTS, released May 2026) now exists and targets Ubuntu 26.04 with Python 3.14. Jazzy remains supported on 24.04 until 2029-05-31. Kilted ends 2026-12-31. | Recommended default: Jazzy in Docker. See ADR-007 (to be written in Stage 2). |
| robosuite and robomimic compatibility is "a known pain point" | robosuite 1.5.2 is the PyPI release; robomimic 0.4+ supports robosuite 1.5; robomimic's own docs and Dockerfile still reference old Python (3.8, 3.9). | Use robomimic only to fetch and read datasets; do not depend on its training code. Gate G8 tests this. |
| Triton and CUDA extension both on "week 1 gate" | PyTorch 2.13 stopped bundling `ptxas` in the cu13 binary. The pip wheel does not provide `nvcc` either. | Two extra pitfalls to test in gates G5 and G6. |

## 5. Glossary

- **Action chunk:** a short sequence of future actions predicted at once.
- **Flow matching:** train a network to predict a velocity field that transports noise to data; sample by integrating an ODE.
- **DDPM/DDIM:** denoising diffusion training, with deterministic DDIM sampling.
- **ACT:** Action Chunking with Transformers; regression-style chunk predictor.
- **EMA:** exponential moving average of weights, used at evaluation time.
- **Resume equivalence:** an interrupted-and-resumed run matches an uninterrupted run for the next M steps to a stated tolerance.
- **N (sampler steps):** how many denoising steps the sampler takes per action chunk. The policy is trained with 100 steps; N is an inference-time choice, so it trades latency for fidelity (E3).
- **DDIM vs DDPM sampling:** DDPM's step injects fresh noise every iteration, so it cannot be truncated below its trained length; DDIM's is deterministic, which is what makes an N sweep meaningful. Same trained weights, different sampler.
- **Chunk / action horizon:** the policy predicts `pred_horizon` (16) actions at once and the loop executes `action_horizon` (8) of them before replanning.
- **Deadline miss:** one chunk whose measured inference time exceeded the control period (100 ms here). Counted per episode, from measured time, not from a simulation.
- **Temporal ensembling:** averaging the predictions of several overlapping chunks for the same control step, instead of letting the newest chunk replace the old ones (E12).
- **Episode-level interval:** a bootstrap interval that resamples eval episodes only. It cannot speak to training-run variance; it is what a single borrowed checkpoint allows.
- **Floor effect:** a measurement whose subject scores near zero, so a difference cannot be resolved no matter how many episodes are added. The ablation's 4-7 % success is one.
- **`PolicyLike`:** the structural interface (`obs_horizon`, `pred_horizon`, `action_dim`, `predict_chunk`, `num_parameters`) that eval, deployment and the latency ladder depend on; satisfied by trained checkpoints and by the pretrained adapter alike.
- **Gate:** a pass/fail smoke test that must pass before the dependent work starts.
- **SDPA:** `torch.nn.functional.scaled_dot_product_attention`.

## 6. Open questions (answered 2026-10-28)

- **Which Ubuntu version is the host?** None: the host is Windows 11 build 26200, Git Bash, one
  laptop (recorded 2026-10-18, `docs/02` section 2). ROS 2 and Docker therefore stay out of
  scope (D7, G9a NOT RUN).
- **Exact GPU model and VRAM?** RTX 5060 Laptop, 8151 MiB, driver 617.42, compute capability
  (12, 0) -- measured 2026-10-19, `docs/02` section 2.
- **Linux native or WSL2?** Neither. Native Windows, no WSL2 (2026-10-18).
- **Decisions D1-D8 open?** Closed 2026-10-19 and 2026-10-27: D1-D11 all carry a `CHOSEN` entry
  with a date in `docs/06` section 1.

## 7. Next actions

- [x] Owner: fill 02 sections 2 and 8. Done 2026-10-19 (driver 617.42, gates G1-G5 re-run).
- [x] Agent (Stage 2): ADR-001 to ADR-008. Done 2026-10-18, consolidated as D1-D11 in `docs/06`
  rather than as separate ADR files; D11 added 2026-10-27.
- [x] Agent (Stage 3): Diffusion Policy PushT recipe, pi0 and ACT summaries, async chunking
  survey. Recipe and the pi0/ACT reading list are in `docs/03` section 2 and `docs/07`
  section 3, sourced from replicating papers and the LeRobot port and tagged `[S]`/`[U]`
  accordingly; the async-chunking survey is still `[U]` -- no paper was read line by line,
  and that is why E12's third strategy is implemented from first principles and labelled as
  such rather than attributed to a paper.

## 8. Acceptance criteria (for this document)

- A person with no prior context can follow "Start here" without reading anything else first.
- Every correction in section 4 has a source in 02, section 10.
- No metric in section 3 has a value that was not measured.
