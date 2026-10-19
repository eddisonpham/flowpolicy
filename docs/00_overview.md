# 00. Overview

Status: Stage 1 draft. Written 2026-10-18. See [CHANGELOG](CHANGELOG.md).
Source of truth for scope: the FlowPolicy Agent Brief. This file restates it, then records what Stage 1 research changed.

---

## Document map

| File | Content |
|---|---|
| [01_architecture.md](01_architecture.md) | diagram, interfaces, repo layout, CI, Docker |
| [02_environment_setup.md](02_environment_setup.md) | verified versions, setup, gates G1-G9 |
| [03_data_models_training.md](03_data_models_training.md) | data, DP recipe, policies, trainer, failure tests |
| [04_eval_and_experiments.md](04_eval_and_experiments.md) | pre-registered protocol, experiment matrix E0-E12 |
| [05_inference_and_deployment.md](05_inference_and_deployment.md) | latency ladder, Triton, ROS 2, async chunking |
| [06_decisions_and_risks.md](06_decisions_and_risks.md) | D1-D10 and risk register |
| [07_roadmap.md](07_roadmap.md) | milestones M0-M8, agent rules, packaging |

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

Build and honestly evaluate a from-scratch conditional flow-matching visuomotor policy (pi0-style, small scale) on PushT and robomimic, compare it against a DDPM/DDIM diffusion policy and an ACT-style regression policy with identical data and encoder, optimize inference at batch size 1, and deploy it in a fixed-rate ROS 2 loop with latency and disturbance experiments. All on one consumer GPU with about 7 GB usable VRAM and 24 CPU cores.

## 2. Scope and non-goals

In scope: the four pillars in the brief (model and training, rigorous comparison, inference engineering, deployment).

Non-goals: real-robot work, language-conditioned VLA (decision D5, revisit at week 5), beating published state of the art, custom raw-CUDA attention or GEMM kernels.

## 3. Success criteria

| Criterion | Measured how | Status |
|---|---|---|
| Resumable trainer with resume-equivalence test | CI test on tiny model; tolerance stated | TBD |
| Three-policy comparison, at least 3 training seeds, CIs | Raw per-episode JSONL, scripts regenerate tables | TBD |
| PushT reproduction sanity check against published Diffusion Policy numbers | Within a pre-stated tolerance, or written explanation | TBD |
| Integration-step vs success vs latency figure | Sweep N in {1,2,4,8,10,16} | TBD |
| One Triton kernel with honest benchmark, including the case where it loses | `triton.testing.do_bench` plus equivalence test | TBD |
| ROS 2 (or plain-Python fallback) fixed-rate loop with latency injection | Latency histogram, deadline-miss log | TBD |
| Week 8 packaging complete | README, results tables, failure analysis, demo video | TBD |

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
- **Gate:** a pass/fail smoke test that must pass before the dependent work starts.
- **SDPA:** `torch.nn.functional.scaled_dot_product_attention`.

## 6. Open questions

- Which Ubuntu version is the host? (Decides ROS 2 distro and Docker base image.)
- Exact GPU model and VRAM (brief says RTX 5060-class, about 7 GB usable). Unverified until `nvidia-smi` output is recorded.
- Is the host Linux native, or Windows with WSL2? Driver and Docker GPU passthrough steps differ. Not stated in the brief.
- Decisions D1 to D8 remain open; ADRs come in Stage 2.

## 7. Next actions

- [ ] Owner: complete the day-one checklist above and fill in 02, sections 2 and 8.
- [ ] Agent (Stage 2): ADR-001 to ADR-008 using the findings in 02.
- [ ] Agent (Stage 3): Diffusion Policy PushT recipe, pi0 and ACT summaries, async chunking survey.

## 8. Acceptance criteria (for this document)

- A person with no prior context can follow "Start here" without reading anything else first.
- Every correction in section 4 has a source in 02, section 10.
- No metric in section 3 has a value that was not measured.
