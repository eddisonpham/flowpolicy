# Resume notes: what this project is, and how to say it

One line: **a measurement and deployment stack for a pretrained visuomotor policy -- it answers
"what does a working policy cost to run", and it made the policy it measures compile.**

The bullets below are in XYZ form: *Accomplished X, as measured by Y, by doing Z*. Every number
is from `results/*.jsonl` and reproducible with the recipes in the README. Nothing here is
inflated: the project's own headline includes a null result and two measurements that
contradicted the plan.

---

## Bullets (pick 4-6 for a resume; the first three are the strongest)

- **Cut the cost of evaluating a 262.7 M-parameter visuomotor policy by 20x** -- from 23.9 s to
  1.19 s of GPU time per 100 control steps, and from 53.1 s to 2.6 s per PushT episode -- **by
  measuring the checkpoint's published sampling setup against two alternatives on identical
  episodes and moving to 4-step DDIM sampling at the coverage plateau** (mean max overlap 0.947
  -> 0.961 on the same episodes, 4 episodes per arm).
- **Made `torch.compile` and CUDA-graph capture possible on a policy that could not be
  compiled**, by diagnosing ten host/device synchronizations per action chunk (`diffusers` keeps
  the DDIM schedule on the CPU) and **rewriting the denoising loop with coefficients
  precomputed on the device** -- equal to the reference sampler to fp32 rounding (max 0.02 px on
  a 512 px action space, both on a toy denoiser and on the checkpoint), pinned by 9 unit tests
  and a GPU regression test.
- **Matched a published Diffusion Policy's coverage where the two runs can actually be compared,
  and showed where it does not reproduce**: on the 32 episodes both runs fail, 0.850 against
  0.832 mean coverage; on 200 paired episodes of the checkpoint's own seed block, success is
  104/200 against its 138/200 (exact McNemar p < 0.001) -- a 17-point gap that 50 seeds had
  hidden. Its reported 0.915 coverage is a lower bound, not a measurement, because gym-pusht
  clips reward at 1.0 so every success records exactly 0.95 -- **after restoring eight
  normalization tensors that the current library release silently drops at load time**.
- **Determined that a 100 ms control period admits at most 4 sampler steps on an 8 GB laptop
  GPU**: 65-73 ms per action chunk at N=4 versus 160-183 ms at N=10, which meant 27-33 missed
  deadlines per episode at N=10 and none at N=4 -- **by sweeping the sampler ladder (300
  episodes) against a batch-1 latency ladder with eager, CUDA-graph and compiled execution**.
- **Quantified how a working policy degrades when perception is late, over 270 episodes**:
  success 56.7 % at 0 ms of injected observation latency versus 43.3 % at 200 ms (intervals
  overlapping, coverage flat at 0.90-0.93), **by running a fixed-rate control loop with
  observation-staleness injection and per-chunk deadline accounting**.
- **Showed that replanning four times faster breaks the policy unless chunks are ensembled**:
  naive replacement collapsed from 57 % to 10 % success at a 2-step replan rate, while uniform
  temporal ensembling recovered 53 % at four times the chunk count -- **a 270-episode
  controlled comparison of two deployment strategies on the same policy, seeds and latency**.
- **Designed and executed a 450-episode, three-policy x three-seed ablation** (conditional flow
  matching, DDPM, ACT-style regression, 12 000 steps each) that returned a **null result at
  4-7 % success**; diagnosed it as a floor effect at the compute budget available on one laptop
  GPU, reported it instead of hiding it, and **re-scoped the project around what the evidence
  supported**.
- **Wrote a Triton kernel (fused final denoising step plus de-normalization) that is 2.2-2.4x
  faster than eager PyTorch and 0.76-0.83x slower than `torch.compile`**, and reported it as a
  loss rather than dropping the comparison.
- **Found and fixed a sampling bug in a baseline implementation**: DDPM's step ladder ended at a
  timestep whose cumulative alpha is exactly 1.0, so one and two sampling steps produced
  bit-identical outputs; added a regression test that fails on the old code.
- **Built the measurement plumbing that makes every number above checkable**: 2 300+ raw
  per-episode JSONL rows, all tables, figures and README sections generated from them by script,
  49 CPU unit tests, ruff and `mypy --strict` on 23 files, and a CI job that fails when a
  generated table no longer matches the data.
- **Measured that parallel rollout workers do not speed this workload up** (two workers cost
  1.8x per control step) because a single process already saturates the GPU -- **measured
  deliberately before claiming a faster harness**.
- **Showed that action-level equivalence does not imply episode-level equivalence in a chaotic
  contact task**: two numerically equivalent samplers (0.02 px apart) agreed on only 9 of 20
  paired PushT outcomes, **by running the paired check instead of assuming it and reporting the
disagreement as a resolution limit of 20-episode evaluation**.

## Two-line version

> Built a measurement and deployment stack for a pretrained Diffusion Policy on PushT, quantifying
> what a working visuomotor policy actually costs to serve: 20x cheaper evaluation by finding the
> sampler-step plateau (0.96 coverage), a device-resident DDIM sampler that unlocked
> `torch.compile`/CUDA graphs and was proven equal to the reference implementation, and controlled
> evidence that a 100 ms control period admits only 4 sampler steps and that fast replanning
> requires temporal ensembling (57 % -> 10 % without it).

## Interview talking points

- **Why the pivot.** The first half of the project trained three policies from scratch to 4-7 %
  success. At that level a three-policy comparison cannot order them and a latency experiment
  cannot observe degradation, so the honest move was to stop training and measure a policy that
  works. The ablation is kept in the repo as the reason.
- **The bug that mattered.** `torch.compile` failing was first recorded as "no MSVC on this host".
  It was actually the sampler mixing CPU scalars into CUDA math. Raising the MSVC toolchain would
  have hidden it; rewriting the loop fixed the cause and removed a per-step synchronization.
- **A result that reversed.** On an 11 M-parameter model, CUDA graphs gave 4.7x and compile 5.8x.
  On the 262.7 M-parameter policy they give about 1.1x, because the network is compute-bound. The
  transfer assumption was wrong and is written down as such.
- **What "honest" cost.** Every table prints intervals; the 50-episode samples of one fixed policy
  differ by 10 percentage points, which is stated as the resolution limit rather than hidden by
  reporting point estimates.
