# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M8 packaging, second pass after the D11 pivot (2026-10-27) |
| Current step id | none (evals complete: E0, E3, E9, E10, E12) |
| Green flag active | owner granted a standing green flag on 2026-10-19 ("finish the project end to end, MVP stage") |
| Last completed step | M8.3-M8.5 (2026-10-27): pretrained adapter + E0 gate, E3/E9/E10/E12 on the pretrained checkpoint, generators, `just report`/`just serve`/`just doctor`, README and docs re-framed |
| Evidence | `just test` green (2026-10-27: ruff, mypy strict on 23 files, 33 unit tests); raw rows in `results/*.jsonl` (pretrained) and `results/ablation/*.jsonl` (from-scratch); tables in `results/tables/` |
| Spec installed | docs/00-07 + CHANGELOG.md; docs carry dated deviations where reality differs |

## What changed on 2026-10-27 (D11 pivot)

The project stopped training and put a borrowed policy behind the same interface. The
from-scratch models reached 4-7 % success at 12000 steps, which is a floor effect: two
policies cannot be ordered at 6-11 successes in 150 episodes, and a latency experiment
cannot see a policy degrade when it is not succeeding. The owner asked for the 180.

- Substrate: `lerobot/diffusion_pusht` (Apache-2.0, Diffusion Policy, gym-pusht, 175k steps,
  262.7 M parameters fp32). Model card: 65.4 % success, 0.955 mean max overlap over 500
  episodes.
- New code: `src/flowpolicy/models/pretrained.py` (adapter + normalization read from the
  checkpoint), `PolicyLike` protocol in `models/policies.py`, one pretrained branch in
  `rollout.load_policy`, a `--scheduler` flag on the three runners. The episode loop, the
  deployment loop, the latency protocol and the generators did not change.
- Install: `just setup-pretrained` (lerobot 0.6.1 `--no-deps`, because it caps torch < 2.12
  and this host runs torch 2.14.1+cu130). Deviation in docs/04 section 4, risk R13 in docs/06.
- Removal: E13 (36000 steps + random-crop) scaffolding deleted unrun; `train.log` removed.

## Machine facts (measured)

- Windows 11 build 26200, Git Bash, single host. RTX 5060 Laptop, 8151 MiB total, driver
  617.42 (CUDA UMD 13.4), compute capability (12, 0).
- Python 3.12.13 in `.venv`, torch 2.14.1+cu130, torchvision 0.29.1+cu130,
  `triton-windows==3.8.0.post29` (Windows marker only).
- Pretrained backend (outside the lockfile): lerobot 0.6.1, diffusers 0.42.1, safetensors
  0.8.0, huggingface-hub 2.2.0, draccus 0.11.6. Verified importing and running on torch
  2.14.1+cu130; lerobot's own requirement is torch < 2.12.
- Missing: Docker, ROS 2, `nvcc`, robosuite. G6, G8, G9a-G9c are NOT RUN and are never
  reported as passes.

## What exists and is measured

- **Pretrained path**: adapter with per-call DDIM/DDPM scheduler choice; checkpoint
  normalization (ImageNet mean/std, dataset min/max) applied from `model.safetensors`;
  4 unit tests on those transforms; `hf:<repo_id>` accepted by `evaluate.py`,
  `deploy_eval.py` and `latency.py`.
- **E0 (validation gate)**: 64.0 % success (32/50) and 0.924 mean coverage at DDIM N=10,
  against the model card's 65.4 % / 0.955. `results/e0.jsonl`,
  `results/tables/e0_validation.md`. This is the project's reproduction check.
- **Latency at the operating point** (batch 1, fp32, single process, measured before the
  eval runs): 20.7 ms at N=1, 178.3 ms at N=10, 1766.6 ms at N=100. Peak VRAM 1125 MiB.
- **E3 (DDIM ladder, 300 episodes)**: coverage 0.121 [0.076, 0.165] at N=1, 0.716 [0.627,
  0.804] at N=2, 0.885 [0.847, 0.919] at N=4 and flat from there (0.919 / 0.888 / 0.892 at
  N=8/10/16, all intervals overlapping). Success 0/50 at N=1, 10/50 at N=2, then noise
  (42-60 %). Latency 31.5 -> 726.7 ms under two-worker contention. `results/e3.jsonl`.
- **E9 (latency ladder, batch 1, single process)**: eager 20.2 / 73.5 / 190.5 / 309.4 ms at
  N=1/4/10/16; CUDA graph 17.5 / 66.8 / 181.0 / 304.0 ms, i.e. 1.02-1.16x, against 4.7x on the
  11 M ablation model. `torch.compile` is NOT RUN: `InvalidCxxCompiler: cl is not found`
  (Inductor CPU codegen; the checkpoint's DDIM scheduler puts host tensors in the graph). E9
  plus E3 fix the deployment envelope: only N <= 4 fits a 100 ms period.
- **E10 (270 episodes, injected observation latency 0-200 ms + 20 ms jitter)**: success
  56.7 % at 0 ms, 60.0 % at 50 ms, 56.7 % at 100 ms, 46.7 % at 150 ms, 43.3 % at 200 ms (all
  CIs overlapping); coverage flat at 0.897-0.932. Deadline misses 26-33 per episode at N=10,
  i.e. inference needs ~2x the 100 ms period. `results/deploy.jsonl`.
- **E12 (replan every 2 steps, four overlapping chunks)**: naive replacement 10.0 % [0.0, 20.1]
  at 100 ms and 10.0 % [0.0, 23.3] at 200 ms (coverage 0.749 / 0.553), disjoint from the 8-step
  baseline; uniform temporal ensembling recovers 53.3 % / 43.3 % (coverage 0.926 / 0.880) at 4x
  the chunk count. Separated, unlike the ablation's ambiguous version. It is also the fifth row
  of the computed headline summary, so `just report` and the README table carry it
  (`flowpolicy.eval.report` strips the runner's `-r2` suffix to pair the rows).
- **Infrastructure added 2026-10-27 (no training involved)**: `flowpolicy.eval.report` (computed
  headline summary, used by `just report` and the README), `flowpolicy.deploy.server` +
  `scripts/serve.py` (`just serve`, stdlib-only HTTP serving boundary, contract in docs/05
  section 4), `scripts/doctor.py` (`just doctor`, PASS/FAIL/NOT RUN environment check),
  `docs/MODEL_CARD.md`, and a CI job that regenerates the tables and fails if they are stale.
  Tests: 46 unit tests total (6 for the server contract over a real socket, 7 for the summary
  computations, 4 for the checkpoint normalization).
- **Episode-set variance to keep in mind**: the same configuration at DDIM N=10 scores 64.0 %
  on eval seeds 10000-10049 (E0) and 54.0 % on 20000-20049 (E3). Two 50-episode samples of one
  policy differ by 10 percentage points, which is the resolution limit of every table here.
- **Ablation (from-scratch, kept)**: 3 policies x 3 seeds x 50 paired episodes at 12000
  steps = 450 episodes, success flow 7.3 % [3.3, 12.0], DDPM 4.7 % [1.3, 8.7], regression
  4.0 % [1.3, 7.3], coverage 0.519 / 0.468 / 0.336; all success CIs overlapping. Training
  cost, overfit checks and the Triton kernel (which loses to `torch.compile`, 0.76-0.83x)
  all under `results/ablation/`. Training is still reproducible via `just train-e1`.

## Open items (not blockers)

- One checkpoint exists for PushT, so E0/E3/E9/E10 intervals are episode-level and there is
  no seed-to-seed variance to report. Stated on every table that has pretrained rows.
- The backend lives outside `uv.lock`; `just setup` prunes it. Documented, not solved.
- R5 (gym-pusht vs original PushT equivalence) is still unverified, but E0 now anchors the
  harness to an external published number, which is the stronger check.
- E10 runs the loop in virtual time (no wall-clock pacing); inference is timed for real and
  deadline misses are logged.
- Demo video not recorded (no capture path on this host); docs/00 marks it missing.
- E2, E4-E8, E11 not run; robomimic, ROS 2, Docker and the CUDA C++ extension are out of
  scope on this host (D1, D6, D7). The Triton kernel stays an ablation result.

## Decision log (user answers, dated)

- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit `docs/STATE.md` on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `flowpolicy_complete` is the authoritative spec; docs/ adopts it verbatim.
- 2026-10-18: "proceed with implementation" taken as the green flag for M0.1 only (roadmap row M0).
- 2026-10-18: owner confirms this Windows 11 laptop runs all work; there is no separate Ubuntu host.
- 2026-10-18: "proceed with the recommended ones" -> native Windows for the gates; Docker/G9 deferred.
- 2026-10-19: owner installed the NVIDIA driver 617.42 (the only G1 cause).
- 2026-10-19: owner chose `triton-windows==3.8.0.post29` to clear G3/G5 rather than drop compile/Triton.
- 2026-10-19: owner granted a standing green flag: finish the project end to end at MVP stage without further approval requests.
- 2026-10-23: owner asked for full results; training budget raised from 1800 to 12000 steps (one seed per policy) and eval episodes raised to 50. `just train-e1` resumes interrupted runs.
- 2026-10-23 19:06: owner asked to pause training to go AFK; the DDPM run was stopped at step 8825 with a checkpoint at step 8000.
- 2026-10-23 22:52: training resumed from the step-8000 checkpoint and finished at 12000 steps for all three policies; E1, E3 and E10 re-run on the new checkpoints.
- 2026-10-24: three training seeds at the 12000-step budget; DDPM sampling-ladder bug found and fixed; E1/E3 re-evaluated.
- 2026-10-27: owner asked what the verdict on the project was; answer: process strong, scientific content weak (null on success, floored signal).
- 2026-10-27: owner asked to do a 180 -- use a model that is already good and work on the software infrastructure. Decisions taken with the owner: (1) wrap the `lerobot` package rather than reimplement the model, (2) full re-frame: adapter, validation, re-run E3/E9/E10, rewrite README and docs 00-07, (3) keep the from-scratch work as a documented ablation and delete the unrun E13 scaffolding.
