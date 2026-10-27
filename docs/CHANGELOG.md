# CHANGELOG

Paths in entries below refer to where files lived on the date of the entry. The 2026-10-27 entry lists every move (from-scratch measurements now under `results/ablation/`).

## 2026-10-18: Stage 1

- Added `00_overview.md` (day-one checklist, scope, success criteria template, corrections to the brief).
- Added `02_environment_setup.md` (compatibility matrix, pitfalls, install steps, gates G1 to G9, results table, references).
- Findings that change the brief: PyTorch is at 2.14.1 and no longer ships CUDA 12.8 wheels (use `cu130`); ROS 2 Lyrical Luth now exists; PyTorch 2.13 stopped bundling `ptxas` in cu13 builds.
- Open: items listed under "Not verified in Stage 1" in `02_environment_setup.md` section 10.
- Next: Stage 2 (ADR-001 to ADR-008).

## 2026-10-18: Stage 2 (consolidated)

- Added 01, 03, 04, 05, 06, 07. ADRs for D1-D10 are consolidated in 06; reading list and packaging are in 07.
- Diffusion Policy recipe sourced from replicating papers and the LeRobot port, not the original paper (flagged [S]/[U]); agent must confirm against the paper before E0.
- Unverified items remain tagged [U] (pi0 timestep distribution, ACT details, dataset URLs/licenses, async chunking literature, Triton pin, uv/Docker/Nsight specifics).

## 2026-10-18: M0 scaffold (agent)

- Repo scaffold per M0: `pyproject.toml` (3.12, hatchling, ruff, mypy strict on `src/`, pytest markers), `uv.lock`, `justfile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `README.md`, `AGENTS.md`, `src/flowpolicy/__init__.py`, `tests/unit/test_package.py`.
- Acceptance met: `just setup && just test` green from a clean clone; CI green (run 39179413619).
- Deviation: the `smoke` CI job in 01 §4 is absent until a model exists (M3); the `unit`, `lint`, `types` jobs cover M0.

## 2026-10-18: M1.1 gates G1-G5 (agent) — BLOCKED

- Deviation from the docs' platform assumption: the owner has one machine, a **Windows 11 laptop** (build 26200, MIGW64 shell), not Ubuntu. No WSL2. Consequences: docs/02 §5-6 and the Docker/ROS 2 path (D7) do not apply as written; §2 fact sheet and §8 gate table are filled with measured values, and this entry is the dated deviation that 07 §0 rule 1 requires.
- torch 2.14.1+cu130 and torchvision 0.29.1+cu130 install from `https://download.pytorch.org/whl/cu130` on Windows (`win_amd64` wheel exists), pinned in `pyproject.toml` and `uv.lock`. The `--index` syntax in 02 §5.2 is [U] as written; the working form is `[[tool.uv.index]]` plus `[tool.uv.sources]`, tagged [V] 2026-10-18.
- G1-G5 FAIL, single root cause: `cudaGetDeviceCount() returned cudaErrorNotSupported, likely using older driver`. Driver is 577.13; CUDA 13.0 needs the R580 branch. Risk R1 has no fallback, so no model or data work starts until G1 is green.
- G5 has a second cause: `triton` is not installed by the Windows cu130 wheel. Do not install it separately (02 §3.1); revisit after the driver fix.
- G6-G9c NOT RUN: no `nvcc`, no `docker`, no simulators yet. G9 is deferred by owner decision to M1.5.
- Doc defect found: 02 §9 refers to `10_risk_register.md`, which does not exist; risks live in 06 §2.

## 2026-10-19: driver updated to 617.42, gates G1-G5 re-run (agent)

- Owner installed the NVIDIA App on 2026-10-18 and the display driver on 2026-10-19; `nvidia-smi` now reports **617.42** (CUDA UMD 13.4) on the RTX 5060 Laptop GPU, up from the factory 577.13. The R580 minimum in 02 §3.1 is met.
- Same five gate scripts, unedited: G1 PASS, G2 PASS, G3 FAIL, G4 PASS, G5 FAIL. `torch.cuda.is_available()` is now True and `torch.cuda.get_device_capability()` returns (12, 0), which clears pitfall 1 for good.
- G3 and G5 now have a single, different cause: `triton` is not installed in the Windows cu130 environment. This is not a driver fault and was not worked around; 02 §3.1 forbids installing Triton ad hoc, so the choice (Windows Triton wheel vs dated scope deviation) is left to the owner and recorded in `docs/STATE.md`.
- 02 §2 (driver, compute capability, VRAM rows) and §8 (G1-G5 rows, history note) updated with these measurements. G6-G9c remain NOT RUN.

## 2026-10-19: triton-windows pin, G3 and G5 green

- Owner decision: install a Windows Triton build rather than drop compile/Triton from scope, because 02 §8 makes a red G3/G5 a stop-the-line result.
- Added `triton-windows==3.8.0.post29` to `[project].dependencies` with a `sys_platform == 'win32'` marker, so the Linux CI jobs keep torch's own Triton resolution. `uv.lock` updated by `uv add`.
- G3 now passes all three modes (`default`, `reduce-overhead`, `max-autotune`) and G5 passes with `triton 3.8.0`; the assembler works, so risk R2's `TRITON_PTXAS_PATH` fallback is not needed here.
- `just gates` is five PASS and `just test` is green. 02 §2 (Triton row), §3.1 (pin, marked [V] with the [U] upstream-version question) and §8 (G3, G5) updated.
- This replaces the "do not install Triton separately" guidance in 02 §3.1 for this host; the reason is the dated deviation recorded there.

## 2026-10-19: M2 data and model plumbing (agent)

- Added the Zarr window loader, train-only normalizer with round-trip tests, ResNet-18 GN multi-frame encoder, 1-D U-Net and transformer chunk heads, and the three policies (flow, DDPM/DDIM, ACT-style regression) behind one `BasePolicy` interface (`loss(batch)` / `predict_chunk(obs)`).
- `scripts/data/prepare_pusht.py` downloads and converts `lerobot/pusht` to `data/pusht/pusht.zarr` (32 MB, MIT). Hydra configs live in `conf/` with structured dataclasses in `src/flowpolicy/config.py`.
- Measured memory table (docs/03 §4): batch 32 = 2317 MiB / 7.80 steps/s, batch 64 = 4414 MiB / 4.00 steps/s, batch 128 = 8616 MiB / 0.05 steps/s and flagged not fitting. Model is 10.95 M parameters. Batch 64 was used for E1.
- Unit tests cover window indices, normalizer round-trip, policy shapes/losses, seed disjointness, and resume equivalence (`rtol=1e-5`). `just test` green (23 tests).

## 2026-10-19: M3 trainer and overfit sanity (agent)

- Trainer: bf16 autocast, AdamW, warmup + cosine, grad clip 10, EMA 0.999 (EMA used for every eval), periodic + final checkpoints, SIGINT/SIGTERM checkpoint-and-exit, JSONL scalar log, checkpoint carries model/EMA/optimizer/scheduler/step/RNG/hydra config/stats/git hash/`uv.lock` hash.
- Overfit check on 8 episodes, 10x-reduction criterion: flow 1.230 -> 0.065 (0.053x) and DDPM 1.096 -> 0.091 (0.083x) at 1200 steps; regression 0.376 -> 0.029 (0.077x) at 2000 steps (it reached 0.109x at 1200, so the budget was raised for that policy only and the step count is reported per row).
- Bug found by the post-reboot resume: a checkpoint loaded with `map_location=cuda` holds CUDA RNG tensors, which `set_state` rejects. `Trainer.resume` now moves them to CPU. The interrupted DDPM run resumed from step 1200 and its uncheckpointed 1200-1625 tail was dropped from the JSONL instead of being kept.

## 2026-10-19: M4/M5 E1, E3 and the E0 fallback (agent)

- Nine runs (3 policies x 3 seeds), 1800 steps, batch 64, ~9 min each (1.4 GPU-h total). `results/train/*.jsonl` has every logged step.
- E1: 270 episodes (30 per seed), fixed eval seeds 10000-10029, EMA weights, N=10. **No policy reaches the 95% coverage success threshold.** Mean coverage: regression 0.290, flow 0.252, DDPM 0.132; success 1.1% for regression and 0% for the other two, every 95% CI including 0. Table: `results/tables/e1_main.md`.
- E0 is not reproduced and is not claimed: the DDPM baseline trains for ~1.5 epochs, not the reference ~200k steps. This is the M4 fallback "proceed with relative comparisons only" plus the written explanation in docs/04 §4.
- E3: 720 episodes over N in {1,2,4,8,10,16} on the E1 checkpoints, 20 episodes per seed. Latency scales linearly with N (flow 10.7 ms at N=1 to 66.6 ms at N=16, measured with 2 workers). The pre-registered hypothesis "flow degrades less than DDIM as N falls" is **not supported** at this budget: flow coverage 0.381 at N=1 vs 0.333 at N=10, non-monotonic and within noise; no configuration succeeds.
- Deviation: eval workers reduced from 4 to 2 after the owner reported the laptop was unresponsive; timings are therefore under two-process contention and are labelled as such.

## 2026-10-19: M6 inference ladder and Triton kernel (agent)

- E9, idle GPU, batch 1, 50 warmup + 200 timed calls, mean/p95: eager 33.46/34.48 ms at N=10; `compile-reduce-overhead` 5.80/5.89 ms (5.8x, 25.8 s compile); `compile-max-autotune` 5.98/6.33 ms (133.6 s); `compile-default` 13.64/14.60 ms; CUDA graph 7.09/7.24 ms (4.7x). CUDA-graph N sweep: 2.29 ms at N=1 to 10.21 ms at N=16. `results/latency.jsonl`.
- One Triton kernel (fused last Euler step + de-normalization) passes equivalence at fp32 `atol=1e-5` (max diff 1.2e-4) and beats eager 2.2-2.4x, but **loses to compiled PyTorch** (0.83x at batch 1, 0.76x at batch 64). Reported as a negative result; `results/kernel.jsonl`.
- First compile of the kernel failed: Triton rejects `typing.Any` annotations inside `@triton.jit` functions. The pointer arguments are unannotated and the function carries a `no-untyped-def` ignore for mypy.

## 2026-10-19: M7 deployment loop and E10 (agent)

- ROS 2/Docker are unavailable (G9a NOT RUN), so the documented plain-Python fallback (D7) runs the same policy, env and JSONL interface: fixed 100 ms tick, action queue, deadline-miss counter, observation-staleness latency injection with uniform jitter, naive replace vs temporal ensembling.
- E10: 315 episodes (3 checkpoints x 3 eval seeds x 5 episodes) at 0/50/100/150/200 ms and 20 ms jitter. Mean coverage 0.293/0.334/0.287/0.334/0.322; zero successes and **zero deadline misses** because measured inference (45-55 ms) stays under the 100 ms period. Temporal ensembling does not recover anything measurable (0.299 at 100 ms, 0.272 vs 0.322 naive at 200 ms). Null result, reported as such.
- Cold-start bug: the first E10 pass timed the first chunk cold (871 ms), which put the mean above p95. The loop now warms up untimed per episode and `results/deploy.jsonl` was regenerated from scratch.

## 2026-10-19: M8 packaging and MVP close (agent)

- `scripts/make_tables.py` and `scripts/make_figures.py` regenerate every table (`results/tables/`) and figure (`results/figures/`) from `results/*.jsonl`; `scripts/make_readme.py` injects them into `README.md`. No number in the README is hand-typed.
- README carries the measured tables, the honest null results (no E1 success, Triton loses to compile, E10 flat), the E0 explanation, and the limitations list. Demo video is the one packaging item not produced (no capture path on this host); it is marked as missing in docs/00.
- Remaining gaps, stated rather than hidden: E2, E4-E8, E11, E12 not run; robomimic/ROS 2/CUDA C++ out of scope on this host; gym-pusht equivalence (R5) still unverified.

## 2026-10-23: 12000-step budget, E1/E3/E10 re-run, E12 made real (agent)

- Training budget raised from 1800 to 12000 steps on the owner's request: batch 64, 3.2 steps/s, 62-64 min per run, 3.1 GPU-h for the three policies. One training seed per policy (seed 0); the pre-registered three are a documented deviation in docs/04 §4. Peak VRAM 4458 MiB, final loss 0.054 flow / 0.062 DDPM / 0.024 regression.
- `scripts/train_e1.sh` now resumes instead of retraining: a run with `last.pt` is skipped, otherwise it resumes from the newest `step_*.pt`. The DDPM run was interrupted at step 8825 and resumed from step 8000.
- E1 (150 episodes, 50 per checkpoint, N=10): **flow is the first policy with non-zero success**, 8.0% (4/50) [2.0%, 16.0%]; regression 4.0% (2/50) [0.0%, 10.0%]; DDPM 0% (0/50). Mean coverage flow 0.471, regression 0.303, DDPM 0.284. Intervals are episode-level.
- Budget ablation, same pipeline: at 1800 steps flow was 0% / coverage 0.252 and DDPM 0% / 0.132; at 12000 steps flow is 8% / 0.471. The earlier null result was the budget, not a broken pipeline.
- E3 (600 episodes): flow retains coverage as N falls (0.401 at N=1 vs 0.468 at N=10, -14%) while DDPM loses twice as much (0.172 vs 0.239, -28%), which supports the graceful-degradation hypothesis in coverage. Success is too sparse to order N: flow 14% at N=4 and N=16 but 0% at N=10, non-monotonic and inside the +-6 pp resolution of 50 episodes. Latency 9.6 -> 69.4 ms.
- E10 (420 episodes, naive replan): coverage 0.439 at 0 ms and 0.385-0.417 at 50-200 ms, success 0-3.3% with overlapping CIs. Deadline misses 3-8 per ~2280 chunk inferences (individual slow chunks; mean inference 45-53 ms against a 100 ms period), down from the "zero misses" claim that only held at the 1800-step checkpoints.
- E12 bug found and fixed: temporal ensembling replanned only when the action queue emptied, so each step was covered by exactly one chunk and the ensemble was a copy of naive (coverage 0.411 at 100 ms, 0.413 at 200 ms, identical success). Added `replan_every`; naive and temporal-ensemble were re-run at a 2-step replan rate with four overlapping chunks. Naive replacement collapses (coverage 0.212 / 0.154, 0/120 successes); temporal ensembling recovers most of it (0.386 / 0.323, 4/120) but stays below the 8-step naive baseline (0.411 / 0.413) and costs 4x chunk inference.
- Tables, figures, README and docs/04 §3 claims regenerated from the new JSONL; docs/04 §4 records the new deviations (single training seed, replan-rate change, ~19% zero-coverage episodes). `just test` green.

## 2026-10-24: three training seeds, DDPM ladder bug fixed (agent)

- E1 now runs the pre-registered three training seeds at the 12000-step budget (seeds 1 and 2 trained at 3.2 steps/s, 62-64 min each), 50 paired episodes per checkpoint, 450 episodes total. Success: flow 7.3% [3.3%, 12.0%] (11/150), DDPM 4.7% [1.3%, 8.7%] (7/150), regression 4.0% [1.3%, 7.3%] (6/150); coverage 0.519 / 0.468 / 0.336. All three success CIs overlap, so the earlier "flow is the only policy with non-zero success" claim is retracted: the policies are ordered by point estimate and by coverage (regression clearly below both generative policies), not separated on success.
- `DDPMPolicy._sample` bug: the ladder ran from t=99 down to t=0, but `alphas_cumprod[0] == 1.0` exactly, so the last step was a no-op and every N behaved as N-1 steps; N=1 and N=2 produced bit-identical samples (coverage 0.2294 both). The ladder now ends at t=1. Guarded by `test_ddpm_step_count_is_effective`, which fails on the old code. DDPM E1 and E3 rows were re-evaluated (sampling-only change, no retraining).
- E3 re-run for DDPM: 0.229 [0.177, 0.284] at N=1, 0.213 [0.168, 0.264] at N=2, 0.467 [0.395, 0.536] at N=4, 0.550 [0.463, 0.632] at N=10 - DDIM needs at least 4 steps. Flow's ladder (0.401-0.543) has overlapping intervals at every N, so E3 cannot order N for flow; the table now prints coverage intervals so the reader can see that.
- `scripts/make_tables.py` prints a 95% bootstrap interval for E3 coverage; README, tables, figures and docs/04 §3/§4 regenerated from the new JSONL.

## 2026-10-27: pivot to a pretrained policy, then measure what a good policy actually costs (agent)

- **The owner asked for a 180: stop training, use a checkpoint that is already good, make the
  infrastructure the deliverable (D11).** The trigger is the ablation: three policies at 12000
  steps reach 4-7 % success, and at that level a comparison cannot order two policies and a
  latency experiment cannot see a policy degrade. The substrate is now
  `lerobot/diffusion_pusht` (Apache-2.0, 262.7 M parameters, fp32, trained on gym-pusht for
  175k steps; the model card reports 65.4 % success and 0.955 mean max overlap over 500
  episodes). Nothing is trained here any more.
- `src/flowpolicy/models/pretrained.py` is new: a `PolicyLike` adapter that puts the HF
  checkpoint behind the repo's interface. `models/policies.py` gained the structural
  `PolicyLike` protocol; `eval/rollout.py`, `eval/latency.py` and `deploy/control_loop.py`
  were re-typed to it. The episode loop, the latency protocol, the control loop, the tables
  and the figures were **not** changed -- one load branch in `load_policy` was enough, which is
  the architecture claim in docs/01 holding up.
- **Normalization is the only interesting part of the adapter.** lerobot 0.6.1 no longer
  creates the `normalize_inputs` / `unnormalize_outputs` modules the 2024 checkpoint was saved
  with, so `from_pretrained` logs those eight tensors as unexpected keys and drops them.
  The adapter reads them from `model.safetensors` (ImageNet mean/std for images, dataset
  min/max for state and actions) and applies them itself. A unit test covers the three
  transforms (`tests/unit/test_pretrained_normalization.py`, 4 tests, 33 total); CI stays
  backend-free because lerobot is imported lazily.
- **E0 is the acceptance gate and it passes.** The checkpoint through our gym-pusht wrapper,
  our harness and our metric: 64.0 % success (32/50) and 0.924 mean coverage at DDIM N=10,
  against the published 65.4 % / 0.955 for the checkpoint's own DDPM@100 setup. That also
  closes the reproduction item that the from-scratch route never reached (docs/00 section 3).
- **Dependencies:** lerobot 0.6.1 requires `torch < 2.12` while this project pins
  `torch 2.14.1+cu130`, so `just setup-pretrained` installs it `--no-deps` with the packages
  its inference path imports. Documented as a deviation (docs/04 section 4) and as R13 in
  docs/06, because it is verified working, not supported.
- **What moved:** from-scratch measurements now live in `results/ablation/` (e1, e3, deploy,
  latency, kernel, memory) and are presented as the ablation that motivated the pivot;
  pretrained rows are written to `results/{e0,e3,latency,deploy}.jsonl` with
  `train_seed = -1` and `weights = pretrained`, so no table can quietly mix them with a
  training-seed comparison. `E13` (36000 steps plus random-crop augmentation) was scaffolded
  before the pivot, never run, and has been removed rather than left half-finished.
- **Measured, pretrained checkpoint (2026-10-27).** E3 (300 episodes, DDIM): coverage 0.121
  [0.076, 0.165] at N=1, 0.716 at N=2, 0.885 at N=4, flat from there (0.888-0.919, intervals
  overlapping) -- one step is catastrophic, four reaches the plateau. E9 (batch 1): eager
  20.2 / 73.5 / 190.5 / 309.4 ms and CUDA graph 17.5 / 66.8 / 181.0 / 304.0 ms at N=1/4/10/16,
  i.e. graphs are worth 1.02-1.16x here against 4.7x on the 11 M ablation model; `torch.compile`
  is NOT RUN (Inductor needs MSVC for the CPU partition the checkpoint's DDIM scheduler creates,
  `InvalidCxxCompiler: cl is not found`), reported as NOT RUN by `scripts/latency_ladder.sh`
  rather than as a row. E10 (270 episodes): success 56.7 / 60.0 / 56.7 / 46.7 / 43.3 % at 0/50/
  100/150/200 ms injected latency with overlapping intervals and flat coverage (0.897-0.932), but
  26-33 deadline misses per episode because inference needs ~2x the control period. E12: naive
  replacement at a 2-step replan rate collapses to 10.0 % (coverage 0.749 / 0.553, disjoint from
  the baseline) while temporal ensembling recovers 53.3 % / 43.3 % (0.926 / 0.880). Together E3 +
  E9 fix the deployable envelope: only N <= 4 fits a 100 ms period, exactly where coverage has
  plateaued.
- **Infrastructure added the same day, none of it training.** `flowpolicy.eval.report` computes
  the headline summary that `just report` prints and that the README injects (so the five
  headline answers are generated, not typed); `flowpolicy.deploy.server` is a stdlib-only HTTP
  serving boundary with a documented JSON contract (docs/05 section 4) and six socket-level
  contract tests; `scripts/doctor.py` reports environment PASS/FAIL/NOT RUN in seconds;
  `docs/MODEL_CARD.md` covers the borrowed policy and this stack; CI gained a job that
  regenerates the tables and fails if the checked-in ones are stale. `kernel_bench.py` and
  `bench_memory.py` were repointed at `results/ablation/` after the move. Unit tests: 45.
- Docs re-framed around the pivot: 00 (status, goals, success criteria), 01 (interface and
  adapter, acceptance ticked), 02 (new section 3.4 on the optional backend), 03 (marked as the
  ablation path), 04 (matrix, deviations, priority order), 06 (D11, R13, R14), 07 (rule 4
  amended: borrowed policy code is now allowed, the harness/loop/tables stay ours).
- **E12 became the fifth headline answer, and the README prose stopped restating numbers.**
  `flowpolicy.eval.report` gained a row for the `-r2` rows (naive replacement 10 % at 100 ms
  and 200 ms against 57 % / 43 % for the 8-step baseline, temporal ensembling recovering
  53 % / 43 %), so the deployment finding is in `just report` and in the README's generated
  headline table rather than only in the E10 prose; `tests/unit/test_report.py` covers it
  (7 summary tests, 46 unit tests total). The paragraphs around the tables now describe the
  direction of each table and its exceptions without reprinting measured numbers, which is
  AGENTS.md rule 3 applied to prose as well as tables. Also recorded: the E12 deadline-miss
  counts (120 for the ensembling rows, 141-147 for naive-r2) in docs/04 and docs/MODEL_CARD.
  The ratios quoted in the docs were re-checked against `results/latency.jsonl`: 1.02-1.15x for
  CUDA graphs on the pretrained policy against 4.7x for the same comparison on the 11 M
  ablation model, where `torch.compile` was 5.8x faster than eager.
