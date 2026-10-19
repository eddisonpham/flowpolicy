# Docs changelog

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
