# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M8 packaging; MVP complete 2026-10-19 |
| Current step id | none |
| Green flag active | owner granted a standing green flag on 2026-10-19 ("finish the project end to end, MVP stage") |
| Last completed step | M2-M7 implementation + E1, E3, E9, E10 + generated tables/figures/README |
| Evidence | `just test` green (23 passed; ruff and mypy clean, 2026-10-19); raw rows in `results/*.jsonl`; tables in `results/tables/`; figures in `results/figures/` |
| Spec installed | docs/00-07 + CHANGELOG.md from the uploaded kit; docs carry dated deviations where reality differs |

## Machine facts (measured)
- Windows 11 build 26200, Git Bash, single host. RTX 5060 Laptop, 8151 MiB total, driver 617.42 (CUDA UMD 13.4), compute capability (12, 0).
- Python 3.12.13 in `.venv`, torch 2.14.1+cu130, torchvision 0.29.1+cu130, `triton-windows==3.8.0.post29` (Windows marker only).
- Missing: Docker, ROS 2, `nvcc`, robosuite. G6, G8, G9a-G9c are NOT RUN and are never reported as passes.

## What exists and is measured
- Data: `lerobot/pusht` -> Zarr (`data/pusht/pusht.zarr`), train-only min/max stats, 96x96, To=2, Tp=16.
- Models: shared ResNet-18 GN multi-frame encoder; 1-D U-Net (flow, DDPM) and transformer chunk head (regression); 10.95 M parameters for flow/DDPM, 9.84 M for regression.
- Trainer: bf16, EMA 0.999, cosine + warmup, grad clip, resumable checkpoints, JSONL logs. Resume-equivalence unit test at `rtol=1e-5`.
- E1 (270 episodes, 3 policies x 3 seeds, 30 episodes/seed, N=10): success 0% flow, 0% DDPM, 1.1% regression; mean coverage 0.252 / 0.132 / 0.290. No success CI excludes 0. `results/tables/e1_main.md`.
- E3 (720 episodes, N in {1,2,4,8,10,16}): no successes; flow coverage 0.381 at N=1 vs 0.333 at N=10 (non-monotonic, hypothesis not supported). Latency grows with N (flow 10.7 -> 66.6 ms). `results/tables/e3_steps.md`.
- E9 (idle GPU, batch 1): eager 33.46 ms at N=10, compile-reduce-overhead 5.80 ms (5.8x), CUDA graph 7.09 ms (4.7x). `results/tables/e9_latency.md`.
- Triton fused Euler/de-norm kernel: equivalent (max diff 1.2e-4) and 2.2-2.4x faster than eager, but 0.76-0.83x of compiled PyTorch (it loses). `results/tables/kernel.md`.
- E10 (315 episodes, 0-200 ms injected latency + 20 ms jitter): mean coverage 0.287-0.334, zero successes, zero deadline misses (inference 45-55 ms < 100 ms period). Temporal ensembling does not help at 100 or 200 ms. `results/tables/e10_deploy.md`.
- Memory: batch 32 = 2317 MiB / 7.80 steps/s; batch 64 = 4414 MiB / 4.00 steps/s; batch 128 flagged not fitting. `results/tables/m2_memory.md`.
- Overfit 8 episodes (10x criterion): flow 0.053x and DDPM 0.083x at 1200 steps, regression 0.077x at 2000 steps.
- Packaging: `scripts/make_tables.py`, `scripts/make_figures.py`, `scripts/make_readme.py`; `just tables`, `just figures`, `just readme`.

## Open items (not blockers for the MVP)
- Demo video not recorded (no capture path on this host); docs/00 marks it as missing.
- E0 reproduction is not claimed; the 1800-step budget is ~1.5 epochs against ~200k in the reference. Written explanation in docs/04 §4.
- E2, E4-E8, E11 and E12 not run; robomimic, ROS 2, Docker and the CUDA C++ extension are out of scope on this host (D1, D6, D7).
- R5 (gym-pusht vs original env equivalence) is still unverified; reported as a finding.
- E1 latencies were measured with 2 worker processes (owner asked to reduce load); E9 is the single-process idle-GPU number and should be quoted for inference claims.

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
