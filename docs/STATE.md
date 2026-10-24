# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M8 packaging; MVP complete 2026-10-19 |
| Current step id | none |
| Green flag active | owner granted a standing green flag on 2026-10-19 ("finish the project end to end, MVP stage") |
| Last completed step | 12000-step training + E1, E3, E10 rerun + regenerated tables/figures/README |
| Evidence | `just test` green (2026-10-23; ruff, mypy and 23 unit tests); raw rows in `results/*.jsonl`; tables in `results/tables/`; figures in `results/figures/` |
| Spec installed | docs/00-07 + CHANGELOG.md from the uploaded kit; docs carry dated deviations where reality differs |

## Machine facts (measured)
- Windows 11 build 26200, Git Bash, single host. RTX 5060 Laptop, 8151 MiB total, driver 617.42 (CUDA UMD 13.4), compute capability (12, 0).
- Python 3.12.13 in `.venv`, torch 2.14.1+cu130, torchvision 0.29.1+cu130, `triton-windows==3.8.0.post29` (Windows marker only).
- Missing: Docker, ROS 2, `nvcc`, robosuite. G6, G8, G9a-G9c are NOT RUN and are never reported as passes.

## What exists and is measured
- Data: `lerobot/pusht` -> Zarr (`data/pusht/pusht.zarr`), train-only min/max stats, 96x96, To=2, Tp=16.
- Models: shared ResNet-18 GN multi-frame encoder; 1-D U-Net (flow, DDPM) and transformer chunk head (regression); 10.95 M parameters for flow/DDPM, 9.84 M for regression.
- Trainer: bf16, EMA 0.999, cosine + warmup, grad clip, resumable checkpoints, JSONL logs. Resume-equivalence unit test at `rtol=1e-5`.
- 12000-step runs (batch 64, seed 0): 3.2 steps/s, 62-64 min per run, peak VRAM 4458 MiB, final loss 0.054 flow / 0.062 DDPM / 0.024 regression.
- E1 (150 episodes, 3 policies x 1 seed, 50 episodes/checkpoint, N=10): success flow 8.0% [2.0, 16.0] (4/50), regression 4.0% [0.0, 10.0] (2/50), DDPM 0.0% (0/50); mean coverage 0.471 / 0.303 / 0.284. Intervals are episode-level (one training seed). `results/tables/e1_main.md`.
- E3 (600 episodes, N in {1,2,4,8,10,16}): flow keeps coverage as N falls (0.401 at N=1 vs 0.468 at N=10) while DDPM loses twice as much (0.172 vs 0.239). Success is too sparse to order N: flow 14% at N=4 and N=16, 0% at N=10. No DDPM successes. Latency grows with N (flow 9.6 -> 69.4 ms). `results/tables/e3_steps.md`.
- E9 (idle GPU, batch 1): eager 33.46 ms at N=10, compile-reduce-overhead 5.80 ms (5.8x), CUDA graph 7.09 ms (4.7x). `results/tables/e9_latency.md`.
- Triton fused Euler/de-norm kernel: equivalent (max diff 1.2e-4) and 2.2-2.4x faster than eager, but 0.76-0.83x of compiled PyTorch (it loses). `results/tables/kernel.md`.
- E10 (540 episodes, 0-200 ms injected latency + 20 ms jitter, 2 policies): coverage 0.439 at 0 ms and 0.385-0.417 at 50-200 ms, success 0-3.3% with overlapping CIs; 3-8 deadline misses per ~2280 chunk inferences under naive replace (individual slow chunks, not a systematic overrun). `results/tables/e10_deploy.md`.
- E12 (partial, 240 episodes at 100/200 ms): at a 2-step replan rate naive replacement collapses (coverage 0.212 / 0.154, 0 successes) while temporal ensembling over four overlapping chunks recovers most of it (0.386 / 0.323, 4/120 successes) but does not beat the 8-step naive baseline (0.411 / 0.413). Ensembling costs 4x chunk inference. The original ensemble rows were a no-op (no chunk overlap) and were replaced, see docs/04 §4.
- Budget ablation: the same pipeline at 1800 steps gave flow 0% / coverage 0.252 and DDPM 0% / 0.132; at 12000 steps flow is 8% / 0.471. The earlier null result was the budget, not the pipeline. Archived rows under `logs/archive/`.
- Memory: batch 32 = 2317 MiB / 7.80 steps/s; batch 64 = 4414 MiB / 4.00 steps/s; batch 128 flagged not fitting. `results/tables/m2_memory.md`.
- Overfit 8 episodes (10x criterion): flow 0.053x and DDPM 0.083x at 1200 steps, regression 0.077x at 2000 steps.
- Packaging: `scripts/make_tables.py`, `scripts/make_figures.py`, `scripts/make_readme.py`; `just tables`, `just figures`, `just readme`.

## Open items (not blockers for the MVP)
- E1/E3/E10 run one training seed at the 12000-step budget, not the pre-registered three; CIs are episode-level. A 3-seed sweep is about 6 GPU-h.
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
- 2026-10-23: owner asked for full results on this iteration; training budget raised from 1800 to 12000 steps (one seed per policy) and eval episodes raised to 50. `just train-e1` now resumes interrupted runs from their newest periodic checkpoint.
- 2026-10-23 19:06: owner asked to pause training to go AFK; the DDPM run was stopped at step 8825 with a checkpoint at step 8000 and the GPU left idle. Resume instructions in `logs/RESUME.md`.
- 2026-10-23 22:52: training resumed from the step-8000 checkpoint and finished at 12000 steps for all three policies; E1, E3 and E10 re-run on the new checkpoints.
