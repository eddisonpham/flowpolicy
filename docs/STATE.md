# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M1 (G1 green; G3/G5 blocked on Triton) |
| Current step id | none |
| Green flag active | no |
| Last completed step | M0.1 repo scaffold |
| Last attempted step | M1.1 gates G1-G5 — partially green after the driver update, not signed off |
| Evidence | gate run 2026-10-19 (driver 617.42): G1 PASS, G2 PASS, G3 FAIL, G4 PASS, G5 FAIL; both FAILs are `triton` missing |
| Spec installed | docs/00-07 + CHANGELOG.md, byte-identical to the uploaded kit |

## Machine facts (measured 2026-10-18, driver updated 2026-10-19)
- OS: Windows 11, build 26200, MINGW64 (Git Bash). Single host for all work (owner, 2026-10-18): native Windows, no Ubuntu box, WSL2 not chosen. Dated deviation from the Ubuntu assumption in docs/02.
- CPU 24 cores. GPU NVIDIA GeForce RTX 5060 Laptop, 8151 MiB total. Free VRAM measured at 3428, 2640 and 6.88 GiB on three checks — measure at run time, never reuse a figure. Host is an Acer Predator PHN16-73.
- Compute capability 12.0 (sm_120), now readable through torch as `(12, 0)`; `arch_list` contains `sm_120` (pitfall 1 in 02 §4 cleared).
- Driver **617.42**, installed 2026-10-19 with CUDA UMD 13.4; meets the R580 minimum. Previous 577.13 was the G1 cause and is gone.
- Tooling: uv 0.11.30, just 1.36.0, git 2.54.0, Python 3.12.13 in `.venv`.
- torch 2.14.1+cu130 and torchvision 0.29.1+cu130 installed from `download.pytorch.org/whl/cu130` (Windows wheel exists).
- Missing: docker (G9a-G9c, deferred to M1.5), nvcc (G6), triton (G3 and G5).

## M1.1 result: G1 GREEN, G3/G5 BLOCKED ON TRITON
- 2026-10-19 run of the unchanged scripts: G1 PASS, G2 PASS, G3 FAIL, G4 PASS, G5 FAIL.
- G1: device visible, capability (12, 0), `sm_120` in `arch_list`, bf16 SDPA (G2) and CUDA graph capture/replay (G4) both correct.
- G3 and G5 share one cause: no `triton` module in this Windows environment. `torch.compile` on Windows requires it; the cu130 wheel does not ship it and 02 §3.1 says not to install it ad hoc.
- Consequence per 02 §8: "G3, G4, or G5 red must be fixed before any other work". So M1.2 is blocked until Triton-on-Windows is decided (install a Windows Triton wheel vs record a dated deviation dropping compile/Triton from scope).
- G6-G9c still NOT RUN: no `nvcc`, no Docker, no simulators.

## Open questions (waiting on user)
- Triton on Windows: install a Windows Triton build to clear G3/G5, or accept a dated deviation that drops compile/Triton (and the E9 compile ladder) from scope?
- `smoke` CI job from 01 §4 deliberately absent until a model exists (M3).
- Spec correction needed: 02 §9 references `10_risk_register.md`, which does not exist; risks live in 06 §2.

## Decision log (user answers, dated)
- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit `docs/STATE.md` on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `flowpolicy_complete` is the authoritative spec; docs/ adopts it verbatim.
- 2026-10-18: "proceed with implementation" taken as the green flag for M0.1 only (roadmap row M0).
- 2026-10-18: owner confirms this Windows 11 laptop runs all work; there is no separate Ubuntu host.
- 2026-10-18: "proceed with the recommended ones" -> native Windows for the gates; Docker/G9 deferred to M1.5. Taken as the go-ahead for M1.1.
