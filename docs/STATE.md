# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M1 (G1-G5 all PASS; M1.2 not started) |
| Current step id | none |
| Green flag active | no |
| Last completed step | M1.1 gates G1-G5 |
| Last attempted step | M1.1 gates G1-G5 — PASS, awaiting owner go-ahead for M1.2 |
| Evidence | 2026-10-19 `just gates` (driver 617.42, triton-windows 3.8.0.post29): G1 PASS, G2 PASS, G3 PASS, G4 PASS, G5 PASS; `just test` 2 passed |
| Spec installed | docs/00-07 + CHANGELOG.md, byte-identical to the uploaded kit |

## Machine facts (measured 2026-10-18, driver updated 2026-10-19)
- OS: Windows 11, build 26200, MINGW64 (Git Bash). Single host for all work (owner, 2026-10-18): native Windows, no Ubuntu box, WSL2 not chosen. Dated deviation from the Ubuntu assumption in docs/02.
- CPU 24 cores. GPU NVIDIA GeForce RTX 5060 Laptop, 8151 MiB total. Free VRAM measured at 3428, 2640 and 6.88 GiB on three checks — measure at run time, never reuse a figure. Host is an Acer Predator PHN16-73.
- Compute capability 12.0 (sm_120), now readable through torch as `(12, 0)`; `arch_list` contains `sm_120` (pitfall 1 in 02 §4 cleared).
- Driver **617.42**, installed 2026-10-19 with CUDA UMD 13.4; meets the R580 minimum. Previous 577.13 was the G1 cause and is gone.
- Tooling: uv 0.11.30, just 1.36.0, git 2.54.0, Python 3.12.13 in `.venv`.
- torch 2.14.1+cu130 and torchvision 0.29.1+cu130 installed from `download.pytorch.org/whl/cu130` (Windows wheel exists).
- Missing: docker (G9a-G9c, deferred to M1.5), nvcc (G6). Triton is present as `triton-windows==3.8.0.post29` (Windows-only marker).

## M1.1 result: ALL FIVE GATES PASS
- 2026-10-19 run: G1 PASS, G2 PASS, G3 PASS, G4 PASS, G5 PASS, scripts unedited since the first run.
- G1: device visible, capability (12, 0), `sm_120` in `arch_list`, VRAM 7.93 GiB total / 6.88 GiB free at run time.
- G2: bf16 autocast + SDPA produce bf16 output. G4: CUDA graph capture and replay match eager.
- G3 and G5 need Triton, which the cu130 wheel does not ship on Windows. Added `triton-windows==3.8.0.post29` with a `sys_platform == 'win32'` marker, so Linux CI keeps the torch-pinned Triton. Both gates then pass, and the assembler works (risk R2 does not apply on this host).
- G6-G9c still NOT RUN: no `nvcc`, no Docker, no simulators.
- `just test` green (2 passed). M1.2 has not been started; it needs the owner's green flag.

## Open questions (waiting on user)
- Green flag for M1.2 (device/tensor plumbing and dataset skeletons), the next roadmap row.
- G6 needs `nvcc`; decide whether the CUDA extension stretch stays in scope (D6).
- `smoke` CI job from 01 §4 deliberately absent until a model exists (M3).
- Spec correction needed: 02 §9 references `10_risk_register.md`, which does not exist; risks live in 06 §2.

## Decision log (user answers, dated)
- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit `docs/STATE.md` on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `flowpolicy_complete` is the authoritative spec; docs/ adopts it verbatim.
- 2026-10-18: "proceed with implementation" taken as the green flag for M0.1 only (roadmap row M0).
- 2026-10-18: owner confirms this Windows 11 laptop runs all work; there is no separate Ubuntu host.
- 2026-10-18: "proceed with the recommended ones" -> native Windows for the gates; Docker/G9 deferred to M1.5. Taken as the go-ahead for M1.1.
- 2026-10-19: owner installed the NVIDIA driver 617.42 (the driver was the only G1 cause).
- 2026-10-19: owner chose to install a Windows Triton build (`triton-windows==3.8.0.post29`) to clear G3/G5 rather than drop compile/Triton from scope.
