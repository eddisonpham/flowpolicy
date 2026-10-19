# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M1 (blocked at G1) |
| Current step id | none |
| Green flag active | no |
| Last completed step | M0.1 repo scaffold |
| Last attempted step | M1.1 gates G1-G5 — BLOCKED, not done |
| Evidence | gate runs 2026-10-18: G1-G5 all FAIL, one root cause; log in this file and docs/02 §8 |
| Spec installed | docs/00-07 + CHANGELOG.md, byte-identical to the uploaded kit |

## Machine facts (measured 2026-10-18)
- OS: Windows 11, build 26200, MINGW64 (Git Bash). Single host for all work (owner, 2026-10-18): native Windows, no Ubuntu box, WSL2 not chosen. Dated deviation from the Ubuntu assumption in docs/02.
- CPU 24 cores. GPU NVIDIA GeForce RTX 5060 Laptop, 8151 MiB total, 3428 MiB free at rest.
- Compute capability 12.0 (sm_120) measured via `nvidia-smi`; torch `arch_list` contains `sm_120`, so the wheel matches the GPU (pitfall 1 in 02 §4 cleared).
- Driver 577.13 — unchanged after the compatibility check on 2026-10-18. **This is the only remaining blocker:** below the CUDA 13.0 minimum (R580 branch).
- Tooling: uv 0.11.30, just 1.36.0, git 2.54.0, Python 3.12.13 in `.venv`.
- torch 2.14.1+cu130 and torchvision 0.29.1+cu130 installed from `download.pytorch.org/whl/cu130` (Windows wheel exists).
- Missing: docker (G9a-G9c, deferred to M1.5), nvcc (G6), triton (G5 import fails).

## M1.1 result: BLOCKED
- G1-G5 FAIL. torch reports `2.14.1+cu130`, cuda 13.0, but `cudaGetDeviceCount()` returns `cudaErrorNotSupported, likely using older driver`.
- One root cause for G1-G4: the host driver. G2-G4 were never exercised, so their behaviour is unknown, not bad.
- G5 has a second cause: `triton` is not installed by the Windows cu130 wheel.
- Not a cause: wrong wheel. `sm_120` is in the arch list and the GPU reports compute capability 12.0.
- Fix (owner action, cannot be done by the agent): update the NVIDIA driver to R580 or newer and reboot, then re-run `uv run python scripts/gates/g1_gpu.py`. Risk R1 in docs/06 says no fallback: fix before any model or data work.

## Open questions (waiting on user)
- Driver update scheduled? Everything in M1-M8 is blocked until G1 is green.
- Windows driver 577.13 vs the R580 minimum: confirm R580+ exists for this laptop GPU/driver channel.
- `smoke` CI job from 01 §4 deliberately absent until a model exists (M3).
- Spec correction needed: 02 §9 references `10_risk_register.md`, which does not exist; risks live in 06 §2.

## Decision log (user answers, dated)
- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit `docs/STATE.md` on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `flowpolicy_complete` is the authoritative spec; docs/ adopts it verbatim.
- 2026-10-18: "proceed with implementation" taken as the green flag for M0.1 only (roadmap row M0).
- 2026-10-18: owner confirms this Windows 11 laptop runs all work; there is no separate Ubuntu host.
- 2026-10-18: "proceed with the recommended ones" -> native Windows for the gates; Docker/G9 deferred to M1.5. Taken as the go-ahead for M1.1.
