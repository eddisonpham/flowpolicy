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
