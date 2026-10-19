# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M0 (scaffold complete) |
| Current step id | none |
| Green flag active | no |
| Last completed step | M0.1 repo scaffold |
| Evidence | `just test` -> exit 0, 2 passed; clean clone of HEAD -> `just setup && just test` exit 0 |
| Spec installed | docs/00-07 + CHANGELOG.md, byte-identical to the uploaded kit |

## Machine facts (measured 2026-10-18)
- OS: Windows 11, build 26200, shell MINGW64 (Git Bash). Host is **not** the Ubuntu host the spec assumes.
- CPU: 24 cores. GPU: NVIDIA GeForce RTX 5060 Laptop, 8151 MiB total, 3428 MiB free at rest, driver 577.13.
- Tooling: uv 0.11.30, just 1.36.0, git 2.54.0, Python 3.12.13 in `.venv` (uv-managed).
- Missing: docker (blocks G9a/G9b/G9c), nvcc (blocks G6).
- Host `python3` is 3.14.5, `python` 3.13.12; project pins 3.12 via uv.

## M0.1 scope actually delivered
- `pyproject.toml` (3.12, hatchling, ruff, mypy strict on `src`, pytest markers), `uv.lock`, `justfile`, `.pre-commit-config.yaml`, `README.md`, `AGENTS.md`, `.github/workflows/ci.yml`, `src/flowpolicy/__init__.py`, `tests/unit/test_package.py`.
- No runtime dependency yet: torch and the rest enter with the step that first names them (M1.1 gates).

## Open questions (waiting on user)
- Is this Windows laptop the machine that runs GPU work, or is there a separate Ubuntu box? Decides ROS 2 (D7) and the Docker gates.
- Windows driver 577.13 is below the Linux 580.65.06 minimum in 02 §3.1; cu130 behaviour here is [U] until G1 runs.
- CI has never executed: `.github/workflows/ci.yml` is unverified until the first push triggers Actions.
- Spec correction needed: 02 §9 references `10_risk_register.md`, which does not exist; risks live in 06 §2.

## Decision log (user answers, dated)
- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit `docs/STATE.md` on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `flowpolicy_complete` is the authoritative spec; docs/ adopts it verbatim.
- 2026-10-18: "proceed with implementation" taken as the green flag for M0.1 only (roadmap row M0). M1 is not started.
