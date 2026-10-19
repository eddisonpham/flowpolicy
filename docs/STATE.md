# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M0 (not started) |
| Current step id | none |
| Green flag active | no |
| Last completed step | none (setup only, not a plan step) |
| Evidence | commit `25b39f0`; spec docs installed from `flowpolicy_complete` |
| Kit installed | CLAUDE.md, .claude/, docs/ at repo root |
| Spec installed | docs/00-07 + CHANGELOG.md, byte-identical to the uploaded kit |

## Machine facts (measured 2026-10-18)
- OS: Windows 11, build 26200, shell MINGW64 (Git Bash). Host is **not** the Ubuntu host the spec assumes.
- CPU: 24 cores. GPU: NVIDIA GeForce RTX 5060 Laptop, 8151 MiB total, 3428 MiB free at rest, driver 577.13.
- Tooling present: uv 0.11.30, just 1.36.0, ruff 0.16.0, mypy 2.3.0, pytest 9.1.1, git 2.54.0.
- Missing: docker (blocks G9a/G9b/G9c), nvcc (blocks G6).
- Host `python3` is 3.14.5, `python` 3.13.12; project targets 3.12 via uv.

## Open questions (waiting on user)
- Is this Windows laptop the machine that runs GPU work, or is there a separate Ubuntu box? Decides ROS 2 (D7) and the Docker gates.
- Windows driver 577.13 is below the Linux 580.65.06 minimum in 02 §3.1; cu130 wheel behaviour here is [U] until G1 runs.
- Spec correction needed: 02 §9 references `10_risk_register.md`, which does not exist; risks live in 06 §2.

## Decision log (user answers, dated)
- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit this bookkeeping file on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `flowpolicy_complete` is the authoritative spec; docs/ adopts it verbatim.
