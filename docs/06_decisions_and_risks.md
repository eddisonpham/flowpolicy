# 06. Decisions (ADR summary) and risk register

Owner decides; each row has a recommended default. Mark the chosen option and date in the "Decision" column. Links: [00](00_overview.md) [02](02_environment_setup.md) [07](07_roadmap.md)

## 1. Decisions

| ID | Decision | Options | Recommended | Evidence | What would change it | Decision |
|---|---|---|---|---|---|---|
| D1 | Benchmarks | PushT; PushT+robomimic; +ManiSkill | PushT + robomimic | Both have public demos and reference recipes ([03](03_data_models_training.md)); ManiSkill GPU sim is heavy at about 7 GB VRAM [U, not measured] | G8 fails: PushT only | CHOSEN 2026-10-19: PushT only (G8 NOT RUN; fallback in the row) |
| D2 | Robomimic scope | Lift; Lift+Can; +Square; image vs low-dim | Lift + Can image, Square if time | Image obs cost VRAM and CPU; robomimic Lift is the cheapest task | Memory table shows image batch does not fit: low-dim for robomimic | DROPPED 2026-10-19 with D1 (robomimic out of scope; G8 NOT RUN) |
| D3 | Encoder | ResNet-18 GN; small ViT; cached DINOv2 | ResNet-18 GN; DINOv2 cached as E7 | DP recipe uses ResNet-18 GN [S] | Time short: skip E7 | CHOSEN 2026-10-19: ResNet-18 GN from scratch; E7 not run |
| D4 | Action head | 1D U-Net; transformer | U-Net primary, transformer E6 | DP default is ConditionalUnet1D [S] | Transformer wins in overfit test and fits memory | CHOSEN 2026-10-19: 1-D U-Net + FiLM; E6 not run |
| D5 | Language conditioning | none; LIBERO | none in core; revisit week 5 | Adds a benchmark, a text encoder, more compute | Weeks 1-5 exit criteria met early | CHOSEN 2026-10-19: none in core |
| D6 | Stretch pillar | CUDA port; world-model eval; real arm; none | CUDA C++ port if ahead; gate G6 must pass | One kernel in several implementations beats several half-finished ones (brief) | G6 red or behind schedule: none | CHOSEN 2026-10-19: none (G6 NOT RUN, no nvcc); Triton kernel still run for M6 |
| D7 | Deployment layer | ROS 2 Docker; native ROS 2; plain Python | ROS 2 Jazzy in Docker | Jazzy Python 3.12 matches training env, supported to 2029-05-31 [V, see 02]; Lyrical is the current LTS but uses Python 3.14 on Ubuntu 26.04 [V] | G9 red: plain Python loop. Host is Ubuntu 26.04 and owner prefers native: Lyrical | CHOSEN 2026-10-19: plain Python loop (G9a NOT RUN, no Docker) |
| D8 | Tracking | W&B only; W&B + JSONL | both | Offline reproducibility, results regenerate from JSONL | none | CHOSEN 2026-10-19: JSONL only; W&B omitted (dated deviation) |
| D9 | PushT env | gym-pusht; DP repo env | gym-pusht (pip, maintained package) after an equivalence check | LeRobot port reports about 63.8% vs 64.2% with original code on gym-pusht [S] | Check shows a mismatch: use DP env code as reference (reading only, per brief rule 5) | CHOSEN 2026-10-19: gym-pusht 0.1.8; original-env equivalence still [U] (R5) |
| D10 | Python and CUDA | 3.12 + cu130 | as recommended | [02](02_environment_setup.md) section 3 | gate failures | CHOSEN 2026-10-19: Python 3.12 + cu130, verified by G1 |

## 2. Risk register

| # | Risk | L | I | Mitigation | Kill / fallback trigger |
|---|---|---|---|---|---|
| R1 | Wrong CUDA wheel, no kernels for the GPU | M | H | gate G1 arch check | none, fix before anything else |
| R2 | Triton cannot assemble kernels on cu13 (no bundled ptxas) | M | M | install toolkit, `TRITON_PTXAS_PATH` | still failing after 1 day: skip Triton, keep compile + graphs |
| R3 | No `nvcc` / extension build fails | M | L | toolkit or container | drop CUDA stretch |
| R4 | robosuite/robomimic/MuJoCo version conflicts or headless GL | H | M | pinned lock, `--no-deps` robomimic, EGL first | week 1: drop robomimic |
| R5 | gym-pusht differs from original PushT | M | M | equivalence check, reproduction E0 | report as finding |
| R6 | Cannot reproduce about 64% on PushT | M | M | check normalization, EMA, horizons ([03](03_data_models_training.md) section 6) | write up why; proceed with relative comparisons |
| R7 | Model or batch does not fit VRAM | M | H | memory table, smaller batch, checkpointing, 84x84 | switch to low-dim variant |
| R8 | Training too slow for 3 seeds x 3 policies | H | H | measure steps/s in M2; reduce epochs equally for all policies; fewer seeds on secondary tasks | below 3 seeds on E1: report as exploratory |
| R9 | ROS 2 Docker GPU or DDS issues | M | M | `--network host`, gate G9 | defer; plain Python loop (ADR) |
| R10 | Resume-equivalence not bitwise | M | L | state wording rule in 03 | say "tolerance-based" |
| R11 | Overrun of any phase by more than a few days | M | H | roadmap fallbacks | invoke fallback, log in CHANGELOG |
| R12 | Week 8 squeezed | M | H | week 8 protected; start README skeleton at M4 | never cut packaging |

L = likelihood, I = impact (H/M/L). Estimates are judgments, not measurements.

Outcomes as of 2026-10-19: R2 cleared (G5 passes with `triton-windows`). R3 triggered and accepted (no `nvcc`, D6 closed). R4 triggered and accepted (robosim not installed, G8 NOT RUN, D1 = PushT only). R5 remains open (no original-env equivalence check; reported as a finding, not a pass). R6 triggered: the 1800-step DDPM baseline reaches 0% success, so no reproduction is claimed and the fallback "write up why; proceed with relative comparisons" was used. R8 triggered: the budget is 1800 steps; 3 seeds were kept for E1 and every result is labelled scaled-down. R9 triggered: Docker absent, plain Python loop (D7). R10 handled: tests and docs say tolerance-based (`rtol=1e-5`), never bit-exact.

## 3. Acceptance criteria

- [x] Owner has filled the Decision column for D1 to D7 and D9 before M3 (2026-10-19).
- [x] Each triggered risk has a dated entry in `docs/CHANGELOG.md` (R2-R6, R8-R10 are recorded there).
