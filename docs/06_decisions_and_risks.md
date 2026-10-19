# 06. Decisions (ADR summary) and risk register

Owner decides; each row has a recommended default. Mark the chosen option and date in the "Decision" column. Links: [00](00_overview.md) [02](02_environment_setup.md) [07](07_roadmap.md)

## 1. Decisions

| ID | Decision | Options | Recommended | Evidence | What would change it | Decision |
|---|---|---|---|---|---|---|
| D1 | Benchmarks | PushT; PushT+robomimic; +ManiSkill | PushT + robomimic | Both have public demos and reference recipes ([03](03_data_models_training.md)); ManiSkill GPU sim is heavy at about 7 GB VRAM [U, not measured] | G8 fails: PushT only | OPEN |
| D2 | Robomimic scope | Lift; Lift+Can; +Square; image vs low-dim | Lift + Can image, Square if time | Image obs cost VRAM and CPU; robomimic Lift is the cheapest task | Memory table shows image batch does not fit: low-dim for robomimic | OPEN |
| D3 | Encoder | ResNet-18 GN; small ViT; cached DINOv2 | ResNet-18 GN; DINOv2 cached as E7 | DP recipe uses ResNet-18 GN [S] | Time short: skip E7 | OPEN |
| D4 | Action head | 1D U-Net; transformer | U-Net primary, transformer E6 | DP default is ConditionalUnet1D [S] | Transformer wins in overfit test and fits memory | OPEN |
| D5 | Language conditioning | none; LIBERO | none in core; revisit week 5 | Adds a benchmark, a text encoder, more compute | Weeks 1-5 exit criteria met early | OPEN |
| D6 | Stretch pillar | CUDA port; world-model eval; real arm; none | CUDA C++ port if ahead; gate G6 must pass | One kernel in several implementations beats several half-finished ones (brief) | G6 red or behind schedule: none | OPEN |
| D7 | Deployment layer | ROS 2 Docker; native ROS 2; plain Python | ROS 2 Jazzy in Docker | Jazzy Python 3.12 matches training env, supported to 2029-05-31 [V, see 02]; Lyrical is the current LTS but uses Python 3.14 on Ubuntu 26.04 [V] | G9 red: plain Python loop. Host is Ubuntu 26.04 and owner prefers native: Lyrical | OPEN |
| D8 | Tracking | W&B only; W&B + JSONL | both | Offline reproducibility, results regenerate from JSONL | none | RECOMMENDED |
| D9 | PushT env | gym-pusht; DP repo env | gym-pusht (pip, maintained package) after an equivalence check | LeRobot port reports about 63.8% vs 64.2% with original code on gym-pusht [S] | Check shows a mismatch: use DP env code as reference (reading only, per brief rule 5) | OPEN |
| D10 | Python and CUDA | 3.12 + cu130 | as recommended | [02](02_environment_setup.md) section 3 | gate failures | RECOMMENDED |

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

## 3. Acceptance criteria

- [ ] Owner has filled the Decision column for D1 to D7 and D9 before M3.
- [ ] Each triggered risk has a dated entry in `docs/CHANGELOG.md`.
