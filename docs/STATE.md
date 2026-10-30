# STATE (agent updates after each step)

| Field | Value |
|---|---|
| Current milestone | M9 hardening (post-M8), 2026-10-28 |
| Current step id | none (evals complete: E0, E3, E3-replication, E8, E9, E10, E12, E14, serving load, paced loop; rename to `policybench` complete) |
| Green flag active | owner granted a standing green flag on 2026-10-19 ("finish the project end to end, MVP stage") |
| Last completed step | rename `flowpolicy` -> `policybench` (2026-10-28), after M9.7 |
| Evidence | `just verify` green (2026-10-30: ruff, ruff format, mypy strict on 29 files, 84 CPU unit tests, 1 smoke test, generated tables/README byte-identical); raw rows in `results/*.jsonl` (pretrained) and `results/ablation/*.jsonl` (from-scratch); tables in `results/tables/` |
| Spec installed | docs/00-07 + CHANGELOG.md; docs carry dated deviations where reality differs |

## What changed on 2026-10-30 (option H: audited, amended, scaffolded; nothing run yet)

H was the one gated option. It is now an amendment, a design, default-off code with a
*deterministic* gate -- and **measured: the gate passed, so H is a null.** The 200-episode paired
control also landed and is null at the checkpoint's own sampler. The diffusion stage therefore
has one measured sampler cost and **no measured adapter defect** at 200-episode resolution.

- **`just processor-equiv` -> PASS** (`results/processor_equiv.json`): migrated weights identical
  (0.0), images 2.384e-07 (one fp32 ulp), state 0.0, actions 0.0, history ordering 0.0 -- all
  inside the pre-registered 1e-6 bar. The checkpoint's buffers and `lerobot`'s processor
  pipelines are the same arithmetic, so the ~8 pp residual is not normalization.
- **`just lerobot-control-rows` -> 126/200 (63.0 %)**, 200 unique seeds (`results/lerobot_control/
  rows.jsonl`). Against our DDPM@100 (118/200): 78 both, 40 / 48 discordant, exact McNemar
  **p = 0.46 — not separated at the checkpoint's own sampler.** Against DDIM@10: 104 vs 126,
  35 / 57, p = 0.03 — the only separated comparison left, and it is the sampler.
- **The control's own spread is 1.0 pp, and the residual is ~4 pp.** Four repeats of the same
  row-writing command (r2-r5) join the original row file: 126, 128, 125, 123, 127 of 200 =
  **62.9 % ± 1.0 pp** (range 61.5-64.0). Against our DDPM@100 118/200 that is +3.9 pp with an
  episodes-and-runs bootstrap of **[-5.5, +13.0] pp** -- a null that spans zero
  (`results/lerobot_control/spread.json`). The 134/200 run came from the aggregate entry point
  (`just lerobot-control`) and is n=1, not a draw from this band.
- **`just env-equiv` -> identical env** (`results/env_equiv.json`): coverage and success traces
  agree to 0.0 step for step, same termination step, bit-identical first frame. R5's environment
  half is closed; the env term in loop-to-loop comparisons is zero.

- **The audit re-sized the target: ~4 pp, not 15.** Like-for-like -- same sampler, same 200
  environment seeds -- the residual is our DDPM@100 118/200 against the adapter-free control's
  five-run mean of 62.9 %. The 15 pp Fisher figure compared our DDIM@10 against a DDPM@100 control
  and therefore charged the sampler's share to the adapter; the 8 pp the audit first used came
  from the control's single 134/200 run.
- **Neither 200 nor 600 seeds resolve that.** Measured, not modelled: the DDPM arm's +7 pp had 88
  discordant pairs and p = 0.17, and a 4 pp effect would need more than the 600-seed arm was sized
  for. A block of either size would return "not distinguished" whether or not the truth was a
  defect, which is why the deterministic probe is the decision rule.
- **Flat coverage lowers H's prior.** 0.884-0.895 across every arm from 45.5 % to 69.0 % success
  means the losses are near-threshold rather than a systematic transform error -- an arithmetic
  bug usually moves coverage too. Step 0 is a refutation test, and passing it is expected.
- **Code, default off:** `models/normalization.py` (one declared mapping; `BuffersNorm` /
  `ProcessorsNorm`; history flatten/restore; the device-step override; the `config_id` guard),
  `normalization="processors"` threaded through `load_policy` / `evaluate` / `--normalization`,
  `scripts/processor_equiv.py` (step 0), `scripts/paired_test.py` + `just paired` for the 2x2.
- **Amendment recorded** in `docs/04` §4 and on rule 4 of `docs/07`, with its four conditions and
  an explicit "nothing adopted". Audit, design and runbook: `docs/10_processor_adoption.md`.
- Tests: `tests/unit/test_normalization_protocol.py` pins the seam, the history-order reshape and
  the guard. The lerobot half is exercised only by the probe, because CI has no `lerobot`.

## What changed on 2026-10-30 (options pass: three taken, three declined, one gated)

`docs/08` was a menu; this pass worked it in order of risk, one commit per option on branch
`exp/options`, and `docs/09_option_findings.md` records the cost and outcome of each. Taken: F
(multiplicity note), B (paired control rows), C (crash-bounded and resumable blocks, which is
where the real bug was). Declined: D (second task), E (ROS 2 path -- its measurable half already
exists in `bench_serve`/paced loop, and Docker is absent on this host), G (packaging). Gated: H
(adopt lerobot's processors), behind A's pre-registered reading.

- **C found a defect, not an ergonomics gap.** Shards were one contiguous chunk per worker and
  only a finished chunk flushed, so a single-worker block held every row in memory: A's first
  launch sat at zero rows in `.partial` for 25 minutes, and a 3 h block would have lost all of it
  to a closed laptop. Seeds are now cut into `workers * 8` interleaved chunks, rows flush per
  chunk, a re-run of the same `config_id` skips seeds already in `.partial`, and the append is
  sorted. Verified with a realistic partial: `resuming probe_n8b: 3/8`, five episodes run, 8 rows,
  8 unique seeds, partial removed.
- **B makes the adapter-free control pairable.** `policybench/eval/lerobot_rows.py` (pure, 5 tests)
  plus `scripts/lerobot_control_rows.py` wrap `eval_policy` and write this repo's rows, so the
  control stops being an aggregate comparison across different samples. Verified at 4 episodes:
  3/4 success, 3 rows censored at 0.95, `2 both, 1 ours only, 1 control only, p = 1.00`. The
  200-episode run is deliberately not repeated yet -- do it once, after A, so both arms are paired.
- **F cost one paragraph and immediately mattered.** `docs/04` §1 now states the six quoted tests,
  the uncorrected alpha and the Bonferroni arithmetic, plus the rule for blocks that grew. The
  per-block analysis below is the first post-hoc split the note has to cover.
- **A (running) is no longer the same question.** The committed 50-seed DDPM@100 arm is 30/50
  against the checkpoint's 37/50 (p = 0.14) and against our DDIM@10 32/50 (p = 0.83), but those
  seeds are the block where our DDIM is *already* not separated from the checkpoint (p = 0.40), so
  that arm had no gap to explain. Splitting the 200-seed block locates it: 1000-1049 32 vs 37
  (p = 0.40), 1050-1099 25 vs 33 (p = 0.13), **1100-1149 18 vs 34 (p < 0.001)**, 1150-1199 29 vs 34
  (p = 0.40), with coverage lowest on the same block (0.857 against 0.894-0.917). Post-hoc and
  labelled as such; A must therefore be read per block. Pre-registered reading: DDPM@100 separated
  from DDIM@10 on 1100-1149 means truncation is the cause; not separated means H (adapter) is next.
- Costs measured on the way: DDPM@100 is 4.0 s per chunk and ~154 s per episode at two workers
  (2.3x per chunk for 2x the work, i.e. slower than one worker), so the recipe now defaults to one
  worker and the 200-seed block is ~3 h.

## What changed on 2026-10-30 (stage close: one computed verdict, idempotent runs)

A cleanup pass, no new measurements. Every item either removes a way for the repository to
contradict itself or makes a multi-hour run survive a crash.

- **The README's reproduction verdict was wrong, and the generator was the cause.** The
  headline row read "Yes: 64.0 %" because `eval/report.py` computed it from the unpaired
  50-episode E0 block, while `docs/00` had recorded the criterion as PARTIAL since 2026-10-29
  and the paired 200-seed block says p < 0.001. `headline_rows` now takes the paired block
  (`results/e0_published_seeds.jsonl`) and the checkpoint's own block
  (`results/diffusion_published.jsonl`) and renders the computed verdict; the unpaired branch
  says "unpaired only" and names the command that would pair it. Guarded by two new tests.
- **One implementation of each test.** McNemar and Fisher exact moved from
  `scripts/make_tables.py` into `policybench/eval/stats.py`, so `mypy --strict` covers them and
  the table generator and the README cannot word the same p-value differently. The script
  imports them under their old names, so the existing table tests are unchanged.
- **A crashed eval block cannot be mistaken for a complete one.** `rollout.evaluate` flushes
  rows to `<out>.partial` per shard and appends to `<out>` only after every shard finishes, so
  a 2.5 h block loses at most one shard and a partial block is never read as a result.
  `results/*.partial` is ignored.
- **Re-running a recipe no longer doubles its rows.** The recipes that append to a fixed JSONL
  (`eval-dp-published`, `eval-dp-noise`, `eval-dp-published-ddpm`, `eval-vqbet-replication`)
  now prune their own `config_id` prefix first, matching `eval-vqbet`. Duplicate seeds would
  have inflated `n` in every paired test.
- **`just verify`** runs the CI commands plus the artifact diff (`tables`, `readme`) in one
  command, so the pre-commit question is answered locally, not by waiting for CI.
- **The control's setup is testable without the control.** `scripts/lerobot_migrate.py` exposes
  `strip_normalization` and `tests/unit/test_lerobot_migrate.py` pins which keys survive (the
  weights) and which the processors take over (the eight buffers).
- **Committed the rows the docs cite:** `results/e0_published_seeds_noise2.jsonl` and
  `results/lerobot_control/eval_info.json`; `justfile`, `lerobot_control.sh` and `docs/00`,
  `docs/04`, `docs/CHANGELOG.md` carry the same story. `docs/08_next_stage_options.md` records
  the next stage as options with costs and risks.

## What changed on 2026-10-28 (E15: a second policy family, and the architecture page)

**The generality claim is now measured, not asserted.** `lerobot/vqbet_pusht` (behaviour
transformer over discretized action tokens, 5 observation steps, 5-action chunk, 37.5 M
parameters) runs behind the same `PolicyLike` and through the unchanged harness, latency
protocol and control loop. New code: `models/pretrained_vqbet.py` (config filtering + its
own checkpoint normalization) and a `config.json`-`type` dispatch in `load_policy`. Evidence:
`results/vqbet.jsonl` (50 episodes, seeds 100000-100049), `results/vqbet_latency.jsonl`,
`results/vqbet_loop.jsonl`, `results/tables/generality.md`, README E15 section, `docs/04`
matrix row + claim + false-if.

Measured: 54.0 % [40.0 %, 68.0 %] success and 0.851 [0.790, 0.905] coverage against the
checkpoint's published 57.0 % over 500 episodes. 9.2 ms per chunk eager at batch 1 (199 MiB
peak), 1.83 ms per control step against the diffusion checkpoint's 9.19.

**Corrected, then resolved at a real sample size.** The first version of this entry reported a
"10 pp success gap" against the checkpoint's block. It was wrong twice over: gym-pusht reports
`reward = clip(coverage / 0.95, 0, 1)`, so the published `max_reward` column is coverage
divided by 0.95 (the "4 pp coverage gap" was that scaling), and the 50-episode paired test
could not order the runs anyway (8 vs 13 discordant, p = 0.38).

Both half-fixes were then measured properly on 2026-10-29:

- **VQ-BeT, 200 shared seeds** (`results/vqbet.jsonl`, seeds 100000-100199): 108/200
  (54.0 % [47.0 %, 61.0 %]) against the checkpoint's own 113/200 (56.5 %), 107 agreeing, 44
  succeeding only here and 49 only there, exact McNemar p = 0.68. Mean coverage 0.818 against
  0.780. The two runs are not distinguished.
- **Diffusion, 200 shared seeds** (`results/e0_published_seeds.jsonl`, seeds 1000-1199, which
  are inside the checkpoint's block and the original E0 block is not): 104/200 (52.0 %
  [45.0 %, 58.5 %]) against the checkpoint's own 138/200 (69.0 %), 106 agreeing, 30 succeeding
  only here and 64  only there, exact McNemar p < 0.001. Coverage 0.895 [0.872, 0.917] here against 0.915 there,
  but theirs is a **censored lower bound**: gym-pusht clips reward at 1.0, so all 138 of its
  successes record exactly 0.95 (and the card's 0.955 implies only 0.907), and only its failures
  carry a real number. On the 32 episodes both runs fail -- the only like-for-like coverage
  comparison -- it is 0.850 against 0.832. **The 50-seed version of this test was underpowered
  and its "not distinguished" reading was wrong**: the first 50 seeds reproduce exactly (32/50),
  and the gap sits where coverage lands just under the success threshold -- 64 episodes sit in
  [0.90, 0.95), where this repo succeeds 0 times and the checkpoint 42. So neither success nor
  coverage is shown to reproduce; no root cause is established. A second pass on 2026-10-30
  that shifts only the policy noise seed (same 200 seeds, same sampler) gives 91/200 (45.5 %),
  so the rate is a property of the policy, not of the Gaussian it happened to draw.
- **VQ-BeT replication, disjoint 200-seed block** (`results/vqbet_replication.jsonl`, seeds
  100200-100399): 108/200 (54.0 %) against 108/200 (54.0 %) on the first block, two-sided
  Fisher exact p = 1.00; coverage 0.777 against 0.818. The success rate repeats on episodes the
  first block never saw.
- `scripts/published_blocks.py` (`just published-blocks`) pins both blocks to
  `results/{vqbet,diffusion}_published.jsonl`; `results/tables/generality.md` prints both tests,
  and `make_tables` derives the verdict from p rather than asserting it.

Also in this pass: `docs/ARCHITECTURE.md` (one page: layers, seams, invariants, extension
points), the README subtitle and `docs/MODEL_CARD.md` now state the deployment-cost framing
explicitly, and four unit tests cover the config filtering the adapter depends on (61 tests).

## What changed on 2026-10-28 (clean-clone dry run, and the CI dependency it found)

The README quickstart was executed in a fresh `git clone` of `7a0792a` in a temp directory, in
order, with no edits:

| Step | Result |
|---|---|
| `uv sync --python 3.12` | exit 0, 97 packages, `policybench` installed editable |
| `just test` | 57 passed (18.5 s cold) |
| `just smoke` | 1 passed |
| `just tables && just figures && just readme` | exit 0, `git diff --exit-code` clean |
| `just setup-pretrained` | exit 0, lerobot 0.6.1 imports, torch stays 2.14.1+cu130 |
| `just doctor` | 6 PASS (python, cuda, sim, lockfile, backend, checkpoint), 2 NOT RUN (dataset, MSVC) |
| `just report` | exit 0, headline table |
| `scripts/evaluate.py`, 2 episodes (stand-in for `just eval-e0`) | exit 0, coverage 0.942, success 1/2 |

Nothing broke -- but only after the `diffusers` fix earlier in this session: on the pre-fix tree
a clean `uv sync` leaves `just test` unable to collect, which is exactly what CI's `unit` job
runs. Not exercised: the full 100-episode `just eval-e0` (~50 min), `just data` and the ablation
path, `just test-local`, `just serve` (blocks), `just gates`.

## What changed on 2026-10-28 (renamed to PolicyBench)

The package and project name now say what the repository does. The old name described the
from-scratch flow-matching models, which became an ablation on 2026-10-27; the stack itself
measures and deploys a borrowed diffusion policy.

- `src/flowpolicy/` -> `src/policybench/` (`git mv`), all `flowpolicy.*` imports and docs paths
  rewritten, `pyproject.toml` name/description/hatch target updated, `uv.lock` relocked.
- Project title in `README.md` (via `docs/README_template.md`), `CLAUDE.md`, `scripts/report.py`
  and the package docstrings now read PolicyBench. `models.policies.FlowPolicy` keeps its name:
  that class is the flow-matching ablation model and the name is accurate.
- The uploaded brief stays verbatim at `flowpolicy_complete/`; `docs/00` points at it there.
- `docs/CHANGELOG.md` carries a note that older entries were path-rewritten, not rewritten
  history.
- Verified: `just test` 57 passed, tables and figures regenerate byte-identically (no data
  contained the old name), README diff is the title plus ten path mentions.
- **Found while doing it:** the rename forced a real `uv sync`, which prunes the optional
  backend -- and `just test` then fails at collection, because `tests/unit/test_fast_sampler.py`
  imports `diffusers` while the lockfile has neither `diffusers` nor `lerobot`. CI's `unit` job
  runs exactly `uv sync --frozen` + `pytest -m "unit and not gpu"`, so that job has been red
  since `9e5be8c` regardless of the rename. Not fixed here; see open items.

## What changed on 2026-10-28, M9 (hardening after packaging)

Seven steps, in order, each either producing rows or making an already-written sentence true.

- **M9.1 spec reconciliation.** `docs/04` E9 still said `torch.compile` was NOT RUN, the
  README's reading order said the same, and `docs/00` advertised E0-E12. Fixed, E14 given a
  matrix row, twelve acceptance boxes across docs/00-03 ticked with evidence or annotated as
  unmet, and the two dangling `10_risk_register.md` references retargeted at `docs/06` section 2.
- **M9.2 CI smoke job.** `tests/integration/test_train_smoke.py` (20 steps, loss falls, resume
  matches at `rtol=1e-5`), new `smoke` marker, `just smoke`, CI job. This is the job docs/01
  section 4 promised on 2026-10-18 and deferred "until a model exists".
- **M9.3 E8 precision.** `--dtype fp32|bf16|fp16` through the adapter and runners (fast sampler
  only), `scripts/bench_precision.py`, `results/precision.jsonl`. Half precision buys 1.3-1.4x
  per chunk and half the VRAM but not a control period: the best N=10 row is still 1.4x over
  100 ms, so the envelope stays N <= 4. Success over 40 paired episodes per dtype is
  indistinguishable (35 % fp32, 50 % bf16, 45 % fp16, intervals overlapping).
- **M9.4 E3 replication** on a second, independent 50-episode block (seeds 30000-30049):
  coverage replicates to within 0.04 at every step count, success swings 6-14 pp, so the
  plateau is real and the success ordering above N=2 is episode noise.
- **M9.5 serving load test.** `scripts/bench_serve.py` spawns the real server and measures
  client vs server time: 17.8 ms of JSON/socket per request and two clients serialising on the
  GPU, with a 655 ms cold first request excluded by warmup.
- **M9.6 paced loop.** `LoopConfig(paced=True)` / `--paced`: real sleeps, per-tick lateness.
  N=4 holds the 100 ms period; N=10 falls behind and stays behind.
- **M9.7 E12 third arm.** `strategy="blend"` cross-fades the queue into the new chunk. It
  recovers coverage (0.875 / 0.864) but not success (26.7 % / 10.0 %), unlike temporal
  ensembling (53.3 % / 43.3 %), so the pre-registered "inpainting-style is best" hypothesis is
  refuted for the implementable version and ensembling remains the requirement.

## What changed on 2026-10-28 (the compile gap closed, and the value baseline)

The owner asked for two things the earlier packaging did not deliver: a **measured baseline**
for the stack's own usefulness, and a plain answer to "what is this and why does it matter".
Both are now in the repo; the second is a README section and `docs/RESUME.md`.

- **Root cause behind the `torch.compile` failure, not just a workaround.** `diffusers` keeps
  `alphas_cumprod` on the host and indexes it with a Python int per denoising step, so lerobot's
  DDIM loop mixed ten host scalars into CUDA math per chunk. That both forced a host/device
  synchronization per step and gave Inductor a CPU partition that wants MSVC. The 2026-10-27
  note called this "an environment limitation"; it was a sampler-implementation problem.
- `src/policybench/models/fast_sampler.py` precomputes the DDIM coefficients on the sampling
  device and runs the loop in tensor math only. Proven equal to `diffusers` at five step counts
  and two beta schedules (`tests/unit/test_fast_sampler.py`, 9 tests), including the case where
  it disagrees if a coefficient is applied twice -- a bug this file had on first draft and the
  test caught.
- **E14 (`results/stack.jsonl`) is the usefulness baseline.** Same checkpoint, same GPU, same
  episodes, three ways to run it: the published configuration (DDPM@100), the stock
  lerobot->diffusers DDIM path, and the fast sampler with CUDA graphs and `torch.compile`.
  Latency rows plus throughput rows (s/episode, episodes/hour, success, coverage).
- The adapter gained `sampler="stock"|"fast"`; every existing row keeps `stock`, so E0/E3/E10
  remain comparable to what they were. `--sampler` is threaded through `evaluate`,
  `deploy_eval` and `latency`; RNG and noise handling are unchanged.
- `scripts/demo_rollout.py` (`just demo`) renders one episode to an annotated GIF so the
  behaviour is visible, not only tabulated.

## What changed on 2026-10-27 (D11 pivot)

The project stopped training and put a borrowed policy behind the same interface. The
from-scratch models reached 4-7 % success at 12000 steps, which is a floor effect: two
policies cannot be ordered at 6-11 successes in 150 episodes, and a latency experiment
cannot see a policy degrade when it is not succeeding. The owner asked for the 180.

- Substrate: `lerobot/diffusion_pusht` (Apache-2.0, Diffusion Policy, gym-pusht, 175k steps,
  262.7 M parameters fp32). Model card: 65.4 % success, 0.955 mean max overlap over 500
  episodes.
- New code: `src/policybench/models/pretrained.py` (adapter + normalization read from the
  checkpoint), `PolicyLike` protocol in `models/policies.py`, one pretrained branch in
  `rollout.load_policy`, a `--scheduler` flag on the three runners. The episode loop, the
  deployment loop, the latency protocol and the generators did not change.
- Install: `just setup-pretrained` (lerobot 0.6.1 `--no-deps`, because it caps torch < 2.12
  and this host runs torch 2.14.1+cu130). Deviation in docs/04 section 4, risk R13 in docs/06.
- Removal: E13 (36000 steps + random-crop) scaffolding deleted unrun; `train.log` removed.

## Machine facts (measured)

- Windows 11 build 26200, Git Bash, single host. RTX 5060 Laptop, 8151 MiB total, driver
  617.42 (CUDA UMD 13.4), compute capability (12, 0).
- Python 3.12.13 in `.venv`, torch 2.14.1+cu130, torchvision 0.29.1+cu130,
  `triton-windows==3.8.0.post29` (Windows marker only).
- Pretrained runtime: lerobot 0.6.1 (outside the lockfile), diffusers 0.42.1 (locked in the
  `dev` group as of 2026-10-28), safetensors 0.8.0, huggingface-hub 2.2.0, draccus 0.11.6.
  Verified importing and running on torch 2.14.1+cu130; lerobot's own requirement is torch < 2.12.
- Missing: Docker, ROS 2, `nvcc`, robosuite. G6, G8, G9a-G9c are NOT RUN and are never
  reported as passes.

## What exists and is measured

- **Pretrained path**: adapter with per-call DDIM/DDPM scheduler choice; checkpoint
  normalization (ImageNet mean/std, dataset min/max) applied from `model.safetensors`;
  4 unit tests on those transforms; `hf:<repo_id>` accepted by `evaluate.py`,
  `deploy_eval.py` and `latency.py`.
- **E0 (validation gate)**: 64.0 % success (32/50) and 0.924 mean coverage at DDIM N=10 on
  seeds 10000-10049, against the model card's 65.4 % / 0.907 coverage. `results/e0.jsonl`,
  `results/tables/e0_validation.md`. This is the project's reproduction check, and the honest
  summary after 2026-10-29 is narrower than it was: on 200 paired episodes of the checkpoint's
  own seed block (1000-1199) neither metric is shown to reproduce: coverage is 0.895 against a
  censored lower bound of 0.915, and success is 104/200 against 138/200 (exact McNemar
  p < 0.001, `results/e0_published_seeds.jsonl`). The earlier 50-seed read of p = 0.40 was a
  power problem.
- **Latency at the operating point** (batch 1, fp32, single process, measured before the
  eval runs): 20.7 ms at N=1, 178.3 ms at N=10, 1766.6 ms at N=100. Peak VRAM 1125 MiB.
- **E3 (DDIM ladder, 300 episodes)**: coverage 0.121 [0.076, 0.165] at N=1, 0.716 [0.627,
  0.804] at N=2, 0.885 [0.847, 0.919] at N=4 and flat from there (0.919 / 0.888 / 0.892 at
  N=8/10/16, all intervals overlapping). Success 0/50 at N=1, 10/50 at N=2, then noise
  (42-60 %). Latency 31.5 -> 726.7 ms under two-worker contention. `results/e3.jsonl`.
- **E9 (latency ladder, batch 1, single process)**: eager 20.2 / 73.5 / 190.5 / 309.4 ms at
  N=1/4/10/16; CUDA graph 17.5 / 66.8 / 181.0 / 304.0 ms, i.e. 1.02-1.16x, against 4.7x on the
  11 M ablation model. These rows use the stock sampler. `torch.compile` now runs (2026-10-28)
  on the fast sampler and is reported in E14: graph and compile are both worth roughly a tenth
  here, because a 262.7 M-parameter U-Net at batch 1 is compute-bound. E9/E14 plus E3 fix the
  deployment envelope: only N <= 4 fits a 100 ms period.
- **E10 (270 episodes, injected observation latency 0-200 ms + 20 ms jitter)**: success
  56.7 % at 0 ms, 60.0 % at 50 ms, 56.7 % at 100 ms, 46.7 % at 150 ms, 43.3 % at 200 ms (all
  CIs overlapping); coverage flat at 0.897-0.932. Deadline misses 26-33 per episode at N=10,
  i.e. inference needs ~2x the 100 ms period. `results/deploy.jsonl`.
- **E12 (replan every 2 steps, four overlapping chunks)**: naive replacement 10.0 % [0.0, 20.1]
  at 100 ms and 10.0 % [0.0, 23.3] at 200 ms (coverage 0.749 / 0.553), disjoint from the 8-step
  baseline; uniform temporal ensembling recovers 53.3 % / 43.3 % (coverage 0.926 / 0.880) at 4x
  the chunk count. Separated, unlike the ablation's ambiguous version. It is also the fifth row
  of the computed headline summary, so `just report` and the README table carry it
  (`policybench.eval.report` strips the runner's `-r2` suffix to pair the rows).
- **E14 (2026-10-28) is the stack's own value baseline**, same checkpoint / GPU / episodes:
  published DDPM@100 53.1 s per episode and 23.9 s of GPU time per 100 control steps; stock
  DDIM@10 5.5 s and 2.65 s; fast-sampler DDIM@4 2.6 s and 1.19 s (20x cheaper than published),
  with coverage 0.947 -> 0.961 and success 50 % -> 75 % on the same 4 episodes. Latency: 65.2 ms
  (graph) / 65.6 ms (compile-reduce-overhead) / 72.4 ms (stock eager) at N=4; 160-183 ms at N=10,
  so only N <= 4 fits a 100 ms period. Two rollout workers cost 1.8x per control step -- on this
  policy parallelism is not a throughput win. Sampler equivalence: action chunks within 0.02 px
  of the library's, but only 9/20 paired episodes agree on success (chaotic task, 20-episode
  resolution), reported rather than hidden.
- **Adapter-free control (2026-10-30)**: `scripts/lerobot_control.sh` runs the same checkpoint
  through `lerobot`'s own eval loop in an isolated `.venv-lerobot`, on seeds 1000-1199:
  134/200 (67.0 %) against the block's 138/200 (69.0 %), Fisher p = 0.75. The reference
  reproduces on this host; this repo's 104/200 (52.0 %) is the shortfall. Aggregate only --
  that lerobot version writes no per-episode rows.
- **Infrastructure added 2026-10-27 (no training involved)**: `policybench.eval.report` (computed
  headline summary, used by `just report` and the README), `policybench.deploy.server` +
  `scripts/serve.py` (`just serve`, stdlib-only HTTP serving boundary, contract in docs/05
  section 4), `scripts/doctor.py` (`just doctor`, PASS/FAIL/NOT RUN environment check),
  `docs/MODEL_CARD.md`, and a CI job that regenerates the tables and fails if they are stale.
  Tests: 46 unit tests total (6 for the server contract over a real socket, 7 for the summary
  computations, 4 for the checkpoint normalization).
- **Episode-set variance to keep in mind**: the same configuration at DDIM N=10 scores 64.0 %
  on eval seeds 10000-10049 (E0) and 54.0 % on 20000-20049 (E3). Two 50-episode samples of one
  policy differ by 10 percentage points, which is the resolution limit of every table here.
- **Ablation (from-scratch, kept)**: 3 policies x 3 seeds x 50 paired episodes at 12000
  steps = 450 episodes, success flow 7.3 % [3.3, 12.0], DDPM 4.7 % [1.3, 8.7], regression
  4.0 % [1.3, 7.3], coverage 0.519 / 0.468 / 0.336; all success CIs overlapping. Training
  cost, overfit checks and the Triton kernel (which loses to `torch.compile`, 0.76-0.83x)
  all under `results/ablation/`. Training is still reproducible via `just train-e1`.

## Open items (not blockers)

- **Resolved 2026-10-28: the `unit` CI job could not pass from a clean checkout.**
  `tests/unit/test_fast_sampler.py` imports `diffusers` to compare against the library's
  scheduler, and `diffusers` was installed only by `just setup-pretrained`, which CI never runs.
  Fixed by pinning `diffusers==0.42.1` in the `dev` group, so the equivalence test (9 tests)
  now runs under `uv sync --frozen` instead of being skipped. Verified in a CI-shaped env:
  `uv sync --frozen` then `pytest -m "unit and not gpu"` -> 57 passed.
- One checkpoint exists for PushT, so E0/E3/E9/E10 intervals are episode-level and there is
  no seed-to-seed variance to report. Stated on every table that has pretrained rows.
- **Resolved 2026-10-30: the diffusion reference block can be regenerated on this host.** The
  2026-10-29 entry said it could not; both blockers are cleared, in an isolated environment.
  `av` 15.1.0 lives in `.venv-lerobot` (built by `scripts/build_lerobot_env.sh`, so the project
  env keeps its own `av`), and the missing `policy_preprocessor.json` is produced by
  `scripts/lerobot_migrate.py`, which runs lerobot's own `migrate_policy_normalization` and then
  finishes the model save that crashes inside draccus (`Couldn't encode 84`, on this
  checkpoint's `crop_shape: [84, 84]`). `just lerobot-control` runs `lerobot-eval` on the
  migrated copy for seeds 1000-1199, with `use_async_envs=false` because Windows has no
  `forkserver` context. It returned 134/200 (67.0 %) against that block's 138/200 (69.0 %),
  Fisher p = 0.75: **the reference reproduces on this host.** Nothing in the project's locked
  environment changed. The remaining gap is on this repo's side, but it is not a 15 pp adapter
  shortfall: 15 pp compares DDIM@10 with the control's DDPM@100. The 200-seed DDPM@100 cell
  filled on 2026-10-30 (`docs/09` option A) scores 118/200 (59.0 %) against DDIM@10's 104/200 on
  the same episodes, exact McNemar p = 0.17, so the published sampler buys ~7 pp at 4.4x the
  per-chunk cost and ~8 pp is not sampling -- and that ~8 pp is not separated from the
  adapter-free control at the same sampler (paired p = 0.46).
- **The success gap is not the noise stream this repo draws (2026-10-30).** Same 200 seeds, same
  sampler, only the per-episode policy noise seed shifted: 91/200 (45.5 %) against 104/200
  (52.0 %), exact McNemar p = 0.21, coverage 0.884 against 0.895, and only 109/200 episodes
  agree. The noise decides individual episodes -- it is the loosest joint in the pipeline -- but
  it does not move the rate, and 69 % sits outside both blocks. With normalization buffers, the
  84x84 crop, action slicing, observation rendering, observation history, truncated sampling
  (DDPM@100 p = 0.14) and now the noise stream excluded, the 200-seed gap was unexplained as of
  that run. The 200-seed DDPM@100 block then bought ~7 pp (59.0 % against 52.0 %, p = 0.17); the
  ~4 pp that left is not separated from the adapter-free control at that sampler (126/200, five
  repeats at 62.9 % ± 1.0 pp, paired p = 0.46, bootstrap [-5.5, +13.0] pp) and is no larger than
  the control's own measured spread, so it is not attributed to the adapter. `docs/09` option A has
  the split.
- `lerobot` lives outside `uv.lock` (it caps torch < 2.12) and `just setup` prunes it;
  `diffusers` is locked in `dev`, so the unit gate no longer depends on the backend. The
  lerobot half is still documented rather than solved.
- R5 (gym-pusht vs original PushT equivalence) is still unverified, but E0 now anchors the
  harness to an external published number, which is the stronger check.
- E10/E12 run the loop in virtual time; the paced rows (`results/paced.jsonl`) are separate
  and smaller (12 episodes per step count), and the N=4 vs N=10 success difference there is
  directional rather than separated.
- E8 resolves about 15 pp at 40 paired episodes, not the 2 pp the pre-registered hypothesis
  named: it can show "no gross regression" but not equality.
- E12's third arm is a cross-fade over queued actions, not model-level inpainting; the
  literature survey that would justify a stronger strategy is still `[U]`.
- The serving load numbers are one quiet run on this laptop (`results/serve.jsonl`), not a
  benchmark under contention.
- The demo is a 55-frame GIF of one episode (`results/figures/demo_rollout.gif`, `just demo`),
  not a narrated video.
- E14's per-arm episode count is small (4 for the cost arms, 20 for the equivalence check), so
  its success columns are indicative; the cost columns are per control step and do not depend on
  episode length.
- E2, E4-E7, E11 not run; robomimic, ROS 2, Docker and the CUDA C++ extension are out of
  scope on this host (D1, D6, D7). The Triton kernel stays an ablation result. E8 now has rows
  (on the pretrained checkpoint, not the ablation models it was registered against).

## Decision log (user answers, dated)

- 2026-10-18: install agent kit by promoting `agent_kit/` to repo root; push `main` to origin.
- 2026-10-18: commit `docs/STATE.md` on direct user instruction, without `GO <id>`.
- 2026-10-18: uploaded `policybench_complete` is the authoritative spec; docs/ adopts it verbatim.
- 2026-10-18: "proceed with implementation" taken as the green flag for M0.1 only (roadmap row M0).
- 2026-10-18: owner confirms this Windows 11 laptop runs all work; there is no separate Ubuntu host.
- 2026-10-18: "proceed with the recommended ones" -> native Windows for the gates; Docker/G9 deferred.
- 2026-10-19: owner installed the NVIDIA driver 617.42 (the only G1 cause).
- 2026-10-19: owner chose `triton-windows==3.8.0.post29` to clear G3/G5 rather than drop compile/Triton.
- 2026-10-19: owner granted a standing green flag: finish the project end to end at MVP stage without further approval requests.
- 2026-10-23: owner asked for full results; training budget raised from 1800 to 12000 steps (one seed per policy) and eval episodes raised to 50. `just train-e1` resumes interrupted runs.
- 2026-10-23 19:06: owner asked to pause training to go AFK; the DDPM run was stopped at step 8825 with a checkpoint at step 8000.
- 2026-10-23 22:52: training resumed from the step-8000 checkpoint and finished at 12000 steps for all three policies; E1, E3 and E10 re-run on the new checkpoints.
- 2026-10-24: three training seeds at the 12000-step budget; DDPM sampling-ladder bug found and fixed; E1/E3 re-evaluated.
- 2026-10-27: owner asked what the verdict on the project was; answer: process strong, scientific content weak (null on success, floored signal).
- 2026-10-27: owner asked to do a 180 -- use a model that is already good and work on the software infrastructure. Decisions taken with the owner: (1) wrap the `lerobot` package rather than reimplement the model, (2) full re-frame: adapter, validation, re-run E3/E9/E10, rewrite README and docs 00-07, (3) keep the from-scratch work as a documented ablation and delete the unrun E13 scaffolding.
- 2026-10-28: owner asked to fix the outstanding issues, produce a measured usefulness baseline, and explain the software in plain terms plus XYZ resume bullets. Scope taken: close the compile gap at its cause, add E14 as the baseline, render a demo artifact, and write the explanation down (`README.md` top section, `docs/RESUME.md`) rather than only in chat.
- 2026-10-28: owner asked to plot the remaining phases, then "proceed with the full implementation of EVERYTHING you planned" with M9.3 (E8 precision on the pretrained checkpoint) named explicitly. Scope taken: M9.1-M9.7 in the proposed order, one measurement per step, every claim entered in docs/04 with a "false if" condition, and the unmeasured remainder (E2, E4-E7, E11, narrated video) left declared rather than attempted.
- 2026-10-28: owner asked whether the project should be renamed, then chose `policybench` (package and project) with the full scope: package directory, pyproject, lock, README title and docs prose.
