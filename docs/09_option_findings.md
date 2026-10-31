# 09. Option findings: what each next-stage option cost, and what it bought

Companion to `docs/08_next_stage_options.md` (the menu). This page records what was actually
tried on 2026-10-30, in order of risk, and what it changed. Every item below is one commit on
branch `exp/options`, so any of them can be reverted on its own without touching the others.

Ordering rule used: anything that only writes prose or a new isolated file goes first; anything
that changes the harness (`src/`) goes after it still has tests; anything that costs GPU hours or
adds a third-party surface goes last, and only if the cheaper items did not already answer it.

| Option | Risk | Status | Commit |
|---|---|---|---|
| F. Multiplicity / repeated-looks note | lowest (docs only) | **done** | `33f7215` |
| B. Per-episode rows from the adapter-free control | low (new isolated file) | **done, verified** | `d1e27f0` |
| C. Crash-bounded, resumable eval blocks | medium (changes the harness) | **done, verified** | `8bd1ea3` |
| A. DDPM@100 on the 200 published seeds | none in code, 2 h 58 m GPU | **done** | (this commit; rows in `results/e0_published_seeds_ddpm.jsonl`) |
| H. Adopt lerobot's processors | medium, contingent on A | **gated** | — |
| D. A second task | high | **not started** | — |
| E. Real deployment path (ROS 2) | high | **partial already measured** | — |
| G. Packaging / PyPI / container | medium, non-scientific | **not started** | — |

## F. Multiplicity and repeated looks — done

`docs/04` §1 now names the six hypothesis tests the README quotes, states that each is reported at
alpha 0.05 uncorrected on its own pre-registered block, and gives the reader the Bonferroni
arithmetic (bar 0.008; only E0's p < 0.001 clears it, so the correction would change no sentence).
It also fixes the rule for blocks that grew after a first look: final size only, earlier looks in
the changelog, never stopped or extended because of a p-value.

**It bit immediately.** The per-block analysis under A is a post-hoc split of one 200-seed block
into four; it is exploratory and is labelled that way wherever it appears.

## B. Per-episode rows from the adapter-free control — done, verified

`lerobot-eval` 0.6.1 writes only `eval_info.json`, so the control that separates this repo's
adapter from the checkpoint's own pipeline could be compared only in aggregate (134/200 vs
138/200) while every other comparison here is paired by env seed. The per-episode records are
computed inside `eval_policy` and then discarded.

Two files, no patch of the library: `policybench/eval/lerobot_rows.py` (pure conversion, five unit
tests) and `scripts/lerobot_control_rows.py` (wraps `eval_policy`, writes this repo's row schema,
prints the paired McNemar). The reward-scaling inversion and the 0.95 censor are the same ones
`scripts/published_blocks.py` already carries.

Verified end to end on the real loop, 4 episodes at batch 2, seeds 1000-1003:
`3/4 = 75.0 % success, 3 rows censored at 0.95`, paired `2 both, 1 ours only, 1 control only ->
exact McNemar p = 1.00`. Cost: ~4.5 min under GPU contention with A.

**Not yet run at 200.** Three of the four seeds succeeding is consistent with the 67 % aggregate,
and re-running the whole control is a ~2 h job. Do it once, after A, so both arms of the
adapter-vs-sampler question are paired at the same time.

## C. Crash-bounded, resumable eval blocks — done, verified

The menu asked for the cheap version of resumable runs. Looking at the code first found a real
hole rather than an ergonomics gap: `evaluate` sharded seeds as one contiguous chunk per worker
and flushed after each chunk, so a **single-worker** block flushed nothing until its last episode.
That is not hypothetical — A's first launch sat at `0` rows in `.partial` for 25 minutes, and a
3-hour block would have lost all of it to a closed laptop.

Changed `rollout.evaluate`: seeds are cut into `workers * 8` interleaved chunks, rows flush as each
chunk completes, a re-run of the same `config_id` skips the seeds already in `.partial` (and
ignores rows from any other `config_id`), and rows are sorted before the append so the block's
order no longer depends on which chunk finished first. Five unit tests on the chunking and the
resume read; the pool loop itself is exercised by the probe below.

Verified with a realistic partial (three complete rows for `probe_n8b`, seeds 5000-5002, as a
deterministic stand-in for a crash): the run printed `resuming probe_n8b: 3/8 seeds already in
.partial, 5 to run`, ran five episodes, produced 8 rows with 8 unique seeds, one `config_id`, and
deleted the partial. The first, deliberately malformed probe (rows missing `latency_mean_ms`)
crashed `scripts/evaluate.py`'s summary *after* the GPU work — the harness never writes such a
row, so this is a property of the probe, not a defect, but see the open item below.

Residual: a crash costs at most one chunk (25 of 200 episodes at the default), not one episode.

## A. DDPM@100 on the 200 published seeds — done, and it splits the gap

The only open scientific item was a 15-point success gap known to sit in *this repo's path* (the
adapter-free control reproduces the checkpoint on the same seeds) but not where in that path.

**Cost, measured rather than estimated.** The DDPM@100 arm costs 4.0 s per chunk and ~154 s per
episode with two workers; E14's clean single-process number is 1766 ms per chunk, and two workers
measured 2.3x per chunk for 2x the work, i.e. slower in wall clock. The recipe therefore defaults
to one worker (`d1e27f0`), and the 200-seed block is a ~3 h job (2 h 58 m measured, one worker).

**What the already-committed 50-seed arm says, and why it is not the answer.** That block
(`results/e0_published_seeds_ddpm.jsonl` at `7094d5d`) is 30/50 (60.0 %) against the checkpoint's
37/50 (74.0 %) on the same seeds, exact McNemar p = 0.14, and against this repo's DDIM@10 on the
same seeds 30/50 vs 32/50, p = 0.83. Read alone that looks like "truncation is not the cause". It
is weaker than that: **the 50 seeds it used are the block where this repo's DDIM is already not
separated from the checkpoint** (32/50 vs 37/50, p = 0.40), so the arm had almost no gap to
explain. It is a null on the easy block, not a null on the question.

**Exploratory per-block split (post-hoc, four tests).** Splitting the 200-seed paired block into
50-seed blocks locates the gap:

| Seeds | Ours DDIM@10 | Checkpoint | Discordant (ours / theirs) | Exact McNemar | Our mean coverage |
|---|---|---|---|---|---|
| 1000-1049 | 32/50 (64.0 %) | 37/50 (74.0 %) | 9 / 14 | p = 0.40 | 0.914 |
| 1050-1099 | 25/50 (50.0 %) | 33/50 (66.0 %) | 7 / 15 | p = 0.13 | 0.894 |
| **1100-1149** | **18/50 (36.0 %)** | **34/50 (68.0 %)** | **5 / 21** | **p < 0.001** | **0.857** |
| 1150-1199 | 29/50 (58.0 %) | 34/50 (68.0 %) | 9 / 14 | p = 0.40 | 0.917 |

Three of four blocks are not separated; the aggregate is carried by one block, where 21 episodes
succeed only for the checkpoint and 5 only for this repo. That is a different shape of problem from
"uniformly 15 points worse", and it changes what A is for: the arm has to be read **per block**
too, because a sampler deficit acting uniformly and an adapter defect acting on one seed band
predict different splits. (The block split is post-hoc; four tests at alpha 0.05 put the
Bonferroni bar at 0.0125, which only the 1100-1149 block clears.)

**Result (200 seeds, one worker, 2 h 58 m, `results/e0_published_seeds_ddpm.jsonl`).**

| Seeds | DDIM@10 (this repo) | DDPM@100 | Discordant (ours / theirs) | Exact McNemar |
|---|---|---|---|---|
| **1000-1199 (pre-registered)** | 104 (52.0 %) | **118 (59.0 %)** | 37 / 51 | p = 0.17 |
| 1000-1049 | 32 (64.0 %) | 30 (60.0 %) | 12 / 10 | p = 0.83 |
| 1050-1099 | 25 (50.0 %) | 35 (70.0 %) | 7 / 17 | p = 0.064 |
| 1100-1149 | 18 (36.0 %) | 30 (60.0 %) | 5 / 17 | p = 0.017 |
| 1150-1199 | 29 (58.0 %) | 23 (46.0 %) | 13 / 7 | p = 0.26 |

Mean coverage is flat (0.895 at DDIM@10, 0.893 at DDPM@100) and the arm costs 4.4x per chunk at
batch 1 (1819 ms against 417 ms; both are in the rows).

**Against the pre-registered reading.** Neither branch fires as written. Branch 1 wanted ~69 %
overall *and* separation on 1100-1149: the arm lands at 59.0 % with p = 0.17 overall, and the
band's p = 0.017 does not clear the four-look Bonferroni bar of 0.0125 that `docs/04` §1 sets.
Branch 2 wanted no separation on the band: there is one, and it is the largest effect in the table.
The honest reading is the one neither branch offered:

- **The sampler schedule is part of the gap, not all of it.** The checkpoint's own 100-step DDPM
  sampler buys +7.0 pp overall (52.0 -> 59.0) and +24 pp on 1100-1149 (36.0 -> 60.0), which is most
  of that band's 32-point deficit. It leaves ~10 pp to the checkpoint's 69 % unexplained, and on
  1150-1199 the arm *loses* 6 pp (p = 0.26), so band-level effects at n = 50 are themselves noisy.
  What the table supports is "truncated sampling is a real, partial cause" -- not "the band is the
  sampler".
- **It does not rescue the reproduction claim.** The published sampler is 4.4x the per-chunk cost
  and still 10 pp short of the checkpoint's own block here, so E0's criterion stays PARTIAL and
  DDIM@10 stays the deployment-relevant point estimate (`docs/05`). What changes is attribution:
  the ~15 pp was "this path against the reference"; it is now "~7 pp sampler choice, ~8 pp adapter
  or process", and only the second half is a defect.
- **H's target is now smaller and located.** It has to explain ~8-10 pp at DDPM@100 on the
  1100-1149 band, with a paired design, rather than a 15 pp block average.

## D, E, G. Assessed and not started

**D (a second task).** Unchanged verdict: most of the cost is environment archaeology (robosuite
and robomimic absent, G8 NOT RUN), and E15 already showed the *policy-family* axis with two
checkpoints under one `PolicyLike`. A second PushT checkpoint would add a family, not a task, and
a second task would mostly test the env wrapper, which is the least interesting seam here. Not
now; it is the right option only if the goal changes from "a deployment stack measured on one
task" to "a harness for several benchmarks".

**E (real deployment path).** The defensible half is already measured: `bench_serve` reports
17.8 ms of JSON/socket per request against in-process GPU time, and the paced loop reports
wall-clock lateness. The full version needs ROS 2, which is Docker-only in this repo and Docker is
not installed on this host, so its main claim could not be verified here. Dropped rather than
half-built.

**G (packaging).** Publishing `policybench` promises API stability for a harness with a GPU and a
simulator dependency; the CPU container cannot be built or verified on this host either. This is
reach, not evidence, and `docs/07` rule 1 says an option is worth taking if it produces rows or
makes a written sentence true. Skipped.

## H. Adopt lerobot's processors — now the leading hypothesis

Medium cost, and it trades away the boundary `docs/07` rule 4 draws (measurement machinery stays
ours), so it needs a dated amendment like D11's. A did not exonerate the sampler, but it did not
clear it either: ~7 pp of the 15 was sampler choice, ~8 pp survives at the checkpoint's own
sampler, and the band effect that is left is 60.0 % against 68.0 %. So H is the only hypothesis
cheap enough to run before D, and its target is a paired ~8-10 pp gap on 1100-1149 at DDPM@100.

**Measured and declined (2026-10-30).** Three results, none of them needing the episode arm:

- **`just processor-equiv` passed at 1e-6** (`results/processor_equiv.json`): the checkpoint's eight
  buffers and `lerobot`'s processor pipelines agree on 20 real frames -- weights 0.0, images
  2.4e-07 (one fp32 ulp), state 0.0, actions 0.0, history ordering 0.0. The mapping is not the
  defect, so the pre-registered rule stops H here.
- **The paired control is null at the checkpoint's sampler** (`results/lerobot_control/rows.jsonl`,
  200 rows): 126/200 for the control against our DDPM@100 118/200 -- 78 both, 40 / 48 discordant,
  exact McNemar **p = 0.46**. Only DDIM@10 against that control separates (104 vs 126, p = 0.03),
  and that comparison *is* the sampler. Five repeats of that loop (r2-r5 plus the original row
  file) score 126, 128, 125, 123 and 127 -- **62.9 % ± 1.0 pp** -- so the gap is +3.9 pp with a
  [-5.5, +13.0] pp episodes-and-runs bootstrap that spans zero. The 134/200 scored earlier came
  from the other entry point (`just lerobot-control`, aggregate only) and is one unrepeated run,
  not a sample of this loop's spread (`docs/04` §4).
- **`just env-equiv` shows the env wrapper is identical** (coverage and success traces agree to
  0.0, same termination step, identical first frame), so no part of any loop-to-loop gap is the
  environment.

Net: the ~4 pp that survived option A is inside the combined noise of the noise stream, the
control's own repeats (sd 1.0 pp) and 200-episode resolution. No located defect; H is closed as an
explanation and kept only as a tested seam. The audit makes that resolution explicit rather than
implied: a 200-episode block here resolves ~13 pp, so this null excludes a defect of that size and
nothing smaller (`docs/04` §1, `just power-audit`).

**The design, for the record (2026-10-30).** `docs/10_processor_adoption.md` carries the
amendment, an audit of the decision and the adoption design; the code exists default-off
(`models/normalization.py`, `normalization="processors"`) and **nothing has been measured with
it**. The audit moved two numbers rather than the direction:

- The like-for-like residual is **~4 pp, not 15** — our DDPM@100 (118/200) against the control's
  five-run mean (125.8/200, 62.9 % ± 1.0 pp) at the same sampler on the same seeds. The 15 pp
  figure compares our DDIM@10 against a DDPM@100 control and so charges the sampler's share to
  the adapter; the 8 pp this audit was first written with came from the control's single 134/200
  run, which sits ~4 sd outside its own repeated band.
- A 200-seed A/B cannot resolve even 4 pp (the DDPM arm's 7 pp came with 88 discordant pairs and
  p = 0.17), and 4 pp costs 243 episodes when the two arms disagree on 5 % of them and 2,157 when
  they disagree on 44 %. So the arm is sized by a pilot rather than fixed at 600 (`docs/10` §5.1,
  `just arm-power`), both arms run at the same count -- another reason the deterministic probe is
  the decision rule and the arm never ran.
  Coverage being flat (0.884-0.895)
  across arms from 45.5 % to 69.0 % lowers H's prior: an arithmetic error usually moves coverage
  too. Step 0 is therefore a refutation test, and passing it is the expected outcome.

## Decision (staff-engineer read of the above)

Take **C + F + B + A** -- all four are done in this batch -- and take **H next**, with the pairing
work B enables done first. Leave **D, E and G** declined; nothing in A's result changes their
tradeoffs.

Why this combination and not a single item: F is the cheapest thing in the batch and it immediately
changed how A is allowed to be reported (four post-hoc looks, bar 0.0125, which is exactly what
stops "the sampler fixes the band" from being written). C was not on the menu as a science item and
turned out to be the difference between a 3 h block that survives a closed laptop and one that does
not -- it paid for itself during this batch. B converts an aggregate control into a paired one, so
H's eventual verdict can be a McNemar instead of a Fisher test on 134 vs 138. A cost 3 GPU-h and
bought the one thing no doc could supply: the gap has a size per component (7 + 8 pp), a location
(the 1100-1149 band) and an honest null in the other two bands.

The ordering is the point. A alone would have produced "59 %, still not 69 %" and no next step;
F + A produced "7 pp sampler, ~8 pp residual, on this band, at this bar". $0 of that came from the
high-risk options: D, E and G would each have cost days and none of them would have touched the
gap, because all three change what is being measured rather than where the difference comes from.

**The owed measurement is now in, and it moved the conclusion rather than settling it.**
`just lerobot-control-rows` at 200 episodes produced 126/200 (63.0 %) for the control against our
DDPM@100 118/200, exact McNemar p = 0.46: at the checkpoint's own sampler this repo and the
reference are **not separated**. The control separates only from DDIM@10 (104 vs 126, p = 0.03),
which is a statement about truncated sampling, not about the adapter. Five repeats of that loop put
its own spread at 1.0 pp (62.9 % ± 1.0 pp), which makes the residual +3.9 pp with a
[-5.5, +13.0] pp interval that spans zero; the 134/200 scored earlier is a single run of the
aggregate entry point, not a draw from that band (`docs/04` §4). Combined with the probe's 2.4e-07
and the env check's 0.0,
the honest summary of the diffusion stage is: one measured sampler cost, no measured adapter
defect, at 200-episode resolution.
