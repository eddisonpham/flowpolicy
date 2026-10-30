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
| A. DDPM@100 on the 200 published seeds | none in code, ~3 GPU-h | **running** | (rows land in `results/e0_published_seeds_ddpm.jsonl`) |
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

## A. DDPM@100 on the 200 published seeds — running

The only open scientific item is a 15-point success gap that is known to sit in *this repo's path*
(the adapter-free control reproduces the checkpoint on the same seeds) but not where in that path.

**Cost, measured rather than estimated.** The DDPM@100 arm costs 4.0 s per chunk and ~154 s per
episode with two workers; E14's clean single-process number is 1766 ms per chunk, and two workers
measured 2.3x per chunk for 2x the work, i.e. slower in wall clock. The recipe therefore defaults
to one worker (`d1e27f0`), and the 200-seed block is a ~3 h job.

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

**Pre-registered reading of A, written before its rows exist:**

1. If DDPM@100 lands near 69 % on the whole block **and** is separated from DDIM@10 on 1100-1149,
   the sampler is the cause: truncation costs the points, and E0's verdict becomes "the checkpoint
   reproduces only at its own 100-step sampler".
2. If DDPM@100 is not separated from DDIM@10 on 1100-1149, the sampler is exonerated and H
   (adapter) is the next thing to run, against that block rather than the whole thing.
3. Either way the row is reported per block as well as in aggregate.

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

## H. Adopt lerobot's processors — gated on A

Medium cost, and it trades away the boundary `docs/07` rule 4 draws (measurement machinery stays
ours), so it needs a dated amendment like D11's. It is now the **leading** hypothesis rather than a
contingency, because C-A's proxy evidence above shows no sampler effect on the block that had a gap
worth measuring. Do not start it before A returns.

## Decision

Proceed with **A (running) + B (done) + C (done) + F (done)**, and hold H behind A's
pre-registered reading above. Do not start D, E or G.

The combination matters more than any single item: F makes the per-block look legitimate to
report, C makes the block survive the 3 hours it takes, B makes the control comparable on the same
seeds once A lands, and A is the only item that produces a new scientific row. Four cheap items
turn one expensive measurement from a number into an attribution.
