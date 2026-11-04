# 06. The next stage: what to scaffold, what it costs, and what order

Status: **2026-11-03.** Written after the first full 12-stage pipeline run and the sampler
sensitivity work. These are plans, not permissions: `AGENTS.md` still requires a `GO <step-id>`
before an edit lands, and a plan that is not in this set has not been costed.

## What is already true, so the next stage does not re-buy it

- **The stack runs end to end.** `just pipeline --profile full` completed 12/12 stages in 73
  minutes and re-derived the committed tables byte for byte.
- **The numbers reproduce from HEAD.** The reproducibility block agrees on 50/50, 200/200, 300/300
  and 200/200 episodes for E0, E0-published, E3 and E15, and 45/45 for E10 once its stage was
  pinned to the sampler the committed block was measured with.
- **Sampler sensitivity is measured, not asserted.** Two deploy blocks that differ only in which
  DDIM loop ran them disagree on 23 of 45 paired episodes (+33.3 pp towards the stock loop, exact
  McNemar p = 0.003, psi = 0.51); every one of those episodes also ended at a different env step.
  The 45-episode block resolves 29.1 pp at 80 % power, so the gap is larger than the bar, and
  *measuring* it rather than detecting it would take ~34 episodes per arm
  (`results/tables/sampler_sensitivity.md`).
- **The hardware-gated gaps are unchanged:** `L1`, `A1`'s evidence half and `R1` still need a Linux
  host with a CUDA card (`agents/README.md`).

## The options

| # | Option | Buys | Cost | Risk | Depends on |
|---|---|---|---|---|---|
| O1 | **Recipe stamping + a block manifest** | Every row says which recipe produced it; comparisons refuse mismatched recipes by rule instead of by review | 0.5-1 day, CPU only (estimated) | Low -- absent fields stay "unknown", which is the convention already used for `train_seed = -1` | none |
| O2 | **One generic lerobot adapter** (`A1`) | A third policy family becomes configuration; normalization has one implementation; RTC becomes reachable | 1-2 days CPU + ~10 min GPU for the golden blocks (estimated) | Medium -- it rewrites the path that produced every published row | none (the `EnvLike` seam landed 2026-11-02) |
| O3 | **Serving leftovers** (`S1` step 4) | Per-request deadline, drain on shutdown, clamped `steps`; the deployment claim stops being aspirational | 0.5 day, CPU only (estimated) | Low -- the surrounding guard tests already exist | none |
| O4 | **The Linux pilot** (`L1`) | The only path to a VLA number; tests the runbook against reality | 0.5 setup days + ~15 min GPU (computed) | High -- it can fail on the host's `robosuite`/`libero` versions, which is itself a result | a Linux host with CUDA |
| O5 | **RTC against the cross-fade** (`R1`) | Answers whether upstream RTC beats the hand-built blend at the same latency | 1 day + 6-9 GPU-h (computed) | Medium | O2 and O4 |
| O6 | **A second environment** (MetaWorld) | A task seam nobody has asked for yet | days (estimated) | High -- no published cell to check the adapter against | O2 |

### Why each is worth or not worth its cost

- **O1 is the cheapest protection against the failure this stage just had.** The 23/45 "reproducibility
  failure" was a missing `--sampler` flag: a row that records its own recipe cannot produce that
  detour, and `just manifest` would make a block's recipe readable without re-reading the script
  that wrote it. It buys no new capability, which is exactly why it should go first -- it protects
  work already done rather than work not yet started.
- **O2 is the largest capability gain on the CPU side.** Two adapters re-deriving normalization is
  the reason `docs/10` had to design a processor adoption at all. The tradeoff is blast radius: it
  must land behind a golden test that reproduces the E0 and E15 blocks exactly, or it is a
  rewrite that quietly moves every number.
- **O3 is small and closes a documented hole.** It is here because it is cheap and CPU-only, not
  because it is urgent.
- **O4 and O5 are the only options that need hardware,** so they are ordered by dependency, not by
  preference. Nothing on the CPU side makes them start sooner.
- **O6 is declined for now.** `docs/11` §7 already records why a third *task* without a published
  cell adds code rather than evidence; the useful half of it is the environment seam, which O2
  builds.

## Recommended order

1. **O1** -- 0.5-1 day, CPU. Protects every number already published; makes every later comparison
   self-checking.
2. **O3** -- 0.5 day, CPU. Folds naturally into the same batch as O1.
3. **O2** -- 1-2 days, CPU-testable, GPU only for the golden blocks.
4. **O4**, then **O5**, whenever a Linux host with a CUDA card exists.

**Orders considered and rejected.** (a) *O2 before O1*: the adapter would be rewritten before its
rows can prove they did not move, i.e. the change with the largest blast radius would run before the
check that detects blast damage. (b) *O4 first*: it has no CPU prerequisite, so putting it first
changes only what is queued, not what completes -- and it is blocked today regardless. (c) *O3
alone*: it closes a hole without protecting a number, so it rides next to O1 rather than occupying
a stage of its own.

## Acceptance for the stage

- `just verify` green, and two consecutive full pipeline runs agree on every committed block with
  zero discordant episodes (`summary.json`'s reproducibility block, which now carries each block's
  resolution).
- Every committed file in `results/` has a manifest entry stating its recipe, seed block, n and
  provenance; a block whose recipe is missing is written as unknown, never inferred.
- O2 lands only with a golden test: E0 and E15 blocks reproduce exactly through the new adapter.
- O3 lands with a deadline test and a drain test in CI, and `steps` bounded in the server.
- O4, if a host appears, produces one recorded pilot row and a `docs/12` amended with what actually
  broke.

## Kill conditions

- **O1**: if a manifest cannot be produced from committed rows without hand annotation, the gap is
  the rows themselves -- stop and fix the writers first.
- **O2**: if the golden test cannot reproduce E0 within one episode, the adapter is not a
  refactor; abandon it rather than re-baseline the published blocks.
- **O3**: if the deadline has to be enforced by sleeping, the loop's virtual-time design is the
  problem, not the server -- record that and stop.
- **O4**: if `scripts/smolvla_pilot.py --stats-only` fails on the host, the pinned lerobot and the
  checkpoint's processor layout disagree; that is a result to write down, not a script to patch.

## Total

O1 + O3 + O2 is roughly **2-3.5 CPU days** and no GPU (estimated, from the table above); O4 is 0.5
setup days and ~15 GPU-minutes; O5 is 1 day and 6-9 GPU-hours. Nothing here re-prices V1-V6:
`docs/11` §6 already did that, and this list is about gaps in what the repository promises, not
about new research directions.
