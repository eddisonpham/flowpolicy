# Next-stage plans: what this program is missing, and in what order

Status: **2026-11-02. Drafted from the gap inventory in `00_backlog.md`, then revised by a review
pass recorded in `REVIEW.md`.** These are plans, not permissions: `AGENTS.md` still requires a
`GO <step-id>` and `.claude/GREENLIGHT` before any edit lands.

| # | Plan | Step id | Depends on | Cost | Status |
|---|---|---|---|---|---|
| 00 | Gap inventory | -- | -- | -- | reference |
| 01 | Run the Linux pilot to a recorded row | `L1` | a Linux host with a CUDA card | 0.5 setup days + ~15 min GPU (computed) | blocked on hardware |
| 02 | One generic lerobot adapter + an `EnvLike` seam | `A1` | `L1` only for its evidence half | 1-2 days (estimated) | startable now, evidence blocked |
| 03 | Measure RTC against the cross-fade | `R1` | `L1`, `A1` | 1 day + 6-9 GPU-hours (computed at the band) | blocked on `L1` |
| 04 | Close the serving wrapper's real gaps | `S1` | none | 0.5-1 day (estimated) | **done 2026-11-02**, step 4 open (owner's call) |
| 05 | Make the pretrained path visible to CI, and stamp provenance | `C1` | none | 1 day (estimated) | **done 2026-11-02** except the CI job's first green run |
| 06 | The next stage: recipe stamping, serving leftovers, the generic adapter | `O1`-`O6` | none for O1-O3; O4/O5 need a Linux + CUDA host | 2-3.5 CPU days + 0.5 setup days (estimated) | **recommended order written 2026-11-03** (`06_next_stage_plan.md`) |

**Implemented 2026-11-02: `C1` and `S1`** (plus `A1` step 1, the `EnvLike` seam, which the
determinism test needed to run without a simulator). What remains is `L1`, `A1`'s adapter and
`R1` -- all three gated on a Linux host with a CUDA card, i.e. on the same missing machine.

**Superseded for the next stage by `06_next_stage_plan.md` (2026-11-03)**, which re-orders the
remaining CPU work ahead of the hardware-gated plans now that the end-to-end runner exists: `O1`
(recipe stamping + manifest) -> `O3` (serving leftovers) -> `O2` (generic adapter) -> `L1` -> `R1`.
The reasoning it replaces is quoted here because the old order was right for its own date -- `C1`
really was the only plan protecting finished work, and `L1` really was the only one turning an
estimate into a measurement.

**Original order (2026-11-02): `C1` -> `L1` (if a host is free) -> `S1` -> `A1` -> `R1`.** `C1` can start
today and is the only plan that protects work already done; `L1` goes next whenever a machine is
available, because it is the only plan that turns an estimate into a measurement and should not
queue behind polish; `S1` is small enough to run in `L1`'s gaps; `A1` and `R1` are worth nothing
until there is a machine to score them on. `REVIEW.md` section 2 records this order *and* the one I
rejected, with the reason.

## Reading a plan

Each plan file carries the same five headings -- `## Goal`, `## Non-goals`, `## Steps`, `## Cost`,
`## False if / kill` -- so a plan is machine-checkable as well as readable. A plan missing one of
them is the kind of plan this repo has already had to correct:

1. **Goal** -- the one sentence the step is graded against.
2. **Non-goals** -- what the step must *not* grow into.
3. **Steps** -- ordered, each with an acceptance check that is a command or a file.
4. **Cost** -- tagged `measured` or `estimated`; a number without a tag is a guess (docs/11 §6).
5. **False if / kill** -- the observation that ends the step early, so a failed step is a result.

## What these plans deliberately do not include

Real-robot work, training a VLA, TLS or multi-tenant auth on the local server, or a third policy
family without a published cell to check it against. Each is a place this repo could grow, not a
gap it has: `00_backlog.md` records why each is out.
