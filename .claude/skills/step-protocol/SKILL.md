---
name: step-protocol
description: Use before ANY implementation work, and whenever the user asks what to do next or to plan a step. Produces the exact Step Plan the user must approve with GO before anything is built.
---
# Step protocol

Steps are ids from `docs/07_roadmap.md`, split into sub-steps (e.g. M1.1, M1.2) small enough to finish in one sitting. If the milestone is bigger, propose the split first.

## Phase 1: PROPOSE (read-only)
1. Read `docs/STATE.md`, the milestone row in `docs/07_roadmap.md`, and the docs it links.
2. Run skill `ambiguity-check`. If any blocking question exists, output only the questions. Stop.
3. Otherwise output exactly this Step Plan, then stop:

```
STEP <id>: <title>
Goal: <1 line>
What I will do (exact):
  - create/edit: <paths>
  - run: <commands>
  - install: <packages with versions> (or none)
What YOU must do (before/after): <exact commands/clicks, or "nothing">
Decisions I assumed (need your OK): <numbered, each with recommended default>
Cost: <minutes, GPU use, disk, network>
Risks / what could go wrong: <max 3>
Done when (checks I will run and show): <commands + expected pass condition>
Out of scope (I will NOT touch): <list>
To approve: reply `GO <id>` and run `! echo <id> > .claude/GREENLIGHT`
```
Plan length: max 25 lines. No prose around it.

## Phase 2: EXECUTE (only after `GO <id>` and the hook allows writes)
- Do exactly the plan. If you need anything not in it, stop, return to PROPOSE with a revised plan.
- Run the "Done when" checks; capture decisive output.
- Update `docs/STATE.md`.

## Phase 3: HAND BACK
- Reply with skill `terse-report`.
- Remind: `! rm .claude/GREENLIGHT` to revoke the green flag.
- Propose the next step id in one line. Do not start it.
