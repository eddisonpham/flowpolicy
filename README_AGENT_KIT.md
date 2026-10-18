# Agent kit: how to use

Copy into the repo root: `CLAUDE.md`, `.claude/`, `docs/STATE.md` (alongside the planning docs).

Loop for every step:
1. `/plan M1.1`  (agent proposes; read-only)
2. You answer its questions / edit the plan.
3. You approve: type `GO M1.1` in chat AND run `! echo M1.1 > .claude/GREENLIGHT` (the `!` runs it as you, so the agent cannot do it).
4. Agent executes only that step, then reports (`/status` format) and stops.
5. Revoke: `! rm .claude/GREENLIGHT`. Repeat.

Safety net: `.claude/hooks/gate.py` blocks file edits and mutating shell commands whenever `.claude/GREENLIGHT` is absent (only `docs/STATE.md` and read-only commands pass). Add `.claude/GREENLIGHT` to `.gitignore`.
Test it once: with no flag, ask the agent to create any file; it should be blocked. Hook field names and the `!` shortcut depend on your Claude Code version: confirm against current Claude Code docs.
