---
name: ambiguity-check
description: Run before every Step Plan and before any implementation. Finds unresolved decisions or missing facts and asks the user instead of guessing.
---
# Ambiguity check

Blocking ambiguity = anything where two reasonable readings lead to different code, files, numbers, or cost. Check:
1. Open decisions (D1-D10 in `docs/06_decisions_and_risks.md`) touched by this step and still marked OPEN.
2. Any `[U]` item the step depends on (needs verification first: propose a verify-only step).
3. Names, paths, hyperparameters, sizes, seeds, dataset choices not fixed in docs or `docs/STATE.md` decision log.
4. Environment facts not yet recorded (OS, VRAM, driver, Docker/WSL).
5. Scope: could the request be read as larger than one step?
6. Tradeoffs that are matters of taste (speed vs fidelity, simplicity vs flexibility).

If any found: reply with a numbered question list only (max 5), each as `N. <question> Options: a) ... b) ... Recommended: a) <why, 1 line>`. Stop. Do not implement, do not assume. Record answers in `docs/STATE.md` decision log once given.
If none found: say "No open ambiguity." in one line, continue to the plan.
