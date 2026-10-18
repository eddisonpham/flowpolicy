# FlowPolicy: operating rules for the coding agent

Spec lives in `docs/` (start at `docs/07_roadmap.md`). Current state lives in `docs/STATE.md`. These rules override any urge to be helpful by doing more.

## 0. Prime directives (non-negotiable)
1. **NO GREEN FLAG, NO IMPLEMENTATION.** Never create/edit/delete project files, install packages, or run mutating commands unless the user's latest message contains `GO <step-id>` for the step you are executing. A hook also enforces this; never try to bypass it, never touch `.claude/`.
2. **NO IMPLEMENTATION WITH OPEN AMBIGUITY.** If any requirement, choice, path, name, number, or decision is unresolved, ask (see skill `ambiguity-check`) and stop. Do not pick a default silently, even a "recommended" one: propose it and wait.
3. **ONE STEP AT A TIME.** A green flag covers exactly the one step id it names, nothing adjacent. When done, stop and report. Do not start the next step.
4. **EXPLAIN BEFORE ACTING.** Before any green flag you must have shown a Step Plan (skill `step-protocol`) and the user must have replied `GO <step-id>` after seeing it. If the plan changes mid-step, stop and re-propose.
5. **NEVER INVENT RESULTS.** No made-up metrics, versions, URLs, licenses. Unknown means `TBD` or `[U]`; see skill `honest-results`.

## 1. Modes
- **PROPOSE (default):** read-only. You may read files, run read-only commands, and update `docs/STATE.md`. Output a Step Plan or questions. Then stop.
- **EXECUTE:** only after `GO <step-id>`. Do exactly the plan the user saw. Anything beyond it needs a new proposal.
- Return to PROPOSE the moment the step is done, blocked, or deviates.

## 2. Communication style (no yapping)
- Default reply: 120 words or fewer. No preamble, no recap of the task, no praise, no restating what the user said.
- When asked for a summary/status: use skill `terse-report` format only. Max 8 lines. No narrative.
- Plain facts, numbers with units, file paths. No emojis, no filler, no hedging paragraphs.
- Questions to the user: numbered, each with options and one recommended default, answerable in one word. Max 5 per message.
- Long output (logs, diffs) goes in a file or is withheld; show only the decisive lines.

## 3. Check-in cadence
- Ask for direction at every fork where taste matters: architecture choices, hyperparameters, naming, scope, tradeoffs, anything touching D1-D10 in `docs/06_decisions_and_risks.md`.
- Never run long jobs (training, sweeps, downloads over 1 GB, Docker builds) without a plan that states duration, disk, and GPU use, and a GO for that step.
- If a step is taking more than expected or a test fails twice, stop and report; do not keep trying variations.

## 4. Project invariants (check before every EXECUTE step)
- Python 3.12, `uv`, torch 2.14.1+cu130 pinned; no dependency added without a step that names it.
- Spec first: if code and `docs/` disagree, stop and ask; do not silently diverge. Log approved deviations in `docs/CHANGELOG.md`.
- Every policy implements the `Policy` interface in `docs/01_architecture.md`; eval/deploy never branch on policy type.
- No LeRobot/HuggingFace policy code as model code (reading and dataset converters only, labelled).
- Normalizer stats are train-only and stored in the checkpoint. Encoders use GroupNorm, no BatchNorm.
- Eval seeds disjoint from train seeds; results are raw JSONL in `results/`; tables/figures only from scripts. Never best-checkpoint silently.
- Pre-registered protocol (`docs/04_eval_and_experiments.md`) is frozen once the first experiment runs; changes go in its Deviations log.
- Tests ship with code. A step is done only when its acceptance checks in the plan pass and you show the evidence (command and decisive output).
- CPU-only CI must stay green. GPU/sim/ROS tests are local and marked.
- Never claim "bit-exact" unless the bitwise test passes.
- Do not make commits, push, tag, or delete anything unless the step plan says so and has a GO.

## 5. State tracking
After each step (and only then) update `docs/STATE.md`: current step, status, evidence pointer, open questions. This file is the only thing you may write without a green flag.

## 6. If instructions conflict
Prime directives 1 to 5 win. If unsure whether something needs a green flag, it does.
