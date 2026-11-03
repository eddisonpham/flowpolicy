# REVIEW -- the high-level pass over these plans

**Who.** A second reading of `agents/01`-`05`, done against this repository's own rules
(`AGENTS.md` 1-6, `docs/06`'s kill-criteria discipline, `docs/11` §6's measured-vs-estimated tag)
rather than against a general checklist.

**What it changed.** Six revisions, all applied to the plans as they now stand. The ones that
mattered are recorded with the reasoning, because a review that only says "looks good" is not a
review.

## 1. Verdicts

| Plan | Verdict | Finding |
|---|---|---|
| `01` Linux pilot | **revise** | Its acceptance check was vacuously true: "regenerate tables, byte-identical" cannot fail, because no generator reads pilot rows [V, `docs/12` §9]. Now the check names the doc edit and says explicitly that byte-identity proves *no table moved*. Its kill criterion also closed the arm on a single failed import -- too eager; the runbook's ladder comes first. |
| `02` Generic adapter | **revise, then accept** | Step 2 would have built a third normalization implementation. That is the exact defect class fixed one day earlier -- three readers of the processor JSONs, each with its own wrong assumption [V, commit `fc521f6`]. The step now says *reuse* `load_processor_norm` and extend only where a checkpoint ships no processor files. |
| `02` (second finding) | **revise** | It demanded the published cell be "inside its own reported interval". For `HuggingFaceVLA/smolvla_libero` there is no interval: the card is the generic stub with no eval table [V, `docs/11` §4.4], so the reference is lerobot's ~81.5 % figure. The plan now requires the band to be stated *before* the run, with its reason. |
| `03` RTC | **revise** | It sized the block from a 50-episode discordance point estimate. At ~20 % discordance, 50 episodes give roughly a +-14 pp interval -- so the point estimate is not a measurement, which is the mistake this repo already corrected once when it sized a block from a single guess. The pilot is now >= 100 episodes and the block is sized from the interval, at its upper bound. |
| `04` Serving | **accept after cuts** | It had grown a wish list. Auth/TLS, batching and a metrics endpoint were removed: they imply a deployment story the repo does not claim. What remains is four CPU-testable edges. |
| `05` CI + repro | **accept, order changed** | Correct content, wrong priority in the first draft: it sat after the serving plan. It is the cheapest plan in the set and the only one that protects work already done, so it goes first. |

## 2. Ordering: the version I rejected

The draft order was `C1 -> S1 -> L1 -> A1 -> R1`. That puts polish ahead of the only step that
converts an estimate into a measurement, and `L1` is blocked on hardware that may be free today and
gone tomorrow. `S1` is small enough to fit inside `L1`'s gaps, so it does not need to be earned
first. The accepted order is `C1 -> L1 (if a host is free) -> S1 -> A1 -> R1`, and the condition in
parentheses is the honest part: if no host appears, `L1` is not skipped, it is blocked, and the
stage should end at `S1` + `C1` rather than drift into `A1` without evidence.

## 3. Cross-cutting findings

1. **Two of the three defects found in this repo came from an invariant having more than one
   owner** -- the sidecar gate, and before it the normalization mapping. `02` was the plan most
   likely to repeat that pattern, which is why its step 2 is now a reuse requirement rather than a
   implementation step.
2. **Acceptance checks must be able to fail.** Three drafts had a check that could not fail: the
   pilot's table regeneration, "one published cell" without a stated band, and `05`'s job going
   green while `continue-on-error` hid an install failure. All three now name the observation that
   would falsify them.
3. **Estimates must stay tagged.** `L1`'s 0.5 days and 15 minutes are computed from §5.1's numbers;
   `A1`, `S1` and `C1` are person-day estimates with nothing behind them. The plans keep the tags
   the repo already uses, so a reader cannot mistake an estimate for a measurement -- which is the
   one rule this project has broken in public once.
4. **Scope creep enters through "hardening".** `04` is where it tried to. The rule applied: a step
   is allowed only if something already promised becomes true; anything that would make a *new*
   promise (multi-tenant serving, a third family, a real robot) is out.

## 4. What would make this plan set wrong

- If a Linux host is not available this stage at all, then `L1`, `A1`'s evidence half and all of
  `R1` are inert, and the correct output of the stage is `C1` + `S1` plus a README that keeps to
  two families. That is not a failure of the plans; it is `docs/06` D12's second branch.
- If `lerobot` cannot be installed in CI at all, `C1` step 1 fails and the adapter path stays
  locally verified only -- in which case the review would re-rank `C1` below `S1` and say why.
- If `L1` measures a per-chunk cost far below the band's centre, `R1`'s prize shrinks (a leftover in
  flight matters less when a chunk already fits the period), and `R1` should be re-priced before it
  is run, not after.

## 5. What I refused to add

A real-robot plan, a training plan, a third adapter without a published cell, and a "make CI run
the GPU tests" item. The first three would each make a promise the repository cannot currently
keep; the fourth is not possible on a CPU runner and a mocked GPU test would be worse than the
honest local-only note the README already carries.
