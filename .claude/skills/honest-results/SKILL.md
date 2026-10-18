---
name: honest-results
description: Use when stating any version, number, metric, claim, license, URL, or result, when running experiments or evals, and when writing README/docs/posts. Prevents invented facts and cherry-picking.
---
# Honest results

- Every fact is one of: measured (point to file/command), verified (URL + access date), or tagged `[U]` / `TBD`. Never recall versions or numbers from memory as fact.
- Metrics come only from `results/*.jsonl` via scripts. No hand-typed numbers.
- Report: mean over seeds, bootstrap 95% CI, n seeds, n episodes, checkpoint rule (LAST and mean of last K=5). Best checkpoint only if labelled "oracle, not a result".
- Eval seeds disjoint from train seeds; harness asserts it.
- Null and negative results get the same prominence. A failed reproduction is reported with a suspected cause.
- Each claim has a "false if" condition (see claims table in `docs/04_eval_and_experiments.md`).
- If a result looks too good, first suspect a bug (leakage, normalization, eval seeds, wrong weights); say so and propose a check before reporting it.
- Wording: "bit-exact" only if the bitwise test passes; otherwise "tolerance-based".
