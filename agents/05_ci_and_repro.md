# 05. `C1` -- make the pretrained path visible to CI, and stamp what produced a row

## Goal

Two things that protect every number already in `results/` without a GPU: a CI job where
the `lerobot`-dependent tests actually run instead of skipping, and a determinism + provenance guard
for eval rows.

## Status (2026-11-02)

**Steps 1-4 implemented.** Evidence is in `docs/CHANGELOG.md` under that date; the short version:

| Step | State |
|---|---|
| 1 CI job | written in `.github/workflows/ci.yml`; **not yet observed on GitHub** -- the first run is what closes it, and it is set to fail rather than to hide an install failure |
| 2 determinism | `tests/unit/test_rollout_determinism.py`, 5 tests; the `torch.manual_seed` line was removed locally to confirm the first test fails without it, then restored |
| 3 provenance | `src/policybench/provenance.py`, shared with the trainer; `tests/unit/test_provenance.py`, 5 tests |
| 4 resolution guard | `just verify` regenerates `results/power_audit.json` and diffs it |

## Non-goals

- Not `lerobot` in the lockfile. It caps `torch<2.12` while the project pins 2.14.1 (R13); the CI
  job builds its own environment, exactly as `just setup-pretrained` does locally.
- Not the 1 GB checkpoint in CI. Weights are local-only; the contract tests use the real
  `config.json` and processor JSONs, which are kilobytes.
- Not a re-run of any block. Rows must not move: this plan adds guards around them.

## Steps

| # | Work | Acceptance |
|---|---|---|
| 1 | CI job `pretrained-contracts`: `uv sync --frozen`, then `bash scripts/setup_pretrained.sh` (`--no-deps`, the local recipe), then the unit gate | the checkpoint-contract tests report **passed, not skipped**. `transformers` is deliberately *not* installed: one test asserts the config loads without it, and a VLA weight load needs a GPU the runner does not have |
| 2 | Determinism test: `rollout_episode` twice with the same seed and a deterministic stub policy, compare the executed action trace and every non-timing field | identical traces; the test fails if the `torch.manual_seed(noise_seed)` line is removed -- that is the regression it guards |
| 3 | Provenance: `_stamp` gains `git` and `lock_sha256` using the same helpers `trainer.save` already has | new rows carry both; **old rows are untouched** and their absence reads as "unknown", never as a mismatch |
| 4 | Enforce the resolution guard: `just verify` runs `scripts/power_audit.py --out <tmp>` and diffs against the committed `results/power_audit.json` | an uninformative null cannot change silently; the diff is part of `verify`'s existing byte-identity check, not a new rule |

## Cost

1 day **estimated**. Steps 1, 2 and 4 are CPU and free; step 1 pays a ~2 GB `lerobot` install per CI
run unless the uv cache is warmed (`enable-cache: true` is already how the other jobs work).

## False if / kill

- `lerobot==0.6.1` cannot install on the CI runner (its `torch<2.12` wheel resolution vs the
  runner's Python) -> keep the job `continue-on-error: true`, record it in `docs/02` §3.4 next to
  R13, and do not pretend the adapter path is covered.
- The determinism test is flaky on CPU (thread count, oneDNN) -> pin `torch.set_num_threads(1)` in
  the test and say so; a flaky determinism test is worse than none.
- Step 3's stamp changes any existing row's shape in a way the table generators read -> the stamp
  goes under a nested `"provenance"` key so no reader can mistake it for a measurement.

## Why step 1 first

Every other plan's evidence depends on the adapter path behaving. Today a change there is invisible
to CI: the tests skip, and the first signal is a local run days later. This is the cheapest
available increase in confidence per hour, and it is the plan that can start this minute.
