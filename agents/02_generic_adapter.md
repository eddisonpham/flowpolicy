# 02. `A1` -- one generic lerobot adapter, behind an `EnvLike` seam

## Goal

Replace the two hand-written adapters with one class that loads any `lerobot` checkpoint
whose `config.json` `type` the package supports, so "the harness does not know which policy it
holds" is a property of the code rather than of two files that happen to agree.

## Non-goals

- Not a generality claim. Acceptance is **one** published cell reproduced within its own interval;
  the README keeps saying two families until then [V, docs/11 §7].
- No new dependency, no `lerobot` in the lockfile (it caps `torch<2.12`, R13).
- No change to `eval/`, `deploy/`, the row schema or the generators. If a change there is needed,
  this plan is wrong.

## Design

Two seams, both additive, neither on the measured path until a caller opts in:

1. **`EnvLike` (structural protocol).** `reset(seed) -> obs`, `step(action) -> (obs, reward, done,
   info)`, `close()`, `fps`, `task_description`. `PushTEnv` already satisfies it [V]; the protocol
   is declared in `models/` next to `PolicyLike` and used as a *type* only. `rollout` and the
   control loop keep calling the concrete class this stage -- changing the call site is a separate
   step with its own golden test, because those two files produced every row in `results/`.
2. **`PretrainedLeRobotPolicy`.** `PreTrainedConfig.from_pretrained(dir_or_id)` ->
   `get_policy_class(config.type)` -> `make_pre_post_processors(...)`. Observations map by
   `config.input_features`: `observation.images.<name>` from the harness's named views,
   `observation.state` padded to `max_state_dim`, task string through the preprocessor's tokenizer.
   Actions come back through the postprocessor and are unpadded to `action_feature.shape[0]`.
   `n_action_steps` is **not** used to slice: `predict_action_chunk` returns the full chunk [V,
   lerobot 0.6.1 source, `_get_action_chunk`].

The existing gates carry over and are the reason this is safe to attempt: `check_processor_files`
(sidecar preflight), `named_sidecars`, the leftover round trip (`PaddedActionSpace`), and the
statistics probe. A generic adapter that bypasses them would be a regression, not a refactor.

## Steps

| # | Work | Acceptance |
|---|---|---|
| 1 | Declare `EnvLike`; assert `PushTEnv` satisfies it by a test-only `isinstance`-style protocol check | new unit test; `just verify` green; no file outside `models/` and `tests/` touched |
| 2 | **Reuse, do not re-derive.** `load_processor_norm` / `ProcessorsNorm` already own the pipeline path and `check_processor_files` gates its sidecars; extend only for a checkpoint that ships no processor files, where the VQ-BeT buffer path is the precedent | a unit test against the real SmolVLA config fixture (already in `test_smolvla_pilot.py`), no GPU. A third normalization implementation fails this step on sight: three readers with their own assumption is the defect class fixed on 2026-11-02 [V, commit `fc521f6`] |
| 3 | Adapter class + dispatch by `config.json` `type`, keeping the diffusion and VQ-BeT paths as the two existing classes | `load_policy` still returns the old classes for those ids; existing rows unchanged |
| 4 | CPU contract test: batch keys, dtype, padding width, instruction, single-chunk refusal | new test file; runs in CI where lerobot is absent -> `importorskip` with a loud reason |
| 5 | GPU acceptance, only after `L1`: one published cell | success rate inside a band **stated before the run**. The card has no eval table [V, `docs/11` §4.4], so the reference is lerobot's ~81.5 % spatial figure, which carries no interval: the band (e.g. +-10 pp) and its justification go into `docs/11` before any episode runs |

## Cost

1-2 days **estimated** for steps 1-4; 0 GPU-hours **measured** for them (they are CPU tests).
Step 5 is `L1`'s cost again plus one episode block; not priced here.

## False if / kill

- A published cell is missed outside its interval and the cause is the wrapper -> the wrapper is not
  policy-agnostic; record it, keep the two families, and do not add a third adapter on top.
- Step 1 cannot be done without editing `rollout.py` -> the seam is wrong for this stage; revert to
  declaring the protocol and leave the call sites alone.
- The adapter needs to reach inside `eval/` or `deploy/` to work -> it is a fork of the harness, not
  an adapter. Stop.

## Risks

| Risk | Mitigation |
|---|---|
| Silent view binding (two views swapped) | named cameras + `input_features` order + a test with distinct pixel values per view (the server's test already does this for the wire format) |
| Padding width confusion (7 vs 32) | the `max_action_dim` bug that precondition 3 already hit once [V, docs/11 §5.2]; the declared width comes from the JSON, always |
| Unnormalizing after the chunk instead of inside the loop | `PaddedActionSpace` is the single owner of that map; the adapter imports it rather than re-deriving |
