# CHANGELOG

Paths in entries below refer to where files lived on the date of the entry. The 2026-10-27 entry lists every move (from-scratch measurements now under `results/ablation/`). Exception: the 2026-10-28 rename entry below rewrote `flowpolicy` to `policybench` in the older entries, so their paths still resolve; the events they describe happened under the earlier name, and the uploaded brief is still `flowpolicy_complete/`.

## 2026-11-03: A block is observable, an end-to-end run exists, and the first one found a real bug

`scripts/pipeline.py` (`just pipeline`) runs the measurement stack end to end -- probes, eval
blocks, latency ladder, deploy rows, generated artifacts -- into a scratch run directory, and ends
by pairing the fresh rows with the committed ones by seed. Two things came out of building it
that no CPU test had caught.

- **The pretrained adapter could not run on a GPU at all.** `BuffersNorm` was constructed from
  `self.image_mean` *before* `self.to(device)`, and `Module.to` **rebinds** buffers rather than
  moving them in place, so the norm object kept the CPU copies and every chunk raised
  `RuntimeError: Expected all tensors to be on the same device, but found at least two devices,
  cuda:0 and cpu`. Introduced with the option-H normalization change and invisible to the 229 CPU
  tests, because a CPU-only move rebinds nothing. The move now happens first, and
  `tests/unit/test_pretrained_normalization.py::test_the_normalization_tracks_the_device` pins it
  with a `gpu`-marked test (device identity by `data_ptr`, plus one real chunk).
- **A 200-episode block was unattended in the literal sense.** Nothing printed until the last
  episode, and a worker crash surfaced as `BrokenProcessPool` from `future.result()` -- no seed, no
  config, and under `spawn` not even the worker's traceback. Now:
  - `src/policybench/logging.py` writes one JSON event per line to a gitignored
    `results/logs/<run_id>_<stamp>.jsonl`: `run_start` with checkpoint, workers, device and git;
    one `episode_done` per episode **from the worker that ran it**; `chunk_done` with rate and ETA;
    `error` with the seed and the full traceback; `run_end` with the summary. `Progress` returns
    the ETA as `None` before the first unit instead of a fabricated zero.
  - **Workers write their own rows.** A chunk that raised on its 12th of 25 episodes used to lose
    all 25; it now loses at most the episode that failed, and `RolloutError` names it.
  - **`evaluate` no longer dies under `spawn` when called without a script.** From a REPL,
    notebook or `python -c`, the main module's file is `<stdin>`/`<string>`, which `spawn` cannot
    re-import. It asks `spawn.get_preparation_data` what it would do and falls back to one
    in-process worker, logging `spawn_unavailable` so the slower shape is never assumed away.
  - `scripts/logs.py` (`just logs`) renders a log: event counts, episodes, success rate, failures
    with their tracebacks. `scripts/pipeline.py` writes `summary.json` after every stage, so a
    killed run still says what it got through.
- **Tests: 245 CPU unit (from 202), plus one `gpu` test.** `test_run_log.py` (schema, multi-handle
  append, traceback capture, torn-line recovery, the ETA arithmetic), `test_rollout_worker_io.py`
  (rows written per episode, prior rows kept on failure, the seed in the error, both branches of
  the spawn guard) and `test_pipeline.py` (quick ⊂ full, every stage's script exists, no stage
  writes to the committed evidence except `tables`, and the pairing rules: shared seeds only,
  step counts never paired across, an absent block is "not compared" rather than zero).

- **Four workers appending to one `.partial` file corrupted it.** The E3 stage died on a torn line
  and reading the block back hit the same fragments: four writers on one file interleave mid-line.
  Each worker now appends to its own shard (one writer per file) and every reader skips a line that
  does not decode, so a torn line costs its own seed and nothing else (`test_eval_resume.py`).
- **The comparison step then crashed after all twelve stages had gone green.** E9's latency rows are
  benchmarks, not episodes -- no `eval_seed`, no `success` -- and `compare_stage` keyed on
  `eval_seed`, so a 49-minute run ended in `KeyError: 'eval_seed'` with `summary.json` never getting
  its reproducibility block. A row without those two fields is now "not an episode", and a block
  with no episode rows reports `not an episode block` instead of a zero.
- **...and once fixed it mis-paired E10, twice.** An absent `sampler_steps` -- the committed deploy
  rows predate the field -- read as `0`, so the 45 fresh rows matched nothing and the run reported
  "no shared seeds" over a real 45-episode overlap; and `latency_ms` was not in the key at all,
  which would have paired a 0 ms episode with a 200 ms one. Pairing is now the episode (seed,
  index, strategy) plus the conditions a block varies *within* one file (sampler steps, injected
  latency), with an unrecorded field read as unknown rather than zero and a committed row never
  paired twice. `just pipeline-compare <run-dir>` rebuilds that block from the rows on disk, which
  is how the crashed comparison was repaired without re-running an episode.
- **`just pipeline-status <run-dir>`.** The stage table from `summary.json` (status, wall-clock,
  detail), the tail of each failed stage's own log, and the comparison: the screen for a run that
  `just logs` is for one block, and safe to read while the run is going.
- **E10's 23-episode disagreement was a configuration change, not HEAD drifting.** With the
  pairing fixed, `deploy` paired 45/45 and disagreed on **23** (exact McNemar p = 0.003; fresh
  35.6 % vs committed 68.9 %). It is not RNG -- the current code reproduces itself episode for
  episode -- and it is not the environment: the paired episodes differ in *where they ended*
  (300 vs 233 env steps) at near-equal final coverage (0.9489 vs 0.9536). The cause is the
  sampler. The deploy stage never passed `--sampler`, so it ran the default fast DDIM loop, while
  every committed E10/E12 row predates that flag and was measured with `diffusers`' stock loop.
  The two loops agree to 1e-5 on the action (`tests/gpu/test_pretrained_sampler_equivalence.py`),
  and PushT's contact dynamics turn that rounding into different episodes -- so the gap is a
  property of the *sampler pair*, not of the repo failing to reproduce itself. Re-running the
  same 45 keys with `--sampler stock --dtype float32` reproduces the committed rows exactly
  (seed 10000 episode 0: 233 steps and the same outcome, row for row). The stage now pins those
  two flags, and `sampler`/`dtype` joined the pairing conditions, so a recorded mismatch cannot be
  paired as one block again; an unrecorded value stays a wildcard, which is what keeps the old
  deploy rows comparable at all. The mis-configured block is kept as
  `results/deploy_fast_sampler_45.jsonl` so the 23/45 disagreement can be recomputed instead of
  taken on trust.
- **`p_text` printed `p = 0.00` for p = 0.0026.** Two decimals is coarser than the band between
  1e-3 and 1e-2, which is where a paired 45-episode result lands. It now keeps three decimals
  there (`p = 0.003`) and two above it, so a decisive result cannot read as a rounding bug.
- **Tests.** `test_pipeline.py` gains the non-episode block, unknown-vs-zero step count, the latency
  condition, the no-double-pairing rule and `--compare-only`; `test_run_log.py` gains the
  run-directory screen (stage table, a failed stage's log tail, a missing summary).

## 2026-11-02: Rows name their commit, the seeding has a test, and the server stops trusting `steps`

`agents/05_ci_and_repro.md` (`C1`) and `agents/04_serving_hardening.md` (`S1`) as one batch: four
guards and three serving edges. **No new measurement and no row touched** -- every number in
`results/` is exactly what it was. `just verify` green, and it now diffs
`results/power_audit.json` as well as the tables and the README.

- **Every row carries `provenance: {git, lock_sha256}`.** Checkpoints have carried those two
  stamps since M3 from a private copy in `training/trainer.py`; `src/policybench/provenance.py`
  is now the one implementation both call, cached per process so a 200-episode block does not fork
  `git` 200 times. Nested under a single key so no generator can read a hash as a measurement, and
  the rows written before today have no key at all -- which reads as "unknown", the same
  convention `step = -1` and `train_seed = -1` already use.
- **The per-episode seeding is a test, not a comment.** `tests/unit/test_rollout_determinism.py`
  runs one episode twice against a stub policy that draws its chunk from the global RNG: identical
  traces when the seeding line is there, different ones when it is not. Confirmed by removing
  `torch.manual_seed(noise_seed)` locally -- the first test fails -- and restoring it. A second
  test asserts a *different* seed does draw different noise, so the first cannot pass on a policy
  that ignores the RNG, and a third pins `noise_seed_offset` to the noise stream only, which is
  the property the sound-stream block of 2026-10-30 depended on.
- **The environment seam that test needed.** `envs/base.py` declares `EnvLike` (`reset(seed)`,
  `step(action)`, `close()`, `fps`, `task_description`); `PushTEnv` gains the task string, and
  `rollout_episode` gains `env_factory=PushTEnv`. Existing call sites are untouched on purpose --
  swapping them is `agents/02`'s step, with its own golden test, because those files produced
  every row here. The payoff today is that the determinism test runs in CI without a simulator.
- **CI runs the adapter tests instead of skipping them.** The new `pretrained-contracts` job
  installs `lerobot==0.6.1` the way `just setup-pretrained` does (`--no-deps`, on top of the
  pinned torch) and re-runs the unit gate, where the checkpoint-contract tests report `passed`.
  `transformers` is deliberately absent: one test asserts the config loads *without* it. **Not yet
  observed on GitHub** -- the first green run is what closes that step, and the job is set to fail
  loudly rather than carry `continue-on-error`, which would hide an install failure.
- **`steps` is bounded, `deadline_ms` is a label, `shutdown()` drains.** A step count outside
  `[1, max_steps]` -- the sampler's own schedule, 100 here, advertised by `/health` -- is a 400
  naming the range instead of being trusted into the model, and a server constructed with a count
  outside its own bound refuses to start. `deadline_ms` echoes `deadline_missed` from the measured
  `latency_ms` and still returns the chunk: refusing a late one would make lateness unobservable
  at the only point it matters. `shutdown()` stops accepting, waits for the requests already
  inside the policy, then closes, which `ThreadingHTTPServer`'s daemon threads would otherwise
  truncate on Ctrl-C.
- **Still open.** `S1` step 4 -- a `LICENSE` and a version above `0.0.1` -- is the owner's call,
  so the repository still has no licence. `L1`, `A1`'s adapter and `R1` remain blocked on a Linux
  host; nothing in this batch moves them.

## 2026-11-02: Precondition 2's config half is checked without transformers, and it found the cadence

The pilot's precondition 2 (`SmolVLAPolicy.from_pretrained`) has been `[NOT RUN]` everywhere this
repo has run, because the weights need the VLM backbone and `transformers`. The config half needs
neither, so `tests/unit/test_smolvla_pilot.py` now loads the real `config.json` through lerobot's
`PreTrainedConfig.from_pretrained` and asserts what it declares: 50/1/1 for
`chunk_size`/`n_action_steps`/`n_obs_steps`, action 7, state 8, two 256x256 views, `max_action_dim`
32, `num_steps` 10, ACTION/STATE `MEAN_STD`, VISUAL `IDENTITY`.

- **`n_action_steps` is 1, not 10.** The pilot's ten-action cadence is available because
  `_get_action_chunk` does not slice by it -- `predict_action_chunk` returns all 50 -- while
  `select_action`'s queue, which the pilot does not use, is what the field sizes. Read from lerobot
  0.6.1, not measured; the pilot measures it.
- **`smolvla` is unregistered until its config module is imported**, so a config-only load raises
  `ValueError: Policy type 'smolvla' ... is not registered`. The pilot is unaffected (its backend
  import pulls the config in), but any config-only check has to import it explicitly.
- The test skips where lerobot is absent -- CI, which installs neither it nor `transformers` -- and
  where `transformers` is present, since then "without transformers" is unprovable. Precondition 2
  stays half-verified: geometry settled, weights not.

## 2026-11-02: The sidecar gate reaches the migration and the control, not just the eval side

Auditing the remaining readers of a processor JSON for the same "the JSON is the artifact"
assumption found it three more times, all on the path that produces and runs the adapter-free
control -- the one experiment whose point is to be the reference with none of this repo's code in
it.

- **`scripts/lerobot_migrate.py`** checked four names and then "verified" the directory by loading
  it through `DiffusionPolicy.from_pretrained`, which succeeds with no statistics. A migration
  interrupted between writing the JSONs and their sidecars would have printed `verified`.
- **`scripts/lerobot_control.sh`** reused `checkpoints/diffusion_pusht_migrated` whenever
  `policy_preprocessor.json` existed and `config.json` was non-empty, so that partial directory was
  never rebuilt -- and the reuse branch is the one every later run takes.
- **`scripts/lerobot_control_rows.py`** refused only a directory with no `policy_preprocessor.json`.

`normalization.check_processor_files` is now public and is the single gate for all four callers
(`load_processor_norm` included), returning the sidecars it checked so a caller can print what it
proved. The shell step runs it through the control env's interpreter before `lerobot-eval`. Two CPU
tests pin it: the list a complete directory returns, and that it refuses a directory with
`config.json` and `model.safetensors` present but the statistics missing. Checked and not affected:
`models/pretrained_vqbet.py` and `load_normalization` read the eight buffers out of
`model.safetensors` and never open a processor JSON.

## 2026-11-02: The stats gate becomes the runbook's first gate, before the download

`docs/12_linux_runbook.md` ran the preconditions only after the 2 GB checkpoint, which is the wrong
order for the one check that costs nothing and invalidates everything: precondition 3 is arithmetic
on two files in the checkpoint repo.

- **New step 2** (`just smolvla-pilot-stats`, `--stats-only --device cpu`) sits between building the
  interpreter and the port walk. It reads `policy_postprocessor.json` and the sidecar it names, a few
  hundred KB -- `PolicyProcessorPipeline` fetches named files and never runs `snapshot_download` --
  so a failure costs a minute instead of the download.
- **Two things the section states because the runbook did not.** `--stats-only` exiting 1 covers both
  `mismatch` (a stop) and `not run` (the sidecar could not be read), distinguished only by the
  printed JSON's `verdict`; and the number is a comparison rather than a tick -- 7.45e-09 here, so a
  materially larger difference on the Linux stack is that interpreter's arithmetic, which is the
  arithmetic the pilot would measure on.
- **Renumbering fixed three stale `step 4` references** (the port-walk section, the exit-code table
  and the failure ladder) that pointed at the interpreter install, which is step 1. Steps are now
  0 preflight, 1 interpreter, 2 stats, 3 port walk, 4 LIBERO config, 5-6 download + preconditions,
  7 measurement.

## 2026-11-02: Precondition 3 verified on this host; the equiv gate now follows `state_file` too

The probe that was just repaired was run against the real checkpoint, and the same sidecar mistake
was found a second time in the processor-equiv preflight.

- **Precondition 3 is `[ok]` without a GPU.** `scripts/smolvla_pilot.py --stats-only --device cpu`
  builds the postprocessor from `policy_postprocessor.json` (setting the two action-tensor
  converters `make_pre_post_processors` would have set) and probes it: MEAN_STD over 7 dims, max
  |diff| 7.45e-09 against the 1e-05 bar. So the arithmetic the pilot depends on is verified; the
  env, the weights and the episode are not, and no cost row exists.
- **The equiv gate checked the JSONs and not the files they name.** `load_processor_norm` required
  `policy_preprocessor.json` / `policy_postprocessor.json` and then handed them to lerobot, which
  loads each step's statistics from the `state_file` sidecar -- `step_3_` on the preprocessor,
  `step_0_` on the postprocessor, so the name has to be read, not derived. A directory missing
  those sidecars would load both pipelines *with no statistics at all*, i.e. the identity, and the
  two implementations would then "disagree" by whatever the unnormalizer contributed. The
  preflight now follows `named_sidecars()` and runs before the lerobot import, so the failure is
  `FileNotFoundError` naming the sidecar even on a host with no lerobot.
- **Six CPU tests** in `tests/unit/test_normalization_protocol.py` use the migrated checkpoint's
  own JSON layout: no `stats` key anywhere, the sidecar named by each step, a name carrying
  `step_7` at index 0 so a derived name would miss, a missing sidecar reported as such, and the
  device override keeping both the `state_file` reference and the declared `norm_map`
  (docs/10 condition 4).

## 2026-11-02: Precondition 3 was reading a file that has no statistics in it

Writing the CPU test for `_statistics_check` first meant reading the checkpoint it would probe, and
the probe could never have passed. **No GPU time, no rows**; the pilot is still unrun.

- **The JSON has no numbers.** `HuggingFaceVLA/smolvla_libero`'s `policy_postprocessor.json` names
  `ACTION: MEAN_STD` and a 7-wide feature, and keeps the statistics in
  `policy_postprocessor_step_1_unnormalizer_processor.safetensors` as `action.mean` / `action.std`.
  The probe scanned the JSON for a mean/std pair, found nothing, and returned `not run` -- which
  `main` prints as `[STOP]` and exits 1 from. It now reads the `state_file` the step names.
- **The probe fed the wrong width.** It passed `max_action_dim` (32), the padded model space, to a
  pipeline configured for 7. The width now comes from the JSON's ACTION feature, which is also what
  the tests pin by using a fake that refuses anything but 7.
- **The MIN_MAX closed form was wrong.** `normalize_processor.py`'s inverse is
  `(z + 1) / 2 * (max - min) + min`, so normalized 0 is the range's *midpoint*, not its minimum.
  The probe now checks -1 / 0 / 1 against min / midpoint / max, via three points instead of two.
- **The tests are shaped like the checkpoint, not like the code.** Eight CPU tests write the real
  sidecar layout (including an 8-wide `observation.state` entry that a longest-pair heuristic would
  have picked) and drive a fake pipeline whose only knob is the arithmetic: a 0.01 % std scale
  error has to read as `mismatch`, the plausible 0-is-min inverse has to as well, and a missing
  `action.mean` has to read as `not run` rather than `ok`.

## 2026-11-02: The Linux path, written as a runbook rather than a list of hopes

`docs/11` 5.1 says the pilot is one install session away on a Linux box and then never says which
session. `docs/12_linux_runbook.md` is that: preflight, the pilot interpreter, the pin check before
the 2 GB download, LIBERO's config file, the preconditions, the measurement, and a failure ladder.
**Not executed** -- this repo has no Linux host -- and section 11 of the runbook names the two steps
that are inference rather than observation.

- **Two interpreters, and the runbook says why.** `lerobot==0.6.1` caps `torch<2.12` while `.venv`
  pins 2.14.1+cu130, so the pilot gets `.venv-pilot` (now gitignored) instead of a resolve that would
  move the torch build under every measured number. The consequence is stated rather than hidden:
  the pilot's wall clock is on the torch lerobot allows, which is the stack the block would run, not
  the stack E8/E9 measured.
- **Cheap gates first.** The port-walk gate runs against the pilot interpreter and must print
  `[PINNED ]` -- seconds, no GPU, no network -- before the checkpoint download; then the three
  preconditions; only then the episode. A wrong robosuite or a CPU-only torch fails at step 1 or 2,
  ahead of the download, which is the whole reason for that order.
- **Two recorded failure modes became steps rather than footnotes.** LIBERO calls `input()` when
  `~/.libero/config.yaml` is missing, so an unattended run dies at `EOFError`; the runbook derives
  that file from the installed package. And `egl-probe` building is the entire Linux/Windows
  difference, so `libero==0.1.1` is installed with its dependencies here and was `--no-deps` there.

## 2026-11-01: The pilot script exists, and says what it refuses to measure

`docs/11` 5.2 asked for one script to replace 5.1's per-chunk band with a measurement.
`scripts/smolvla_pilot.py` is it (`just smolvla-pilot`, `just smolvla-pilot-check`). **No GPU time,
no rows**: it cannot run on this host.

- **What it measures.** One LIBERO-Spatial episode (task 0, initial state 0, 280-step cap) at 28
  chunks of ten actions: policy time synchronized around `predict_action_chunk` plus the
  postprocessor, the observation build, and the ten `env.step` calls, each with mean / median /
  p95 / max. Then `projected_gpu_hours_500` and `..._three_arms`, which are 5.1's GPU-hours column
  with that mean substituted for the band -- labelled projected, because the cadence is still an
  assumption even after the per-chunk cost stops being one.
- **Preconditions in order, with exits that mean different things.** The env constructs on the
  pinned stack (1), the checkpoint loads (2), and the postprocessor reproduces the checkpoint's
  own declared statistics at 1e-5 (3, which is 4.4's step 1). A mismatch at (3) stops the run: it
  would calibrate a different model. A missing LIBERO or `transformers` exits 2 -- what this
  Windows host does -- so "the stack is absent" is never reported as "the stack is broken".
- **Three silent failure modes became errors.** A chunk holding fewer actions than the cadence
  executes (a single action, so a different loop); actions outside LIBERO's `[-1, 1]` box (an
  unnormalized chunk, which times a policy that never moved the arm); and a repeat starting from
  the next initial state, because `LiberoEnv.reset` advances `init_state_id` -- the repeat now
  pins it back so both episodes are the same episode.
- **Not run.** LIBERO's pin cannot load here and `transformers` is not in the project env, so the
  first real run belongs to the Linux path of 5.1. What was verified is the failure path: exit 2
  with `[NOT RUN]` naming the missing module, which is the designed outcome on this host.

## 2026-11-01: The same framing error audited out of the remaining docs

The entry below fixed `docs/11` section 5.1 and left the prose around it alone. This one greps the
tree for the other places where a capability was written as something to buy. No code, no GPU time,
no rows.

- **Seven more passages in `docs/11`.** Section 0's scoping rule said a model "is not adoptable at
  any price this repo is willing to pay"; it now says the limit is the machine, not a willingness to
  obtain one. Section 1 called π0/π0.5/GR00T "the ones worth renting a GPU for" and now calls them
  "the ones that need a 14-22 GB card", with their published cells attached to the requirement
  rather than to a purchase. Sections 4.3, 4.4, 4.4's verdict and 7's ordering said "rented GPU";
  they now name the machine (a Linux box with a CUDA card, a 14 GB-class card). Section 5's bullet
  said 24-48 GB, which contradicted the memory column four lines above it (π0 is 14 GB); it says
  14-48. Section 5's heading "the honest cost" did not say which kind of cost, and now reads "what
  that forces".
- **One currency figure was still in `docs/09`.** "$0 of that came from the high-risk options" was
  the only dollars figure outside this changelog; it reads "None of that came from", which is the
  claim being made anyway.
- **Dated entries were not rewritten.** The entries below this one and the dated sections of
  `docs/STATE.md` still say "priced" and "rented" -- they record what was reasoned on the day, and
  editing them would make the record look right earlier than it was. The live documents (README,
  model card, `docs/06`, `docs/09`, `docs/11`) carry the hardware framing.
- **Done is a grep, not a reading.** `grep -nE 'rent|price|afford|dollars|\$[0-9]'` over the live
  docs now returns only wall-clock uses ("pays warmup", "prices it at a fifth") and this entry's
  description of the correction.

## 2026-11-01: The cost table becomes a hardware table, and the port walk becomes a gate

Three corrections to `docs/11`; no code path changed and no GPU time was spent.

- **5.1 no longer prices spending.** The column that mattered was "how many dollars" and it should
  have been "what does the machine have to be". The table now states requirements (OS, disk, card,
  compute), carries no currency figures, and separates two things that were conflated: memory is
  *not* the exclusion -- measured peaks are 1189 MiB fp32 / 710 MiB bf16 for a 262.7 M policy at
  N=10, so a 450 M policy fits in 8 GB [E] -- while the operating system is. No row implies a
  purchase; a grep confirmed currency figures lived only in four Markdown files and in no script,
  config or dependency.
- **New 5.2 plans the pilot that replaces the per-chunk band.** One episode, LIBERO-Spatial task 0,
  280-step cap at a 10-action cadence (28 chunks), recording policy and env time separately with
  mean/median/p95 and peak VRAM. Under 15 minutes and ~0.004 GPU-hours, so it precedes the block
  rather than being inferred from it; it decides nothing about success, discordance or RTC, and says
  so. Owed: one ~80-line `scripts/smolvla_pilot.py`, not written because it cannot run before the
  platform decision in 5.1.
- **The port walk is now a script rather than a scratch file.** `scripts/libero_port_walk.py`
  (`just port-walk`) re-checks the five walls instead of trusting the record: a static probe per
  removal via `find_spec` / `signature`, then two stages that must fail at `ModuleNotFoundError` (no
  shims) and `KeyError: 'MountedMountedPanda'` (walls 1/2/4 shimmed). `pinned` (robosuite 1.4.x),
  `ladder intact` and `not run` exit 0; `drift` exits 1, because a wall that disappeared silently
  invalidates 4.5's 2-4 day estimate. Writes `results/libero_port_walk.json`.

## 2026-10-18: Stage 1

- Added `00_overview.md` (day-one checklist, scope, success criteria template, corrections to the brief).
- Added `02_environment_setup.md` (compatibility matrix, pitfalls, install steps, gates G1 to G9, results table, references).
- Findings that change the brief: PyTorch is at 2.14.1 and no longer ships CUDA 12.8 wheels (use `cu130`); ROS 2 Lyrical Luth now exists; PyTorch 2.13 stopped bundling `ptxas` in cu13 builds.
- Open: items listed under "Not verified in Stage 1" in `02_environment_setup.md` section 10.
- Next: Stage 2 (ADR-001 to ADR-008).

## 2026-10-18: Stage 2 (consolidated)

- Added 01, 03, 04, 05, 06, 07. ADRs for D1-D10 are consolidated in 06; reading list and packaging are in 07.
- Diffusion Policy recipe sourced from replicating papers and the LeRobot port, not the original paper (flagged [S]/[U]); agent must confirm against the paper before E0.
- Unverified items remain tagged [U] (pi0 timestep distribution, ACT details, dataset URLs/licenses, async chunking literature, Triton pin, uv/Docker/Nsight specifics).

## 2026-10-18: M0 scaffold (agent)

- Repo scaffold per M0: `pyproject.toml` (3.12, hatchling, ruff, mypy strict on `src/`, pytest markers), `uv.lock`, `justfile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `README.md`, `AGENTS.md`, `src/policybench/__init__.py`, `tests/unit/test_package.py`.
- Acceptance met: `just setup && just test` green from a clean clone; CI green (run 39179413619).
- Deviation: the `smoke` CI job in 01 §4 is absent until a model exists (M3); the `unit`, `lint`, `types` jobs cover M0.

## 2026-10-18: M1.1 gates G1-G5 (agent) — BLOCKED

- Deviation from the docs' platform assumption: the owner has one machine, a **Windows 11 laptop** (build 26200, MIGW64 shell), not Ubuntu. No WSL2. Consequences: docs/02 §5-6 and the Docker/ROS 2 path (D7) do not apply as written; §2 fact sheet and §8 gate table are filled with measured values, and this entry is the dated deviation that 07 §0 rule 1 requires.
- torch 2.14.1+cu130 and torchvision 0.29.1+cu130 install from `https://download.pytorch.org/whl/cu130` on Windows (`win_amd64` wheel exists), pinned in `pyproject.toml` and `uv.lock`. The `--index` syntax in 02 §5.2 is [U] as written; the working form is `[[tool.uv.index]]` plus `[tool.uv.sources]`, tagged [V] 2026-10-18.
- G1-G5 FAIL, single root cause: `cudaGetDeviceCount() returned cudaErrorNotSupported, likely using older driver`. Driver is 577.13; CUDA 13.0 needs the R580 branch. Risk R1 has no fallback, so no model or data work starts until G1 is green.
- G5 has a second cause: `triton` is not installed by the Windows cu130 wheel. Do not install it separately (02 §3.1); revisit after the driver fix.
- G6-G9c NOT RUN: no `nvcc`, no `docker`, no simulators yet. G9 is deferred by owner decision to M1.5.
- Doc defect found: 02 §9 refers to `10_risk_register.md`, which does not exist; risks live in 06 §2.

## 2026-10-19: driver updated to 617.42, gates G1-G5 re-run (agent)

- Owner installed the NVIDIA App on 2026-10-18 and the display driver on 2026-10-19; `nvidia-smi` now reports **617.42** (CUDA UMD 13.4) on the RTX 5060 Laptop GPU, up from the factory 577.13. The R580 minimum in 02 §3.1 is met.
- Same five gate scripts, unedited: G1 PASS, G2 PASS, G3 FAIL, G4 PASS, G5 FAIL. `torch.cuda.is_available()` is now True and `torch.cuda.get_device_capability()` returns (12, 0), which clears pitfall 1 for good.
- G3 and G5 now have a single, different cause: `triton` is not installed in the Windows cu130 environment. This is not a driver fault and was not worked around; 02 §3.1 forbids installing Triton ad hoc, so the choice (Windows Triton wheel vs dated scope deviation) is left to the owner and recorded in `docs/STATE.md`.
- 02 §2 (driver, compute capability, VRAM rows) and §8 (G1-G5 rows, history note) updated with these measurements. G6-G9c remain NOT RUN.

## 2026-10-19: triton-windows pin, G3 and G5 green

- Owner decision: install a Windows Triton build rather than drop compile/Triton from scope, because 02 §8 makes a red G3/G5 a stop-the-line result.
- Added `triton-windows==3.8.0.post29` to `[project].dependencies` with a `sys_platform == 'win32'` marker, so the Linux CI jobs keep torch's own Triton resolution. `uv.lock` updated by `uv add`.
- G3 now passes all three modes (`default`, `reduce-overhead`, `max-autotune`) and G5 passes with `triton 3.8.0`; the assembler works, so risk R2's `TRITON_PTXAS_PATH` fallback is not needed here.
- `just gates` is five PASS and `just test` is green. 02 §2 (Triton row), §3.1 (pin, marked [V] with the [U] upstream-version question) and §8 (G3, G5) updated.
- This replaces the "do not install Triton separately" guidance in 02 §3.1 for this host; the reason is the dated deviation recorded there.

## 2026-10-19: M2 data and model plumbing (agent)

- Added the Zarr window loader, train-only normalizer with round-trip tests, ResNet-18 GN multi-frame encoder, 1-D U-Net and transformer chunk heads, and the three policies (flow, DDPM/DDIM, ACT-style regression) behind one `BasePolicy` interface (`loss(batch)` / `predict_chunk(obs)`).
- `scripts/data/prepare_pusht.py` downloads and converts `lerobot/pusht` to `data/pusht/pusht.zarr` (32 MB, MIT). Hydra configs live in `conf/` with structured dataclasses in `src/policybench/config.py`.
- Measured memory table (docs/03 §4): batch 32 = 2317 MiB / 7.80 steps/s, batch 64 = 4414 MiB / 4.00 steps/s, batch 128 = 8616 MiB / 0.05 steps/s and flagged not fitting. Model is 10.95 M parameters. Batch 64 was used for E1.
- Unit tests cover window indices, normalizer round-trip, policy shapes/losses, seed disjointness, and resume equivalence (`rtol=1e-5`). `just test` green (23 tests).

## 2026-10-19: M3 trainer and overfit sanity (agent)

- Trainer: bf16 autocast, AdamW, warmup + cosine, grad clip 10, EMA 0.999 (EMA used for every eval), periodic + final checkpoints, SIGINT/SIGTERM checkpoint-and-exit, JSONL scalar log, checkpoint carries model/EMA/optimizer/scheduler/step/RNG/hydra config/stats/git hash/`uv.lock` hash.
- Overfit check on 8 episodes, 10x-reduction criterion: flow 1.230 -> 0.065 (0.053x) and DDPM 1.096 -> 0.091 (0.083x) at 1200 steps; regression 0.376 -> 0.029 (0.077x) at 2000 steps (it reached 0.109x at 1200, so the budget was raised for that policy only and the step count is reported per row).
- Bug found by the post-reboot resume: a checkpoint loaded with `map_location=cuda` holds CUDA RNG tensors, which `set_state` rejects. `Trainer.resume` now moves them to CPU. The interrupted DDPM run resumed from step 1200 and its uncheckpointed 1200-1625 tail was dropped from the JSONL instead of being kept.

## 2026-10-19: M4/M5 E1, E3 and the E0 fallback (agent)

- Nine runs (3 policies x 3 seeds), 1800 steps, batch 64, ~9 min each (1.4 GPU-h total). `results/train/*.jsonl` has every logged step.
- E1: 270 episodes (30 per seed), fixed eval seeds 10000-10029, EMA weights, N=10. **No policy reaches the 95% coverage success threshold.** Mean coverage: regression 0.290, flow 0.252, DDPM 0.132; success 1.1% for regression and 0% for the other two, every 95% CI including 0. Table: `results/tables/e1_main.md`.
- E0 is not reproduced and is not claimed: the DDPM baseline trains for ~1.5 epochs, not the reference ~200k steps. This is the M4 fallback "proceed with relative comparisons only" plus the written explanation in docs/04 §4.
- E3: 720 episodes over N in {1,2,4,8,10,16} on the E1 checkpoints, 20 episodes per seed. Latency scales linearly with N (flow 10.7 ms at N=1 to 66.6 ms at N=16, measured with 2 workers). The pre-registered hypothesis "flow degrades less than DDIM as N falls" is **not supported** at this budget: flow coverage 0.381 at N=1 vs 0.333 at N=10, non-monotonic and within noise; no configuration succeeds.
- Deviation: eval workers reduced from 4 to 2 after the owner reported the laptop was unresponsive; timings are therefore under two-process contention and are labelled as such.

## 2026-10-19: M6 inference ladder and Triton kernel (agent)

- E9, idle GPU, batch 1, 50 warmup + 200 timed calls, mean/p95: eager 33.46/34.48 ms at N=10; `compile-reduce-overhead` 5.80/5.89 ms (5.8x, 25.8 s compile); `compile-max-autotune` 5.98/6.33 ms (133.6 s); `compile-default` 13.64/14.60 ms; CUDA graph 7.09/7.24 ms (4.7x). CUDA-graph N sweep: 2.29 ms at N=1 to 10.21 ms at N=16. `results/latency.jsonl`.
- One Triton kernel (fused last Euler step + de-normalization) passes equivalence at fp32 `atol=1e-5` (max diff 1.2e-4) and beats eager 2.2-2.4x, but **loses to compiled PyTorch** (0.83x at batch 1, 0.76x at batch 64). Reported as a negative result; `results/kernel.jsonl`.
- First compile of the kernel failed: Triton rejects `typing.Any` annotations inside `@triton.jit` functions. The pointer arguments are unannotated and the function carries a `no-untyped-def` ignore for mypy.

## 2026-10-19: M7 deployment loop and E10 (agent)

- ROS 2/Docker are unavailable (G9a NOT RUN), so the documented plain-Python fallback (D7) runs the same policy, env and JSONL interface: fixed 100 ms tick, action queue, deadline-miss counter, observation-staleness latency injection with uniform jitter, naive replace vs temporal ensembling.
- E10: 315 episodes (3 checkpoints x 3 eval seeds x 5 episodes) at 0/50/100/150/200 ms and 20 ms jitter. Mean coverage 0.293/0.334/0.287/0.334/0.322; zero successes and **zero deadline misses** because measured inference (45-55 ms) stays under the 100 ms period. Temporal ensembling does not recover anything measurable (0.299 at 100 ms, 0.272 vs 0.322 naive at 200 ms). Null result, reported as such.
- Cold-start bug: the first E10 pass timed the first chunk cold (871 ms), which put the mean above p95. The loop now warms up untimed per episode and `results/deploy.jsonl` was regenerated from scratch.

## 2026-10-19: M8 packaging and MVP close (agent)

- `scripts/make_tables.py` and `scripts/make_figures.py` regenerate every table (`results/tables/`) and figure (`results/figures/`) from `results/*.jsonl`; `scripts/make_readme.py` injects them into `README.md`. No number in the README is hand-typed.
- README carries the measured tables, the honest null results (no E1 success, Triton loses to compile, E10 flat), the E0 explanation, and the limitations list. Demo video is the one packaging item not produced (no capture path on this host); it is marked as missing in docs/00.
- Remaining gaps, stated rather than hidden: E2, E4-E8, E11, E12 not run; robomimic/ROS 2/CUDA C++ out of scope on this host; gym-pusht equivalence (R5) still unverified.

## 2026-10-23: 12000-step budget, E1/E3/E10 re-run, E12 made real (agent)

- Training budget raised from 1800 to 12000 steps on the owner's request: batch 64, 3.2 steps/s, 62-64 min per run, 3.1 GPU-h for the three policies. One training seed per policy (seed 0); the pre-registered three are a documented deviation in docs/04 §4. Peak VRAM 4458 MiB, final loss 0.054 flow / 0.062 DDPM / 0.024 regression.
- `scripts/train_e1.sh` now resumes instead of retraining: a run with `last.pt` is skipped, otherwise it resumes from the newest `step_*.pt`. The DDPM run was interrupted at step 8825 and resumed from step 8000.
- E1 (150 episodes, 50 per checkpoint, N=10): **flow is the first policy with non-zero success**, 8.0% (4/50) [2.0%, 16.0%]; regression 4.0% (2/50) [0.0%, 10.0%]; DDPM 0% (0/50). Mean coverage flow 0.471, regression 0.303, DDPM 0.284. Intervals are episode-level.
- Budget ablation, same pipeline: at 1800 steps flow was 0% / coverage 0.252 and DDPM 0% / 0.132; at 12000 steps flow is 8% / 0.471. The earlier null result was the budget, not a broken pipeline.
- E3 (600 episodes): flow retains coverage as N falls (0.401 at N=1 vs 0.468 at N=10, -14%) while DDPM loses twice as much (0.172 vs 0.239, -28%), which supports the graceful-degradation hypothesis in coverage. Success is too sparse to order N: flow 14% at N=4 and N=16 but 0% at N=10, non-monotonic and inside the +-6 pp resolution of 50 episodes. Latency 9.6 -> 69.4 ms.
- E10 (420 episodes, naive replan): coverage 0.439 at 0 ms and 0.385-0.417 at 50-200 ms, success 0-3.3% with overlapping CIs. Deadline misses 3-8 per ~2280 chunk inferences (individual slow chunks; mean inference 45-53 ms against a 100 ms period), down from the "zero misses" claim that only held at the 1800-step checkpoints.
- E12 bug found and fixed: temporal ensembling replanned only when the action queue emptied, so each step was covered by exactly one chunk and the ensemble was a copy of naive (coverage 0.411 at 100 ms, 0.413 at 200 ms, identical success). Added `replan_every`; naive and temporal-ensemble were re-run at a 2-step replan rate with four overlapping chunks. Naive replacement collapses (coverage 0.212 / 0.154, 0/120 successes); temporal ensembling recovers most of it (0.386 / 0.323, 4/120) but stays below the 8-step naive baseline (0.411 / 0.413) and costs 4x chunk inference.
- Tables, figures, README and docs/04 §3 claims regenerated from the new JSONL; docs/04 §4 records the new deviations (single training seed, replan-rate change, ~19% zero-coverage episodes). `just test` green.

## 2026-10-24: three training seeds, DDPM ladder bug fixed (agent)

- E1 now runs the pre-registered three training seeds at the 12000-step budget (seeds 1 and 2 trained at 3.2 steps/s, 62-64 min each), 50 paired episodes per checkpoint, 450 episodes total. Success: flow 7.3% [3.3%, 12.0%] (11/150), DDPM 4.7% [1.3%, 8.7%] (7/150), regression 4.0% [1.3%, 7.3%] (6/150); coverage 0.519 / 0.468 / 0.336. All three success CIs overlap, so the earlier "flow is the only policy with non-zero success" claim is retracted: the policies are ordered by point estimate and by coverage (regression clearly below both generative policies), not separated on success.
- `DDPMPolicy._sample` bug: the ladder ran from t=99 down to t=0, but `alphas_cumprod[0] == 1.0` exactly, so the last step was a no-op and every N behaved as N-1 steps; N=1 and N=2 produced bit-identical samples (coverage 0.2294 both). The ladder now ends at t=1. Guarded by `test_ddpm_step_count_is_effective`, which fails on the old code. DDPM E1 and E3 rows were re-evaluated (sampling-only change, no retraining).
- E3 re-run for DDPM: 0.229 [0.177, 0.284] at N=1, 0.213 [0.168, 0.264] at N=2, 0.467 [0.395, 0.536] at N=4, 0.550 [0.463, 0.632] at N=10 - DDIM needs at least 4 steps. Flow's ladder (0.401-0.543) has overlapping intervals at every N, so E3 cannot order N for flow; the table now prints coverage intervals so the reader can see that.
- `scripts/make_tables.py` prints a 95% bootstrap interval for E3 coverage; README, tables, figures and docs/04 §3/§4 regenerated from the new JSONL.

## 2026-10-27: pivot to a pretrained policy, then measure what a good policy actually costs (agent)

- **The owner asked for a 180: stop training, use a checkpoint that is already good, make the
  infrastructure the deliverable (D11).** The trigger is the ablation: three policies at 12000
  steps reach 4-7 % success, and at that level a comparison cannot order two policies and a
  latency experiment cannot see a policy degrade. The substrate is now
  `lerobot/diffusion_pusht` (Apache-2.0, 262.7 M parameters, fp32, trained on gym-pusht for
  175k steps; the model card reports 65.4 % success and 0.955 mean max overlap over 500
  episodes). Nothing is trained here any more.
- `src/policybench/models/pretrained.py` is new: a `PolicyLike` adapter that puts the HF
  checkpoint behind the repo's interface. `models/policies.py` gained the structural
  `PolicyLike` protocol; `eval/rollout.py`, `eval/latency.py` and `deploy/control_loop.py`
  were re-typed to it. The episode loop, the latency protocol, the control loop, the tables
  and the figures were **not** changed -- one load branch in `load_policy` was enough, which is
  the architecture claim in docs/01 holding up.
- **Normalization is the only interesting part of the adapter.** lerobot 0.6.1 no longer
  creates the `normalize_inputs` / `unnormalize_outputs` modules the 2024 checkpoint was saved
  with, so `from_pretrained` logs those eight tensors as unexpected keys and drops them.
  The adapter reads them from `model.safetensors` (ImageNet mean/std for images, dataset
  min/max for state and actions) and applies them itself. A unit test covers the three
  transforms (`tests/unit/test_pretrained_normalization.py`, 4 tests, 33 total); CI stays
  backend-free because lerobot is imported lazily.
- **E0 is the acceptance gate and it passes.** The checkpoint through our gym-pusht wrapper,
  our harness and our metric: 64.0 % success (32/50) and 0.924 mean coverage at DDIM N=10,
  against the published 65.4 % / 0.955 for the checkpoint's own DDPM@100 setup. That also
  closes the reproduction item that the from-scratch route never reached (docs/00 section 3).
- **Dependencies:** lerobot 0.6.1 requires `torch < 2.12` while this project pins
  `torch 2.14.1+cu130`, so `just setup-pretrained` installs it `--no-deps` with the packages
  its inference path imports. Documented as a deviation (docs/04 section 4) and as R13 in
  docs/06, because it is verified working, not supported.
- **What moved:** from-scratch measurements now live in `results/ablation/` (e1, e3, deploy,
  latency, kernel, memory) and are presented as the ablation that motivated the pivot;
  pretrained rows are written to `results/{e0,e3,latency,deploy}.jsonl` with
  `train_seed = -1` and `weights = pretrained`, so no table can quietly mix them with a
  training-seed comparison. `E13` (36000 steps plus random-crop augmentation) was scaffolded
  before the pivot, never run, and has been removed rather than left half-finished.
- **Measured, pretrained checkpoint (2026-10-27).** E3 (300 episodes, DDIM): coverage 0.121
  [0.076, 0.165] at N=1, 0.716 at N=2, 0.885 at N=4, flat from there (0.888-0.919, intervals
  overlapping) -- one step is catastrophic, four reaches the plateau. E9 (batch 1): eager
  20.2 / 73.5 / 190.5 / 309.4 ms and CUDA graph 17.5 / 66.8 / 181.0 / 304.0 ms at N=1/4/10/16,
  i.e. graphs are worth 1.02-1.16x here against 4.7x on the 11 M ablation model; `torch.compile`
  is NOT RUN (Inductor needs MSVC for the CPU partition the checkpoint's DDIM scheduler creates,
  `InvalidCxxCompiler: cl is not found`), reported as NOT RUN by `scripts/latency_ladder.sh`
  rather than as a row. E10 (270 episodes): success 56.7 / 60.0 / 56.7 / 46.7 / 43.3 % at 0/50/
  100/150/200 ms injected latency with overlapping intervals and flat coverage (0.897-0.932), but
  26-33 deadline misses per episode because inference needs ~2x the control period. E12: naive
  replacement at a 2-step replan rate collapses to 10.0 % (coverage 0.749 / 0.553, disjoint from
  the baseline) while temporal ensembling recovers 53.3 % / 43.3 % (0.926 / 0.880). Together E3 +
  E9 fix the deployable envelope: only N <= 4 fits a 100 ms period, exactly where coverage has
  plateaued.
- **Infrastructure added the same day, none of it training.** `policybench.eval.report` computes
  the headline summary that `just report` prints and that the README injects (so the five
  headline answers are generated, not typed); `policybench.deploy.server` is a stdlib-only HTTP
  serving boundary with a documented JSON contract (docs/05 section 4) and six socket-level
  contract tests; `scripts/doctor.py` reports environment PASS/FAIL/NOT RUN in seconds;
  `docs/MODEL_CARD.md` covers the borrowed policy and this stack; CI gained a job that
  regenerates the tables and fails if the checked-in ones are stale. `kernel_bench.py` and
  `bench_memory.py` were repointed at `results/ablation/` after the move. Unit tests: 45.
- Docs re-framed around the pivot: 00 (status, goals, success criteria), 01 (interface and
  adapter, acceptance ticked), 02 (new section 3.4 on the optional backend), 03 (marked as the
  ablation path), 04 (matrix, deviations, priority order), 06 (D11, R13, R14), 07 (rule 4
  amended: borrowed policy code is now allowed, the harness/loop/tables stay ours).
- **E12 became the fifth headline answer, and the README prose stopped restating numbers.**
  `policybench.eval.report` gained a row for the `-r2` rows (naive replacement 10 % at 100 ms
  and 200 ms against 57 % / 43 % for the 8-step baseline, temporal ensembling recovering
  53 % / 43 %), so the deployment finding is in `just report` and in the README's generated
  headline table rather than only in the E10 prose; `tests/unit/test_report.py` covers it
  (7 summary tests, 46 unit tests total). The paragraphs around the tables now describe the
  direction of each table and its exceptions without reprinting measured numbers, which is
  AGENTS.md rule 3 applied to prose as well as tables. Also recorded: the E12 deadline-miss
  counts (120 for the ensembling rows, 141-147 for naive-r2) in docs/04 and docs/MODEL_CARD.
  The ratios quoted in the docs were re-checked against `results/latency.jsonl`: 1.02-1.15x for
  CUDA graphs on the pretrained policy against 4.7x for the same comparison on the 11 M
  ablation model, where `torch.compile` was 5.8x faster than eager.

## 2026-10-28: compile gap closed at its cause, E14 usefulness baseline, demo, resume notes

- **`torch.compile` on the pretrained policy went from NOT RUN to measured, and the earlier
  diagnosis was wrong in an instructive way.** The 2026-10-27 entry blamed the host (no MSVC).
  The blocker was `diffusers` keeping `alphas_cumprod` on the CPU and indexing it with a Python
  int per step: lerobot's DDIM loop therefore mixed ten host scalars into CUDA math per chunk,
  which costs a host/device synchronization per step *and* gives Inductor a CPU partition that
  needs a C++ compiler. `src/policybench/models/fast_sampler.py` precomputes the DDIM
  coefficients on the sampling device and runs the loop in tensor math only. Equality with
  `diffusers` is a test, not a claim: `tests/unit/test_fast_sampler.py` (9 tests) covers five
  step counts, two beta schedules, the timestep grid and the coefficient layout. The first draft
  applied `sqrt(1-a)/sqrt(a)` after multiplying by `1/sqrt(a)` as well; the test caught it.
- **E14 (`scripts/bench_stack.py`, `results/stack.jsonl`, `results/tables/e14_stack.md`)** is the
  baseline the earlier packaging lacked. Same checkpoint, same GPU, same episodes, three ways to
  run it: the published configuration (DDPM@100), the stock lerobot->diffusers DDIM path, and
  the fast sampler with CUDA graphs and `torch.compile`. Latency rows report per-chunk cost,
  control rate and the share of a 100 ms period; throughput rows report seconds per episode and
  episodes/hour with success and coverage attached, so the cheap arms cannot be quietly worse
  policies. `results/figures/e14_stack_throughput.png` puts the ratio on a log axis.
- **What the numbers say** (see the table for values): the published configuration costs seconds
  per chunk and an episode takes minutes; the step-count dial buys two orders of magnitude in
  experiment cost at a coverage cost that E3 already quantified; graphs and compile are worth
  roughly a tenth each because the 262.7 M-parameter network is compute-bound, so launch
  overhead is not where a 100 ms period is won or lost. That last point contradicts the
  from-scratch ablation's 4.7x/5.8x and is reported as the reversal it is.
- `PretrainedDiffusionPolicy` gained `sampler="stock"|"fast"` (default `stock`, so every earlier
  row stays comparable); `--sampler` is threaded through `evaluate`, `deploy_eval`, `latency` and
  `bench_stack`. `rollout_episode` now records `wall_s` and `chunks` per episode, which is what
  the throughput arms aggregate.
- **A demo artifact exists**: `scripts/demo_rollout.py` (`just demo`) records a real rollout to
  `results/figures/demo_rollout.gif`, with step, coverage, queue depth, measured chunk latency
  and deadline misses drawn on the frame. This closes the "no capture path on this host" item
  from M8 in docs/00 and docs/07.
- **Plain-language docs**: the README now opens with what the software is, a component table and
  a stack table, then "what it buys you" before the experiment sections; `docs/RESUME.md` holds
  XYZ-format bullets. `scripts/doctor.py` treats MSVC as optional and says why.
- **The sampler swap was checked on episodes, and it does not fully agree.** Action chunks from
  the fast and stock loops differ by at most 0.02 px on the 512 px action space (fp32 summation
  order), pinned by `tests/gpu/test_pretrained_sampler_equivalence.py`. On 20 paired episodes in
  a contact-rich simulator that rounding is enough to move outcomes: 70 % vs 55 % success and
  9/20 episodes agreeing, coverage 0.905 vs 0.856. Reported in E14's equivalence block rather
  than averaged away, and stated as a resolution limit: 20 episodes cannot certify a 1e-5 action
  difference, so the sampler is justified by the arithmetic test plus the E0 gate, not by
  counting successes.

## 2026-10-28: M9 hardening (docs reconciled, CI smoke job, E8, E3 replication, serving load, pacing, blend)

- **The docs were reconciled with `results/` before anything new was measured.** Three claims
  still described the pre-2026-10-28 state: the E9 claim row said `torch.compile` was NOT RUN,
  the README read-me said the same, and `docs/00` advertised a matrix of E0-E12 while E14
  existed. All three fixed, E14 gained its own matrix row, and twelve acceptance boxes across
  docs/00-03 were either ticked with dated evidence or annotated as unmet with the reason (the
  Diffusion Policy recipe is still `[S]`-sourced; the "fresh agent can run M0/M1" check needs a
  clean-clone dry run that has not been done). Two references to a `10_risk_register.md` that
  never existed now point at `docs/06` section 2.
- **CI `smoke` job, promised by docs/01 section 4 since 2026-10-18 and deferred "until a model
  exists".** `tests/integration/test_train_smoke.py` trains 20 steps on synthetic data, asserts
  the loss falls by at least 5 %, and asserts an interrupted-and-resumed run matches an
  uninterrupted one at `rtol=1e-5`. New `smoke` pytest marker, `just smoke` recipe (~5 s on
  CPU), and a CI job that runs it. This closes a recorded deviation rather than adding a claim.
- **E8: precision, on the pretrained checkpoint (`results/precision.jsonl`,
  `results/tables/e8_precision.md`).** bf16/fp16 cut a chunk from 78.1 to 65.6 ms eager (73.4
  to 55.6 ms with a CUDA graph) at N=4, and from 209.0 to 152.3 ms eager (194.4 to 138.4 ms
  graph) at N=10, while peak VRAM falls from 1189 to 710 MiB. **The envelope does not move:** the
  best half-precision N=10 row is 136.7 ms, still 1.4x over a 100 ms period, so fewer sampler
  steps remain the only knob. On 40 paired episodes per dtype: success 35.0 % fp32 [22.5, 50.0],
  50.0 % bf16 [35.0, 65.0], 45.0 % fp16 [30.0, 60.0], coverage 0.900 / 0.885 / 0.901 -- a
  no-gross-regression result, not an equality proof (40 episodes cannot resolve the 2 pp the
  hypothesis named). Implementation: `--dtype` threaded through adapter and runners, allowed
  only on the fast sampler, and the sinusoidal timestep embedding kept in fp32 with its output
  cast at the module boundary (the unet raises a dtype mismatch otherwise).
- **E3 replication on an independent 50-episode block (`results/e3_replication.jsonl`).**
  Coverage replicated (0.761 / 0.897 / 0.902 / 0.896 at N=2/4/8/10 against 0.716 / 0.885 / 0.919
  / 0.888); success did not (24/38/48/40 % against 20/42/60/54 %, every pair overlapping). The
  plateau and the two-step collapse are properties of the policy; the "N=8 is best" ordering
  was episode noise, and now that is measured twice instead of asserted.
- **The serving boundary is load-tested (`scripts/bench_serve.py`, `results/serve.jsonl`).**
  One client: 91.5 ms p50 client round trip against 74.0 ms server-side chunk time, i.e. 17.8 ms
  of JSON/socket per request and 11 requests/s. Two clients: 171.6 ms p50, because
  `ThreadingHTTPServer` accepts both and they serialize on one GPU. A cold server pays 655 ms
  on its first request; the script warms up before timing, like the latency protocol.
- **The control loop has a wall-clock paced mode (`--paced`, `results/paced.jsonl`).** At N=4
  the loop holds the 100 ms period (0.2 late ticks per episode; 2 of 12 episodes slipped once,
  mean lateness 1.1 ms). At N=10 it is late on every episode (52.5 late ticks, 65.8 ms mean
  lateness, p95 102.9 ms): once a chunk overruns, the lateness is a floor it carries rather than
  a spike it recovers from. Pacing also puts the miss into the outcome rather than only the
  counter -- success 3/12 at N=10 against 10/12 at N=4, coverage unchanged -- directionally,
  since twelve episodes leave the intervals just overlapping.
  The first tick's offset was fixed before the measurement (`steps * period`, not
  `(steps - 1) * period`), precisely so the reported lateness is the loop's, not the clock's.
- **E12's third strategy exists and is named honestly (`--strategy blend`).** Model-level
  "inpainting" needs a policy conditioned on committed actions, which this checkpoint is not, so
  the arm is a receding-horizon cross-fade over the first four queued actions. Measured at a
  2-step replan rate: coverage 0.875 / 0.864 at 100/200 ms (naive replacement 0.749 / 0.553),
  success 26.7 % / 10.0 % (ensembling 53.3 % / 43.3 %). Smoothing the splice helps coverage but
  not success, so the pre-registered "inpainting-style is best" hypothesis is refuted for this
  implementation and ensembling stays the requirement.
- **A real bug the new load test surfaced, fixed.** `do_POST` answered 404 for an unknown path
  without reading the request body; under HTTP/1.1 keep-alive the leftover body was then parsed
  as the next request line, so a client posting to the wrong path could see a connection reset
  instead of the documented 404. The body is now always read and discarded before the path
  check. `test_unknown_path_is_404` had been failing intermittently in the full suite (about
  one run in two) and now passes three consecutive suite runs.
- Tests: 57 CPU unit tests (from 49) -- blend arithmetic, `resolve_dtype`, the E8 headline row,
  and the report tests rewritten to look rows up by experiment instead of by position. `just
  lint`, `just types`, `just test` and `just smoke` green; tables and README regenerate.

## 2026-10-28: renamed `flowpolicy` -> `policybench`

- The package name described the from-scratch flow-matching models, which became an ablation on
  2026-10-27; the repository is now a measurement and deployment stack around a borrowed
  diffusion policy. `src/flowpolicy/` moved to `src/policybench/` with `git mv`, every
  `flowpolicy.*` import and docs path was rewritten, `pyproject.toml` (name, description, hatch
  target) and `uv.lock` were updated, and `just test` passes 57 tests afterwards.
- `models.policies.FlowPolicy` was deliberately **not** renamed: that class is the conditional
  flow-matching ablation model, and the name is still the right one. A blanket search-and-replace
  would have silently renamed the model too.
- Project title updated in `docs/README_template.md` (README is generated), `CLAUDE.md`,
  `scripts/report.py` and the package docstrings. The uploaded brief keeps its original name at
  `flowpolicy_complete/`, and `docs/00` now says so explicitly instead of citing a document that
  no longer matches the tree.
- No measured number changed: no `results/*.jsonl` row contained the old name, and
  `just tables && just figures` regenerated byte-identically. The README diff is the title plus
  ten path mentions.
- Found while verifying: a real `uv sync` prunes the optional backend, and `just test` then fails
  at collection because `tests/unit/test_fast_sampler.py` imports `diffusers`, absent from
  `uv.lock`. CI's `unit` job has the same shape, so it has been red since `9e5be8c`. Recorded as
  an open item in `docs/STATE.md` rather than fixed inside a rename.

## 2026-10-28: E15 -- a second policy family through the same harness

- **Why:** every measurement so far came from one checkpoint, so "the harness is general" was
  an argument, not a result. `lerobot/vqbet_pusht` (Apache-2.0: behaviour transformer over
  discretized action tokens, `n_obs_steps=5`, `action_chunk_size=5`, 37.5 M parameters) is a
  different family from Diffusion Policy and was plugged in behind the same `PolicyLike`.
- **Code:** `models/pretrained_vqbet.py` (drops the 2024 checkpoint's stale `mlp_hidden_dim`
  before `draccus` parses the config; applies the six normalization buffers the modern library
  loads as unexpected keys) and a dispatch on the checkpoint's own `config.json` `type` in
  `load_policy`. Nothing in `eval/`, `deploy/`, the runners or the generators changed, which is
  the point of the experiment.
- **Measured:** 54.0 % [40.0 %, 68.0 %] success and 0.851 [0.790, 0.905] coverage over 50
  episodes on seeds 100000-100049; the checkpoint's published block is 57.0 % over 500 episodes
  and 64 % on exactly those 50 seeds, with 29/50 paired episodes agreeing. Chunk latency 9.2 ms
  eager at batch 1 (199 MiB peak VRAM), i.e. 1.83 ms per control step against the diffusion
  checkpoint's 9.19. The control loop runs it unchanged: 6 loop rows, zero deadline misses
  (`results/vqbet_loop.jsonl`).
- **Correction, same day: the same-seed comparison is paired, and there is no resolved gap.**
  The first version reported a "10 pp success gap to the published same-seed block and a 4 pp
  coverage gap" as unexplained. Both parts were wrong. The checkpoint ships `eval_info.json`
  with one row per episode, so the comparison can be paired: 27/50 here against 32/50 there on
  the 50 shared seeds, 29 agreeing, 8 succeeding only here and 13 only there, exact McNemar
  p = 0.38. And gym-pusht reports `reward = clip(coverage / 0.95, 0, 1)`, so the published
  `max_reward` column is coverage divided by 0.95 -- the "4 pp coverage gap" was that scaling.
  Rescaled, coverage is 0.851 here against 0.842 there. `scripts/vqbet_published.py` pins the
  published block to `results/vqbet_published.jsonl` so the test regenerates offline (it is
  committed rather than fetched, so `just tables` stays network-free).
- **Also:** `docs/ARCHITECTURE.md` (one page, code as built), the README subtitle and
  `docs/MODEL_CARD.md` state the deployment-cost framing, four unit tests for the config
  filtering, and a `just eval-vqbet` recipe.
- Tests: 61 CPU unit tests (from 57). `just lint`, `just types`, `just test` green; tables,
  figures and README regenerate.

## 2026-10-29: both paired tests at their real sample size

- **VQ-BeT, 50 -> 200 shared seeds.** The 50-episode block could not order this repo's run
  against the checkpoint's own block (8 vs 13 discordant, p = 0.38). The block was grown to
  seeds 100000-100199: 108/200 (54.0 % [47.0 %, 61.0 %]) against the checkpoint's 113/200
  (56.5 %) on the same seeds, 107 agreeing, 44 succeeding only here and 49 only there, exact
  McNemar p = 0.68 -- not distinguished. Mean coverage 0.818 against 0.780, so this repo
  reaches the goal slightly further on the episodes it wins. `just eval-vqbet` now defaults to
  200 episodes, and `results/vqbet.jsonl` is that block (the earlier 50 rows were re-run rather
  than kept, so the config id moved from `vqbet_pusht_n50` to `vqbet_pusht_n200`).
- **Diffusion gets the same treatment.** E0 has always run on seeds 10000-10049, which is
  outside `lerobot/diffusion_pusht`'s own block (1000-1499), so that comparison could only be
  aggregate-to-aggregate. `just eval-dp-published` re-ran 50 episodes on seeds 1000-1049
  (`results/e0_published_seeds.jsonl`): 32/50 (64.0 %) against the checkpoint's 37/50 (74.0 %),
  27 agreeing, 9 vs 14 discordant, exact McNemar p = 0.40; coverage 0.914 against 0.913.
- **A units error corrected in E0 too.** The model card's 0.955 is `max_reward`, i.e.
  coverage / 0.95, so the coverage it implies is 0.907 -- every place that compared our 0.924
  coverage against "0.955" was comparing two different quantities. Fixed in the README,
  `docs/00`, `docs/04`, `docs/06`, `docs/MODEL_CARD`, `docs/RESUME`, the headline generator
  (`src/policybench/eval/report.py`) and the E0 figure's reference line.
- **Code:** `scripts/vqbet_published.py` -> `scripts/published_blocks.py` (`just
  published-blocks`), which pins both blocks to `results/{vqbet,diffusion}_published.jsonl`.
  `make_tables.paired_note` is per-family, the "Published" column is computed from the pinned
  block instead of a hand-typed constant, and the note derives its verdict from p, so the table
  cannot claim a gap the data does not support.
- Tests: 61 CPU unit tests, lint and mypy strict green after the change.

## 2026-10-29 (later): the diffusion paired test at 200 seeds, and a VQ-BeT replication

- **The 50-seed diffusion "not distinguished" was underpowered.** `just eval-dp-published` was
  extended to 200 episodes on the checkpoint's own seeds (1000-1199, still inside its 1000-1499
  block). Result: 104/200 (52.0 % [45.0 %, 58.5 %]) against the checkpoint's 138/200 (69.0 %),
  106 agreeing, 30 succeeding only here and 64 only there, exact McNemar p < 0.001. The first 50
  seeds of that same run reproduce the earlier block exactly (32/50), so this is not drift: at
  five times the sample the gap is real.
- **What the gap is made of.** Our coverage is 0.895 [0.872, 0.917]; theirs reports 0.915, which
  is a **censored lower bound** (next bullet), so the two means cannot show a 2 pp deficit. On the
  32 episodes both runs fail -- the only like-for-like coverage comparison -- it is 0.850 against
  0.832. Success is a hard `coverage > 0.95` threshold, and the outcomes differ there: 64 episodes
  land in [0.90, 0.95), where this repo crosses the threshold 0 times and the checkpoint 42. No
  root cause is established, so it is recorded as open rather than explained.
- **Correction: the published coverage column is clipped.** gym-pusht computes
  `reward = clip(coverage / 0.95, 0, 1)`, so all 327 of the diffusion block's successes (138 of
  them on our 200 seeds) report exactly 0.95 whatever their true coverage; VQ-BeT's block is the
  same (285/285). Their mean is a lower bound, and the earlier "coverage reproduces (0.895 against
  0.915)" reading in this entry was not supported. `published_blocks.py` now writes
  `coverage_censored`, `paired_note` says "lower bound", and the bin table averages their failures
  only, printing `-` where a bin has none.
- **Consequence for E0's headline.** The project no longer claims it reproduces the checkpoint's
  headline: success is 52.0 % against 69.0 %, and coverage is 0.895 against a censored bound of
  0.915 (the card's 0.907 is the same kind of bound). `docs/00` moves the criterion from PASS to
  PARTIAL, and the README, `docs/04`, `docs/STATE`, `docs/MODEL_CARD` and `docs/RESUME` say so.
- **VQ-BeT replicates.** A disjoint 200-seed block (`just eval-vqbet-replication`, seeds
  100200-100399, `results/vqbet_replication.jsonl`) gives 108/200 (54.0 %) against the first
  block's 108/200, two-sided Fisher exact p = 1.00; coverage 0.777 against 0.818. The success
  rate is a property of the policy, not of the seed block.
- **Diagnosis: neither action scaling nor coverage calibration.** Their coverage is censored at
  0.95 (all 138 shared-seed successes report exactly 0.95), so their mean is a bound. On the 32
  episodes both runs fail -- the only uncensored comparison -- it is 0.850 here against 0.832,
  which rules out a multiplicative action-scale error and a global normalization offset. The
  checkpoint's own DDPM@100 on the same 50 seeds (`just eval-dp-published-ddpm`,
  `results/e0_published_seeds_ddpm.jsonl`) gives 30/50 (60.0 %) against their 37/50 (74.0 %),
  p = 0.14 -- no better than DDIM@10's 32/50 and ten times slower, so truncated sampling is not
  it either. What is left is an upper-tail shift: 42 of the 64 episodes in [0.90, 0.95) are
  successes for the checkpoint, and +0.02 coverage would flip 35 of the 64. Source not
  identified.
- **Code:** `make_tables.replication_note` computes the independent-block test and `paired_note`
  prints `p < 0.001` instead of rounding to `0.00`; both notes carry the seed range, so a
  200-episode note cannot be mistaken for a 50-episode table row. Five unit tests for the Fisher
  helper and the replication wording (`tests/unit/test_tables.py`), plus `pythonpath = ["."]` in
  the pytest config so tests can import `scripts.make_tables`.
- **Housekeeping:** the two run logs a previous `git add -A` swept in are removed and
  `results/logs/` is ignored.
- Tests: 66 CPU unit tests (from 61).

## 2026-10-29 (later): the gap is bounded, not explained

- **The planned discriminator does not run, and that is the finding.** Running the checkpoint's
  own `lerobot-eval` here fails twice. `av` 19.0.1 breaks the import (`module 'av' has no
  attribute 'option'`; the library caps `av<16`); pinning `av` 15.1.0 locally gets past that and
  then `make_pre_post_processors` raises `ProcessorMigrationError` -- the 2024 checkpoint ships no
  `policy_preprocessor.json`, which is the exact mismatch `models/pretrained.py` bridges. Their
  eval path cannot regenerate their published block in this library generation, so the 69 %
  reference is a pinned artifact rather than a run this host can repeat. The `av` pin was a local
  experiment: `uv.lock` is untouched and `uv sync` restores 19.0.1.
- **Every mechanism the audit could reach is excluded.** The eight normalization buffers have the
  right shapes (means/stds are `(3,1,1)`, so the subtraction is per channel) and the config's own
  MEAN_STD/MIN_MAX mapping; `crop_shape=[84,84]` is applied inside `DiffusionRgbEncoder.forward`
  so both paths get it; `generate_actions` already returns indices 1-8 and the rollout consumes
  0-7 of that window, so there is no one-step offset; gym-pusht renders at 512 and resizes, so
  observation size is not a state difference; and `select_action`'s 2-frame queue pads exactly
  the way the rollout deque does. Together with DDPM@100 at p = 0.14 that leaves nothing in the
  adapter or the loop to point at.
- Recorded as unexplained in `docs/04` section 4 with the exclusion list, and in `docs/STATE`'s
  open items, rather than closed with a guess.

## 2026-10-30: the noise stream is out, and an adapter-free control exists

- **Same seeds, different Gaussian: 45.5 % against 52.0 %.** `just eval-dp-noise` re-ran the 200
  published seeds (1000-1199) with `--noise-seed-offset 500000`, which leaves every environment
  seed untouched and shifts only the sampler's per-episode initial noise
  (`results/e0_published_seeds_noise2.jsonl`). Result: 91/200 (45.5 %) against 104/200 (52.0 %),
  exact McNemar p = 0.21, Fisher p = 0.23; coverage 0.884 against 0.895. Per-seed agreement is
  only 109/200 (54.5 %), with 52 baseline-only and 39 offset-only successes -- so the initial
  noise does decide individual episodes, and it is the loosest joint in the pipeline -- but it
  does not move the rate, and 69 % lies outside both blocks (23 vs 70 discordant against the
  checkpoint, p < 0.001). The one component of the comparison nobody had controlled is now
  controlled, and it is not the explanation: whatever causes the gap has to survive both streams.
- **`rollout.evaluate(noise_seed_offset=...)`** defaults to 0, so every earlier row keeps its
  exact number. Rows now carry `noise_seed` and `noise_seed_offset`, so two blocks over the same
  seeds cannot be mistaken for a resample of one another.
- **The adapter-free control now runs.** The earlier attempt recorded two blockers and stopped
  there; both are now cleared, and neither by changing what is measured. The checkpoint ships no
  processor files, so `scripts/lerobot_migrate.py` runs lerobot's own
  `migrate_policy_normalization` and finishes the model save it cannot complete (draccus fails to
  encode this checkpoint's `crop_shape: [84, 84]`, leaving a 0-byte `config.json`). The migrated
  directory loads through `DiffusionPolicy.from_pretrained` at 262.7 M parameters with the
  published `n_action_steps=8` / `horizon=16`. `scripts/lerobot_control.sh` (and `just
  lerobot-control`) then runs `lerobot-eval` in an isolated `.venv-lerobot` -- lerobot 0.6.1,
  av 15.1.0, torch 2.14.1+cu130, built by `scripts/build_lerobot_env.sh` -- on seeds 1000-1199,
  with `use_async_envs=false` because the async vector env asks for a `forkserver` context that
  does not exist on Windows. The project's own environment is never touched by any of it.
- **The control reproduces the checkpoint: 67.0 % against 69.0 %.** `just lerobot-control` on
  seeds 1000-1199 gives 134/200 (67.0 %), mean coverage 0.885, 2042 s. Against the checkpoint's
  block on the same seeds (138/200, 69.0 %) Fisher exact p = 0.75, so their published number is
  reproducible on this host; the 2026-10-29 entry that called it "a pinned artifact of a
  different library generation" is superseded by this run. Against this repo's 104/200 (52.0 %)
  on the same episodes, Fisher p = 0.003. **The 15 pp shortfall is therefore in this repo's path,
  not in the reference and not in the seeds** -- but it is not yet assigned to a stage: the
  control runs the checkpoint's own DDPM@100 while E0's 52 % is DDIM@10, and the existing
  50-seed DDPM@100 row (30/50 against 37/50, p = 0.14) is too small to split them. The decisive
  cell is this repo's adapter at DDPM@100 over all 200 seeds, and it is not run yet.
## 2026-10-30 (later): the stage closes with one source per verdict

- **The README's reproduction row was wrong, and the generator caused it.** It read "Yes:
  64.0 %" because `eval/report.py` computed the verdict from the unpaired 50-episode E0 block;
  `docs/00` had said PARTIAL since 2026-10-29 and the paired 200-seed block says p < 0.001.
  `headline_rows` now takes both the paired block and the checkpoint's own block and renders
  the computed verdict, and the unpaired branch says "unpaired only" plus the command that
  would pair it. This is the same class of bug the tables-fresh CI job exists to prevent: a
  claim regenerated from the weakest row.
- **McNemar and Fisher exact moved to `policybench/eval/stats.py`**, so `mypy --strict` covers
  them and the table generator and the README compute a p-value once. `scripts/make_tables.py`
  imports them under the old names; its tests are unchanged.
- **`rollout.evaluate` flushes to `<out>.partial` per shard** and appends only when every shard
  finishes, so a crashed multi-hour block loses one shard and a partial block can never be read
  as a complete one. `results/*.partial` is ignored.
- **The append-style recipes prune their own rows first** (`eval-dp-published`, `eval-dp-noise`,
  `eval-dp-published-ddpm`, `eval-vqbet-replication`). Without that, a re-run duplicated seeds
  and silently inflated `n` in the paired tests.
- **`just verify`** = the CI commands plus the generated-artifact diff, so a commit can be
  checked without waiting for CI. **`tests/unit/test_lerobot_migrate.py`** pins the migration's
  key filter. `docs/08_next_stage_options.md` records the next stage as options with tradeoffs.
- **Files:** `results/e0_published_seeds_noise2.jsonl` and
  `results/lerobot_control/eval_info.json` are now committed (they are cited above).
  `results/lerobot_control/eval_info.json` is the control's record. This lerobot
  version writes only aggregates (`per_task`, `per_group`, `overall`) -- unlike the checkpoint's
  own 2024 `eval_info.json`, which carried one row per episode -- so the control can be compared
  to this repo in aggregate only, not per seed. The episode videos under
  `results/lerobot_control/videos/` are run artifacts and are not part of the result.

## 2026-10-30: next-stage options pass (three taken, three declined, one gated)

- **`docs/09_option_findings.md`** records the risk-ordered pass over `docs/08`: cost, outcome and
  verdict per option, one commit each on `exp/options`.
- **C found a defect.** `rollout.evaluate` sharded one contiguous chunk per worker and flushed only
  at chunk end, so a single-worker block flushed nothing until its last episode (observed live: an
  empty `.partial` for 25 minutes of a 3 h run). Seeds are now `workers * 8` interleaved chunks,
  rows flush per chunk, a re-run of the same `config_id` resumes from `.partial`, and the append is
  sorted. Tests: `tests/unit/test_eval_resume.py`.
- **B makes the adapter-free control pairable.** `policybench/eval/lerobot_rows.py` +
  `scripts/lerobot_control_rows.py` (and `just lerobot-control-rows`) wrap lerobot's `eval_policy`
  to capture the per-episode records it currently discards, in this repo's schema. Verified at 4
  episodes; the 200-episode control run waits until A lands so both arms are paired together.
- **F answers multiplicity while it is cheap.** `docs/04` §1 names the six quoted tests, the
  uncorrected alpha, the Bonferroni bar and the rule for blocks that grew after a first look.
- **A's 50-seed arm is not evidence about the sampler.** 30/50 at DDPM@100 vs 32/50 at DDIM@10 on
  the same seeds (p = 0.83), but those seeds are the block where our DDIM is already not separated
  from the checkpoint (p = 0.40). The 200-seed gap is concentrated in seeds 1100-1149 (18/50 vs
  34/50, p < 0.001; post-hoc, labelled). The 200-seed DDPM@100 block now runs one worker (~3 h).
- **A landed, and the gap splits ~7 / ~8.** The 200-seed DDPM@100 block (`just
  eval-dp-published-ddpm`, 2 h 58 m on one worker, `results/e0_published_seeds_ddpm.jsonl`) is
  118/200 (59.0 %) against DDIM@10's 104/200 (52.0 %) on the same episodes: 37/51 discordant, exact
  McNemar p = 0.17, coverage flat (0.893 vs 0.895), 4.4x the per-chunk cost (1819 ms vs 417 ms).
  Neither pre-registered branch fires. The sampler is a partial cause: +7 pp overall and +24 pp on
  the 1100-1149 band (36.0 -> 60.0 %, p = 0.017, outside the 0.0125 four-look bar, against the
  checkpoint's 68.0 % there), while 1150-1199 moves the other way (29 -> 23, p = 0.26). So ~8-10 pp
  is not sampling, E0 stays PARTIAL, and H is the only hypothesis left worth running.
- Declined with reasons in `docs/09`: D (second task), E (ROS 2 -- its measurable half is already
  `bench_serve` and the paced loop, and Docker is absent here), G (packaging). H is now the next
  option, not a gated one. Owed before it: the 200-episode `just lerobot-control-rows` run that
  makes the adapter-free control paired by seed instead of aggregate (134 vs 138 by Fisher).

## 2026-10-30: option H -- audit, amendment, and the seam behind a default-off flag

- **`docs/10_processor_adoption.md`** carries the amendment to rule 4 (four conditions, nothing
  adopted), an audit of the H decision, the adoption design and the runbook.
- **The audit re-sized the target and lowered the prior.** The like-for-like residual is 8 pp
  (our DDPM@100 118/200 against the adapter-free control's 134/200), not 15; coverage is flat at
  0.884-0.895 across arms spanning 45.5 % to 69.0 % success, which argues against an arithmetic
  defect; and 200 seeds cannot resolve 8 pp (the DDPM arm's 7 pp had 88 discordant pairs,
  p = 0.17), so the episode arm is 600 seeds and the deterministic probe is the decision rule.
- **`models/normalization.py`** is the seam: `BuffersNorm` (default, unchanged arithmetic) and
  `ProcessorsNorm` (lerobot's pipelines, opt-in), plus `flatten_history`/`restore_history`, the
  device-step override and the `config_id` guard. `normalization="processors"` threads through
  `load_policy`, `evaluate` and `--normalization`; the VQ-BeT adapter refuses it.
- **`scripts/processor_equiv.py`** (`just processor-equiv`) is step 0: migrated weights against
  the published ones, then images, state, actions, the library's own round trip and history
  ordering, bar 1e-6, result in `results/processor_equiv.json`, non-zero exit on a difference.
- **`scripts/paired_test.py`** (`just paired`) is the one implementation of the paired 2x2 that
  both the control comparison and H's episode arm need, so no p-value is typed by hand.
- Tests: `tests/unit/test_normalization_protocol.py` (8) pin the seam, the history ordering and
  the transposed-reshape trap; 92 unit tests total. **No measurement was taken with any of it** --
  no row in `results/` came from the processor path, and step 0 has not been run.

## 2026-10-30: three probes, three nulls -- H declined, the residual is noise-sized

- **`just processor-equiv` passes at 1e-6** (`results/processor_equiv.json`, 20 real frames): the
  checkpoint's eight normalization buffers and `lerobot`'s migrated processor pipelines agree --
  weights 0.000e+00, images 2.384e-07 (one fp32 ulp), state 0.0, actions 0.0, history ordering
  0.0. Option H's pre-registered rule stops there: the mapping is not the defect, nothing is
  adopted, `normalization="processors"` stays default-off and unmeasured. The seam and its 9 unit
  tests stay because they are cheap and default-off.
- **The paired control is null at the checkpoint's sampler.** `just lerobot-control-rows` wrote
  200 rows (seeds 1000-1199): control 126/200 (63.0 %) against our DDPM@100 118/200 -- 78 both,
  40 / 48 discordant, exact McNemar p = 0.46. Only DDIM@10 separates (104 vs 126, p = 0.03), which
  is the sampler, not the adapter.
- **The control's own reruns disagree by 4 pp.** The same command scored 134/200 earlier with
  identical settings, so `results/lerobot_control/eval_info.json` now holds the second run (126)
  and the first survives only in this file. That spread is the same size as the residual, which is
  why the like-for-like null is the reading to keep.
- **`just env-equiv` shows the env wrapper is identical** (`results/env_equiv.json`): one seed,
  fixed 300-action sequence, coverage and success traces agree to 0.0, same termination step,
  bit-identical first frame. R5's environment half is closed.
- **New tooling:** `scripts/paired_test.py` + `just paired` (one implementation of the paired 2x2),
  `scripts/env_equiv.py` + `just env-equiv`, `scripts/processor_equiv.py` + `just processor-equiv`.
  Docs: `docs/04` §4 carries all three entries, `docs/06` records the R5/R14 outcomes, `docs/10`
  carries the amendment and the design, `docs/09` records H as measured and declined.

## 2026-10-30: the control is repeated -- spread 1.0 pp, so the residual is ~4 pp, not 8

- **Four more runs of the identical control command** (`scripts/control_repeats.sh`, r2-r5, one
  worker, ~38 min each, `results/lerobot_control/repeats/`), so the adapter-free control is now
  five runs of the row-writing loop: 126, 128, 125, 123, 127 of 200 -- **62.9 %, sd 1.0 pp**, range
  61.5-64.0 %, t interval over runs [61.7 %, 64.1 %].
- **`scripts/control_spread.py` (`just control-spread`) states the null with an interval:** each
  run, the control's own across-run interval, and the like-for-like gap resampled over episodes
  *and* runs (`stats.paired_delta_ci`). Result: **+3.9 pp, 95 % [-5.5, +13.0] pp** -- it spans zero
  (`results/lerobot_control/spread.json`).
- **The "4 pp run-to-run spread" was not spread, and the 8 pp residual goes with it.** The
  134/200 (67.0 %) run came from `just lerobot-control`, the *aggregate* entry point, in 2042 s,
  while every row-writing run takes ~2312 s; it sits ~4 sd outside the five-run band. Reading 134
  and 126 as one command disagreeing with itself conflated two entry points into the same loop, and
  the ~8 pp computed against the 134 run is superseded by +3.9 pp against the repeated mean. The
  path difference is recorded as *unexplained*: a repeat of the aggregate path is running before
  anything is claimed about it.
- Updated in place: `docs/04` §4 (amended paired-control entry plus a new repeats entry), `docs/09`
  (H's measured-and-declined block and the closing read), `docs/10` §0.1/§0.2/§2/§5, `docs/STATE`,
  the README template (and the generated README) and the model card. No code path changed.

## 2026-10-30: the blocks are audited, and 67.0 % turns out to be a draw

- **`just power-audit`** (`scripts/power_audit.py`, `stats.resolvable_effect`, `results/power_audit.json`)
  states the smallest effect each paired block can resolve. **Every 200-episode block in this repo
  resolves 13.1-13.5 pp at 80 % power**, because any two of these runs disagree on 44-47 % of
  episodes. Four quoted nulls observe less than that -- sampler 7 pp, noise stream 6.5 pp,
  like-for-like 4 pp, VQ-BeT 2.5 pp -- so they are coarse nulls, and `docs/04` §1 now says a reader
  may not read them as "no difference".
- **The aggregate control entry point was repeated: 134/200 then 126/200 (63.5 %)**
  (`results/lerobot_control/eval_info_plain_200.json`), the second inside the row writer's
  61.5-64.0 % band. So the 67.0 % was one high draw, not a property of that entry point: seven runs
  of the loop span 61.5-67.0 % around a 63.5 % mean, the paired analysis stays on the five runs that
  wrote rows, and the `Why the two paths differ` question in `docs/04` closes as "they don't".
- **Tree cleaned.** `tmp_det/` (an earlier determinism probe's working copy) and
  `results/lerobot_control/repeats/eval_info.json` (a byproduct; r5's rows are the record) removed.
  `.gitignore` gained `/tmp_*/` and per-output-dir `eval_info.json`, and lost the redundant
  `results/lerobot_control/videos/` line that `**/videos/` already matched. `git check-ignore` was
  run both ways: the scratch and byproduct paths are ignored, and every tracked row and the
  tracked aggregate stay unshadowed.
- `results/lerobot_control/plain_r2/` was deleted once its aggregate had a named copy
  (`eval_info_plain_200.json`): the output dir was scratch, the JSON is the record.

## 2026-10-30: tables carry their resolution, and VQ-BeT runs to the reference's limit

- **The generated tables print a `Resolves pp` column** (`scripts/make_tables.py`: one
  `paired_resolution` helper feeding both the column and the paired note), so a "not distinguished"
  cell cannot be read as "no difference": the generality table shows 13.5 pp for diffusion and
  8.6 pp for VQ-BeT beside the comparison each qualifies, and the note says the null is only that
  coarse. Four tests pin it, including identical arms, where the formula would divide by zero.
- **VQ-BeT re-run on the checkpoint's whole published block** (`just eval-vqbet 500 100000`, 500
  episodes, ~6 min): 274/500 (54.8 % [50.6 %, 59.2 %]) against 285/500 (57.0 %) on the same seeds,
  261 agreeing, 114 / 125 discordant, exact McNemar p = 0.52, mean coverage 0.798 against 0.776.
  Resolution improves from 13.4 pp to **8.6 pp**, which is the ceiling: 500 is every episode the
  published reference has.
- **2.5 pp is not reachable against this checkpoint.** At 46.5 % discordance it needs 5,838 paired
  episodes, so the only route is an adapter-free VQ-BeT reference this repo runs itself (~1.75
  GPU-h per arm at 201 episodes per 3.6 min, measured). Not run. `docs/04` §3's E15 claim and its
  falsifier now say that instead of asking for "a further 200-seed block".

## 2026-10-30: `docs/11` surveys the models this wrapper could carry

- **Added `docs/11_sota_models.md`**: the landscape of open-weight policies as of today, what each
  one costs to adopt behind `PolicyLike`, and the two protocol seams that do not exist yet (a
  language slot and N camera views). No code, no rows, no claims moved -- it is a research page with
  sources tagged [V]/[S]/[U] and a "false if" condition per option.
- **The finding that changes scope:** the pinned `lerobot` 0.6.1 already ships `pi0`, `pi05`,
  `pi0_fast`, `smolvla`, `groot`, `wall_x`, `xvla`, `molmoact2` and `envs/libero.py`, so the model
  zoo costs no new dependency and no new policy code -- one adapter keyed off the checkpoint's own
  `config.json` type covers four of them. Their configs (chunk 50/40, 32- or 132-wide padded
  actions, 10 flow steps) are the same shape the existing protocol already expresses.
- **The finding that constrains it:** π0 needs 14 GB at inference (openpi asks for >8 GB, `lerobot`
  documents 14 GB) and this host has 8 GB, so SmolVLA is the only VLA measurable here, and it has no
  PushT checkpoint -- the VLA arm needs a different task (LIBERO), which is the same environment
  archaeology that made option D expensive. Recorded as a decision for the owner rather than a
  plan, because it changes the README's one-consumer-GPU promise.

## 2026-10-31: the observation contract gains named cameras and a language slot

- **`models/observation.py`** (`ObservationSpec`, `spec_for`, `check_cameras`, `check_frames`,
  `check_instruction`): the contract was implicit and single-view -- `obs` had exactly `image`
  and `state`, and nothing could carry a task string -- so no VLA could be expressed at all.
  Views are now **named** and validated against what the policy declares in an optional
  `camera_names`; the instruction travels as the `instruction=` call argument, because `obs` is
  `dict[str, Tensor]` and a string cannot live there. It is called `instruction` rather than
  `task` because rows already use `task: "pusht"` for task identity.
- **Default-off, by construction.** Both attributes are optional, so `("image",)` +
  unconditional is the default every existing adapter, row and test already satisfies: nothing
  was re-measured and no row's meaning changed. The diffusion and VQ-BeT adapters were not
  edited; they absorb `instruction` through `**sampler_kw`.
- **Refusals, not guesses, at all three boundaries.** Rollout and the control loop refuse
  declared views the PushT env cannot supply; the HTTP server takes `cameras: {name: frames}`
  beside the legacy `image`, returns 400 when a view is missing or a required instruction is
  blank, and advertises `cameras` / `instruction_required` on `GET /health`. Feeding one frame
  into two slots is the failure `docs/10` exists to prevent, so it is a hard error: a checkpoint
  fed the same view twice returns plausible actions and a success rate that means nothing.
- **Tests:** new `tests/unit/test_observation_contract.py` (15 cases, CPU) and five socket-level
  cases in `tests/unit/test_deploy_server.py`, where the two views carry distinct pixel values
  so a duplicated or swapped view cannot pass. Rationale and scope: `docs/11` section 3.

## 2026-11-01: RTC against the cross-fade has no like-for-like form, and that is the finding

- **The comparison was resolved by reading the shipped code, not by running it.** `lerobot` 0.6.1
  ships RTC under `policies/rtc/` (there is no `lerobot/rtc/`): `RTCProcessor.denoise_step` wraps a
  *velocity* callable, derives `x1_t = x_t - time * v_t`, and returns
  `v_t - min(1/tau, max_guidance_weight) * (prev_chunk_left_over - x1_t) * weights` under
  `torch.enable_grad()`, prefix from `get_prefix_weights` (`LINEAR`, `execution_horizon` 10,
  `max_guidance_weight` 10.0). Its only entry point is `policies/common/flow_matching.py::
  euler_integrate`, imported by `eo1`, `multi_task_dit`, `pi0`, `pi05`, `smolvla`; `rtc_config` is
  declared only by `evo1`, `groot`, `molmoact2`, `pi0`, `pi0_fast`, `pi05`, `smolvla`. **The
  `policies/diffusion/` module has no RTC reference at all** [V, grep 2026-11-01].
- **So there is nothing to compare on the same episodes.** The 60 `blend-r2` rows are DDIM
  epsilon-prediction on `lerobot/diffusion_pusht`, which has no velocity field for the guidance
  term; our three `predict_chunk` implementations are closed `@torch.no_grad()` samplers with no
  per-step hook, so they cannot wrap an Euler step the way RTC needs; and the only flow
  checkpoints on this host are
  `e1_flow_s0..s2`, the ablation models D11 retired at 4-7 % success. A like-for-like row needs a
  second policy (SmolVLA on LIBERO), which is `docs/11` V3 and needs V4's environment work first.
- **Recorded instead of forced:** `docs/05` section 2 and `docs/11` section 4.3 now carry the
  obstruction with its evidence, `docs/11` no longer promises that a same-episodes run would
  upgrade the cross-fade paragraph, and its V3 row names the second policy as a prerequisite. No
  GPU time spent, no row written, and the cross-fade's own result (coverage up, success flat
  against temporal ensembling) is unchanged.

## 2026-11-01: RTC on SmolVLA scoped, and the cost column tagged measured vs estimated

- **The RTC loop is reachable on SmolVLA without new sampler code.** `SmolVLAPolicy.predict_action_chunk`
  forwards `prev_chunk_left_over`, `inference_delay` and `execution_horizon` into
  `euler_integrate(..., rtc_enabled=...)`, and `select_action` asserts RTC is off, so the blessed
  entry point is the whole-chunk call that `PolicyLike` already is [V]. `rtc_config` defaults to
  `None` and `init_rtc_processor()` re-attaches to a built model, so enabling it is a construction
  change, not a checkpoint edit. The obstructions recorded the same day are about *our* adapters
  driving *our* diffusion samplers; `docs/05` section 2 now says so explicitly.
- **What blocks V3 is the environment, not the strategy.** `HuggingFaceVLA/smolvla_libero` exists
  (apache-2.0, base `lerobot/smolvla_base`) but its card is the generic lerobot stub with no eval
  table, so the acceptance cell is `lerobot`'s docs figure (~81.5 % spatial) [S]; the checkpoint wants
  **two** 3x256x256 views (state 8, action 7) and a 48-token language processor, while `transformers`
  (a lerobot extra) and `robosuite` / `mujoco` / `libero` / `bddl` are all absent here, and PyPI
  `libero` is sdist-only 0.1.1 [V].
- **The published recipe has no chunk left over to guide.** Its `config.json` sets
  `n_action_steps: 1` against `chunk_size` 50 [V], so RTC's premise is absent until the arm raises the
  cadence -- and raising it departs from the published recipe, which means the naive baseline must be
  re-measured at that cadence rather than quoted from the cell.
- **The block is sized, the GPU-hours are not.** `stats.resolvable_effect` gives 5.6 pp at 500 paired
  episodes with 20 % discordance (8.3 pp at 44 %), so LIBERO-Spatial's own 10x50 is sufficient; but
  SmolVLA's per-chunk cost has no anchor on this GPU (E9's 181-417 ms is a 263 M diffusion policy) and
  the 13-chunks-per-episode unit cost was measured on 300-step PushT episodes [V, computed today].
- **`docs/11` section 6's cost column is annotated.** Two cells are measurements (V1's protocol half at
  zero GPU-hours; V4's absent dependencies), the rest are person-day estimates, and the measured unit
  economics -- 13 chunks/episode, 181 / 417 / 1819 ms per chunk, a 200-episode block at 2 h 58 m, VQ-BeT
  500 episodes in ~6 min -- are listed with the explicit warning that none of it transfers to LIBERO.
  Section 4.4 carries the full checklist. No code, no rows, no GPU time.

## 2026-11-01: V4 priced -- the port is the expensive path

- **The port was walked, and it is 2-4 days rather than a shim.** Five walls, each observed: the
  removed `single_arm_env` module (a ~30-line shim clears it), `mount_types` -> `base_types`, the
  removed `robots.single_arm.SingleArm`, the removed `load_controller_config`, and `lite_physics`
  defaulting to `True` where 1.4.0 had no such flag. The third is architectural: a naive
  `SingleArm = FixedBaseRobot` clears every import and then raises `KeyError: 'MountedMountedPanda'`,
  because 1.4.0's `SingleArm` is a full robot implementation and 1.5.2 split that layer into `Robot`
  + `FixedBaseRobot` + composite controllers over concrete subclasses. LIBERO dispatches through that
  generic class, so re-expressing its two robot models means translating controller configs inside a
  third party's environment, at the seam where the physics live. `docs/11` 4.5.
- **The Linux path is 0.5 setup days and no extra hardware.** `robosuite==1.4.0` is what LIBERO
  declares, so all five walls disappear. The deciding variable is the OS, not the card: the measured
  memory anchors (1189 MiB fp32 / 710 MiB bf16 for a 262.7 M policy at N=10) put a 450 M policy
  inside 8 GB [E], so this laptop is excluded by Windows rather than by memory. The recommendation is
  to pick the platform, not to port. `docs/11` 5.1.
- **The SmolVLA block now has GPU-hours.** 500 episodes x 28 chunks at 150 / 250 / 700 ms is 1.4 /
  1.75 / 3.5 h per arm; three arms plus the naive re-baseline and a pilot is **6-9 h central, ~25 h
  pessimistic**. The per-chunk band is still an extrapolation, and one pilot episode is what replaces
  it. Section 6's V3 and V4 cost cells now cite computed figures instead of blanks.

## 2026-11-01: the VLA adapter's leftover contract decided on CPU, and it passes

- **What was undecided, now is.** RTC's guidance acts on `x_t` inside the flow loop -- the
  checkpoint's space (normalized over 7 dims, zero-padded to 32) -- while `deploy/control_loop.py`
  holds raw env units and `lerobot` slices the pad off and unnormalizes before returning. So the
  adapter must rebuild the leftover, and rebuilding it wrongly is silent: plausible actions, a
  broken success rate, the defect shape `docs/10` was written about. `scripts/smolvla_leftover_probe.py`
  (`just leftover-probe`) decides it on CPU, with no checkpoint, no episode, no `transformers` and
  no GPU time.
- **`src/policybench/models/leftover.py` is the seam.** `PaddedActionSpace` owns
  normalize-then-pad and slice-then-unnormalize; one affine covers both declared normalizations
  (`MIN_MAX` is `MEAN_STD` about `(low + high) / 2` with `(high - low) / 2` as the spread); a
  non-invertible table is refused rather than clamped. The probe imports it, so it measures what
  the adapter will ship instead of a copy of it.
- **Nine identities at 1e-5, eight exactly 0.0** (worst 2.4e-07, the float32 round trip)
  [V, `results/smolvla_leftover_probe.json`]: raw -> model -> raw, model -> raw -> model, the pad
  written as zeros, garbage in dims 7-31 discarded, the guided step equal to its closed form, a
  leftover equal to `x1_t` giving exactly zero guidance, the final Euler step unguided, the
  schedule's tail zero, and a 7-dim leftover guiding identically to the same 7 dims widened with
  our own pad.
- **The gradient inside RTC is the identity -- measured, not assumed.** `denoise_step` detaches
  `x_t` before `requires_grad_(True)`, so the velocity has no graph to `x_t`, `correction == err`,
  and the guided step is `v - clamp(w) * (leftover - x1_t) * weights` to 0.0. The action expert's
  Jacobian never enters, so only the leftover's units and the prefix weights shape guidance.
- **The failure mode is loud.** A leftover equal to `x1_t` gives exactly 0 in model space and 1.896
  (model units) when the same tensor is supplied raw, so a unit error is visible from one synthetic
  step rather than only from a success rate.
- **Two facts that change how the V3 arm reads.** `weights` is `(1, 50, 1)`: the prefix schedule is
  per chunk step, not per action dim, so `execution_horizon` cuts the chunk. And the guidance weight
  is `10.0` (clamped) on the first Euler step and `0.0` on the last -- a fresh leftover acts at full
  strength where the sample is still noise, and the final step is unguided whatever it is.
- **What it does not check is stated in the report itself.** The statistics are synthetic
  (`stats: synthetic`), so a wrong *statistic* would still pass; re-running the nine identities
  against `HuggingFaceVLA/smolvla_libero`'s own postprocessor numbers is the first step once the
  2 GB download lands, and before the first blocked episode.
- **New:** 10 unit tests in `tests/unit/test_leftover_space.py` (CI-safe: no `lerobot`). `just
  verify` green: ruff, ruff format, mypy strict, 141 unit + 1 smoke test, generated tables and
  README byte-identical. No rows, no GPU time.

## 2026-11-01: V4 attempted on this host, and the LIBERO wall is the platform rather than the install

- **Attempted, not estimated.** V4's dependency cost was the last unmeasured cell in `docs/11`
  section 6, so the environment was tried in an isolated env (`.venv-libero`, py3.12, uv, Windows
  11). No policy ran and no row was written; the deliverable is an install-and-render matrix.
- **The parts expected to fail are fine.** `transformers==5.5.4` installs within lerobot's
  `>=5.4.0,<5.6.0`; `mujoco.Renderer` returns a shaded offscreen frame (max 201, mean 33.8) with no
  EGL configured; and `robosuite==1.5.2 + mujoco==3.3.7` yields a real
  `agentview_image (128,128,3) uint8 max=255` from a camera-observation `Lift` env [V].
- **The part expected to be easy is the wall.** LIBERO pins `robosuite==1.4.0`, whose
  `binding_utils.py` loads a bundled `mujoco.dll` through `ctypes.WinDLL`; the sdist ships no Windows
  DLL, so `import robosuite.environments` raises `FileNotFoundError` before an env exists [V].
- **`egl-probe` is what blocks the installers.** `robomimic==0.3.0` and `libero==0.1.1` both depend on
  it, and it builds with `cmake ..; make -j`, which exits non-zero here [V]. `libero --no-deps`
  installs, and LIBERO's source never references `MUJOCO_GL` or `egl` [V, grep]: the EGL requirement
  is metadata, not code.
- **LIBERO gets surprisingly far.** Its wheel ships `bddl_files` and `init_files`, so no dataset
  download is needed; `libero_spatial` loads 10 tasks and task 0's bddl path resolves [V]. It then
  dies on one import: `robosuite.environments.manipulation.single_arm_env`, a module 1.5.2 *removed*
  rather than renamed (`manipulation_env.py` defines `ManipulationEnv(RobotEnv)`), while
  `BDDLBaseDomain(SingleArmEnv)` is LIBERO's entire env layer [V]. `lerobot.envs.libero` imports
  `OffScreenRenderEnv` at module scope, so this repo's wrapper fails at the same line [V]. The first
  import also prompts with `input()` when `~/.libero/config.yaml` is absent, which is an `EOFError`
  for anything unattended [V].
- **Consequence:** V4 cannot be pinned on Windows without owning third-party environment code, so it
  becomes a platform decision (Linux or a rented box) rather than a scoping step. `docs/11` section
  4.4 carries the matrix, section 6's V4 row is now measured, and V3's row says it is gated on that
  decision. `.venv-libero/` and `tmp_libero/` are gitignored scratch; no code, no rows, no GPU time.
