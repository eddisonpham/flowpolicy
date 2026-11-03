# 00. Gap inventory: every missing component, with evidence

This is the list the plans are derived from. A row is a **gap** only if something promised or
implied by `docs/` cannot currently be delivered; a thing that is merely nice to have is recorded
as out of scope at the bottom, not as a gap.

| # | Missing component | Evidence it is missing | Blocks | Plan |
|---|---|---|---|---|
| 1 | An executed Linux run of the SmolVLA pilot | `docs/12_linux_runbook.md` has never been run; `robosuite==1.4.0` cannot load on Windows [V, docs/11 §4.4]. Preconditions 3 and 2-config are green here, precondition 1 and the weights are not | Every VLA number; docs/11 §5.1's 150/250/700 ms per-chunk band | `01` |
| 2 | A generic lerobot adapter | Two hand-written adapters (`models/pretrained.py`, `models/pretrained_vqbet.py`), each re-deriving normalization; dispatch is a branch in `rollout.load_policy` [V]. The observation contract can express a VLA but nothing consumes it | The "wrap real models" claim beyond two families; RTC | `02` |
| 3 | An environment seam | `rollout.py`, `deploy/control_loop.py` and the server tests name `PushTEnv` and `MAX_STEPS` directly; a second task means editing the harness that produced every row [V] | Any second environment (LIBERO, MetaWorld); V2's remote client | `02` |
| 4 | `lerobot` in CI | lerobot pins `torch<2.12`, the project pins 2.14.1, so CI has neither lerobot nor `transformers`; the tests that load a real checkpoint (`test_smolvla_pilot.py::test_the_real_config_loads...`, the sidecar tests' lerobot half) skip there [V] | Regression detection on the adapter path | `05` |
| 5 | A determinism test for eval rows | Per-episode seeding exists and is documented as fixing a real defect [V, `rollout.rollout_episode`], but nothing re-runs an episode and compares the trace | Confidence in every paired test | `05` |
| 6 | Provenance on eval rows | `config_id`, `noise_seed_offset`, `scheduler`, `dtype` are stamped; the git SHA and `uv.lock` hash are recorded for training checkpoints only [V, `trainer.save`] | Re-deriving which code produced a row after a rebase | `05` |
| 7 | A resolution guard that can fail a build | Tables print a resolution column [V, `generated tables`], and `scripts/power_audit.py` flags uninformative nulls, but nothing runs it in `just verify` or CI | The exact mistake this repo has already published and corrected once | `05` |
| 8 | RTC measured against the cross-fade | `rtc_config` exists on SmolVLA [V, lerobot 0.6.1]; `models/leftover.py`'s round trip passes at 2.4e-07 [V, `results/smolvla_leftover_probe.json`]; no arm has been run | docs/05's cross-fade paragraph, V3 | `03` |
| 9 | A serving boundary with deadlines and shutdown | The inference lock and body cap landed 2026-11-02 [V]; there is no per-request deadline, no drain on shutdown, and `steps` is unclamped | The deployment claim's honesty under more than one client | `04` |
| 10 | Packaging: licence, entry points, version | `version = "0.0.1"`, no `LICENSE` file, no `[project.scripts]`; scripts run because `pytest`/`sys.path` puts `src` on the path [V] | Anyone installing rather than cloning | `04` (licence is the owner's call) |
| 11 | A data path that is not PushT-only | `load_zarr` reads the full image array into RAM; one converter exists (`scripts/data/prepare_pusht.py`) [V] | A second task's data | recorded, not planned |

## The two gaps that are not a plan

- **A real robot.** No hardware, no ROS 2 (D7), and the loop's virtual-time mode would have to be
  replaced by a paced one against a real clock and a real safety envelope. Out of scope for this
  stage: it would need a different risk register, not a step.
- **A third policy family.** Generality is a claim about the *harness*, and it is currently
  supported by two families with published cells. A third without a published cell adds code, not
  evidence -- which is exactly the trap `docs/11` §7 names.

## Why these and not others

`docs/11` §6 already prices V1-V6. This inventory does not re-price them; it answers a narrower
question: *of the things the repository promises or implies, which are not true yet?* V5 (port π0's
flow loop) and V6 (train a VLA) stay declined for the reasons recorded there -- they violate the
rules the project is built on, not the budget.
