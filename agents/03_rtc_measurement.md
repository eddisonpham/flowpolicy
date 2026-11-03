# 03. `R1` -- measure RTC against the cross-fade on the same episodes

## Goal

Turn `docs/05`'s cross-fade paragraph and the RTC caveat into one measured comparison:
RTC (upstream's real-time chunking), naive replacement, temporal ensembling, and our cross-fade,
on the same episodes with the same checkpoint.

## Non-goals

- Not a new sampler. `lerobot` ships the RTC processor; the work is wiring, measurement and an
  honest report, not re-deriving guidance.
- Not on the diffusion checkpoint. RTC is flow-only and needs a grad-enabled loop [V, docs/05 §3];
  the diffusion path has no RTC reference at all, so a second policy is a prerequisite, not a
  convenience.
- Not a latency claim about the network. The transport-free loop is the object.

## Prerequisites

1. `L1` completed: a host, `.venv-pilot`, a measured per-chunk cost.
2. `A1` step 3: a policy object that accepts `inference_delay` / `rtc_processor` through
   `predict_action_chunk` [V, `modeling_smolvla.py` 232].
3. The leftover gate green: raw -> model -> raw and its inverse at 2.4e-07 [V,
   `results/smolvla_leftover_probe.json`].
4. The arm size from `just arm-power`, not from a guess: 500 episodes resolves 5.6 pp at 20 %
   discordance [V, `stats.resolvable_effect`], which is only enough if the arms actually disagree
   that often.

## Steps

| # | Work | Acceptance |
|---|---|---|
| 1 | CPU parity test for the loop: a fake flow policy that honours `inference_delay`, run through naive / ensemble / cross-fade / RTC wiring; assert the executed action sequence per arm | new unit test, CI-safe, no lerobot needed |
| 2 | A pilot of **>= 100 episodes per arm** on the GPU: measures discordance *and* the RTC path's per-chunk overhead | `results/rtc_pilot.jsonl`. 50 episodes would leave roughly a +-14 pp interval around a ~20 % discordance, so the pilot's *interval* -- not its point estimate -- is what step 3 consumes |
| 3 | Re-run `just arm-power` at the pilot's **upper** discordance bound | a block size with its resolution printed, not assumed. Upper, because `paired_binary_n` grows with discordance (more episodes change hands, so the paired difference carries more noise) -- sizing at the top of the interval cannot under-power the block |
| 4 | Full block, four arms, paired by seed | exact McNemar per arm pair, each with `resolves_pp` |
| 5 | Report: success **and** latency per arm | RTC's premise is a leftover in flight; a success win with no latency advantage is a different finding and must be written as one |

## Cost

1 day **estimated** for wiring and the CPU test; 6-9 GPU-hours **computed** at §5.1's central band
for a 500-episode block, ~25 pessimistic. Both numbers move once `L1` replaces the band -- which is
why `L1` precedes this plan.

## False if / kill

- RTC does not beat temporal ensembling on success at the same latency -> conditioning is not the
  mechanism, `docs/05` was right, and the cross-fade paragraph keeps its caveat. Record.
- Every arm lands within the block's resolution of every other -> the honest statement is "no
  measurable difference at n", with n and the resolution quoted. Not "RTC works".
- The RTC path cannot be reached without patching `lerobot` -> stop; patching a third party's
  sampler is what rule 4 forbids, and a local fork would make the row unreproducible.

## Sequencing note

This plan is last on purpose. It is the most expensive step, the most dependent, and the one whose
value is highest if `L1`'s band turns out to be far from 250 ms -- a per-chunk cost that already
fits the period makes "a leftover in flight" a much smaller prize.
