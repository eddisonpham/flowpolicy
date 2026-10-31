# 10. Option H: amendment, and the processor-adoption design

Status 2026-10-30: **drafted, not started.** No code, no rows, no dependency change. This page
exists so H can be judged and reverted like the other options in `docs/09`, and so the boundary it
trades away is written down *before* the first line of it exists.

Everything below is a plan with a pre-registered stopping rule. The stopping rule is the important
part: H's cheapest test is a numeric comparison, and if that comparison passes, H is a null and the
adoption should not happen at all.

## 0. Audit of the decision, after the numbers (2026-10-30)

Written once the options batch was in, and it changes two things: what H's *prior* is, and how
big a block would be needed to test it.

### 0.1 Every arm on the same 200 episodes

| Arm | Loop | Sampler | Success | Mean coverage | Source |
|---|---|---|---|---|---|
| this repo (E0) | ours | DDIM@10 | 104/200 = 52.0 % | 0.895 | `results/e0_published_seeds.jsonl` |
| this repo, second noise stream | ours | DDIM@10, offset 500000 | 91/200 = 45.5 % | 0.884 | `results/e0_published_seeds_noise2.jsonl` |
| this repo, checkpoint's sampler | ours | DDPM@100 | 118/200 = 59.0 % | 0.893 | `results/e0_published_seeds_ddpm.jsonl` |
| adapter-free control, aggregate entry point, run twice | `lerobot-eval` | DDPM@100 | 134/200 = 67.0 %, then 126/200 = 63.5 % | 0.885 / 0.944 | `results/lerobot_control/eval_info_plain_200.json`, `docs/04` §4 |
| adapter-free control, paired run | `lerobot-eval` + row writer | DDPM@100 | 126/200 = 63.0 % | 0.885 | `results/lerobot_control/rows.jsonl` |
| adapter-free control, repeats r2-r5 | same | DDPM@100 | 128, 125, 123, 127 / 200 | -- | `results/lerobot_control/repeats/` |
| adapter-free control, five-run mean | same | DDPM@100 | 125.8/200 = 62.9 % (sd 1.0 pp) | -- | `results/lerobot_control/spread.json` |
| the checkpoint's own block | published | DDPM@100 | 138/200 = 69.0 % | 0.915 (censored) | `results/diffusion_published.jsonl` |

The like-for-like cell is **118 vs 126**, one adapter apart at the same sampler on the same 200
environment seeds: +3.9 pp against the control's five-run mean of 62.9 % (sd 1.0 pp), 40 / 48
discordant, exact McNemar p = 0.46, and an episodes-and-runs bootstrap of [-5.5, +13.0] pp that
spans zero. The 15 pp Fisher figure compares our DDIM@10 against a DDPM@100 control and so charges
the sampler's share to the adapter; the 8 pp this doc was first written with came from the
control's single 134/200 run, which is one unrepeated sample of a *different entry point* and sits
~4 sd outside the repeated band (`docs/04` §4). Whatever H is chasing, it is smaller than 8 pp,
and this design does not resolve it.

### 0.2 Four things the earlier version of this decision did not account for

1. **A flat coverage column argues against an arithmetic defect.** Coverage sits in a band of
   0.884-0.895 across every arm from 45.5 % to 69.0 % success. A wrong mapping (channel order,
   0-255 vs 0-1, `max-min` vs `max-min+eps`) moves coverage as well, usually more than it moves
   success. What these rows show is a policy that gets almost as close and crosses the 0.95 line
   less often -- the near-threshold story already in `docs/04`. So section 2's prior is too
   generous: the probe below is a *refutation* test, and passing it is the expected result.2. **200 seeds cannot resolve 8 pp.** Measured, not modelled: the DDPM arm's +7 pp came with 88
   discordant pairs and p = 0.17. An 8 pp effect needs roughly three times the episodes to clear
   alpha, so a 200-seed A/B of this path would return "not distinguished" whether or not the truth
   was a defect -- the worst thing a pre-registration can design for. Section 5 is re-sized to 600
   *(the repeats later that day make the target ~4 pp rather than 8, so §5.1 replaces the fixed 600
   with a pilot-chosen 600-2,200; the arm never ran, so that sizing is recorded rather than
   exercised)*
seeds and the deterministic probe is made the primary rule.
3. **The control draws a different noise stream from ours.** `lerobot-eval` batches 8 episodes and
draws DDPM noise per batch; we seed torch per episode. Same env seeds, different Gaussian -- and
that stream alone moved our rate 52.0 -> 45.5 % (p = 0.21). Pairing removes the environment term,
not this one, so 67.0 % bounds the residual rather than measuring it.4. **The control also swaps the environment wrapper** (R5, still open): same `gym_pusht` 0.1.8
   underneath, different `reset`/success path. Part of any loop-to-loop difference could be the
   wrapper. **Measured 2026-10-30: it is not.** `just env-equiv` steps one seed through both
   wrappers with a fixed 300-action sequence (`results/env_equiv.json`): coverage and success
   traces identical step for step (max abs diff 0.0), identical termination step, identical first
   observation, same observation keys. The env term in every loop-to-loop comparison is zero, and
   R5's environment half is closed.

### 0.3 What the audit changes

Direction unchanged, shape tightened: **H proceeds as a deterministic probe first.** If the two
implementations agree within 1e-6 the residual is not normalization, nothing is adopted, and the
number is the evidence. If they differ, the defect is located to a line and only then is an
episode block worth GPU hours -- sized by the pilot in §5.1, not fixed at 200.

That was built default-off: `models/normalization.py` (the seam, see §4), the adapter's
`normalization="processors"` opt-in, `scripts/processor_equiv.py` (§3), `just paired` for the
2x2, and `just eval-dp-processors` for §5.

**Then it was measured, and the answer is no.** The probe passed (§3), the paired control came
back null at the checkpoint's own sampler (§0.1, p = 0.46), and the env wrapper turned out to be
identical. Nothing was adopted; `normalization="processors"` remains default-off and unmeasured.
The full record is the three dated entries in `docs/04` §4.

## 1. The amendment

Rule 4 of `docs/07` (as amended by D11 on 2026-10-27) says that what stays ours is "the rollout
harness, the control loop, the latency protocol, **the normalization glue** and the table/figure
generators". H moves exactly one item off that list: the normalization glue, for the diffusion load
path only, may be `lerobot`'s processor pipeline.

The four conditions are part of the amendment, not preferences:

1. **Scope is one function of one file.** Only the mapping between raw observations/actions and the
   network's inputs inside `models/pretrained.py`. `eval/`, `deploy/`, the JSONL row schema, the
   control loop, `fast_sampler.py` and the generators are untouched. If the change needs anything
   outside `models/`, it is not this amendment.
2. **Default off.** The current buffer path (`models/pretrained.py` reading the eight
   `normalize_inputs.*` / `normalize_targets.*` / `unnormalize_outputs.*` tensors) stays the
   default until a paired A/B (§5) clears. A caller opts in explicitly; no environment variable, no
   import-time branch that a reader cannot see at the call site.
3. **Revert is one commit.** The adoption lands as one commit on its own branch, default off, so
   `git revert` restores the measured path and the old rows stay valid.
4. **The library's copy is not patched.** The migrated `policy_preprocessor.json` /
   `policy_postprocessor.json` are used as `scripts/lerobot_migrate.py` produced them. Any local
   deviation (the device hop, §4.3) is constructed in *our* code and recorded as such, never
   edited into the JSON.

Recorded as a dated entry in `docs/04` §4 and as an amendment note on rule 4 in `docs/07`.

## 2. Why H is worth an amendment at all

After `docs/09` option A the open gap is ~8-10 pp: at the checkpoint's own DDPM@100 this repo
scores 118/200 (59.0 %) against the checkpoint's 138/200 (69.0 %), and on the 1100-1149 band 60.0 %
against 68.0 %. Roughly 7 pp of the original 15 was the sampler; the rest is "adapter or process".
*(Re-measured later on 2026-10-30 against the repeated control: ~4 pp, not 8 -- see §0.1. The
reasoning below is unchanged, its target is smaller.)*

`docs/04` §4 excludes candidates by *outcome*: the 84x84 crop is harmless because the checkpoint
crops anyway, action slicing because `n_action_steps=8` matches, observation rendering because the
only size-dependent drawing is for humans, history because `select_action` pads the same way. Two
of those exclusions are numeric and reproducible; the others are arguments.

**What was never compared numerically is the normalization itself.** The adapter applies MEAN_STD
to images and MIN_MAX to state and actions by hand, from the eight buffers in `model.safetensors`.
After `scripts/lerobot_migrate.py`, `lerobot` applies the *same* declared mapping from
`policy_preprocessor.json` — the file lists image `VISUAL` 3x96x96 `MEAN_STD`, state `STATE` 2
`MIN_MAX`, action `ACTION` 2 `MIN_MAX`, and the same three in the postprocessor, with the state in
`policy_preprocessor_step_3_normalizer_processor.safetensors`. Two hand-written implementations of
one declared mapping is precisely the shape of a silent defect: channel order, the `eps`, the
ImageNet constants, whether images arrive as 0-255 or 0-1, whether the divisor is `max-min` or
`max-min+eps`. Any of those is a few lines and none of them is visible in a 15 pp aggregate.

E0's 64.0 % against the published 65.4 % was the gate that the mapping is *right*, and it is a
gate on the mean, not on the arithmetic. So H has a falsifiable first question that costs minutes,
not GPU hours.

## 3. Step 0: the equivalence probe (decides whether H continues)

Before any adoption, compare the two implementations directly, with no episodes and no simulator:

- Take 20 observations from real rollout frames (the same ones `results/figures` were made from,
  or just 20 recorded `obs` tensors captured once in the probe itself).
- Run both paths on them: (a) the adapter's buffers, (b) `lerobot`'s `policy_preprocessor.json` /
  `policy_postprocessor.json` loaded through `PolicyProcessorPipeline.from_pretrained`.
- Report three max-abs differences: normalized images, normalized state, and the round trip
  `actions -> unnormalize(normalize(actions))` against the raw actions.

**Pre-registered reading.**

- **Every difference ≤ 1e-6 (fp32 rounding only):** the two implementations agree. H cannot explain
  the residual, the adoption would be a maintenance change with no measured benefit, and no code is
  changed. Record the null in `docs/09` and move the residual to the process bucket (the remaining
  candidate is *when* the queue is refilled relative to the env step, which is a loop question, not
  a normalization one).
- **Any difference > 1e-6, or a difference that changes sign/branch:** that *is* the defect, H stops
  being an adoption and becomes a bug fix with a located cause. Then §4 applies, and the A/B in §5
  is how much it was worth.

**Result (2026-10-30, `just processor-equiv`, `results/processor_equiv.json`).**

| Check | Max abs diff | Bar | Result |
|---|---|---|---|
| migrated weights vs published | 0.000e+00 | exact | ok |
| images (MEAN_STD) | 2.384e-07 | 1e-06 | ok |
| state (MIN_MAX) | 0.000e+00 | 1e-06 | ok |
| actions (MIN_MAX inverse) | 0.000e+00 | 1e-06 | ok |
| history ordering (batched vs per frame) | 0.000e+00 | 1e-06 | ok |

`pass: true`. The 2.4e-07 on images is one fp32 ulp at normalized magnitudes around 4 -- a
different operation order, not a different mapping. **Verdict: H is a null.** The two
implementations are the same arithmetic, so the residual cannot be normalization; by the rule
above, nothing is adopted and §5 never runs.

Shape of the work if it is taken: one script (`scripts/processor_equiv.py`, ~60 lines, run with
`.venv-lerobot/Scripts/python.exe` because only that env has `lerobot`) plus one unit test on the
part that can run without the library (the buffer path), and the numbers written into `docs/04`.
No `src/` change. Cost: under an hour, no GPU time beyond loading 1.05 GB of weights once.

## 4. Design scaffold (only if step 0 fails)

### 4.1 The seam that does not move

`PolicyLike` stays as it is, and so does the invariant behind it: **normalization lives inside the
policy, so observations and actions cross every boundary in raw units.** The processor pipeline is
a detail of the diffusion adapter's interior, not a new stage in the harness.

```
PolicyLike.predict_chunk(obs)                    unchanged
      │
PretrainedDiffusionPolicy
      │  self._norm  ◄── the only thing H changes
      │
      ├── BuffersNorm      eight tensors from model.safetensors   (today, default)
      └── ProcessorsNorm   two lerobot PolicyProcessorPipelines    (H, opt-in)
```

One protocol, three methods, so the adapter's `predict_chunk` cannot tell them apart:

```python
class Normalization(Protocol):
    def image(self, image: Tensor) -> Tensor:      ...   # [B,T,3,H,W], 0-255 in, normalized out
    def state(self, state: Tensor) -> Tensor:      ...   # [B,T,2]
    def actions(self, chunk: Tensor) -> Tensor:    ...   # [B,H,2], network units -> pixels
```

`BuffersNorm` is today's code moved behind that protocol, unchanged. `ProcessorsNorm` wraps the two
pipelines. The swap is a constructor argument on the adapter, threaded through
`load_policy(..., normalization="buffers" | "processors")`, defaulting to `"buffers"`.

### 4.2 Where the scaffolding will hurt (the honest part)

Three concrete mismatches, all known before writing code:

1. **Batching.** The pipelines take a single unbatched observation dict per step
   (`to_batch_processor` exists precisely because callers pass one frame). The adapter holds
   `n_obs_steps=2` frames at once. The shim must either call the pipeline per history step and
   stack, or reshape `[B,T,...] -> [B*T,...]` and back. Reshaping is one line and wrong in a way
   that still runs: `[B,T]` and `[T,B]` are the same size, so this is exactly the class of bug that
   needs the step-0 probe extended to a *second* check — 4 observations run individually must equal
   the same 4 stacked (max abs diff ≤ 1e-6). That check is part of the acceptance, not an extra.
2. **Device hop.** The shipped postprocessor ends with `device_processor(device="cpu")`, because
   `lerobot-eval` wants numpy-ready output. In our loop that would be a `.cpu()` and a `.to(cuda)`
   per chunk: measured cost, and it would contaminate E10's deadline-miss count. The design
   therefore constructs the postprocessor with that final step set to `cuda` (our code, our
   object), and the chunk-latency difference between the two arms is reported with the A/B so the
   cost of the adoption is visible rather than assumed absent.
3. **Version surface.** The pipeline format is 2026 `lerobot`'s. `lerobot` is installed outside
   `uv.lock` (R13), so the adoption puts a lockfile-external artifact in the *measured* path for the
   first time — today only the control does that. `BuffersNorm` stays, and stays the default until
   the A/B clears, so a broken pipeline cannot break the default path. R13's trigger applies: if the
   combination stops importing, the processor path is dropped, not worked around.

### 4.3 What does not change

`PolicyLike`, the row schema (`results/*.jsonl` has no normalization field today and will not gain
one — provenance is the `config_id` prefix), `eval/rollout.py`'s sharding, `deploy/`, the samplers,
`stats.py`, the tables and the README generator. The VQ-BeT adapter is out of scope: it does its own
normalization from its own `config.json` and has no processor files, so "the two converge" is a
later question, not this amendment.

## 5. Pre-registered A/B (only if step 0 fails)

If the arithmetic differs, the question changes from "is it the normalization?" to "how many points
did it cost?" — the same shape of test as every other claim here. **It does not run unless §3
diverges:** spending GPU hours on a path whose arithmetic already matches is how a null gets
converted into a false positive.

### 5.1 Size: a 4 pp effect costs 250-2,200 episodes, so a pilot decides

600 was sized when the residual looked like 8 pp. The repeated control moved it to ~4 pp
(`docs/04` §4), and a smaller effect needs *more* episodes, not fewer. Paired-binary power
(Connor's formula, `stats.paired_binary_n`, printed by `just arm-power`) prices that at 80 % power:

| effect | psi=0.05 | psi=0.10 | psi=0.20 | psi=0.30 | psi=0.44 |
|---|---|---|---|---|---|
| 4 pp | 243 | 489 | 979 | 1,470 | 2,157 |
| 8 pp | 59 | 121 | 243 | 366 | 538 |

The 9x range is set by `psi`, the fraction of episodes on which the two arms disagree -- not by the
effect. 44 % is the discordance the DDIM-vs-DDPM comparison actually showed, but those arms differ
by a *sampler*; two implementations of one mapping that agree to 2.4e-07 can only flip episodes
sitting within that arithmetic difference of the 0.95 line, which should be far fewer. Which it is
cannot be known before running, so the size is chosen by a pilot instead of by either prior:

- **Pilot: the first 200 paired episodes.** Report `psi_hat`, the discordant fraction. The branch
  depends on `psi_hat` alone -- never on the direction or the p-value -- so the final test keeps its
  nominal size and no result is looked at twice.
- `psi_hat <= 0.10` -> **600 episodes/arm**: power 0.86 at psi = 0.10, 1.8 GPU-h for both arms.
- `0.10 < psi_hat <= 0.30` -> **1,600 episodes/arm**: power >= 0.80, 4.8 GPU-h.
- `psi_hat > 0.30` -> **2,200 episodes/arm, run twice per arm**: power 0.81 at psi = 0.44, 6.6 GPU-h
  per pass, 13.2 GPU-h for the block. The repeat is not optional at that size -- the episode-level
  standard error there is ~1.0 pp, the same as the control's measured run-to-run sd, so one pass
  cannot separate an effect from a fluke; both passes must agree in sign.

- **Primary endpoint:** paired McNemar at the pilot's n, processor path against the buffer path, at
  **DDIM@10** -- the deployment-relevant point and the configuration E0's PARTIAL verdict is about.
  Seeds are contiguous from 1000, both arms run at the same count (`eval-dp-processors` and
  `eval-dp-buffers`), and the block is reported at its final size only. Both arms are ours, so it
  needs no published reference rows.
- **Secondary, deliberately not a test:** the DDPM@100 cell is where the residual was measured, but
  at a resolvable size it costs ~29 GPU-h for the same question, so it is reported as a directional
  confirmation at 200 episodes and never as a p-value. The 1100-1149 band stays **exploratory**
  (four post-hoc looks; Bonferroni bar 0.0125).
- **What would count as H working:** the processor path separated upward on the primary endpoint
  (p < 0.05) and the DDPM@100 cell moving toward 69 %. Partial: separated on the band but not in
  aggregate, in which case the claim stays "normalization was one of several stage costs", the same
  wording A forced.
- **What would count as H failing:** no separation either way. Then the adoption is declined and
  the residual is written into `docs/04` as measured-and-unexplained, with the probe numbers as the
  evidence that the mapping is not the cause.
- **Cost:** at the measured 417 ms per chunk and ~13 chunks per episode, 600 episodes at DDIM@10 is
  about 1 GPU-h on one worker; the same block at DDPM@100 would be about 9 GPU-h at the measured
  1819 ms per chunk and is only run if the primary endpoint moves. Both recipes default to one
  worker so neither arm pays the 2.3x-per-chunk contention two workers showed."

Passing the A/B does not by itself retire E0's PARTIAL verdict — it would *change the attribution*
of the residual, which is what `docs/09` says H is for.

## 6. Cost, risk, revert

| Item | Estimate | Basis |
|---|---|---|
| Step 0 probe + test | < 1 h, no GPU block | 20 observations, one weight load |
| Adoption (if it continues) | 2-4 h of code, ~60 lines net | one protocol, two implementations, one shim |
| A/B, primary endpoint | 1.8-13.2 GPU-h | pilot-scaled n (§5.1) at 417 ms per chunk, 13 chunks/episode, both arms |
| Risk | a device hop and a reshape bug that runs silently | both pinned by a test in §4.2 |
| Revert | one commit, default off | condition 3 of the amendment |

## 7. Runbook: what to type, in order

Nothing in this page has been run. Each command below is the whole of its step, and each one
either produces a number in `results/` or ends the option.

```bash
just processor-equiv          # step 0: deterministic, ~1 min, no GPU block
```

Writes `results/processor_equiv.json` and exits non-zero on any difference above 1e-6.

- **`pass: true`** -> stop. H is a null: record the number in `docs/09`, keep the seam (it is
  testable and default-off), and move the residual to the process bucket.
- **`pass: false`** -> the difference names the defect. Fix it in `ProcessorsNorm`'s construction
  (or report it as this repo's bug), then §5:

```bash
just eval-dp-processors
just paired results/e0_published_seeds.jsonl results/e0_processors.jsonl ddim@10 processors@10
```

The pilot's branch sets the episode count for both (`just arm-power`, §5.1), so the first is
0.9-6.6 GPU-h per arm. The second prints the 2x2 and the exact p-value; nothing is typed by hand.
If the processor arm clears, the DDPM@100 block follows before anything in `docs/05` or the
README changes, because the deployment point is DDIM@10 but the residual was measured at DDPM@100.

Checks a reader should expect after either branch: `just verify` (unit tests cover the seam, the
history-order reshape and the `config_id` guard), and `docs/04` §4 carrying the outcome.

## 8. What would make me decline H anyway

- **Step 0 passes.** Then H is a rewrite of working code with no measured benefit, and the boundary
  in rule 4 stays exactly where D11 left it.
- **The residual turns out to be loop timing.** If the probe agrees, the next candidate is not
  normalization but the moment the action queue is refilled relative to the environment step, which
  is a `deploy/` and `rollout.py` question and does not need this amendment at all.
- **A second checkpoint arrives first.** The processor files also exist for VQ-BeT-style checkpoints,
  and comparing *two* families through one normalization path would test the adapter boundary more
  than adopting the pipeline for one diffuser does.
