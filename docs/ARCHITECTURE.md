# Architecture, as built

One page. `docs/01` is the design as specified on 2026-10-18; this file describes the code
that exists on 2026-10-28, including what changed under the D11 pivot. If the two disagree,
this file is describing reality and `docs/01` is describing intent.

## Shape

Four layers around one narrow interface. The interface is the design: everything above it
(eval, latency, control loop, server) knows nothing about which policy it holds, and
everything below it knows nothing about experiments.

```
                conf/*.yaml  (ablation only)
                      │
             ┌────────▼─────────┐
             │ training/        │  Trainer, EMA, resume, checkpoints
             └────────┬─────────┘
                      │ checkpoint: config + stats + weights
                      ▼
  ┌────────────────────────────────────────────────────────────┐
  │ models/    PolicyLike  ◄─ PretrainedDiffusionPolicy (lerobot)
  │                        ◄─ PretrainedVQBeTPolicy  (lerobot)
  │                        ◄─ BasePolicy → Flow / DDPM / Regression
  │            fast_sampler.py (DDIM coefficients on-device)   │
  └───────┬──────────────────────────────┬─────────────────────┘
          │ predict_chunk(obs)           │ predict_chunk(obs)
          ▼                              ▼
  ┌───────────────┐              ┌───────────────────────┐
  │ eval/         │              │ deploy/               │
  │ rollout       │              │ control_loop          │
  │ metrics (CI)  │              │ server (stdlib HTTP)  │
  │ latency       │              └──────────┬────────────┘
  └───────┬───────┘                         │
          └──────────► results/*.jsonl ◄─────┘
                            │
        eval/report.py → make_tables → make_figures → make_readme
                            │
                    tables, figures, README
```

## The three seams

| Seam | Contract | Implementations |
|---|---|---|
| `PolicyLike` (`models/policies.py`) | `obs_horizon`, `pred_horizon`, `action_dim`, `predict_chunk(obs, **kw)`, `num_parameters()`, `eval()`, `to()` | diffusion adapter, VQ-BeT adapter, three from-scratch policies |
| `EnvLike` (`envs/base.py`) | `reset(seed)`, `step(action)`, `close()`, `fps`, `task_description`; `{image, state}` in, `{success, coverage}` out | `PushTEnv` (gym-pusht, D1); `rollout_episode(env_factory=...)` is the only injection point so far |
| JSONL rows (`results/`) | one dict per episode: seeds, success, coverage, latency, `provenance {git, lock_sha256}` | written by runners, read by every generator |

`load_policy` (`eval/rollout.py`) accepts a trained checkpoint path or `hf:<repo_id>`, and for
an HF id reads `type` from the checkpoint's own `config.json` to choose an adapter. No caller
passes a policy kind.

The environment is a *type* now (`EnvLike`) rather than a name inside the loop: `rollout_episode`
takes `env_factory`, defaulting to `PushTEnv`. That is what lets the determinism test run the real
loop without the simulator, and what a second task would plug into -- the call sites themselves are
deliberately unchanged, because they produced every row here.

## Layers

**Models.** `BasePolicy` owns normalization, encoder and `predict_chunk`
(`encode → _sample → denormalize`). The pretrained adapters own *their checkpoint's*
normalization instead: 0.6.1 `lerobot` dropped it from `from_pretrained`, so both read the
buffers from `model.safetensors` and apply the checkpoint's own mapping. `fast_sampler.py` is
a DDIM loop whose coefficients are precomputed on device, which is what lets it be captured by
`CUDAGraph` and compiled without a CPU partition.

**Measurement (`eval/`).** `evaluate` shards eval seeds across spawn workers; `rollout_episode`
seeds torch per episode (so results do not depend on worker count), refills an action queue
every `action_horizon` steps and times each chunk around a device synchronize. `metrics`
computes a two-level bootstrap (training seeds, then episodes) plus Wilson intervals.
`latency.benchmark` is a separate protocol: static inputs, 50 warmup + 200 timed calls, eager /
compile / graph.

**Deployment (`deploy/`).** `LoopConfig` + `run_episode` implement a fixed-period loop:
observation delay is `ceil(latency_ms / period_ms)` ticks, inference is timed for real, every
overrun increments `deadline_misses`, and `paced=True` turns that counter into measured
lateness. Three chunk strategies share one path (naive, temporal ensemble, cross-fade).
`server.py` is a stdlib `ThreadingHTTPServer` around any `PolicyLike` with a documented JSON
contract (`GET /health`, `POST /chunk`), raw pixels in and out.

**Observability (`logging.py`, `scripts/pipeline.py`).** `RunLog` appends one JSON event per line
to a gitignored `results/logs/`, opened by the parent *and* by each worker, so every episode's line
names the process that ran it; `Progress` turns completed/total into a rate and an ETA. Workers
write their own rows and log the failing seed with a traceback (`RolloutError`), which replaced a
`BrokenProcessPool` that named nothing. `scripts/pipeline.py` runs the measurement stack as stages
into a scratch run directory -- never on top of `results/` -- and pairs the fresh rows with the
committed ones by seed, so a re-run can disagree with the evidence without overwriting it.

**Evidence → publication.** `eval/report.py` holds the result-file registry and the computed
headline answers (a verdict is computed from the strongest comparison available -- the same-episode
block when it exists -- never from the unpaired one), `eval/stats.py` holds the exact tests, `scripts/make_tables.py` renders tables, `make_figures.py` plots,
`make_readme.py` splices them into the README. CI regenerates all three and fails on any diff.

**Training (subordinate).** Dataclasses + Hydra, Zarr window sampler, ResNet-18/GroupNorm
encoder, 1D U-Net or transformer head, EMA trainer with resumable checkpoints that record git
and lock hashes. Kept because the ablation is evidence for the pivot, not because anything
current depends on it.

## Invariants

- Eval and deploy never branch on policy type; they see the protocol.
- Normalization lives inside the policy, so observations and actions cross boundaries in raw
  units (0–255 images, pixel actions).
- Every row records its seeds; eval seeds are checked disjoint from train seeds, not assumed.
- The optional backend is imported inside functions, so a missing `lerobot` fails at the call
  site with a message naming `just setup-pretrained`.
- One generator per artifact, no hand-typed numbers; a missing result file renders as "not
  measured", never as zero.
- CPU-only CI covers the contracts, including a real-socket server test; GPU and simulator
  tests carry separate markers.
- A long block is observable: every episode leaves a log line, and a failure names its seed.
- A run never writes to `results/*.jsonl`; the pipeline writes to its own run directory and
  compares itself against the committed rows.

## Deliberately absent

No batching or queueing server, no ROS 2 node, no distributed eval, no experiment tracker
beyond JSONL, no model registry. Each is a recorded decision in `docs/06`, not an oversight.

## Extending it

- **New policy:** implement `PolicyLike`, add a branch in `load_policy` (or an adapter keyed by
  the checkpoint's `config.json` type). `docs/04` E15 is the worked example.
- **New task:** write an env wrapper with the same four methods, keep the JSONL schema.
- **New experiment:** a runner that appends rows, a table function in `make_tables.py`, and an
  entry in `RESULT_FILES` so `just report` and the README pick it up.
