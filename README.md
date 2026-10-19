# FlowPolicy

A from-scratch conditional flow-matching visuomotor policy (pi0-style, small scale),
compared against a DDPM/DDIM diffusion policy and an ACT-style regression baseline on
PushT and robomimic, with inference optimization at batch size 1 and a fixed-rate
deployment loop.

**Status: M0 (repo scaffold).** No models, no experiments, no numbers yet. Every metric
in the docs is `TBD` until measured; see `docs/00_overview.md` section 3.

## Where the truth lives

- `docs/07_roadmap.md` — milestones M0-M8, packaging, reading list. Start here.
- `docs/02_environment_setup.md` — pinned versions, setup, week-1 gates G1-G9.
- `docs/01_architecture.md` — module boundaries, the `Policy` interface, CI, Docker.
- `docs/04_eval_and_experiments.md` — pre-registered protocol, E0-E12.
- `docs/06_decisions_and_risks.md` — D1-D10 and the risk register.
- `CLAUDE.md` — operating rules for the coding agent (plan, `GO <step>`, then act).

## Quick start

```bash
uv sync --python 3.12   # or: just setup
just test               # ruff + mypy + unit tests
```

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just).
GPU, simulator and ROS 2 work is local-only; CI is CPU-only by design.

## Limitations

Simulation only. Small models. Benchmark-scale demos, not a product.
