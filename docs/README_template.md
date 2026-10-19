# FlowPolicy

A from-scratch conditional flow-matching visuomotor policy (pi0-style, small scale),
compared against a DDPM/DDIM diffusion policy and an ACT-style regression baseline on
PushT, with batch-1 inference engineering and a fixed-rate deployment loop.

**Status: MVP complete, small-scale.** Every number below is generated from
`results/*.jsonl` by `scripts/make_tables.py` and `scripts/make_figures.py`; no metric is
hand-typed. The models are deliberately small and trained for a short budget on one
8 GB laptop GPU, so the success rates are a scaled-down result, not a reproduction of
the published Diffusion Policy numbers. See **Limitations**.

All work ran on one Windows 11 laptop (RTX 5060 Laptop, 8 GB, driver 617.42), Python
3.12, torch 2.14.1+cu130. GPU, simulator and deployment tests are local only; CI is
CPU-only.

## Quick start

```bash
uv sync --python 3.12       # or: just setup
just test                   # ruff, mypy strict, CPU unit tests (CI gate)
just data                   # download + convert lerobot/pusht to Zarr (about 32 MB)
just train-e1               # 3 policies x 3 seeds, ~9 min per run on this host
just eval-e1                # parallel rollouts -> results/e1.jsonl
just tables && just figures # regenerate every table/figure from JSONL
just readme                 # regenerate README.md from the generated tables
```

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just).
The week-1 gates are in `scripts/gates/`; `just gates` runs G1-G9 and honestly reports
NOT RUN where the host lacks the prerequisite.

## Results (measured)

### E1: flow vs DDPM vs regression, PushT

{{E1_TABLE}}

![E1 success](results/figures/e1_success.png)

### E3: sampler steps vs success and latency

{{E3_TABLE}}

![E3 success](results/figures/e3_success_vs_steps.png)
![E3 latency](results/figures/e3_latency_vs_steps.png)

### E9: batch-1 latency ladder

{{E9_TABLE}}

### E10: injected observation latency in the fixed-rate loop

{{E10_TABLE}}

![E10 success](results/figures/e10_success_vs_latency.png)
![E10 latency histogram](results/figures/e10_latency_hist.png)

### M2: measured memory and throughput

{{M2_TABLE}}

### M3: overfit sanity check (8 episodes)

{{OVERFIT_TABLE}}

### E9 stretch: fused Triton Euler/de-normalization kernel

{{KERNEL_TABLE}}

## Honest limitations

- **Simulation only**, 96x96 PushT, small models (about 11 M parameters), 1800 training
  steps per run (about 1.5 epochs). The reference DDPM recipe trains for ~200k steps; E0 is
  therefore not a reproduction and is reported as a written explanation in `docs/04`.
- **At this budget no policy reaches the 95% coverage success threshold** in E1, so the
  E1 comparison is reported on mean coverage and latency rather than success. This is the
  main limitation of the result and is stated as a null result, not hidden.
- Every seed result comes with a bootstrap interval over seeds and episodes; with three
  seeds those intervals are wide.
- Robomimic, ROS 2, Docker and `nvcc` were unavailable on this host. The documented
  fallbacks were used: PushT only (D1), plain Python loop (D7), no CUDA C++ stretch (D6).
  G8 and G9 are NOT RUN, never reported as passes.
- E10 injects latency as observation staleness in control ticks and runs the loop in
  virtual time; inference time is measured for real. Deadline misses are logged.
- No W&B; JSONL only (D8 deviation).

## Reproduce

```bash
just setup && just test
just gates                    # G1-G5 should pass on this GPU class; G6/G8/G9 report NOT RUN
just data
just train-e1
just eval-e1 && just eval-e3 && just latency-ladder && just eval-e10
just memory && just kernel && just overfit
just tables && just figures && just readme
```

Raw per-episode rows are in `results/*.jsonl`; checkpoints are local (ignored by git).
Training logs are in `results/train/`.

## Layout

```
conf/                  Hydra configs + experiment files
src/flowpolicy/        data, models, training, eval, deploy, kernels
scripts/gates/         G1-G9 gate scripts
scripts/               data prep, train, evaluate, latency, deploy, tables, figures
tests/                 unit (CI), gpu/integration (local)
docs/                  the full spec (00-07), STATE and CHANGELOG
results/               raw JSONL, generated tables and figures
```

## Docs

- `docs/07_roadmap.md` - milestones and packaging checklist
- `docs/02_environment_setup.md` - versions, setup, gates G1-G9 and results
- `docs/01_architecture.md` - the `Policy` interface and module boundaries
- `docs/04_eval_and_experiments.md` - pre-registered protocol, experiments and deviations
- `docs/06_decisions_and_risks.md` - D1-D10 decisions and risks
- `docs/STATE.md` - current state and decision log

## License

Code: not decided by the owner yet. Dataset: `lerobot/pusht` is MIT (dataset card,
accessed 2026-10-19); it derives from the Diffusion Policy PushT demos.
