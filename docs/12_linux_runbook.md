# 12. Linux runbook: fresh machine to the pilot's first recorded row

Status: **not executed end to end on a Linux host** (this repo has only ever run on the Windows
host in `docs/02` section 2). Every command below is composed from three things that are recorded,
not recalled: the versions `docs/11` section 4.4 measured by installing them, the pins this repo
already runs, and the pilot's own preconditions. Where a step is an inference rather than an
observation it says so. Read the whole of section 8 before treating a green run as a result.

Links: [11_sota_models](11_sota_models.md) (why this exists), [02_environment_setup](02_environment_setup.md),
[CHANGELOG](CHANGELOG.md).

---

## 1. What the machine has to be

From `docs/11` section 5.1, which is a hardware statement rather than a spending one:

| Requirement | Value | Why |
|---|---|---|
| OS | Linux x86_64 | `robosuite==1.4.0` is LIBERO's pin, and it cannot load on Windows (11 section 4.4) |
| Card | a CUDA card; 8 GB is enough | SmolVLA peaks well under the 1189 MiB the PushT diffusion policy measured at N=10 (E8); the checkpoint is ~0.9 GiB as bf16 weights |
| Disk | ~15 GB | torch wheels ~3 GB, the pilot env ~3 GB, the checkpoint ~2 GB, HF cache headroom |
| Network | Hugging Face reachable | the checkpoint is a 2 GB download |
| GL | an offscreen backend that MuJoCo can use | the env renders two camera views per step |

Time: **0.5 setup days** in `docs/11` 5.1's table. Steps 0-3 below are about an hour; the rest is a
download and a 15-minute measurement.

## 2. The split that decides everything: two interpreters, not one

The pilot needs `lerobot`, `transformers` and `robosuite==1.4.0` in **one** interpreter, and that
interpreter must not be the project's `.venv`:

- `.venv` pins `torch==2.14.1+cu130` (`pyproject.toml`), and `lerobot==0.6.1` caps `torch<2.12`.
- Resolving lerobot into `.venv` would silently change the torch build under every number already
  measured, which is the reason `scripts/build_lerobot_env.sh` exists for the adapter-free control.

So the pilot gets `.venv-pilot`: a third env, gitignored on the same rule as `.venv-lerobot/` and
`.venv-libero/`. Nothing in the pilot imports this repo's package (`scripts/smolvla_pilot.py` adds
`src/` to the path but imports only numpy, torch and, lazily, lerobot), so the env needs no
`uv sync`.

**Consequence to accept deliberately:** the pilot's wall clock is measured on the torch lerobot
resolves (2.11.x), not on `.venv`'s 2.14.1. That is correct for this number -- it is the stack the
block would actually run on -- but it is not the stack E8/E9 measured, so the pilot's per-chunk cost
is not comparable to the PushT latency rows. It is a new measurement on a new stack, which is what
section 5.2 asked for.

## 3. Step 0 -- preflight (5 min)

```bash
nvidia-smi                                  # a card, and a driver that reports a CUDA version
uv --version && just --version              # both present
df -h . | tail -1                           # >= 15 GB free
cd <repo>                                   # every command below runs from the repo root
```

`just` is a task runner, not a dependency of the code; if it is missing, run the underlying commands
(`<interp> scripts/smolvla_pilot.py ...`) -- the `just` recipes only pass `py=` through.

## 4. Step 1 -- build the pilot interpreter

```bash
uv venv .venv-pilot --python 3.12
export VIRTUAL_ENV="$PWD/.venv-pilot"

# Lerobot first: it carries the torch that its own resolution allows (2.11.x, <2.12).
uv pip install "lerobot==0.6.1"
# `av<16` explicitly: lerobot only declares it behind an extra, and without it the eval path dies
# with `module 'av' has no attribute 'option'` -- the reason is recorded in scripts/build_lerobot_env.sh.
uv pip install "av>=15,<16"
# lerobot's declared range, and the version docs/11 4.4 measured installing (5.5.4).
uv pip install "transformers>=5.4.0,<5.6.0"
# LIBERO *with* its dependencies: `egl-probe` builds on Linux, which is the whole of the platform
# difference -- on Windows it had to be `--no-deps` (11 section 4.4).
uv pip install "libero==0.1.1"
# LIBERO's own pin. If the line above already resolved to 1.4.0 this is a no-op; if it did not, this
# is the command that makes the stack the one the published cell was produced on.
uv pip install "robosuite==1.4.0"
```

Verify, before downloading anything:

```bash
.venv-pilot/bin/python - <<'PY'
import av, robosuite, torch, transformers, lerobot
print("lerobot", lerobot.__version__, "| transformers", transformers.__version__)
print("torch", torch.__version__, "| cuda available", torch.cuda.is_available())
print("robosuite", robosuite.__version__)
assert torch.cuda.is_available(), "CPU-only torch: the pilot would measure CPU, stop here"
assert robosuite.__version__.startswith("1.4."), "not LIBERO's pin, the numbers would be another stack"
PY
```

A CPU-only `torch` here is the one failure worth stopping on: the pilot would run, produce a
believable row, and the GPU-hours projection would be off by an order of magnitude.

## 5. Step 2 -- prove the pin before paying for the download

`docs/11` 4.5's port ladder only matters on a robosuite that is *not* LIBERO's pin. Run the gate
against this interpreter; it is seconds, CPU and network-free:

```bash
.venv-pilot/bin/python scripts/libero_port_walk.py
```

Expected: `[PINNED ] robosuite 1.4.0 is LIBERO's own pin: the ladder does not apply.` and exit 0.
Anything else -- `ladder intact`, `drift`, or a 1.5.x version -- means the interpreter is not on
LIBERO's stack, and every number that follows would come from a stack nobody else runs. Fix step 4
before continuing. (`just port-walk` points at `.venv-libero`, the Windows probe env, so it is the
wrong interpreter here on purpose.)

## 6. Step 3 -- LIBERO's config file, or the run dies on a prompt

LIBERO calls `input()` when `~/.libero/config.yaml` is missing, so an unattended run gets
`EOFError` before any env exists [V, `docs/11` 4.4]. Write the file, deriving the paths from the
installed package rather than hardcoding them:

```bash
.venv-pilot/bin/python - <<'PY'
import importlib.util, pathlib
spec = importlib.util.find_spec("libero")
assert spec and spec.origin, "libero is not importable in this interpreter"
root = pathlib.Path(spec.origin).parent / "libero"
cfg = pathlib.Path.home() / ".libero"
(cfg / "datasets").mkdir(parents=True, exist_ok=True)
(cfg / "config.yaml").write_text(
    f"benchmark_root: {root}\n"
    f"bddl_files: {root / 'bddl_files'}\n"
    f"init_states: {root / 'init_files'}\n"
    f"datasets: {cfg / 'datasets'}\n"
    f"assets: {root / 'assets'}\n",
    encoding="utf-8",
)
print((cfg / "config.yaml").read_text())
PY
```

Then confirm the task definitions actually ship (they do -- the bddl and init files are inside the
wheel, no dataset download, [V]): `.venv-pilot/bin/python -c "from libero.libero import benchmark;
print(len(benchmark.get_benchmark_dict()['libero_spatial']().tasks))"` must print `10`.

`LIBERO_CONFIG_PATH` pointing at a directory works instead of `~/.libero`, and is what the port-walk
gate uses; the pilot does not set it, so the file has to exist at one of those two places.

## 7. Steps 4-5 -- the checkpoint, then the preconditions

Download explicitly, so that a failure inside the pilot means "the checkpoint is broken" rather than
"the network was down":

```bash
.venv-pilot/bin/python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('HuggingFaceVLA/smolvla_libero'))"
```

Then the three preconditions alone (minutes, no episode, no row):

```bash
just smolvla-pilot-check py=".venv-pilot/bin/python"
```

Three `[ok]` lines are required, in this order: the env constructs on the pinned stack, the
checkpoint loads, and the postprocessor reproduces the checkpoint's own declared statistics at 1e-5.
Exit codes mean different things and should not be conflated:

| Exit | Meaning | What to do |
|---|---|---|
| 0 | preconditions green, nothing measured (`--check-only`) | go to step 6 |
| 1 | a precondition **failed on this host** | section 9's ladder; a stats `mismatch` is a stop, not a retry |
| 2 | the stack is absent from this interpreter (`[NOT RUN]`, naming the module) | step 4 did not install, or you are on the wrong interpreter |

Precondition 3 is `docs/11` 4.4's step 1 and it stops the run on purpose: if the postprocessor
disagrees with the statistics in the checkpoint's `policy_postprocessor.json`, the pilot would
calibrate a different model. Do not reach for `--skip-stats` to get past it; that flag exists for the
case where the statistics were verified by hand for another reason, and the script says so.

## 8. Step 6 -- the measurement, and the first recorded row

```bash
just smolvla-pilot py=".venv-pilot/bin/python"
```

One LIBERO-Spatial episode, task 0, initial state 0, the 280-step cap, 28 chunks of ten actions.
Under 15 minutes wall at the pessimistic 700 ms per chunk in `docs/11` 5.1. It writes
`results/smolvla_pilot.jsonl` with `kind: pilot`, and prints:

- `policy` / `build` / `env` / `chunk`, each with mean, median, p95, max -- the split is the point:
  it says whether a change would move the policy or the environment;
- first call (the one that pays warmup), p95 drift against the first five chunks, peak VRAM, GPU name;
- `projected_gpu_hours_500` and `..._three_arms`, which are 5.1's table with the measured mean
  substituted. They stay labelled *projected* because the cadence is still an assumption.

Read it in this order:

1. **Mean per chunk.** Inside 150-700 ms: 5.1's band was right and its table stands with one row
   replaced by a measurement. Outside it: recompute the table from the measured value -- that is the
   pilot working, not failing.
2. **Above ~2 s per chunk:** the plan changes shape. A ten-action cadence stops being defensible and
   RTC's premise (a leftover in flight) goes with it. The script prints this warning itself.
3. **Drift beyond ±20 %:** the repeat was taken automatically, from the same initial state, and the
   row with the smaller |drift| is the one the script tells you to quote. If the repeat ran, say so
   when quoting the number.
4. **`finished_early: true`:** the episode ended before the cap, so the chunk count is short and the
   projection is off by that ratio. Not an error, but not 28 chunks either.

Nothing regenerates a table from pilot rows yet: `scripts/make_tables.py` does not read
`smolvla_pilot.jsonl`, so this row is quoted by hand into `docs/11` 5.1 (and `docs/STATE.md`), which
is what 5.2 promised. Rows in `results/*.jsonl` are tracked by git; the `.gitignore` rules drop only
logs, `*.tmp` and `*.partial`.

## 9. The failure ladder

| Symptom | Cause | Next |
|---|---|---|
| `[NOT RUN] No module named 'libero'` at precondition 1 | step 4 skipped, or you are on `.venv` | re-run with `py=".venv-pilot/bin/python"` |
| `EOFError` / a prompt during LIBERO import | no `~/.libero/config.yaml` | step 3 |
| `[FAIL] precondition 1 ... GL` / `EGL` / a blank frame | no offscreen GL for MuJoCo | `MUJOCO_GL=egl` first, then `MUJOCO_GL=osmesa`; this is G8 in `docs/02` section 7 and it was never green on this repo's host |
| `ModuleNotFoundError: robosuite...single_arm_env` | robosuite is 1.5.x, so the pin did not take | re-install `robosuite==1.4.0`; step 2 already checks this |
| `FileNotFoundError: ...mujoco.dll` | you are on Windows | not a Linux runbook problem; 11 section 4.4 |
| `[FAIL] precondition 3 ... differs by N` | the postprocessor is not the checkpoint's statistics | stop; the pilot would calibrate another model |
| `[FAIL] precondition 2` | checkpoint download incomplete or `transformers` missing | step 7's download, then re-check |
| `cuda available False` | CPU torch resolved | step 4's verify; do not run |

## 10. What this runbook does not promise

- **It has not been executed.** No Linux host has run it end to end; the pilot has never produced a
  row anywhere (11 section 5.2). What is verified about the pilot is its failure path: `[NOT RUN]`
  and exit 2 on a host without the stack.
- **Two steps are inference, not observation.** That `libero==0.1.1` resolves on Linux without
  `--no-deps` follows from `egl-probe` needing build headers (present on Linux, absent on Windows);
  and that `robosuite==1.4.0` pulls a `mujoco` that this card can render offscreen is the ordinary
  Linux story rather than a measured one here. Both fail at step 1 or step 2 -- loudly, before the
  download -- which is why they are ordered before it.
- **It does not decide anything the pilot cannot.** 5.2's list stands: not the block's discordance,
  not RTC's benefit, not SmolVLA's success rate, and not another card's per-chunk cost. Running this
  twice on two cards is two measurements, not one transferred number.
- **It is not a one-command script.** The install block in step 4 is meant to be pasted, and no
  committed script backs it yet; the repo has `scripts/build_lerobot_env.sh` for the PushT control
  env and nothing for the LIBERO stack.
