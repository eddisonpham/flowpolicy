# 01. `L1` -- run the Linux pilot to a recorded row

## Goal

Produce the first `kind: pilot` row in `results/smolvla_pilot.jsonl`, so `docs/11` §5.1's
per-chunk band (150 / 250 / 700 ms) is replaced by a measurement rather than left as the one
extrapolated term in every GPU-hours figure in the repo.

## Non-goals

- Not an eval: n = 1 cannot measure success, and the row must not be quoted as one.
- No LIBERO port to Windows. That path was walked and priced at 2-4 days with an architectural wall
  at the end [V, docs/11 §4.5]; it is not this step.
- No second card, no second task, no second policy.

## Prerequisites

1. A Linux x86_64 host with a CUDA card of >= 8 GB and ~15 GB free disk.
2. Network reachable to the HF Hub (2 GB checkpoint + two small processor files).
3. The repo at a commit whose `just verify` is green. Today: `6f6cc44`.

## Steps

| # | Command | Acceptance |
|---|---|---|
| 1 | `docs/12` §3: `nvidia-smi`, `uv --version`, `just --version`, `df -h .` | four lines of output, one card, >= 15 GB free |
| 2 | `docs/12` §4: build `.venv-pilot` (lerobot 0.6.1, torch 2.11.x, `transformers` 5.4-5.6, `libero==0.1.1`, `robosuite==1.4.0`) | the section's verify block prints `cuda available True` and `robosuite 1.4.*` |
| 3 | `just smolvla-pilot-stats py=".venv-pilot/bin/python"` | `[ok] precondition 3`, `max |diff|` no larger than ~1e-7 (7.45e-09 measured here [V]) |
| 4 | `.venv-pilot/bin/python scripts/libero_port_walk.py` | `[PINNED ]`, exit 0 |
| 5 | `docs/12` §7: write `~/.libero/config.yaml`, then assert `len(tasks) == 10` | prints `10` |
| 6 | Download the checkpoint explicitly, then `just smolvla-pilot-check py=".venv-pilot/bin/python"` | three `[ok]` lines, exit 0 |
| 7 | `just smolvla-pilot py=".venv-pilot/bin/python"` | one row appended; the printed `chunk` block and `projected_gpu_hours_*` |
| 8 | Quote the row into `docs/11` §5.1 and `docs/STATE.md` by hand; run `just verify` | the doc edit names the measured mean and p95; `just verify` green with tables byte-identical -- which proves *no table moved*, because no generator reads pilot rows yet [V, `docs/12` §9]. That is a smaller claim than "the row is in the tables", and it is the true one |

**Row acceptance, not just "it ran".**

- `finished_early: false`, or the short chunk count is stated next to the projection.
- `chunk.mean` is inside 150-700 ms -> §5.1's band was right and one row of its table is replaced
  by a measurement. Outside it -> recompute the table from the measured value; that is the pilot
  working, not failing.
- Above ~2 s per chunk: the ten-action cadence stops being defensible and RTC's premise with it.
  Write that down before it is discovered in `R1`.
- Drift beyond +-20 %: the automatic repeat already ran from the same initial state; quote the row
  the script names and say the repeat happened.

## Cost

0.5 setup days + <= 15 minutes of GPU, both **computed** from `docs/11` §5.1 (install sizes and
28 chunks x 700 ms pessimistic). Not measured: this has never run.

## False if / kill

- If `robosuite==1.4.0` will not import on the Linux host either, work `docs/12` §10's ladder
  first (`MUJOCO_GL=egl`, then `osmesa`, then a clean re-install of the pin). Only when those are
  exhausted is the pin itself the finding: record it in `docs/06` D12 and stop -- the port is not
  worth 2-4 days against a 15-minute measurement.
- If precondition 3 prints `mismatch`, stop before the download -- the pilot would calibrate a
  different model, and that is a finding about the checkpoint or the installed lerobot, not about
  the card.
- If the episode runs but the row's `chunk.mean` is dominated by environment steps, the pilot has
  measured the wrong half; check `build`/`env` against `policy` before quoting anything.

## Rollback

Additive by construction: a new row file plus dated doc entries. `results/smolvla_pilot.jsonl` can
be removed without touching any existing measurement.
