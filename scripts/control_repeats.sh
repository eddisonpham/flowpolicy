#!/usr/bin/env bash
# Repeat the adapter-free control with the identical command, so its run-to-run spread can be
# measured instead of guessed (docs/04 section 4: two runs disagreed by 4 pp, which is the size
# of the residual the control was invoked to measure).
#
# Why repeats at all: the loop is not deterministic. The same 20 seeds scored 14/20 then 13/20,
# with seven seeds flipping and one coverage value moving 0.28 -- PushT's contact dynamics
# amplify kernel-level float differences across 300 steps. Nothing in this repo can be configured
# to remove that, so the honest move is to measure it and put it in the interval.
#
# Each repeat writes `results/lerobot_control/repeats/r<K>.jsonl`. `scripts/control_spread.py`
# then reads every file it finds there plus the original `rows.jsonl`. Re-runs are cheap to
# extend: pass the first K to run and the loop appends nothing it does not overwrite.
set -euo pipefail

PY=".venv-lerobot/Scripts/python.exe"
FROM="${1:-2}"
TO="${2:-5}"
EPISODES="${EPISODES:-200}"
SEED="${SEED:-1000}"
BATCH="${BATCH:-8}"
OUT_DIR="results/lerobot_control/repeats"

if [ ! -x "$PY" ]; then
  echo "missing $PY -- run: bash scripts/build_lerobot_env.sh" >&2
  exit 1
fi
mkdir -p logs "$OUT_DIR"

for k in $(seq "$FROM" "$TO"); do
  echo "== repeat $k: $EPISODES episodes from seed $SEED, batch $BATCH, $(date +%H:%M:%S)"
  "$PY" scripts/lerobot_control_rows.py \
    --episodes "$EPISODES" --seed "$SEED" --batch "$BATCH" \
    --out "$OUT_DIR/r$k.jsonl" > "logs/control_r$k.log" 2>&1
  echo "repeat $k exit=$? $(date +%H:%M:%S)"
  grep -E "control:|paired on" "logs/control_r$k.log" || true
done

echo "== read them with: just control-spread"
