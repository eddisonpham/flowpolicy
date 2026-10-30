#!/usr/bin/env bash
# Build the isolated, pinned environment for the adapter-free control (docs/04 section 4).
#
# Separate from `.venv` on purpose: lerobot 0.6.1 caps `torch<2.12` and `av<16`, and the
# project pins torch 2.14.1+cu130 to keep every measured number comparable. Installing
# lerobot's resolution into `.venv` would silently change the torch build under the
# experiments, so the control gets its own venv and the project env is never touched.
#
# torch is reinstalled afterwards from the cu130 index with `--no-deps`, the same trick
# `scripts/setup_pretrained.sh` uses: lerobot's inference path only needs torch APIs that are
# stable across 2.11 -> 2.14, and a CPU torch would make a 200-episode DDPM@100 control take
# hours longer than it needs to.
set -euo pipefail

VENV=".venv-lerobot"
uv venv "$VENV" --python 3.12
export VIRTUAL_ENV="$PWD/$VENV"

# `av<16` is explicit because lerobot only declares it behind an extra: without it the
# evaluation path dies importing `lerobot.datasets` (`module 'av' has no attribute 'option'`),
# which is the same failure the project env hits with its own av 19.
uv pip install "lerobot==0.6.1" "gym-pusht==0.1.8" "av>=15,<16"
uv pip install --no-deps --reinstall "torch==2.14.1" "torchvision==0.29.1" \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install "diffusers==0.42.1"

"$VENV/Scripts/python.exe" -c "
import av, torch, lerobot
print('pinned control env:', 'lerobot', lerobot.__version__, '| av', av.__version__,
      '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
"
