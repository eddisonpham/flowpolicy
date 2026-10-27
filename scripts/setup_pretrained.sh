#!/usr/bin/env bash
# Install the optional pretrained-policy backend on top of the uv sync.
#
# Why this is not in pyproject: lerobot 0.6.1 requires torch < 2.12 and this project
# pins torch 2.14.1+cu130 (docs/02). A normal resolve therefore wants to downgrade
# torch/torchvision, which would invalidate every number measured on this GPU. Installing
# `--no-deps` keeps the pinned stack and adds only the packages lerobot's inference path
# actually imports. Verified working: torch 2.14.1+cu130, lerobot 0.6.1, diffusers 0.42.1.
#
# `uv sync` (just setup) prunes these, so re-run this script after a sync.
set -euo pipefail

uv pip install --no-deps "lerobot==0.6.1"
uv pip install "diffusers>=0.42" "draccus==0.11.6" "einops==0.8.2" "safetensors>=0.4" \
  "opencv-python-headless>=4.9" termcolor

uv run python - <<'PY'
import torch
import lerobot
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: F401

print(f"pretrained backend ready: lerobot {lerobot.__version__}, torch {torch.__version__}")
PY
