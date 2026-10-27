"""Measure peak VRAM and steps/s for batch sizes 32/64/128 (docs/03 section 4).

Writes results/ablation/memory.jsonl. The "fits" flag uses free VRAM measured before
the run minus 0.5 GiB, not the nominal 8 GiB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from flowpolicy.config import ModelConfig
from flowpolicy.data.normalizer import Stats
from flowpolicy.data.window import load_zarr, split_episodes
from flowpolicy.models.policies import build_policy


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = "cuda"
    arrays = load_zarr("data/pusht/pusht.zarr")
    train_eps, _ = split_episodes(len(arrays.episode_ends), 0.1, 0)
    rng = np.random.default_rng(0)
    stats = Stats.from_json("data/pusht/stats.json")
    policy = build_policy("flow", stats, 2, 16, 2, 2, ModelConfig()).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    free_before, _total = torch.cuda.mem_get_info()
    out = Path("results/ablation/memory.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    def batch(n: int) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, len(train_eps), n)
        frames = np.concatenate(
            [
                np.arange(arrays.episode_ends[ep - 1] if ep else 0, arrays.episode_ends[ep])
                for ep in idx
            ]
        )[:n]
        obs_idx = np.stack(
            [np.clip(np.arange(f - 1, f + 1), 0, arrays.episode_ends[-1] - 1) for f in frames]
        )
        act_idx = np.stack(
            [np.clip(np.arange(f, f + 16), 0, arrays.episode_ends[-1] - 1) for f in frames]
        )
        image = (
            torch.from_numpy(np.ascontiguousarray(arrays.image[obs_idx]))
            .permute(0, 1, 4, 2, 3)
            .float()
            / 255.0
        )
        return {
            "image": image.to(device),
            "state": torch.from_numpy(arrays.state[obs_idx]).to(device),
            "action": torch.from_numpy(arrays.action[act_idx]).to(device),
        }

    rows = []
    for n in (32, 64, 128):
        data = batch(n)
        policy.train()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = policy.loss(data)["loss"]
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        steps = 10
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = policy.loss(data)["loss"]
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        row = {
            "model": "flow",
            "params": policy.num_parameters(),
            "batch": n,
            "image_size": 96,
            "obs_horizon": 2,
            "pred_horizon": 16,
            "peak_vram_mb": peak,
            "steps_per_s": steps / elapsed,
            "fits": bool(peak < free_before / 2**20 - 512),
            "free_vram_mb_before": free_before / 2**20,
        }
        rows.append(row)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row))
        del data
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
