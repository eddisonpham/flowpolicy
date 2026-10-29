"""G7: PushT closed loop with a random policy (docs/02 section 7).

Success is not expected; the gate is that reset/step/render/JSONL work.
Records one line per episode in results/gates/g7.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from policybench.envs.pusht import MAX_STEPS, PushTEnv


def main() -> int:
    rng = np.random.default_rng(0)
    out = Path("results/gates/g7.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(3):
        env = PushTEnv()
        obs = env.reset(seed=10_000 + seed)
        steps = 0
        coverage = 0.0
        success = 0.0
        while steps < MAX_STEPS:
            action = rng.uniform(0.0, 512.0, size=(2,)).astype(np.float32)
            obs, _reward, done, info = env.step(action)
            steps += 1
            coverage = max(coverage, info["coverage"])
            success = max(success, info["success"])
            if done:
                break
        env.close()
        rows.append(
            {
                "task": "pusht",
                "seed": 10_000 + seed,
                "policy": "random",
                "steps": steps,
                "success": success,
                "coverage": coverage,
                "image_shape": list(obs["image"].shape),
                "state_shape": list(obs["state"].shape),
            }
        )
    with out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    shape_ok = all(r["image_shape"] == [96, 96, 3] and r["state_shape"] == [2] for r in rows)
    print(f"{'PASS' if shape_ok else 'FAIL'}: {len(rows)} random episodes, wrote {out}")
    return 0 if shape_ok else 1


if __name__ == "__main__":
    sys.exit(main())
