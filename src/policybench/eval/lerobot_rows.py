"""Turn `lerobot-eval`'s per-episode records into this repo's row schema.

The adapter-free control (`scripts/lerobot_control.sh`) runs the checkpoint through `lerobot`'s
own eval loop, but that version writes only `eval_info.json` -- aggregates plus one record per
episode, no JSONL. The records it computes are the same quantities this repo records, so the
control can be made *paired* with `results/e0_published_seeds.jsonl` instead of compared in
aggregate: same env seeds, both runs, McNemar instead of Fisher.

Two field notes, the same ones `scripts/published_blocks.py` carries, checked against the
installed gym-pusht: `reward = clip(coverage / 0.95, 0, 1)`, so `max_reward * 0.95` is the max
coverage reached, and the clip censors the successful half at 0.95. `coverage_censored` marks
those rows so a coverage comparison never treats a lower bound as an exact value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# gym-pusht: reward = clip(coverage / 0.95, 0, 1). Inverting it needs the same constant.
SUCCESS_THRESHOLD = 0.95


def control_row(episode: Mapping[str, Any], *, config_id: str, checkpoint: str) -> dict[str, Any]:
    """One `per_episode` record -> one row that pairs with this repo's rows on `eval_seed`."""
    max_reward = float(episode["max_reward"])
    success = float(bool(episode["success"]))
    return {
        "eval_seed": int(episode["seed"]),
        "success": success,
        # Exact below the threshold, a lower bound at 0.95 for successes (see module docstring).
        "coverage": min(1.0, max_reward * SUCCESS_THRESHOLD),
        "coverage_censored": success == 1.0,
        "max_reward": max_reward,
        "sum_reward": float(episode["sum_reward"]),
        "config_id": config_id,
        "policy": "lerobot-control",
        "source": checkpoint,
        "checkpoint": checkpoint,
        "weights": "pretrained",
        "train_seed": -1,
        "scheduler": "ddpm",
        "sampler": "stock",
        "dtype": "float32",
        "task": "pusht",
    }


def paired_counts(
    ours: Mapping[int, float], theirs: Mapping[int, float]
) -> tuple[int, int, int, int]:
    """2x2 counts over the seeds both runs cover: (both, ours_only, theirs_only, neither).

    Seeds one run has and the other does not are dropped rather than counted as failures: a
    seed missing from one run is not evidence about the policy.
    """
    both = ours_only = theirs_only = neither = 0
    for seed in sorted(set(ours) & set(theirs)):
        we_hit, they_hit = bool(ours[seed]), bool(theirs[seed])
        if we_hit and they_hit:
            both += 1
        elif we_hit:
            ours_only += 1
        elif they_hit:
            theirs_only += 1
        else:
            neither += 1
    return both, ours_only, theirs_only, neither


def load_success(path: Any) -> dict[int, float]:
    """Read `eval_seed -> success` from a results JSONL, for pairing two runs."""
    import json
    from pathlib import Path

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {int(row["eval_seed"]): float(row["success"]) for row in rows}
