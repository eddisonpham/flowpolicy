"""Pin the published VQ-BeT evaluation block as JSONL, so the paired test is reproducible.

`lerobot/vqbet_pusht` ships `eval_info.json` with one row per episode. E15 compares this
repo's 50 episodes against the checkpoint's rows for the *same* eval seeds; downloading that
file at table time would make `just tables` need the network, so it is fetched once here and
committed as `results/vqbet_published.jsonl` (docs/04 section 3).

Two field notes, both checked against the installed gym-pusht (`envs/pusht.py` line 257):
`reward = clip(coverage / 0.95, 0, 1)`, so the published `max_reward` is **coverage divided
by 0.95**, not coverage. `success` is the env's `is_success` flag, i.e. `coverage > 0.95`,
which is the same criterion this repo uses. Comparing our `coverage` column against their
`max_reward` column directly would therefore invent a 5 % gap.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_ID = "lerobot/vqbet_pusht"
OUT = Path("results/vqbet_published.jsonl")
SUCCESS_THRESHOLD = 0.95


def main() -> None:
    path = Path(hf_hub_download(HF_ID, "eval_info.json"))
    info = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for episode in info["per_episode"]:
        # Undo gym-pusht's reward scaling: max_reward * 0.95 is the max coverage it reached.
        rows.append(
            {
                "source": HF_ID,
                "eval_seed": int(episode["seed"]),
                "success": int(bool(episode["success"])),
                "max_reward": float(episode["max_reward"]),
                "coverage": min(1.0, float(episode["max_reward"]) * SUCCESS_THRESHOLD),
            }
        )
    rows.sort(key=lambda row: row["eval_seed"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    hit = sum(row["success"] for row in rows)
    print(f"wrote {OUT} ({len(rows)} episodes, {hit}/{len(rows)} = {hit / len(rows):.1%})")
    print(f"seeds {rows[0]['eval_seed']}-{rows[-1]['eval_seed']}")


if __name__ == "__main__":
    main()
