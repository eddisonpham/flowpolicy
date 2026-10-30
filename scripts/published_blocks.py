"""Pin each published checkpoint's per-episode eval block, so the paired tests are reproducible.

`lerobot/diffusion_pusht` and `lerobot/vqbet_pusht` each ship `eval_info.json` with one row per
episode. E0 and E15 compare this repo's episodes against the checkpoint's rows for the *same*
eval seeds; downloading that file at table time would make `just tables` need the network, so it
is fetched once here and committed under `results/` (docs/04 section 3).

Two field notes, both checked against the installed gym-pusht (`envs/pusht.py` line 257):
`reward = clip(coverage / 0.95, 0, 1)`, so the published `max_reward` is **coverage divided by
0.95**, not coverage. `success` is the env's `is_success` flag, i.e. `coverage > 0.95`, which is
the same criterion this repo uses. Comparing our `coverage` column against their `max_reward`
column directly invents a 5 % gap. The clip also censors the successful half: a success records
`max_reward = 1.0`, so its coverage is only bounded below at 0.95 -- `coverage_censored` marks
those rows, and only the failed episodes carry an exact coverage value.

Both blocks are 500 episodes: diffusion on seeds 1000-1499, VQ-BeT on 100000-100499.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

BLOCKS = {
    "lerobot/vqbet_pusht": Path("results/vqbet_published.jsonl"),
    "lerobot/diffusion_pusht": Path("results/diffusion_published.jsonl"),
}
SUCCESS_THRESHOLD = 0.95


def main() -> None:
    for repo, out in BLOCKS.items():
        path = Path(hf_hub_download(repo, "eval_info.json"))
        info = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for episode in info["per_episode"]:
            # Undo gym-pusht's reward scaling: max_reward * 0.95 is the max coverage reached.
            success = bool(episode["success"])
            rows.append(
                {
                    "source": repo,
                    "eval_seed": int(episode["seed"]),
                    "success": int(success),
                    # Reward is clipped at 1.0, so every success reports exactly 0.95 here
                    # whatever its true coverage was. Failures are exact.
                    "coverage_censored": success,
                    "max_reward": float(episode["max_reward"]),
                    "coverage": min(1.0, float(episode["max_reward"]) * SUCCESS_THRESHOLD),
                }
            )
        rows.sort(key=lambda row: row["eval_seed"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        hit = sum(row["success"] for row in rows)
        print(
            f"wrote {out} ({len(rows)} episodes, {hit}/{len(rows)} = {hit / len(rows):.1%}, "
            f"seeds {rows[0]['eval_seed']}-{rows[-1]['eval_seed']})"
        )


if __name__ == "__main__":
    main()
