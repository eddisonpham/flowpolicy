"""Per-episode rows from the adapter-free control (docs/04 section 4, docs/09 option B).

`lerobot-eval` 0.6.1 writes only `eval_info.json`, so the control at `results/lerobot_control/`
could only be compared in aggregate while every other comparison in this repo is paired by env
seed. The records it computes *are* per episode; this script wraps `eval_policy` to capture them
and appends them in this repo's row schema, so the control pairs with
`results/e0_published_seeds.jsonl` and gets McNemar instead of a Fisher comparison across
different samples.

Nothing of this repo's policy code joins the measured path: `lerobot` still builds the policy,
the env, the processor pipeline and the loop. Only the *writer* is hooked.

Run it with the isolated interpreter, never the project's (lerobot caps torch < 2.12):

    .venv-lerobot/Scripts/python.exe scripts/lerobot_control_rows.py --episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policybench.eval.lerobot_rows import (
    SUCCESS_THRESHOLD,
    control_row,
    load_success,
    paired_counts,
)
from policybench.eval.stats import mcnemar_exact, p_text


def _summary(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    hits = sum(row["success"] for row in rows)
    coverages = sorted(row["coverage"] for row in rows)
    median = coverages[n // 2] if n else float("nan")
    censored = sum(1 for row in rows if row["coverage_censored"])
    return (
        f"{int(hits)}/{n} = {hits / n:.1%} success, median coverage {median:.3f} "
        f"({censored} rows censored at {SUCCESS_THRESHOLD})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--checkpoint", default="checkpoints/diffusion_pusht_migrated")
    parser.add_argument("--out", type=Path, default=Path("results/lerobot_control/rows.jsonl"))
    parser.add_argument("--config-id", default=None)
    parser.add_argument(
        "--ours",
        type=Path,
        default=Path("results/e0_published_seeds.jsonl"),
        help="this repo's rows on the same seeds, for the paired test",
    )
    args = parser.parse_args()

    migrated = Path(args.checkpoint)
    if not (migrated / "policy_preprocessor.json").exists():
        raise SystemExit(
            f"missing migrated checkpoint in {migrated} -- run `just lerobot-control` once "
            "(it builds the processors this loop needs)"
        )
    config_id = args.config_id or f"lerobot_control_n{args.episodes}"

    from lerobot.scripts import lerobot_eval

    original = lerobot_eval.eval_policy
    written: list[dict[str, Any]] = []

    def hooked(*call_args: Any, **call_kwargs: Any) -> Any:
        result = original(*call_args, **call_kwargs)
        rows = [
            control_row(episode, config_id=config_id, checkpoint=str(migrated))
            for episode in result["per_episode"]
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
        written.extend(rows)
        print(f"control rows: {len(written)} written to {args.out}", flush=True)
        return result

    lerobot_eval.eval_policy = hooked
    if args.out.exists():
        args.out.unlink()
    sys.argv = [
        "lerobot-eval",
        f"--policy.path={migrated}",
        "--env.type=pusht",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.batch_size={args.batch}",
        "--eval.use_async_envs=false",
        f"--seed={args.seed}",
        f"--output_dir={args.out.parent}",
    ]
    lerobot_eval.eval_main()

    if not written:
        raise SystemExit("lerobot-eval returned no per-episode records; nothing written")
    print(f"\ncontrol: {_summary(written)}")

    if args.ours.exists():
        ours = load_success(args.ours)
        theirs = {row["eval_seed"]: row["success"] for row in written}
        both, ours_only, theirs_only, neither = paired_counts(ours, theirs)
        shared = both + ours_only + theirs_only + neither
        p = mcnemar_exact(ours_only, theirs_only)
        print(
            f"paired on {shared} shared seeds against {args.ours}: "
            f"{both} both, {ours_only} ours only, {theirs_only} control only, {neither} neither "
            f"-> exact McNemar {p_text(p)}"
        )


if __name__ == "__main__":
    main()
