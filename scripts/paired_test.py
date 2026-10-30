"""Paired McNemar between two results files, on the seeds both cover.

Every pre-registered comparison in docs/04 is paired by eval seed, and until now each one was
run as an ad-hoc snippet. One implementation, so the same 2x2 cannot be counted two ways (the
rule that moved `mcnemar_exact` into `policybench/eval/stats.py`).

Seeds present in only one file are dropped rather than counted as failures.

    uv run python scripts/paired_test.py --a results/e0_published_seeds.jsonl \\
        --b results/e0_published_seeds_ddpm.jsonl --label-a ddim@10 --label-b ddpm@100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policybench.eval.lerobot_rows import load_success, paired_counts
from policybench.eval.stats import mcnemar_exact, p_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    args = parser.parse_args()

    a = load_success(args.a)
    b = load_success(args.b)
    both, a_only, b_only, neither = paired_counts(a, b)
    shared = both + a_only + b_only + neither
    if shared == 0:
        raise SystemExit(f"{args.a} and {args.b} share no eval_seed; nothing to pair")

    pa = (both + a_only) / shared
    pb = (both + b_only) / shared
    p = mcnemar_exact(a_only, b_only)
    print(
        json.dumps(
            {
                "seeds": shared,
                "a": {"label": args.label_a, "success_rate": round(pa, 4)},
                "b": {"label": args.label_b, "success_rate": round(pb, 4)},
                "discordant": {"a_only": a_only, "b_only": b_only},
                "both": both,
                "neither": neither,
                "p_value": None if p is None else float(p),
                "p_text": p_text(p),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
