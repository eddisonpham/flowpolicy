"""Quantify what a numerically equivalent sampler changes in the deploy block (docs/04).

The fast DDIM loop and `diffusers`' loop agree to ~1e-5 of an action
(`tests/gpu/test_pretrained_sampler_equivalence.py`), and PushT turns that into different episodes.
This pairs two deploy blocks by episode and reports the gap with the resolution of the block, so
the sensitivity is a number rather than a remark.

    uv run python scripts/sampler_sensitivity.py \\
        --fast results/deploy_fast_sampler_45.jsonl results/deploy_fast_sampler_ext.jsonl \\
        --stock results/deploy.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from policybench.eval.sampler_ablation import pairs, render, summarize


def read(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", type=Path, nargs="+", required=True, help="fast-sampler blocks")
    parser.add_argument(
        "--stock", type=Path, nargs="+", default=[Path("results/deploy.jsonl")], help="stock blocks"
    )
    parser.add_argument("--boot", type=int, default=4000, help="bootstrap resamples")
    parser.add_argument("--out", type=Path, default=Path("results/sampler_sensitivity.json"))
    args = parser.parse_args()

    summary = summarize(pairs(read(args.fast), read(args.stock)), n_boot=args.boot)
    print(render(summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
