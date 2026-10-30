"""How much does the adapter-free control move between runs, and with it the like-for-like gap?

`docs/04` section 4 records that the control scored 134/200 and then 126/200 for the same
command, which is the same size as the ~8 pp residual it was invoked to measure. Two runs cannot
say whether that is the control's own noise or a change in the environment, so this reads K
repeats of the identical command and states the gap with an interval that includes it.

Two intervals, because they answer different questions:

* Percentile bootstrap over episodes *and* runs (`stats.paired_delta_ci`) for the gap itself,
  mean(control) - mean(ours): the term a null has to be stated with.
* A t-interval on the control's own mean rate across runs, which is the raw repeatability of the
  reference loop and does not involve this repo's rows at all.

Both are reported as-is. The gap interval is still a lower bound on the total uncertainty: this
repo's own arm is a single DDPM@100 run, so its noise-stream variance is unmeasured at that
sampler (only the DDIM@10 stream has been repeated: 52.0 % and 45.5 %).

    uv run python scripts/control_spread.py --ours results/e0_published_seeds_ddpm.jsonl \
        --runs results/lerobot_control/rows.jsonl results/lerobot_control/repeats/r*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Any

from policybench.eval.lerobot_rows import load_success, paired_counts
from policybench.eval.stats import mcnemar_exact, p_text, paired_delta_ci

# Two-sided 95 % t quantiles for df 1..10. A table instead of scipy: this is the only place the
# repo needs a t value, and the runs are few enough that the normal approximation would mislead.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def t_interval(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Mean of per-run rates with a 95 % t interval over runs (df = K - 1)."""
    k = len(values)
    if k < 2:
        return (float("nan"), float("nan"))
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    t = T95[min(k - 1, 10)]
    half = t * sd / math.sqrt(k)
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, default=Path("results/e0_published_seeds_ddpm.jsonl"))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["results/lerobot_control/rows.jsonl", "results/lerobot_control/repeats/r*.jsonl"],
        help="control row files; globs are expanded",
    )
    parser.add_argument("--out", type=Path, default=Path("results/lerobot_control/spread.json"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.runs:
        paths.extend(sorted(Path(p) for p in glob.glob(pattern)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit(f"no control rows matched {args.runs}")

    ours = {seed: float(hit) for seed, hit in load_success(args.ours).items()}
    runs = [load_success(path) for path in paths]
    n_seeds = len(ours)

    print(f"ours: {args.ours} ({n_seeds} seeds)")
    print(f"{'run':40s} {'success':>10s} {'delta pp':>9s} {'discordant':>11s}  McNemar")
    rows: list[dict[str, Any]] = []
    rates: list[float] = []
    for path, run in zip(paths, runs, strict=True):
        shared = len(set(ours) & set(run))
        hits = int(sum(run.values()))
        rate = hits / len(run)
        both, ours_only, theirs_only, neither = paired_counts(ours, run)
        p = mcnemar_exact(ours_only, theirs_only)
        delta = rate - sum(ours[s] for s in set(run) & set(ours)) / shared
        rates.append(rate)
        print(
            f"{path!s:40s} {hits:>4d}/{len(run):<3d} {delta * 100:>+8.1f} "
            f"{ours_only:>4d} / {theirs_only:<4d}   {p_text(p)}"
        )
        rows.append(
            {
                "path": str(path),
                "n": len(run),
                "success": hits,
                "rate": rate,
                "shared_seeds": shared,
                "pairing": {
                    "both": both,
                    "ours_only": ours_only,
                    "control_only": theirs_only,
                    "neither": neither,
                },
                "mcnemar_p": p,
                "delta_vs_ours": delta,
            }
        )

    mean_rate = statistics.fmean(rates)
    sd_rate = statistics.stdev(rates) if len(rates) > 1 else float("nan")
    lo_t, hi_t = t_interval(rates)
    lo_b, hi_b = paired_delta_ci(ours, runs, n_boot=args.n_boot)

    print()
    print(
        f"control across {len(rates)} runs: {mean_rate:.1%} (sd {sd_rate * 100:.1f} pp, "
        f"range {min(rates):.1%}-{max(rates):.1%})"
    )
    print(f"  t interval over runs: [{lo_t:.1%}, {hi_t:.1%}]")
    print(
        "like-for-like gap (control - ours), bootstrap over episodes x runs: "
        f"[{lo_b * 100:+.1f}, {hi_b * 100:+.1f}] pp at {1 - 0.05:.0%}"
    )

    payload = {
        "ours": str(args.ours),
        "runs": rows,
        "control_mean_rate": mean_rate,
        "control_sd_pp": sd_rate * 100,
        "control_t_interval": [lo_t, hi_t],
        "delta_ci_pp": [lo_b * 100, hi_b * 100],
        "delta_point_pp": (mean_rate - sum(ours.values()) / n_seeds) * 100,
        "n_boot": args.n_boot,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
