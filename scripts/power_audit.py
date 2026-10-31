"""What could each quoted block in `results/` actually have resolved?

Every "not separated" in this repo was computed on a block whose size was chosen before the effect
was known, and a null is only as informative as the smallest effect the block could see. This reads
the committed row files, recovers each comparison's 2x2, and states the smallest effect its n can
resolve at 80 % power (`stats.resolvable_effect`) next to the effect it actually observed.

A row whose observed difference is below its resolution is not evidence of no difference: it is
evidence that the block was too small for that size of effect, and the audit says so explicitly
instead of leaving the reader to infer it from a p-value.

    uv run python scripts/power_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policybench.eval.lerobot_rows import load_success, paired_counts
from policybench.eval.stats import mcnemar_exact, p_text, resolvable_effect

# (label, rows a, rows b). Same-seed pairs only: these are all paired comparisons, and one
# disjoint-block comparison (the VQ-BeT replication) is deliberately absent because a paired power
# calculation does not apply to it.
COMPARISONS = (
    (
        "E0 adapter DDIM@10 vs the checkpoint's block",
        "results/e0_published_seeds.jsonl",
        "results/diffusion_published.jsonl",
    ),
    (
        "E0 adapter DDPM@100 vs the checkpoint's block",
        "results/e0_published_seeds_ddpm.jsonl",
        "results/diffusion_published.jsonl",
    ),
    (
        "sampler: DDPM@100 vs DDIM@10, both ours",
        "results/e0_published_seeds_ddpm.jsonl",
        "results/e0_published_seeds.jsonl",
    ),
    (
        "noise stream: offset 500000 vs 0, same sampler",
        "results/e0_published_seeds_noise2.jsonl",
        "results/e0_published_seeds.jsonl",
    ),
    (
        "like-for-like: adapter DDPM@100 vs adapter-free control",
        "results/e0_published_seeds_ddpm.jsonl",
        "results/lerobot_control/rows.jsonl",
    ),
    (
        "adapter DDIM@10 vs adapter-free control",
        "results/e0_published_seeds.jsonl",
        "results/lerobot_control/rows.jsonl",
    ),
    (
        "VQ-BeT adapter vs its published block",
        "results/vqbet.jsonl",
        "results/vqbet_published.jsonl",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/power_audit.json"))
    args = parser.parse_args()

    rows = []
    header = (
        f"{'comparison':<52s} {'n':>4s} {'disc':>5s} {'psi':>5s} "
        f"{'delta':>7s} {'p':>10s} {'resolves':>9s}"
    )
    print(header)
    for label, path_a, path_b in COMPARISONS:
        a, b = Path(path_a), Path(path_b)
        if not (a.exists() and b.exists()):
            print(f"{label:<52s} skipped (missing rows)")
            continue
        sa, sb = load_success(a), load_success(b)
        both, a_only, b_only, neither = paired_counts(sa, sb)
        n = both + a_only + b_only + neither
        if n == 0:
            print(f"{label:<52s} skipped (no shared seeds)")
            continue
        discordance = (a_only + b_only) / n
        delta = (b_only - a_only) / n
        p = mcnemar_exact(a_only, b_only)
        mde = resolvable_effect(n, discordance)
        rows.append(
            {
                "comparison": label,
                "a": path_a,
                "b": path_b,
                "n": n,
                "discordant": {"a_only": a_only, "b_only": b_only},
                "discordance": round(discordance, 4),
                "delta_pp": round(delta * 100, 1),
                "mcnemar_p": p,
                "resolves_pp": round(mde * 100, 1),
                # A significant result needs no resolution: the block found the effect. What is
                # flagged is the other case -- a null on a difference the block could not have seen.
                "uninformative_null": p >= 0.05 and abs(delta) < mde,
            }
        )
        print(
            f"{label:<52s} {n:>4d} {a_only + b_only:>5d} {discordance:>5.2f} "
            f"{delta * 100:>+6.1f} {p_text(p):>10s} {mde * 100:>8.1f} pp"
        )

    print()
    for row in rows:
        if row["uninformative_null"]:
            print(
                f"UNINFORMATIVE NULL  {row['comparison']}: {row['delta_pp']:+.1f} pp on "
                f"{row['n']} episodes, but the block resolves only {row['resolves_pp']:.1f} pp "
                f"at 80 % power -- it cannot exclude effects up to that size."
            )

    payload = {
        "power": 0.8,
        "alpha": 0.05,
        "comparisons": rows,
        "uninformative_nulls": [r["comparison"] for r in rows if r["uninformative_null"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
