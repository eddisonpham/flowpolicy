"""How many paired episodes does the processor A/B need to resolve ~4 pp?

`docs/10` section 5 sized the arm at 600 seeds when the residual looked like 8 pp. The control is
now repeated, which moved the residual to ~4 pp (`docs/04` section 4) -- a smaller effect needs
*more* episodes, so 600 no longer resolves what the arm exists to test.

The episode count is dominated by the discordance rate, not by the effect: paired-binary power
scales with `sqrt(psi)`, so a 4 pp difference costs ~250 episodes when the arms disagree on 5 % of
episodes and ~2,200 when they disagree on 44 % of them (the rate the DDIM-vs-DDPM comparison
actually showed, where the two arms differ by a *sampler*, not by a 1-ulp arithmetic). That range
is why the design sizes itself from a pilot instead of from one prior.

So this prints the whole curve, the three sizes the pilot can select between, and the cost of each
at the measured per-chunk latency -- including the repeat the largest branch needs, because at that
size the episode-level standard error is the same as the control's measured run-to-run sd.

    uv run python scripts/arm_power.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policybench.eval.stats import mcnemar_power, paired_binary_n

# Measured on this host (docs/05, docs/09): 417 ms per chunk at DDIM@10, 1819 ms at DDPM@100, and
# ~13 chunks per 300-step episode. Kept here rather than re-measured: this script sizes a block, it
# does not produce a result.
DDIM_CHUNK_MS = 417.0
DDPM_CHUNK_MS = 1819.0
CHUNKS_PER_EPISODE = 13.0
ARMS = 2

DISCORDANCES = (0.05, 0.10, 0.20, 0.30, 0.44)

# (rule, psi at the top of the branch, episodes per arm, passes per arm)
BRANCHES = (
    ("psi_hat <= 0.10", 0.10, 600, 1),
    ("0.10 < psi_hat <= 0.30", 0.30, 1600, 1),
    ("psi_hat > 0.30", 0.44, 2200, 2),
)


def cost_hours(episodes: int, chunk_ms: float, passes: int = 1) -> float:
    seconds = episodes * CHUNKS_PER_EPISODE * chunk_ms / 1000.0
    return ARMS * passes * seconds / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=float, default=0.04, help="residual to resolve, as a rate")
    parser.add_argument("--out", type=Path, default=Path("results/arm_power.json"))
    args = parser.parse_args()

    print(
        f"episodes per arm for an exact McNemar to resolve "
        f"{args.effect * 100:.1f} pp at 80 % power\n"
    )
    print("effect   " + "".join(f"psi={psi:<8.2f}" for psi in DISCORDANCES))
    curve: dict[str, dict[str, int]] = {}
    for effect in (args.effect, 0.08):
        row = {f"{psi:.2f}": paired_binary_n(effect, psi) for psi in DISCORDANCES}
        curve[f"{effect:.2f}"] = row
        print(f"{effect * 100:>4.1f} pp  " + "".join(f"{n:<12d}" for n in row.values()))

    # The size is chosen from the discordance rate alone, never from the direction or the p-value,
    # so the final test keeps its nominal size.
    print()
    print("pilot table (first 200 paired episodes, discordant fraction psi_hat):")
    branches = []
    for rule, psi, n, passes in BRANCHES:
        power = mcnemar_power(n, args.effect, psi, trials=400, seed=0)
        hours = cost_hours(n, DDIM_CHUNK_MS, passes)
        print(
            f"  {rule:<22s} -> {n:>4d}/arm x{passes}: power {power:.2f} at psi={psi:.2f}, "
            f"{hours:.1f} GPU-h at DDIM@10"
        )
        branches.append(
            {
                "rule": rule,
                "episodes_per_arm": n,
                "passes_per_arm": passes,
                "power_at_branch_psi": power,
                "psi": psi,
                "gpu_hours_ddim10": round(hours, 2),
                "gpu_hours_ddpm100": round(cost_hours(n, DDPM_CHUNK_MS, passes), 2),
            }
        )

    payload = {
        "effect_pp": args.effect * 100,
        "episodes_per_arm_by_discordance": curve,
        "branches": branches,
        "assumptions": {
            "chunk_ms_ddim10": DDIM_CHUNK_MS,
            "chunk_ms_ddpm100": DDPM_CHUNK_MS,
            "chunks_per_episode": CHUNKS_PER_EPISODE,
            "arms": ARMS,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
