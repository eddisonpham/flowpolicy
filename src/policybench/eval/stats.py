"""Exact 2x2 tests for the paired and replication comparisons.

Two questions recur in `results/`, and the tests are not interchangeable:

* *Same episodes, both runs* -- this repo against a checkpoint's own eval block over the same
  seeds. The samples are paired, so the test is McNemar over the discordant episodes.
* *Different episodes* -- a replication block over fresh seeds. The samples are independent,
  so the test is Fisher exact.

They live here rather than in `scripts/make_tables.py` so that `mypy --strict` covers them and
both readers of a verdict -- the table generator and the README headline in `eval/report.py` --
compute it with the same code instead of each writing its own sentence.
"""

from __future__ import annotations

import math


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    if row1 == 0 or col1 == 0 or n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-12))


def mcnemar_exact(ours_only: int, theirs_only: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs."""
    n = ours_only + theirs_only
    if n == 0:
        return 1.0
    # 2.0, not 2: `int ** int` is typed Any (a negative exponent returns a float).
    tail = sum(math.comb(n, i) for i in range(min(ours_only, theirs_only) + 1)) / 2.0**n
    return min(1.0, 2 * tail)


def paired_delta_ci(
    ours: dict[int, float],
    control_runs: list[dict[int, float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for mean(control) - mean(ours), resampling episodes *and* runs.

    The reference loop is not deterministic: a rerun of the same command flips episodes near the
    success threshold, because PushT's contact dynamics amplify kernel-level float differences
    across 300 steps (measured 2026-10-30, `docs/04` section 4). A bootstrap over episodes alone
    would therefore quote an interval that is too narrow by exactly the term the verdict depends
    on, so this is a two-stage bootstrap: episodes are drawn with replacement, and for each drawn
    episode one control *run* is drawn at random from the repeats. The result covers both the
    episodic sampling noise and the reference's run-to-run noise.

    Only seeds present in every run are used: a seed missing from one run is not evidence about
    the policy. Returns (low, high), or (nan, nan) when nothing is shared.
    """
    import random

    if not control_runs:
        return float("nan"), float("nan")
    shared = sorted(set(ours).intersection(*(set(run) for run in control_runs)))
    n = len(shared)
    if n == 0:
        return float("nan"), float("nan")

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        ours_hits = control_hits = 0.0
        for _ in range(n):
            episode = shared[rng.randrange(n)]
            ours_hits += float(ours[episode])
            control_hits += float(control_runs[rng.randrange(len(control_runs))][episode])
        deltas.append((control_hits - ours_hits) / n)
    deltas.sort()
    lo = deltas[min(n_boot - 1, int((alpha / 2) * n_boot))]
    hi = deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def p_text(p: float) -> str:
    """Render a p-value. Printing "p = 0.00" reads as a rounding bug; below 1e-3 say so."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
