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


def p_text(p: float) -> str:
    """Render a p-value. Printing "p = 0.00" reads as a rounding bug; below 1e-3 say so."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
