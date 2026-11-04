"""How much does a numerically equivalent sampler change deploy outcomes?

`tests/gpu/test_pretrained_sampler_equivalence.py` proves the fast DDIM loop and `diffusers`' loop
produce the same action chunk to ~1e-5 px. That is a statement about the *sampler*; it says nothing
about the *policy*. PushT is contact-rich, so 1e-5 turns into different trajectories and different
successes, and the deploy block is where that shows up: the 2026-11-03 pipeline run accidentally
measured the fast loop against a committed block measured with the stock one and disagreed on 23 of
45 episodes (CHANGELOG). The question that finding left open is how large the sensitivity is, and
at what resolution a block of that size can state it at all -- which is what this module answers,
so the number is "23/45 changed hands, psi = 0.51, this block resolves X pp" rather than "the
sampler matters".

Pairing is by (eval seed, episode index, injected latency): the same episode under the same control
conditions, which is what makes the 2x2 McNemar's rather than a comparison of two rates across
different blocks. Seeds present in only one block are dropped, never counted as failures.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from policybench.eval.stats import mcnemar_exact, p_text, paired_binary_n, resolvable_effect

#: The fields that identify one deploy episode. `latency_ms` and `strategy` are conditions, not
#: labels: a 0 ms episode and a 200 ms one are different measurements of the same seed, and
#: `results/deploy.jsonl` holds four strategies on the same seeds. Pairing without them matched a
#: fast `naive` episode against a committed `blend-r2` one on the first run of this script and
#: reported a 42 % stock rate where the like-for-like number is 69 %.
KEY_FIELDS = ("eval_seed", "episode", "latency_ms", "strategy")

#: What a row must carry to be an episode at all. `strategy` is in the key but optional in the row:
#: a block measured before the field existed is still pairable, and an absent strategy matches only
#: another absent one rather than every strategy at once.
REQUIRED_FIELDS = ("eval_seed", "episode", "latency_ms")


@dataclass(frozen=True)
class Pair:
    """One deploy episode, run once under each sampler at the same latency."""

    latency_ms: float
    fast: bool
    stock: bool
    steps_fast: int | None = None
    steps_stock: int | None = None
    coverage_fast: float | None = None
    coverage_stock: float | None = None


def _opt_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, float, str], dict[str, Any]]:
    """Index deploy rows by episode and condition; a row without those is not an episode."""
    out: dict[tuple[int, int, float, str], dict[str, Any]] = {}
    for row in rows:
        if "success" not in row or any(row.get(name) is None for name in REQUIRED_FIELDS):
            continue
        key = (
            int(row["eval_seed"]),
            int(row["episode"]),
            float(row["latency_ms"]),
            str(row.get("strategy") or ""),
        )
        out[key] = row
    return out


def pairs(fast_rows: list[dict[str, Any]], stock_rows: list[dict[str, Any]]) -> list[Pair]:
    """Pair the two blocks on the episodes and conditions they both cover."""
    fast, stock = index_rows(fast_rows), index_rows(stock_rows)
    out: list[Pair] = []
    for key in sorted(set(fast) & set(stock)):
        here, there = fast[key], stock[key]
        out.append(
            Pair(
                latency_ms=key[2],  # the strategy is part of the key, so both sides share it
                fast=bool(here["success"]),
                stock=bool(there["success"]),
                steps_fast=_opt_int(here.get("steps")),
                steps_stock=_opt_int(there.get("steps")),
                coverage_fast=_opt_float(here.get("coverage")),
                coverage_stock=_opt_float(there.get("coverage")),
            )
        )
    return out


def bootstrap_delta(
    sample: list[Pair], *, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile interval for mean(stock) - mean(fast), resampling episodes within each latency.

    Stratified rather than pooled: latency is a condition the block varies on purpose, and a pooled
    resample would let a 0 ms episode stand in for a 200 ms one, which is not the comparison.
    """
    strata: dict[float, list[Pair]] = defaultdict(list)
    for pair in sample:
        strata[pair.latency_ms].append(pair)
    if not sample:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        hits = 0.0
        total = 0
        for group in strata.values():
            for _ in range(len(group)):
                pair = group[rng.randrange(len(group))]
                hits += float(pair.stock) - float(pair.fast)
                total += 1
        deltas.append(hits / total)
    deltas.sort()
    lo = deltas[min(n_boot - 1, int((alpha / 2) * n_boot))]
    hi = deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def _radius(delta: float, psi: float) -> int | None:
    """Episodes per arm a block would need to resolve `delta` at this discordance."""
    if delta <= 0 or psi < delta**2:
        # No discordant episodes to test, or the effect is larger than the discordance can carry
        # (impossible under the paired model): the closed form has no answer, so neither do we.
        return None
    return paired_binary_n(abs(delta), psi)


def _cell(sample: list[Pair], *, n_boot: int, seed: int) -> dict[str, Any]:
    n = len(sample)
    fast_ok = sum(pair.fast for pair in sample)
    stock_ok = sum(pair.stock for pair in sample)
    fast_only = sum(1 for pair in sample if pair.fast and not pair.stock)
    stock_only = sum(1 for pair in sample if pair.stock and not pair.fast)
    delta = (stock_ok - fast_ok) / n
    psi = (fast_only + stock_only) / n
    p = mcnemar_exact(fast_only, stock_only)
    lo, hi = bootstrap_delta(sample, n_boot=n_boot, seed=seed)
    resolves = resolvable_effect(n, psi) if psi > 0 else None
    return {
        "n": n,
        "fast_success": fast_ok / n,
        "stock_success": stock_ok / n,
        "agree": n - fast_only - stock_only,
        "fast_only": fast_only,
        "stock_only": stock_only,
        "delta_pp": round(delta * 100, 1),
        "ci_pp": [round(lo * 100, 1), round(hi * 100, 1)],
        "discordance": round(psi, 3),
        "mcnemar_p": p,
        "p_text": p_text(p),
        "resolves_pp": None if resolves is None else round(resolves * 100, 1),
        "episodes_to_resolve_delta": _radius(delta, psi),
    }


def summarize(sample: list[Pair], *, n_boot: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Per-latency cells plus the pooled one, with the resolution of each.

    A cell that shares no episode is absent rather than zero: a missing latency level is a missing
    measurement, not a run where nothing succeeded.
    """
    if not sample:
        return {"n": 0, "status": "no shared episodes", "cells": [], "overall": None}
    strata: dict[float, list[Pair]] = defaultdict(list)
    for pair in sample:
        strata[pair.latency_ms].append(pair)
    cells = [
        {"latency_ms": latency, **_cell(rows, n_boot=n_boot, seed=seed)}
        for latency, rows in sorted(strata.items())
    ]
    overall = _cell(sample, n_boot=n_boot, seed=seed)
    disagreements = [
        pair
        for pair in sample
        if pair.fast != pair.stock and pair.steps_fast is not None and pair.steps_stock is not None
    ]
    return {
        "n": len(sample),
        "cells": cells,
        "overall": overall,
        # The episodes that changed hands are the ones worth looking at: if they also ended at a
        # different env step, the divergence is in the trajectory, not in the final frame's noise.
        "discordant_with_step_data": len(disagreements),
        "discordant_ending_elsewhere": sum(
            1 for pair in disagreements if pair.steps_fast != pair.steps_stock
        ),
        "mean_abs_step_change": (
            round(
                sum(abs(int(p.steps_fast or 0) - int(p.steps_stock or 0)) for p in disagreements)
                / len(disagreements),
                1,
            )
            if disagreements
            else None
        ),
    }


def render(summary: dict[str, Any]) -> str:
    """The table, plus the sentence a reader needs to not over-read it."""
    if not summary.get("n"):
        return "No shared episodes between the two blocks.\n"
    lines = [
        "| Latency ms | n | Fast | Stock | Agree | Fast-only | Stock-only | Delta pp | 95% CI "
        "| McNemar | psi | Resolves |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in summary["cells"]:
        ci = f"[{cell['ci_pp'][0]:+.1f}, {cell['ci_pp'][1]:+.1f}]"
        resolves = "-" if cell["resolves_pp"] is None else f"{cell['resolves_pp']:.1f} pp"
        lines.append(
            f"| {cell['latency_ms']:g} | {cell['n']} | {cell['fast_success']:.1%} "
            f"| {cell['stock_success']:.1%} | {cell['agree']} | {cell['fast_only']} "
            f"| {cell['stock_only']} | {cell['delta_pp']:+.1f} | {ci} | {cell['p_text']} "
            f"| {cell['discordance']:.2f} | {resolves} |"
        )
    overall = summary["overall"]
    ci = f"[{overall['ci_pp'][0]:+.1f}, {overall['ci_pp'][1]:+.1f}]"
    resolves = "-" if overall["resolves_pp"] is None else f"{overall['resolves_pp']:.1f} pp"
    lines.append(
        f"| **all** | {overall['n']} | {overall['fast_success']:.1%} "
        f"| {overall['stock_success']:.1%} | {overall['agree']} | {overall['fast_only']} "
        f"| {overall['stock_only']} | {overall['delta_pp']:+.1f} | {ci} "
        f"| {overall['p_text']} | {overall['discordance']:.2f} | {resolves} |"
    )
    lines += ["", _paragraph(summary)]
    return "\n".join(lines) + "\n"


def _paragraph(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    changed = overall["fast_only"] + overall["stock_only"]
    text = (
        f"Over {overall['n']} paired episodes the two loops disagree on {changed} "
        f"({overall['discordance']:.0%} of them, psi): {overall['stock_only']} succeed only with "
        f"the stock loop and {overall['fast_only']} only with the fast one, for a "
        f"{overall['delta_pp']:+.1f} pp difference in rate, exact McNemar {overall['p_text']}."
    )
    if overall["resolves_pp"] is not None:
        # Which way round this reads depends on the number: a gap above the bar is a difference, a
        # gap below it is a null "at that bar", never an equality. The first 105-episode block
        # landed on the second case (+7.6 pp against a 16.7 pp bar), which is exactly the sentence
        # a reader must not have to reconstruct.
        verdict = (
            "so the difference is read as a difference"
            if abs(overall["delta_pp"]) >= overall["resolves_pp"]
            else "which is smaller than that bar, so the gap is a null at that resolution, not an "
            "equality"
        )
        text += (
            f" At this discordance a block of {overall['n']} resolves "
            f"{overall['resolves_pp']:.1f} pp at 80 % power, {verdict}."
        )
    if overall["episodes_to_resolve_delta"]:
        # A block sized to *detect* the gap is not sized to *measure* it: detection only needs the
        # null to be fallen through, while a point estimate with a tight interval needs the
        # discordance term financed too.
        text += (
            f" Detecting a gap this size at 80 % power would take "
            f"{overall['episodes_to_resolve_delta']} episodes per arm at this discordance."
        )
    if summary.get("discordant_with_step_data"):
        text += (
            f" Of the {summary['discordant_with_step_data']} discordant episodes that recorded "
            f"both lengths, {summary['discordant_ending_elsewhere']} ended at a different env step "
            f"(mean absolute change {summary['mean_abs_step_change']}), so the disagreement is "
            f"in the trajectory rather than in the last frame."
        )
    text += (
        " This is a statement about sampler sensitivity, not about either loop being the better "
        "policy: 1e-5 of action rounding is amplified by PushT's contact dynamics, so a block is "
        "only reproducible if it pins the loop that produced it."
    )
    return text


def table_from_paths(fast_paths: list[Any], stock_paths: list[Any]) -> str:
    """Render the table from row files, which is how `make_tables` keeps it regenerable.

    Both sides take a list: the fast arm and the stock arm are each measured in more than one
    block (the committed deploy rows plus the extension run on fresh seeds), and the table is over
    every episode the two arms share.
    """
    import json
    from pathlib import Path

    def read(path: Any) -> list[dict[str, Any]]:
        file = Path(path)
        if not file.exists():
            return []
        return [
            json.loads(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    fast_rows: list[dict[str, Any]] = []
    for path in fast_paths:
        fast_rows.extend(read(path))
    stock_rows: list[dict[str, Any]] = []
    for path in stock_paths:
        stock_rows.extend(read(path))
    return render(summarize(pairs(fast_rows, stock_rows)))
