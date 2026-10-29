"""Aggregation for results JSONL (docs/04 section 1).

The 95 % interval resamples training seeds with replacement, then episodes
within each chosen seed. With three seeds this is a wide, honest interval.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def assert_seed_disjoint(
    train_seeds: Sequence[int] | set[int], eval_seeds: Sequence[int] | set[int]
) -> None:
    overlap = set(train_seeds) & set(eval_seeds)
    if overlap:
        raise ValueError(f"eval seeds overlap train seeds: {sorted(overlap)}")


def bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    metric: str = "success",
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    values: dict[int, list[float]] = {}
    for row in rows:
        values.setdefault(int(row["train_seed"]), []).append(float(row[metric]))
    if not values:
        return float("nan"), float("nan"), float("nan")
    seeds = sorted(values)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        samples: list[float] = []
        for s in chosen:
            episodes = np.asarray(values[int(s)])
            samples.extend(rng.choice(episodes, size=len(episodes), replace=True).tolist())
        means[b] = float(np.mean(samples))
    point = float(np.mean([float(row[metric]) for row in rows]))
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def group_rows(
    rows: Iterable[dict[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    return grouped
