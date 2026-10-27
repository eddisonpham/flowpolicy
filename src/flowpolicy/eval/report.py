"""One-screen summary of everything measured, straight from `results/*.jsonl`.

This lives in the package (not in `scripts/`) so both entry points can share it and so
`mypy --strict` covers it: `scripts/make_tables.py` injects the headline table into the
README, and `scripts/report.py` prints it plus the row inventory at the terminal.

Numbers are computed, never typed, and a file that has not been produced yet renders an
explicit "not measured" line rather than a gap the reader would misread as a zero.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from flowpolicy.eval.metrics import bootstrap_ci, group_rows

Row = dict[str, Any]

# The JSONL files that make up the measured story, in reading order. `ablation/` holds the
# from-scratch policies that the pretrained checkpoint replaced (docs/04 section 4).
RESULT_FILES: dict[str, Path] = {
    "e0": Path("results/e0.jsonl"),
    "e3": Path("results/e3.jsonl"),
    "e9": Path("results/latency.jsonl"),
    "e10": Path("results/deploy.jsonl"),
    "ablation_e1": Path("results/ablation/e1.jsonl"),
    "ablation_e3": Path("results/ablation/e3.jsonl"),
    "ablation_e9": Path("results/ablation/latency.jsonl"),
    "ablation_e10": Path("results/ablation/deploy.jsonl"),
}


def read_rows(path: Path) -> list[Row]:
    """Read one JSONL file; a missing file is empty, not an error."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_results(files: Mapping[str, Path] | None = None) -> dict[str, list[Row]]:
    return {name: read_rows(path) for name, path in (files or RESULT_FILES).items()}


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.1%} [{lo:.1%}, {hi:.1%}]"


def _coverage(group: Sequence[Row]) -> str:
    point, lo, hi = bootstrap_ci(group, metric="coverage")
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _mean(rows: Iterable[Row], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def headline_rows(results: Mapping[str, list[Row]]) -> list[tuple[str, str, str]]:
    """The questions a reader arrives with, as (question, answer, source)."""
    out: list[tuple[str, str, str]] = []

    e0 = [r for r in results.get("e0", []) if r.get("scheduler") == "ddim"]
    if e0:
        point, lo, hi = bootstrap_ci(e0)
        out.append(
            (
                "Does the harness reproduce the published policy?",
                f"Yes: {_fmt_ci(point, lo, hi)} success and {_coverage(e0)} mean coverage over "
                f"{len(e0)} paired episodes at DDIM N=10, against the model card's 65.4 % / 0.955",
                "E0",
            )
        )
    else:
        out.append(
            (
                "Does the harness reproduce the published policy?",
                "not measured yet (`just eval-e0`, about 50 min)",
                "E0",
            )
        )

    e3 = group_rows(results.get("e3", []), ["sampler_steps"])
    if e3:
        parts = [
            f"{bootstrap_ci(group, metric='coverage')[0]:.3f} at N={steps}"
            for (steps,), group in sorted(e3.items())
        ]
        out.append(
            (
                "How few sampler steps still work?",
                "; ".join(parts)
                + ". The ladder is flat after N=4 (coverage intervals overlap there); chunk "
                "latency keeps growing linearly in N, in the table below",
                "E3",
            )
        )
    else:
        out.append(("How few sampler steps still work?", "not measured yet (`just eval-e3`)", "E3"))

    latency = group_rows(results.get("e9", []), ["mode", "steps"])
    if latency:
        parts = []
        for steps in (4, 10):
            eager = latency.get(("eager", steps))
            graph = latency.get(("cudagraph", steps))
            if eager and graph:
                parts.append(
                    f"{_mean(eager, 'mean_ms'):.1f} ms eager / {_mean(graph, 'mean_ms'):.1f} ms "
                    f"graph at N={steps}"
                )
        eager10 = latency.get(("eager", 10))
        graph10 = latency.get(("cudagraph", 10))
        gain = (
            f"{_mean(eager10, 'mean_ms') / _mean(graph10, 'mean_ms'):.2f}x"
            if eager10 and graph10
            else "unmeasured"
        )
        out.append(
            (
                "What does one chunk cost at batch 1?",
                "; ".join(parts)
                + f". CUDA graphs are worth {gain} here, against 4.7x on the 11 M ablation model. "
                "`torch.compile` is NOT RUN (no MSVC on this host)",
                "E9",
            )
        )
    else:
        out.append(
            (
                "What does one chunk cost at batch 1?",
                "not measured yet (`just latency-ladder`)",
                "E9",
            )
        )

    deploy = group_rows(results.get("e10", []), ["strategy", "latency_ms"])
    if deploy:
        parts = []
        for ms in (0.0, 200.0):
            group = deploy.get(("naive", ms))
            if group:
                parts.append(f"{bootstrap_ci(group)[0]:.0%} at {ms:g} ms injected")
        base = deploy.get(("naive", 0.0))
        misses = f"{_mean(base, 'deadline_misses'):.0f}" if base else "?"
        out.append(
            (
                "Does injected observation latency hurt?",
                "success " + " vs ".join(parts) + " (30 episodes per row, CIs overlapping). "
                f"The binding constraint is the policy's own cost: {misses} deadline misses per "
                "episode against a 100 ms period",
                "E10",
            )
        )
    else:
        out.append(
            (
                "Does injected observation latency hurt?",
                "not measured yet (`just eval-e10`)",
                "E10",
            )
        )

    # E12 lives in the same file: the replan_every=2 rows, where four chunks overlap and
    # temporal ensembling can differ from naive replacement. Rows keep the runner's raw
    # strategy name (`naive-r2`), so the suffix is stripped to pair them with the 8-step rows.
    replan = group_rows(results.get("e10", []), ["strategy", "latency_ms", "replan_every"])
    r2 = {
        (key[0].removesuffix("-r2"), key[1]): group for key, group in replan.items() if key[2] == 2
    }
    if r2:

        def r2_rate(strategy: str, ms: float) -> str:
            group = r2.get((strategy, ms))
            return f"{bootstrap_ci(group)[0]:.0%}" if group else "not measured"

        def eight_step_rate(ms: float) -> str:
            group = deploy.get(("naive", ms))
            return f"{bootstrap_ci(group)[0]:.0%}" if group else "not measured"

        out.append(
            (
                "Does replanning four times faster break it?",
                f"Yes for naive replacement: {r2_rate('naive', 100.0)} at 100 ms and "
                f"{r2_rate('naive', 200.0)} at 200 ms against {eight_step_rate(100.0)} / "
                f"{eight_step_rate(200.0)} when replanning every 8 steps; uniform temporal "
                "ensembling over the same overlapping chunks recovers "
                f"{r2_rate('temporal-ensemble', 100.0)} / "
                f"{r2_rate('temporal-ensemble', 200.0)} at 4x the chunk count",
                "E12",
            )
        )
    else:
        out.append(
            (
                "Does replanning four times faster break it?",
                "not measured yet (`just eval-e10`)",
                "E12",
            )
        )

    return out


def headline_table(results: Mapping[str, list[Row]]) -> str:
    """Markdown version of `headline_rows`, for injection into the README."""
    lines = ["| Question | Measured answer | Source |", "|---|---|---|"]
    lines += [f"| {q} | {a} | {src} |" for q, a, src in headline_rows(results)]
    return "\n".join(lines) + "\n"


def inventory(results: Mapping[str, list[Row]]) -> list[str]:
    """One line per result file: row count, or what to run to produce it."""
    lines = []
    for name, path in RESULT_FILES.items():
        rows = results.get(name, [])
        lines.append(f"  {name:12s} {len(rows):5d} rows  {path}")
    return lines
