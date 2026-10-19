"""Regenerate every table from results/*.jsonl (no hand-edited numbers).

Inputs: results/e1.jsonl, results/e3.jsonl, results/latency.jsonl,
results/deploy.jsonl, results/memory.jsonl. Missing files are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flowpolicy.eval.metrics import bootstrap_ci, group_rows

ROOT = Path("results")
OUT = ROOT / "tables"


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.1%} [{lo:.1%}, {hi:.1%}]"


def e1_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Policy | Seeds | Episodes/seed | Success (95% CI) | Mean coverage |",
        "|---|---|---|---|---|",
    ]
    for policy, group in sorted(group_rows(rows, ["policy"]).items()):
        name = policy[0]
        point, lo, hi = bootstrap_ci(group)
        seeds = len({r["train_seed"] for r in group})
        per_seed = len(group) / seeds
        coverage = sum(float(r["coverage"]) for r in group) / len(group)
        lines.append(
            f"| {name} | {seeds} | {per_seed:.0f} | {fmt_ci(point, lo, hi)} | {coverage:.3f} |"
        )
    return "\n".join(lines) + "\n"


def e3_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Policy | N (sampler steps) | Success (95% CI) | Mean chunk latency ms |",
        "|---|---|---|---|",
    ]
    grouped = group_rows(rows, ["policy", "sampler_steps"])
    for (policy, steps), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        point, lo, hi = bootstrap_ci(group)
        latency = sum(float(r["latency_mean_ms"]) for r in group) / len(group)
        lines.append(f"| {policy} | {steps} | {fmt_ci(point, lo, hi)} | {latency:.1f} |")
    return "\n".join(lines) + "\n"


def latency_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Mode | N | Mean ms | p95 ms | Compile s |", "|---|---|---|---|---|"]
    for (mode, steps), group in sorted(
        group_rows(rows, ["mode", "steps"]).items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        mean = sum(float(r["mean_ms"]) for r in group) / len(group)
        p95 = sum(float(r["p95_ms"]) for r in group) / len(group)
        compile_s = max(float(r.get("compile_s", 0.0)) for r in group)
        lines.append(f"| {mode} | {steps} | {mean:.2f} | {p95:.2f} | {compile_s:.1f} |")
    return "\n".join(lines) + "\n"


def deploy_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Latency ms | Strategy | Success (95% CI) | Mean coverage | Deadline misses / episode |",
        "|---|---|---|---|---|",
    ]
    grouped = group_rows(rows, ["latency_ms", "strategy"])
    for (latency, strategy), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        point, lo, hi = bootstrap_ci(group)
        coverage = sum(float(r["coverage"]) for r in group) / len(group)
        misses = sum(float(r["deadline_misses"]) for r in group) / len(group)
        ci = fmt_ci(point, lo, hi)
        lines.append(f"| {latency:g} | {strategy} | {ci} | {coverage:.3f} | {misses:.2f} |")
    return "\n".join(lines) + "\n"


def kernel_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Batch | Equivalent (max abs diff) | Eager ms | Compile ms | Triton ms "
        "| vs eager | vs compile |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch']} | {'yes' if row['equivalent'] else 'no'} "
            f"({row['max_abs_diff']:.1e}) | {row['eager_ms']:.4f} | {row['compile_ms']:.4f} | "
            f"{row['triton_ms']:.4f} | {row['speedup_vs_eager']:.2f}x | "
            f"{row['speedup_vs_compile']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def overfit_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Policy | Steps | Initial loss | Final loss | Ratio | Overfit 8 episodes |",
        "|---|---|---|---|---|---|",
    ]
    for kind in sorted(summary):
        row = summary[kind]
        lines.append(
            f"| {kind} | {row['steps']} | {row['initial_loss']:.4f} | {row['final_loss']:.4f} | "
            f"{row['ratio']:.3f} | {'yes' if row['overfit_8_episodes'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def memory_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Model + data | Batch | Peak VRAM MiB | Steps/s | Fits |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['model']} 96px | {row['batch']} | {row['peak_vram_mb']:.0f} | "
            f"{row['steps_per_s']:.2f} | {'yes' if row['fits'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "e1_main.md": (ROOT / "e1.jsonl", e1_table),
        "e3_steps.md": (ROOT / "e3.jsonl", e3_table),
        "e9_latency.md": (ROOT / "latency.jsonl", latency_table),
        "e10_deploy.md": (ROOT / "deploy.jsonl", deploy_table),
        "m2_memory.md": (ROOT / "memory.jsonl", memory_table),
    }
    for name, (path, fn) in jobs.items():
        rows = read(path)
        if not rows:
            print(f"skip {name}: {path} missing or empty")
            continue
        text = fn(rows)
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name} ({len(rows)} rows)")

    kernel_rows = read(ROOT / "kernel.jsonl")
    if kernel_rows:
        (OUT / "kernel.md").write_text(kernel_table(kernel_rows))
        print(f"wrote {OUT / 'kernel.md'} ({len(kernel_rows)} rows)")

    overfit_path = ROOT / "overfit" / "summary.json"
    if overfit_path.exists():
        (OUT / "overfit.md").write_text(overfit_table(json.loads(overfit_path.read_text())))
        print(f"wrote {OUT / 'overfit.md'}")


if __name__ == "__main__":
    main()
