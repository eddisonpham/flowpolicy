"""Regenerate every table from results/*.jsonl (no hand-edited numbers).

Pretrained-checkpoint rows live in results/{e0,e3,latency,deploy}.jsonl; the
from-scratch ablation lives in results/ablation/. Missing files are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flowpolicy.eval.metrics import bootstrap_ci, group_rows
from flowpolicy.eval.report import headline_table

ROOT = Path("results")
ABLATION = ROOT / "ablation"
OUT = ROOT / "tables"


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.1%} [{lo:.1%}, {hi:.1%}]"


def e1_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Policy | Seeds | Episodes/seed | Success per seed | Success (95% CI) | Mean coverage |",
        "|---|---|---|---|---|---|",
    ]
    for policy, group in sorted(group_rows(rows, ["policy"]).items()):
        name = policy[0]
        point, lo, hi = bootstrap_ci(group)
        seeds = len({r["train_seed"] for r in group})
        per_seed = len(group) / seeds
        # The CI resamples training seeds, so the spread across them has to be visible.
        parts = group_rows(group, ["train_seed"])
        hits = [
            f"{sum(int(r['success']) for r in parts[k])}/{len(parts[k])}" for k in sorted(parts)
        ]
        coverage = sum(float(r["coverage"]) for r in group) / len(group)
        lines.append(
            f"| {name} | {seeds} | {per_seed:.0f} | {', '.join(sorted(hits))} "
            f"| {fmt_ci(point, lo, hi)} | {coverage:.3f} |"
        )
    return "\n".join(lines) + "\n"


def e0_table(rows: list[dict[str, Any]]) -> str:
    """E0: the pretrained checkpoint through our harness (the adapter's acceptance gate)."""
    lines = [
        "| Scheduler | N | Episodes | Success (95% CI) | Mean coverage (95% CI) "
        "| Mean chunk latency ms |",
        "|---|---|---|---|---|---|",
    ]
    grouped = group_rows(rows, ["scheduler", "sampler_steps"])
    for (scheduler, steps), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        point, lo, hi = bootstrap_ci(group)
        cov_point, cov_lo, cov_hi = bootstrap_ci(group, metric="coverage")
        coverage = f"{cov_point:.3f} [{cov_lo:.3f}, {cov_hi:.3f}]"
        latency = sum(float(r["latency_mean_ms"]) for r in group) / len(group)
        lines.append(
            f"| {scheduler} | {steps} | {len(group)} | {fmt_ci(point, lo, hi)} | {coverage} "
            f"| {latency:.1f} |"
        )
    return chr(10).join(lines) + chr(10)


def e3_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scheduler | N (sampler steps) | Success (95% CI) | Mean coverage (95% CI) "
        "| Mean chunk latency ms |",
        "|---|---|---|---|---|",
    ]
    grouped = group_rows(rows, ["scheduler", "sampler_steps"])
    for (scheduler, steps), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        point, lo, hi = bootstrap_ci(group)
        # Coverage moves by ~0.05 between resamples, so it carries its own interval;
        # without it the ladder looks ordered when it is noise.
        cov_point, cov_lo, cov_hi = bootstrap_ci(group, metric="coverage")
        coverage = f"{cov_point:.3f} [{cov_lo:.3f}, {cov_hi:.3f}]"
        latency = sum(float(r["latency_mean_ms"]) for r in group) / len(group)
        lines.append(
            f"| {scheduler} | {steps} | {fmt_ci(point, lo, hi)} | {coverage} | {latency:.1f} |"
        )
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


def stack_table(rows: list[dict[str, Any]]) -> str:
    """E14: the same policy served three ways, and what each one costs.

    Latency rows answer "does one chunk fit the control period"; throughput rows answer
    "how long does an experiment take", with success and coverage carried so the cheap
    arms cannot be quietly worse policies.
    """
    latency = sorted(
        (r for r in rows if r.get("kind") == "latency"),
        key=lambda r: (int(r["steps"]), float(r["mean_ms"])),
    )
    lines = [
        "| Sampler | Mode | N | Mean chunk ms | p95 ms | Control rate Hz "
        "| Share of 100 ms period | Fits 100 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in latency:
        lines.append(
            f"| {row['sampler']} | {row['mode']} | {row['steps']} | {row['mean_ms']:.1f} "
            f"| {row['p95_ms']:.1f} | {row['control_hz']:.2f} | {row['period_frac']:.2f}x "
            f"| {'yes' if row['fits_100ms'] else 'no'} |"
        )
    lines.append("")

    throughput = sorted(
        (r for r in rows if r.get("kind") == "throughput"),
        key=lambda r: float(r["episode_wall_s"]),
    )
    if throughput:
        lines += [
            "| Arm | Scheduler | N | Workers | Episodes | s/episode | Episodes/hour "
            "| s per 100 control steps | Success | Mean coverage |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in throughput:
            per_100 = row.get("s_per_100_steps")
            per_100_text = f"{float(per_100):.2f}" if per_100 is not None else "n/a"
            lines.append(
                f"| {row['arm']} | {row['scheduler']} | {row['steps']} | {row['workers']} "
                f"| {row['episodes']} | {row['episode_wall_s']:.1f} "
                f"| {row['episodes_per_hour']:.1f} | {per_100_text} "
                f"| {row['success_rate']:.0%} | {row['coverage_mean']:.3f} |"
            )

    equivalence = sorted(
        (r for r in rows if r.get("kind") == "equivalence"), key=lambda r: r["arm"]
    )
    if equivalence:
        lines += [
            "",
            "Same policy, same seeds, two DDIM loops (does the fast sampler change the policy?)",
            "",
            "| Loop | Paired episodes | Success | Mean coverage | Mean chunk ms "
            "| Episodes agreeing on success |",
            "|---|---|---|---|---|---|",
        ]
        for row in equivalence:
            agree = row.get("episodes_agreeing")
            agree_text = f"{agree}/{row['episodes']}" if agree is not None else "-"
            lines.append(
                f"| {row['arm']} | {row['episodes']} | {row['success_rate']:.0%} "
                f"| {row['coverage_mean']:.3f} | {row['chunk_ms']:.1f} | {agree_text} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "e0_validation.md": (ROOT / "e0.jsonl", e0_table),
        "e3_steps.md": (ROOT / "e3.jsonl", e3_table),
        "e9_latency.md": (ROOT / "latency.jsonl", latency_table),
        "e10_deploy.md": (ROOT / "deploy.jsonl", deploy_table),
        "e14_stack.md": (ROOT / "stack.jsonl", stack_table),
        # Ablation: the from-scratch policies and the training-cost measurements that
        # the pretrained checkpoint replaced (docs/04 section 4).
        "e1_ablation.md": (ABLATION / "e1.jsonl", e1_table),
        "m2_memory.md": (ABLATION / "memory.jsonl", memory_table),
    }
    for name, (path, fn) in jobs.items():
        rows = read(path)
        if not rows:
            print(f"skip {name}: {path} missing or empty")
            continue
        text = fn(rows)
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name} ({len(rows)} rows)")
    headline = headline_table(
        {
            "e0": read(ROOT / "e0.jsonl"),
            "e3": read(ROOT / "e3.jsonl"),
            "e9": read(ROOT / "latency.jsonl"),
            "e10": read(ROOT / "deploy.jsonl"),
            "e14": read(ROOT / "stack.jsonl"),
        }
    )
    (OUT / "headline.md").write_text(headline)
    print(f"wrote {OUT / 'headline.md'}")

    kernel_rows = read(ABLATION / "kernel.jsonl")
    if kernel_rows:
        (OUT / "kernel.md").write_text(kernel_table(kernel_rows))
        print(f"wrote {OUT / 'kernel.md'} ({len(kernel_rows)} rows)")
    overfit_path = ROOT / "overfit" / "summary.json"
    if overfit_path.exists():
        (OUT / "overfit.md").write_text(overfit_table(json.loads(overfit_path.read_text())))
        print(f"wrote {OUT / 'overfit.md'}")


if __name__ == "__main__":
    main()
