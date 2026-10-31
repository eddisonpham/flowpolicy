"""Regenerate every table from results/*.jsonl (no hand-edited numbers).

Pretrained-checkpoint rows live in results/{e0,e3,latency,deploy}.jsonl; the
from-scratch ablation lives in results/ablation/. Missing files are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policybench.eval.metrics import bootstrap_ci, group_rows
from policybench.eval.report import headline_table
from policybench.eval.stats import fisher_exact, mcnemar_exact, p_text, resolvable_effect

# Older names, kept because the tests pin the statistics through this module and because the
# table code below reads better with short ones. The implementations live in the package.
_fisher_exact = fisher_exact
_mcnemar_exact = mcnemar_exact

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


def _block(group: list[dict[str, Any]]) -> tuple[str, str]:
    """(success with CI, coverage with CI) for one episode block."""
    point, lo, hi = bootstrap_ci(group)
    cov_point, cov_lo, cov_hi = bootstrap_ci(group, metric="coverage")
    return fmt_ci(point, lo, hi), f"{cov_point:.3f} [{cov_lo:.3f}, {cov_hi:.3f}]"


def e3_replication_table(primary: list[dict[str, Any]], replication: list[dict[str, Any]]) -> str:
    """E3's ladder on a second, independent episode block (2026-10-28).

    The point is not a bigger sample, it is a separate one: if the ladder is a property of the
    policy, an independent block of eval seeds should say the same thing; if it was episode
    noise, this is where it shows.
    """
    lines = [
        "| N | First block (n) success (95% CI) | coverage (95% CI) "
        "| Replication (n) success (95% CI) | coverage (95% CI) | Coverage change |",
        "|---|---|---|---|---|---|",
    ]
    steps = sorted(
        {int(r["sampler_steps"]) for r in primary} | {int(r["sampler_steps"]) for r in replication}
    )
    for n in steps:
        first = [r for r in primary if int(r["sampler_steps"]) == n]
        second = [r for r in replication if int(r["sampler_steps"]) == n]
        cells = []
        delta = "-"
        for group in (first, second):
            if group:
                success, coverage = _block(group)
                cells.append(f"{success} ({len(group)}) | {coverage}")
            else:
                cells.append("not run | -")
        if first and second:
            cov_first = sum(float(r["coverage"]) for r in first) / len(first)
            cov_second = sum(float(r["coverage"]) for r in second) / len(second)
            delta = f"{cov_second - cov_first:+.3f}"
        lines.append(f"| {n} | {cells[0]} | {cells[1]} | {delta} |")
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


def precision_table(rows: list[dict[str, Any]]) -> str:
    """E8: half precision, both halves of the trade (cost and policy quality)."""
    latency = sorted(
        (r for r in rows if r.get("kind") == "latency"),
        key=lambda r: (r["steps"], r["dtype"], r["mode"]),
    )
    lines = [
        "| Dtype | Mode | N | Mean chunk ms | p95 ms | Control Hz | Peak VRAM MiB | Fits 100 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in latency:
        lines.append(
            f"| {row['dtype']} | {row['mode']} | {row['steps']} | {row['mean_ms']:.1f} "
            f"| {row['p95_ms']:.1f} | {1000.0 / row['mean_ms']:.2f} "
            f"| {row['peak_vram_mb']:.0f} | {'yes' if row['fits_100ms'] else 'no'} |"
        )

    episodes = sorted((r for r in rows if r.get("kind") == "episode"), key=lambda r: r["dtype"])
    if episodes:
        lines += [
            "",
            "Same policy, same paired episodes, only the arithmetic dtype changes:",
            "",
            "| Dtype | N | Episodes | Success (95% CI) | Mean coverage | Mean chunk ms | Wall s |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in episodes:
            ci = f"{row['success_rate']:.1%} [{row['success_lo']:.1%}, {row['success_hi']:.1%}]"
            lines.append(
                f"| {row['dtype']} | {row['steps']} | {row['episodes']} | {ci} "
                f"| {row['coverage_mean']:.3f} | {row['chunk_ms']:.1f} | {row['wall_s']:.0f} |"
            )
    return "\n".join(lines) + "\n"


def paced_table(rows: list[dict[str, Any]]) -> str:
    """M9.6: the loop with real sleeps, so lateness is wall-clock rather than a counter."""
    lines = [
        "| N | Period ms | Episodes | Success | Mean coverage | Deadline misses/chunk-episode "
        "| Late ticks/episode | Mean lateness ms | p95 lateness ms | Wall s/episode |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    grouped = group_rows(rows, ["sampler_steps", "period_ms"])
    for (steps, period), group in sorted(grouped.items()):
        success = sum(float(r["success"]) for r in group) / len(group)
        coverage = sum(float(r["coverage"]) for r in group) / len(group)
        misses = sum(float(r["deadline_misses"]) for r in group) / len(group)
        late = sum(float(r["late_ticks"]) for r in group) / len(group)
        lateness = sum(float(r["lateness_mean_ms"]) for r in group) / len(group)
        lateness_p95 = sum(float(r["lateness_p95_ms"]) for r in group) / len(group)
        wall = sum(float(r["wall_s"]) for r in group) / len(group)
        lines.append(
            f"| {steps} | {period:g} | {len(group)} | {success:.0%} | {coverage:.3f} "
            f"| {misses:.1f} | {late:.1f} | {lateness:.1f} | {lateness_p95:.1f} | {wall:.1f} |"
        )
    return "\n".join(lines) + "\n"


def serve_table(rows: list[dict[str, Any]]) -> str:
    """The serving boundary: what a client pays, and what is the GPU."""
    lines = [
        "| Clients | Requests | Client p50 ms | Client p95 ms | Client max ms "
        "| Server p50 ms | Server p95 ms | HTTP overhead p50 ms | Req/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["concurrency"]):
        lines.append(
            f"| {row['concurrency']} | {row['requests']} | {row['client_p50_ms']:.1f} "
            f"| {row['client_p95_ms']:.1f} | {row['client_max_ms']:.1f} "
            f"| {row['server_p50_ms']:.1f} | {row['server_p95_ms']:.1f} "
            f"| {row['http_overhead_p50_ms']:.2f} | {row['requests_per_s']:.2f} |"
        )
    return "\n".join(lines) + "\n"


# Published success rates, from the checkpoints' own model cards / eval blocks. These are
# external constants, not measurements of ours (docs/04 section 3).
PUBLISHED_SUCCESS = {"diffusion": 0.654, "vqbet": 0.570}


def paired_resolution(
    ours: list[dict[str, Any]], published: list[dict[str, Any]]
) -> tuple[int, float, float] | None:
    """Paired n, observed discordance, and the smallest effect that n resolves at 80 % power.

    The third number is what a null has to be read against (`docs/04` section 1): every
    200-episode block in this repo resolves ~13 pp, so a table that prints "not distinguished"
    without it invites the reader to hear "no difference". Returns None when the two blocks share
    no seeds, and a resolution of 0.0 when the arms agree on every episode (there is nothing for
    the formula to resolve, and it divides by the discordance).
    """
    theirs = {int(row["eval_seed"]): row for row in published}
    shared = [row for row in ours if int(row["eval_seed"]) in theirs]
    n = len(shared)
    if n < 2:
        return None
    ours_only = sum(
        1 for row in shared if row["success"] and not theirs[int(row["eval_seed"])]["success"]
    )
    theirs_only = sum(
        1 for row in shared if not row["success"] and theirs[int(row["eval_seed"])]["success"]
    )
    discordance = (ours_only + theirs_only) / n
    if discordance <= 0:
        return n, 0.0, 0.0
    return n, discordance, resolvable_effect(n, discordance)


def paired_note(ours: list[dict[str, Any]], published: list[dict[str, Any]], *, family: str) -> str:
    """Same-seed comparison against a checkpoint's own eval block (docs/04 section 3).

    `published` rows come from `scripts/published_blocks.py`. Their `coverage` column is
    `max_reward * 0.95`, because gym-pusht reports reward as `clip(coverage / 0.95, 0, 1)`;
    comparing our coverage against their raw `max_reward` invents a 5 % gap. The clip also
    censors successes at 0.95, so their mean is a lower bound rather than a measurement.
    """
    theirs = {int(row["eval_seed"]): row for row in published}
    shared = [row for row in ours if int(row["eval_seed"]) in theirs]
    if not shared:
        return ""
    n = len(shared)
    o_ok = sum(1 for row in shared if row["success"])
    t_ok = sum(1 for row in shared if theirs[int(row["eval_seed"])]["success"])
    ours_only = sum(
        1 for row in shared if row["success"] and not theirs[int(row["eval_seed"])]["success"]
    )
    theirs_only = sum(
        1 for row in shared if not row["success"] and theirs[int(row["eval_seed"])]["success"]
    )
    p = _mcnemar_exact(ours_only, theirs_only)
    p_str = p_text(p)
    o_cov = sum(float(row["coverage"]) for row in shared) / n
    t_cov = sum(float(theirs[int(row["eval_seed"])]["coverage"]) for row in shared) / n
    pub_ok = sum(int(row["success"]) for row in published)
    resolution = paired_resolution(ours, published)
    resolves = f"{resolution[2] * 100:.1f} pp" if resolution else "-"
    if p < 0.05:
        verdict = "the two runs are distinguished at this sample size"
    else:
        verdict = (
            "the two runs are not distinguished at this sample size, and the block can only "
            f"exclude differences above {resolves}"
        )
    first, last = (
        min(int(row["eval_seed"]) for row in shared),
        max(int(row["eval_seed"]) for row in shared),
    )
    return (
        f"- `lerobot/{family}_pusht`, {n} shared eval seeds ({first}-{last}, inside the "
        f"checkpoint's own block): this repo {o_ok}/{n} "
        f"({o_ok / n:.1%}) against the checkpoint's own block {t_ok}/{n} ({t_ok / n:.1%}); "
        f"{n - ours_only - theirs_only} episodes agree, {ours_only} succeed only here and "
        f"{theirs_only} only there, exact McNemar {p_str}, so {verdict}. The block resolves "
        f"{resolves} at 80 % power (`stats.resolvable_effect`), so a null here is only that "
        f"coarse. Mean coverage "
        f"{o_cov:.3f} here; their block reports {t_cov:.3f}, which is only a lower bound -- "
        f"gym-pusht clips reward at 1.0, so every success records exactly 0.95 whatever its "
        f"true coverage. Their published block is {pub_ok}/{len(published)} = "
        f"{pub_ok / len(published):.1%} over {len(published)} episodes, and these shared seeds "
        f"are a sub-block of it. Source: `results/{family}_published.jsonl` "
        f"(`scripts/published_blocks.py`)."
    )


COVERAGE_BINS = ((0.0, 0.50), (0.50, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01))


def coverage_bin_table(ours: list[dict[str, Any]], published: list[dict[str, Any]]) -> str:
    """Split a paired block by the coverage we reached, and count success on both sides.

    The diffusion gap is not uniform: it lives at the top of the coverage range, where the hard
    0.95 success threshold turns a small coverage difference into most of the outcome difference.
    A single mean coverage hides that, so the bins make the mechanism checkable from results
    rather than argued in prose.

    Their per-episode coverage is censored at 0.95 (gym-pusht clips reward at 1.0), so the only
    comparable values are on episodes their block fails. That column averages their failures and
    prints "-" where a bin has none; it is the closest thing to a like-for-like comparison.
    """
    theirs = {int(row["eval_seed"]): row for row in published}
    shared = [row for row in ours if int(row["eval_seed"]) in theirs]
    if not shared:
        return ""
    lines = [
        "Where the diffusion gap sits (episodes binned by the coverage this repo reached; "
        "success counted on both sides for the same seeds):",
        "",
        "| Our coverage | Episodes | Success here | Success in its block | Mean coverage here "
        "| Its coverage, failures only |",
        "|---|---|---|---|---|---|",
    ]
    for lo, hi in COVERAGE_BINS:
        group = [row for row in shared if lo <= float(row["coverage"]) < hi]
        if not group:
            continue
        o_ok = sum(int(row["success"]) for row in group)
        t_ok = sum(int(theirs[int(row["eval_seed"])]["success"]) for row in group)
        o_cov = sum(float(row["coverage"]) for row in group) / len(group)
        failures = [
            theirs[int(row["eval_seed"])]
            for row in group
            if not theirs[int(row["eval_seed"])]["success"]
        ]
        t_cov = (
            f"{sum(float(row['coverage']) for row in failures) / len(failures):.3f}"
            if failures
            else "-"
        )
        upper = "1.00" if hi > 1.0 else f"{hi:.2f}"
        lines.append(
            f"| {lo:.2f}-{upper} | {len(group)} | {o_ok} | {t_ok} | {o_cov:.3f} | {t_cov} |"
        )
    return "\n".join(lines) + "\n"


def replication_note(primary: list[dict[str, Any]], replication: list[dict[str, Any]]) -> str:
    """Does the first block's rate repeat on a second, disjoint block of episodes?

    Different seeds mean independent samples, so this is Fisher exact rather than McNemar. The
    point is not a bigger n -- it is asking whether the number was an episode sample or the
    policy.
    """
    if not primary or not replication:
        return ""
    a_ok = sum(int(row["success"]) for row in primary)
    b_ok = sum(int(row["success"]) for row in replication)
    p = _fisher_exact(a_ok, len(primary) - a_ok, b_ok, len(replication) - b_ok)
    a_cov = sum(float(row["coverage"]) for row in primary) / len(primary)
    b_cov = sum(float(row["coverage"]) for row in replication) / len(replication)
    a_seeds = (min(int(r["eval_seed"]) for r in primary), max(int(r["eval_seed"]) for r in primary))
    b_seeds = (
        min(int(r["eval_seed"]) for r in replication),
        max(int(r["eval_seed"]) for r in replication),
    )
    verdict = (
        "the second block does not reproduce the first, so the rate is not a property of the "
        "policy alone"
        if p < 0.05
        else "the two blocks are not distinguished, so the rate repeats within sampling error"
    )
    return (
        f"- Replication on a disjoint block: {a_ok}/{len(primary)} ({a_ok / len(primary):.1%}) "
        f"on seeds {a_seeds[0]}-{a_seeds[1]} against {b_ok}/{len(replication)} "
        f"({b_ok / len(replication):.1%}) on seeds {b_seeds[0]}-{b_seeds[1]}; no episode is in "
        f"both blocks, so the samples are independent (two-sided Fisher exact p = {p:.2f}), and "
        f"{verdict}. Mean coverage {a_cov:.3f} against {b_cov:.3f}. Source: "
        f"`results/vqbet.jsonl` and `results/vqbet_replication.jsonl`."
    )


def generality_table(
    vqbet_rows: list[dict[str, Any]],
    diffusion_rows: list[dict[str, Any]],
    diffusion_latency: list[dict[str, Any]],
    vqbet_latency: list[dict[str, Any]],
    vqbet_published: list[dict[str, Any]] | None = None,
    diffusion_paired: list[dict[str, Any]] | None = None,
    diffusion_published: list[dict[str, Any]] | None = None,
    vqbet_replication: list[dict[str, Any]] | None = None,
) -> str:
    """Two policy families through one harness (docs/04 section 3, E0 and E15).

    Diffusion numbers are E0 (success, coverage) and E9 (clean batch-1 chunk time); VQ-BeT
    numbers are this repo's rows in `results/vqbet*.jsonl`. The per-control-step column is
    the fair comparison: the two checkpoints do not predict the same number of actions per
    forward pass, which is exactly what the interface has to hide.

    `diffusion_paired` is E0 re-run on the *published* block's own seeds: the original E0 rows
    sit on seeds 10000-10049, which the checkpoint's block (1000-1499) does not contain, so
    without that re-run the diffusion column could only be compared aggregate-to-aggregate.
    """
    diffusion = [row for row in diffusion_rows if row.get("scheduler") == "ddim"]
    if not diffusion or not vqbet_rows:
        return ""
    d_lat = [
        row for row in diffusion_latency if row.get("mode") == "eager" and row.get("steps") == 4
    ]
    d_ms = float(d_lat[0]["mean_ms"]) if d_lat else float("nan")
    v_ms = float(vqbet_latency[0]["mean_ms"]) if vqbet_latency else float("nan")
    lines = [
        "| Checkpoint | Family | Episodes | Success (95% CI) | Published | Mean coverage (95% CI) "
        "| Params M | Chunk ms (eager, batch 1) | Actions per chunk | ms per control step "
        "| Resolves pp |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, rows, family, chunk_ms, block, paired in (
        (
            "lerobot/diffusion_pusht",
            diffusion,
            "diffusion",
            d_ms,
            diffusion_published,
            diffusion_paired or [],
        ),
        ("lerobot/vqbet_pusht", vqbet_rows, "vqbet", v_ms, vqbet_published, vqbet_rows),
    ):
        point, lo, hi = bootstrap_ci(rows)
        cov, cov_lo, cov_hi = bootstrap_ci(rows, metric="coverage")
        acts = float(rows[0]["action_horizon"])
        params = float(rows[0]["params"]) / 1e6
        # Prefer the pinned block over the hand-typed constant whenever it is present.
        published_rate = (
            sum(int(row["success"]) for row in block) / len(block)
            if block
            else PUBLISHED_SUCCESS[family]
        )
        # The resolution of this family's *paired* block, in the table rather than only in prose:
        # an underpowered null is otherwise indistinguishable from a real equality.
        resolution = paired_resolution(paired, block or [])
        resolves = f"{resolution[2] * 100:.1f}" if resolution else "-"
        lines.append(
            f"| `{label}` | {family} | {len(rows)} | {fmt_ci(point, lo, hi)} "
            f"| {published_rate:.1%} | {cov:.3f} [{cov_lo:.3f}, {cov_hi:.3f}] "
            f"| {params:.1f} | {chunk_ms:.1f} | {acts:.0f} | {chunk_ms / acts:.2f} | {resolves} |"
        )
    for family, ours, block in (
        ("diffusion", diffusion_paired or [], diffusion_published or []),
        ("vqbet", vqbet_rows, vqbet_published or []),
    ):
        note = paired_note(ours, block, family=family)
        if note:
            lines += ["", note]
        if family == "diffusion":
            bins = coverage_bin_table(ours, block)
            if bins:
                lines += ["", bins]
    replication = replication_note(vqbet_rows, vqbet_replication or [])
    if replication:
        lines += ["", replication]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "e0_validation.md": (ROOT / "e0.jsonl", e0_table),
        "e3_steps.md": (ROOT / "e3.jsonl", e3_table),
        "e9_latency.md": (ROOT / "latency.jsonl", latency_table),
        "e10_deploy.md": (ROOT / "deploy.jsonl", deploy_table),
        "e14_stack.md": (ROOT / "stack.jsonl", stack_table),
        "e8_precision.md": (ROOT / "precision.jsonl", precision_table),
        "serve_load.md": (ROOT / "serve.jsonl", serve_table),
        "paced_loop.md": (ROOT / "paced.jsonl", paced_table),
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
        # newline="\n": `.gitattributes` normalizes the repo to LF; the default Windows
        # translation would rewrite every table as CRLF on each regeneration.
        (OUT / name).write_text(text, newline="\n")
        print(f"wrote {OUT / name} ({len(rows)} rows)")
    headline = headline_table(
        {
            "e0": read(ROOT / "e0.jsonl"),
            # The verdict comes from the paired block, not from the unpaired E0 rows.
            "e0_published": read(ROOT / "e0_published_seeds.jsonl"),
            "diffusion_published": read(ROOT / "diffusion_published.jsonl"),
            "e3": read(ROOT / "e3.jsonl"),
            "e9": read(ROOT / "latency.jsonl"),
            "e10": read(ROOT / "deploy.jsonl"),
            "e8": read(ROOT / "precision.jsonl"),
            "e14": read(ROOT / "stack.jsonl"),
        }
    )
    (OUT / "headline.md").write_text(headline, newline="\n")
    print(f"wrote {OUT / 'headline.md'}")

    vqbet = read(ROOT / "vqbet.jsonl")
    if vqbet:
        (OUT / "generality.md").write_text(
            generality_table(
                vqbet,
                read(ROOT / "e0.jsonl"),
                read(ROOT / "latency.jsonl"),
                read(ROOT / "vqbet_latency.jsonl"),
                vqbet_published=read(ROOT / "vqbet_published.jsonl"),
                diffusion_paired=read(ROOT / "e0_published_seeds.jsonl"),
                diffusion_published=read(ROOT / "diffusion_published.jsonl"),
                vqbet_replication=read(ROOT / "vqbet_replication.jsonl"),
            ),
            newline="\n",
        )
        print(f"wrote {OUT / 'generality.md'}")

    replication = read(ROOT / "e3_replication.jsonl")
    if replication:
        primary = read(ROOT / "e3.jsonl")
        (OUT / "e3_replication.md").write_text(
            e3_replication_table(primary, replication), newline="\n"
        )
        print(f"wrote {OUT / 'e3_replication.md'} ({len(replication)} rows)")

    kernel_rows = read(ABLATION / "kernel.jsonl")
    if kernel_rows:
        (OUT / "kernel.md").write_text(kernel_table(kernel_rows), newline="\n")
        print(f"wrote {OUT / 'kernel.md'} ({len(kernel_rows)} rows)")
    overfit_path = ROOT / "overfit" / "summary.json"
    if overfit_path.exists():
        (OUT / "overfit.md").write_text(
            overfit_table(json.loads(overfit_path.read_text())), newline="\n"
        )
        print(f"wrote {OUT / 'overfit.md'}")


if __name__ == "__main__":
    main()
