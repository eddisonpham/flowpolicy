"""Regenerate every figure from results/*.jsonl (docs/04 section 5).

Writes PNGs to results/figures/. Matplotlib is a dev-only dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowpolicy.eval.metrics import bootstrap_ci, group_rows

ROOT = Path("results")
OUT = ROOT / "figures"


def read(name: str) -> list[dict[str, Any]]:
    path = ROOT / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def e1_bar(rows: list[dict[str, Any]]) -> None:
    labels, points, errs = [], [], []
    for (policy,), group in sorted(group_rows(rows, ["policy"]).items()):
        point, lo, hi = bootstrap_ci(group)
        labels.append(policy)
        points.append(point)
        errs.append([point - lo, hi - point])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(
        labels, points, yerr=np.array(errs).T, capsize=4, color=["#4c72b0", "#dd8452", "#55a868"]
    )
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title("E1 PushT (error bars: 95% bootstrap over seeds/episodes)")
    fig.tight_layout()
    fig.savefig(OUT / "e1_success.png", dpi=150)
    plt.close(fig)


def e3_curves(rows: list[dict[str, Any]]) -> None:
    grouped = group_rows(rows, ["policy", "sampler_steps"])
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for policy in sorted({key[0] for key in grouped}):
        xs, ys, lo, hi = [], [], [], []
        for (_policy, steps), group in sorted((k, v) for k, v in grouped.items() if k[0] == policy):
            point, low, high = bootstrap_ci(group)
            xs.append(steps)
            ys.append(point)
            lo.append(point - low)
            hi.append(high - point)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=policy)
    ax.set_xlabel("sampler steps N")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title("E3: integration steps vs success")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "e3_success_vs_steps.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for policy in sorted({key[0] for key in grouped}):
        xs, ys = [], []
        for (_policy, steps), group in sorted((k, v) for k, v in grouped.items() if k[0] == policy):
            xs.append(steps)
            ys.append(np.mean([r["latency_mean_ms"] for r in group]))
        ax.plot(xs, ys, marker="o", label=policy)
    ax.set_xlabel("sampler steps N")
    ax.set_ylabel("chunk latency (ms)")
    ax.set_title("E3: integration steps vs measured latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "e3_latency_vs_steps.png", dpi=150)
    plt.close(fig)


def e10_curve(rows: list[dict[str, Any]]) -> None:
    grouped = group_rows(rows, ["strategy", "latency_ms"])
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for strategy in sorted({key[0] for key in grouped}):
        xs, ys, lo, hi = [], [], [], []
        for (_strategy, latency), group in sorted(
            (k, v) for k, v in grouped.items() if k[0] == strategy
        ):
            point, low, high = bootstrap_ci(group)
            xs.append(latency)
            ys.append(point)
            lo.append(point - low)
            hi.append(high - point)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=strategy)
    ax.set_xlabel("injected observation latency (ms)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title("E10: latency vs success")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "e10_success_vs_latency.png", dpi=150)
    plt.close(fig)

    samples = [v for row in rows for v in row.get("inference_samples_ms", [])]
    if samples:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.hist(samples, bins=40, color="#4c72b0")
        ax.set_xlabel("chunk inference time (ms)")
        ax.set_ylabel("count")
        ax.set_title("E10: measured inference latency histogram (batch 1)")
        fig.tight_layout()
        fig.savefig(OUT / "e10_latency_hist.png", dpi=150)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e1 = read("e1.jsonl")
    if e1:
        e1_bar(e1)
        print("wrote e1_success.png")
    e3 = read("e3.jsonl")
    if e3:
        e3_curves(e3)
        print("wrote e3 figures")
    e10 = read("deploy.jsonl")
    if e10:
        e10_curve(e10)
        print("wrote e10 figures")


if __name__ == "__main__":
    main()
