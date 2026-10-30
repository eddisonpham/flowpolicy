"""Generate README.md from docs/README_template.md plus the measured tables.

Run after scripts/make_tables.py. Missing tables stay as a visible marker rather
than being silently dropped.
"""

from __future__ import annotations

from pathlib import Path

PLACEHOLDERS = {
    "{{HEADLINE_TABLE}}": "results/tables/headline.md",
    "{{E0_TABLE}}": "results/tables/e0_validation.md",
    "{{E3_TABLE}}": "results/tables/e3_steps.md",
    "{{E3_REPLICATION_TABLE}}": "results/tables/e3_replication.md",
    "{{E9_TABLE}}": "results/tables/e9_latency.md",
    "{{E10_TABLE}}": "results/tables/e10_deploy.md",
    "{{E8_TABLE}}": "results/tables/e8_precision.md",
    "{{SERVE_TABLE}}": "results/tables/serve_load.md",
    "{{PACED_TABLE}}": "results/tables/paced_loop.md",
    "{{E14_TABLE}}": "results/tables/e14_stack.md",
    "{{GENERALITY_TABLE}}": "results/tables/generality.md",
    "{{E1_ABLATION_TABLE}}": "results/tables/e1_ablation.md",
    "{{M2_TABLE}}": "results/tables/m2_memory.md",
    "{{OVERFIT_TABLE}}": "results/tables/overfit.md",
    "{{KERNEL_TABLE}}": "results/tables/kernel.md",
}


def main() -> None:
    template = Path("docs/README_template.md").read_text()
    for marker, path in PLACEHOLDERS.items():
        p = Path(path)
        text = p.read_text() if p.exists() else f"_{path} not generated yet_"
        template = template.replace(marker, text.strip())
    # newline="\n": the repo normalizes to LF (`.gitattributes`), and the default Windows
    # translation writes CRLF, which makes every regeneration show up as a 500-line diff.
    Path("README.md").write_text(template, newline="\n")
    print("wrote README.md")


if __name__ == "__main__":
    main()
