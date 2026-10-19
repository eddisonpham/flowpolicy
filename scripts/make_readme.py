"""Generate README.md from docs/README_template.md plus the measured tables.

Run after scripts/make_tables.py. Missing tables stay as a visible marker rather
than being silently dropped.
"""

from __future__ import annotations

from pathlib import Path

PLACEHOLDERS = {
    "{{E1_TABLE}}": "results/tables/e1_main.md",
    "{{E3_TABLE}}": "results/tables/e3_steps.md",
    "{{E9_TABLE}}": "results/tables/e9_latency.md",
    "{{E10_TABLE}}": "results/tables/e10_deploy.md",
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
    Path("README.md").write_text(template)
    print("wrote README.md")


if __name__ == "__main__":
    main()
