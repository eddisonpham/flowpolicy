"""One-screen summary of everything measured: `just report`.

Reads `results/*.jsonl` only. Missing files are reported as "not measured yet" with the
recipe that produces them, never as zeros.
"""

from __future__ import annotations

from policybench.eval.report import RESULT_FILES, headline_rows, inventory, load_results


def main() -> None:
    results = load_results()
    print("PolicyBench results (every number generated from results/*.jsonl)\n")
    for question, answer, source in headline_rows(results):
        print(f"{source:>4s}  {question}")
        print(f"      {answer}\n")
    print("Row inventory:")
    print("\n".join(inventory(results)))
    print("\nRegenerate the tables/figures/README with: just tables && just figures && just readme")
    missing = [name for name, rows in results.items() if not rows]
    if missing:
        print(
            f"Not measured yet: {', '.join(missing)} (see {RESULT_FILES[missing[0]]} for the path)"
        )


if __name__ == "__main__":
    main()
