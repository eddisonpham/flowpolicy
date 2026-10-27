"""Drop rows whose `config_id` starts with one of the given prefixes.

The eval runners call this before appending: re-running one policy must not
discard the rows of the policies it is not re-evaluating (they used to delete
the whole file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--prefix", nargs="+", required=True)
    args = parser.parse_args()
    if not args.path.exists():
        return
    rows = [json.loads(line) for line in args.path.read_text().splitlines() if line.strip()]
    kept = [
        row
        for row in rows
        if not any(str(row.get("config_id", "")).startswith(p) for p in args.prefix)
    ]
    args.path.write_text("".join(json.dumps(row) + "\n" for row in kept))
    print(f"{args.path}: kept {len(kept)} of {len(rows)} rows")


if __name__ == "__main__":
    main()
