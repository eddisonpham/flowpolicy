"""Drop rows whose `config_id` starts with one of the given prefixes.

The eval runners call this before appending: re-running one policy must not
discard the rows of the policies it is not re-evaluating (they used to delete
the whole file).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--prefix", nargs="+", required=True)
    args = parser.parse_args()
    if not args.path.exists():
        return
    rows = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [
        row
        for row in rows
        if not any(str(row.get("config_id", "")).startswith(p) for p in args.prefix)
    ]
    # Atomic replace: this file is the measurement, and a crash between truncate and flush would
    # leave a completed block as a prefix of itself (`results/*.tmp` is gitignored, like the
    # trainer's checkpoint writes).
    tmp = args.path.with_suffix(args.path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8", newline="\n")
    os.replace(tmp, args.path)
    print(f"{args.path}: kept {len(kept)} of {len(rows)} rows")


if __name__ == "__main__":
    main()
