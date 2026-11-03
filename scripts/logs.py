"""Summarize a run log: what happened, how fast, and what broke.

The pipeline and every eval block write JSONL under `results/logs/`. That is machine-readable and
unreadable, so this is the one screen you look at while a 50-minute run is going:

    python scripts/logs.py                       # newest log in results/logs/
    python scripts/logs.py results/logs/eval_*.jsonl
    python scripts/logs.py --errors              # only failures, with their tracebacks

It reads logs; it never writes one, and it does not touch `results/*.jsonl` -- those are the rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from policybench.logging import DEFAULT_LOG_DIR, read_events


def newest(directory: Path) -> Path | None:
    logs = sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def summarize(path: Path, *, errors_only: bool, tail: int) -> str:
    events = list(read_events(path))
    if not events:
        return f"{path}: no events"
    counts = Counter(event.get("event", "?") for event in events)
    lines = [
        f"{path}",
        f"  events: {len(events)}  " + ", ".join(f"{k}={v}" for k, v in counts.most_common()),
    ]

    episodes = [event for event in events if event.get("event") == "episode_done"]
    if episodes and not errors_only:
        success = sum(float(event.get("success") or 0) for event in episodes)
        walls = [float(event["wall_s"]) for event in episodes if "wall_s" in event]
        lines.append(
            f"  episodes: {len(episodes)}  success {success / len(episodes):.1%}  "
            f"mean wall {sum(walls) / len(walls):.1f} s"
            if walls
            else f"  episodes: {len(episodes)}"
        )
        by_seed: dict[str, list[float]] = defaultdict(list)
        for event in episodes:
            by_seed[str(event.get("run_id"))].append(float(event.get("success") or 0))
        for run_id, values in by_seed.items():
            lines.append(
                f"    {run_id}: {len(values)} episodes, {sum(values) / len(values):.1%} success"
            )

    if not errors_only:
        progress = [event for event in events if event.get("event") == "progress"]
        for event in progress[-tail:]:
            fields = {k: v for k, v in event.items() if k not in {"ts", "run_id", "event"}}
            lines.append(f"  progress: {fields}")

    errors = [event for event in events if "error" in str(event.get("event")) or event.get("error")]
    for event in errors:
        lines.append(f"  ERROR [{event.get('run_id')}] {event.get('error')}")
        trace = str(event.get("traceback") or "").strip().splitlines()
        for line in trace[-6:]:
            lines.append(f"    {line}")
    if not errors:
        lines.append("  errors: none")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="log files (default: newest)")
    parser.add_argument("--errors", action="store_true", help="only counts and failures")
    parser.add_argument("--tail", type=int, default=5, help="how many progress events to show")
    parser.add_argument("--json", action="store_true", help="dump the parsed events instead")
    args = parser.parse_args()

    paths = args.paths or [path for path in [newest(DEFAULT_LOG_DIR)] if path is not None]
    if not paths:
        raise SystemExit(f"no logs under {DEFAULT_LOG_DIR}")
    for path in paths:
        if args.json:
            for event in read_events(path):
                print(json.dumps(event))
        else:
            print(summarize(path, errors_only=args.errors, tail=args.tail))


if __name__ == "__main__":
    main()
