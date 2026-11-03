"""Run logs: one JSONL file per run, one event per line, plus a console progress line.

Every long block in this repo had the same two failures. A 200-episode eval printed nothing until
its last episode, so a stuck worker and a slow one looked identical; and a crash surfaced as a bare
`BrokenProcessPool` from `future.result()`, with no seed, no config and no traceback from the worker
that actually raised. Both are observability bugs, not measurement bugs -- no row changed -- but
they are the difference between a four-hour block you can supervise and one you cannot.

So this module is deliberately small and dependency-free:

``RunLog``
    append-only JSONL under `results/logs/`, one object per event, opened by the parent *and* by
    each worker (each opens the same path in append mode; one line per write keeps the records
    from interleaving). `event` never raises: a full disk must not turn a finished block into a
    failed one, so a write error is reported once and the handle is dropped.

``Progress``
    completed / total, rate and ETA, returned as fields so the caller logs them rather than
    printing them. ETA is `None` until one unit has finished -- a fabricated zero is worse than an
    absent number.

The log is not a result. It describes a run of a recipe that is committed, so `results/logs/` is
gitignored (docs/04 section 1); the things worth keeping are the rows, the tables and the changelog.
Events carry the run id, so a block, its workers' episodes and the stage wrapper that launched it
can be read as one timeline with `scripts/logs.py`.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "DEFAULT_LOG_DIR",
    "Progress",
    "RunLog",
    "default_log_path",
    "read_events",
]

#: Where run logs go. Gitignored: reproducible from the recipe, not a result (docs/04).
DEFAULT_LOG_DIR = Path("results/logs")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def default_log_path(run_id: str, directory: Path | str = DEFAULT_LOG_DIR) -> Path:
    """`<dir>/<run_id>_<UTC stamp>.jsonl` -- one file per run, never appended to by a later one."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(directory) / f"{_slug(run_id)}_{stamp}.jsonl"


@dataclass
class Progress:
    """How far a block has got, and how long the rest should take.

    `advance` returns the fields to log, so the caller decides what to do with them and a test can
    check the arithmetic without capturing stderr.
    """

    total: int
    done: int = 0
    started: float = field(default_factory=time.perf_counter)

    def advance(self, n: int = 1) -> dict[str, Any]:
        self.done = min(self.total, self.done + n)
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        rate = self.done / elapsed
        remaining = self.total - self.done
        return {
            "done": self.done,
            "total": self.total,
            "pct": round(100.0 * self.done / self.total, 1) if self.total else 100.0,
            "elapsed_s": round(elapsed, 1),
            "rate_per_s": round(rate, 3),
            # None, not 0: nothing has finished, so the ETA is unknown rather than instantaneous.
            "eta_s": round(remaining / rate, 1) if self.done else None,
        }


class RunLog:
    """Append-only JSONL events for one run, optionally echoed to the terminal.

    Args:
        path: the JSONL file. Parent directories are created.
        run_id: written into every event, so a worker's lines are attributable.
        echo: whether `progress` prints. Workers pass False; the parent passes True.
        stream: where `progress` prints (default stderr, so stdout stays parseable JSON).
    """

    def __init__(
        self,
        path: Path | str,
        run_id: str,
        *,
        echo: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.echo = echo
        self._stream = stream if stream is not None else sys.stderr
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO | None = self.path.open("a", encoding="utf-8", newline="\n")
        self._warned = False

    # -- writing ---------------------------------------------------------------------------

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        """Append one event and return it (so a caller can also log it elsewhere)."""
        payload: dict[str, Any] = {"ts": _now(), "run_id": self.run_id, "event": name}
        payload.update(fields)
        self._write(payload)
        return payload

    def error(self, exc: BaseException, **fields: Any) -> dict[str, Any]:
        """An event carrying the exception's type, message and full traceback."""
        return self.event(
            "error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            **fields,
        )

    def progress(self, name: str, **fields: Any) -> dict[str, Any]:
        """An event that also prints one human line -- for stage and chunk boundaries only."""
        payload = self.event(name, **fields)
        if self.echo and self._fh is not None:
            body = " ".join(f"{key}={_short(value)}" for key, value in fields.items())
            print(f"[{self.run_id}] {name} {body}".rstrip(), file=self._stream, flush=True)
        return payload

    def close(self, **summary: Any) -> None:
        """Write `run_end` and release the handle. Safe to call twice."""
        if self._fh is None:
            return
        if summary:
            self.event("run_end", **summary)
        self._fh.close()
        self._fh = None

    def _write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(payload, default=str) + "\n")
            self._fh.flush()
        except OSError as exc:  # a full disk must not fail a finished measurement
            if not self._warned:
                self._warned = True
                print(
                    f"[{self.run_id}] log write failed ({exc}); continuing unlogged",
                    file=sys.stderr,
                )

    # -- lifecycle -------------------------------------------------------------------------

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _short(value: Any) -> str:
    text = f"{value:.3g}" if isinstance(value, float) else str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def read_events(path: Path | str) -> Iterator[dict[str, Any]]:
    """Iterate a run log, skipping blank lines and a torn final line.

    A killed process can leave half a line; the rest of the log is still readable, which is the
    whole point of writing one event per line.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
