"""Run the benchmark pipeline end to end: probes, eval blocks, deploy rows, generated tables.

`just verify` proves the *tree* is self-consistent. It does not run a policy. This runs the
measurement stack the way `docs/04` describes it and writes each stage's log under
`results/logs/pipeline_<stamp>/`, so a 50-minute run can be supervised rather than waited on.

Two decisions worth stating:

**Rows go to the run directory, not to `results/*.jsonl`.** The committed rows are evidence; a
re-run has to be able to disagree with them without overwriting them. `--compare` (run at the end
of every profile) pairs the fresh rows with the committed ones by seed and reports how far the
pipeline reproduces itself. That is the only claim this script makes: a difference between fresh
and committed rows is a reproducibility finding about the harness, and `summary.json` records it
rather than a rerun quietly replacing the evidence.

**Stages fail independently.** A failed stage does not stop the pipeline (unless it is
`--fail-fast`), because the point of a single long run is to see *everything* that is broken in one
pass, and the exit code still reports failure. `summary.json` is written after every stage, so a
killed run still says what it got through.

Scope, stated rather than implied: this covers the pretrained and deployment path -- E0, E3, E9,
E10, E15, plus the deterministic probes. Training (E1) is a multi-hour GPU job with its own recipe
(`just train-e1`) and is not part of this.

Usage:
    just pipeline                     # full profile, ~50 min on the 5060
    just pipeline quick               # smoke the whole path in ~5 min
    uv run python scripts/pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policybench.eval.report import read_rows
from policybench.eval.stats import mcnemar_exact, p_text, resolvable_effect
from policybench.logging import RunLog, default_log_path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LEROYOT = REPO / ".venv-lerobot"
CKPT = "hf:lerobot/diffusion_pusht"
VQBET = "hf:lerobot/vqbet_pusht"

#: Fresh rows ↔ committed rows, for the reproducibility comparison at the end of a run.
#: A stage not listed here has no committed counterpart, so it is not compared.
COMMITTED: dict[str, str] = {
    "e0": "results/e0.jsonl",
    "e0-published": "results/e0_published_seeds.jsonl",
    "e3": "results/e3.jsonl",
    "vqbet": "results/vqbet.jsonl",
    "deploy": "results/deploy.jsonl",
    "latency": "results/latency.jsonl",
}


@dataclass(frozen=True)
class Stage:
    """One named step: one or more commands, run in order, sharing one log file."""

    name: str
    description: str
    commands: tuple[tuple[str, ...], ...]
    minutes: float
    gpu: bool = False
    requires: Path | None = None


@dataclass
class StageResult:
    name: str
    status: str  # ok | failed | skipped
    seconds: float
    returncode: int = 0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 1),
            "returncode": self.returncode,
            "detail": self.detail,
        }


def lerobot_python() -> Path | None:
    """The isolated interpreter that has `lerobot` (and may not exist on a fresh clone)."""
    for candidate in (LEROYOT / "Scripts" / "python.exe", LEROYOT / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def build_stages(run: Path, profile: str) -> list[Stage]:
    """The pipeline, as data. `run` is the directory fresh rows and logs go to."""
    rows = run / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    py = PY
    episodes = 100 if profile == "full" else 10
    vq_episodes = 200 if profile == "full" else 20

    stages: list[Stage] = [
        Stage(
            "doctor",
            "read-only environment check (versions, CUDA, backend, checkpoint)",
            ((py, "scripts/doctor.py"),),
            0.3,
        ),
        Stage(
            "unit",
            "CPU unit gate: the measurement code's own contracts",
            ((py, "-m", "pytest", "-m", "unit and not gpu", "-q"),),
            1.0,
        ),
        Stage(
            "leftover",
            "deterministic leftover round trip (VLA adapter gate, CPU)",
            ((py, "scripts/smolvla_leftover_probe.py", "--out", str(run / "leftover.json")),),
            0.2,
        ),
        Stage(
            "processor-equiv",
            "the checkpoint's own buffers vs lerobot's processor pipelines",
            (
                (
                    str(lerobot_python() or "python"),
                    "scripts/processor_equiv.py",
                    "--checkpoint",
                    "checkpoints/diffusion_pusht_migrated",
                    "--out",
                    str(run / "processor_equiv.json"),
                ),
            ),
            0.5,
            requires=LEROYOT,
        ),
        Stage(
            "env-equiv",
            "this repo's env wrapper vs lerobot's, fixed action sequence",
            (
                (
                    str(lerobot_python() or "python"),
                    "scripts/env_equiv.py",
                    "--seed",
                    "1000",
                    "--steps",
                    "300",
                    "--out",
                    str(run / "env_equiv.json"),
                ),
            ),
            0.5,
            requires=LEROYOT,
        ),
        Stage(
            "e0",
            f"E0: the published checkpoint, DDIM@10, {episodes} episodes on seeds 10000+",
            (
                (
                    py,
                    "scripts/evaluate.py",
                    "--ckpt",
                    CKPT,
                    "--config-id",
                    f"e0_ddim10_n{episodes}",
                    "--episodes",
                    str(episodes),
                    "--seed-start",
                    "10000",
                    "--steps",
                    "10",
                    "--workers",
                    "4",
                    "--out",
                    str(rows / "e0.jsonl"),
                ),
            ),
            6.0 if profile == "full" else 0.7,
            gpu=True,
        ),
    ]

    if profile == "full":
        stages += [
            Stage(
                "e0-published",
                "E0 paired: the checkpoint on its own 200-episode block (seeds 1000+)",
                (
                    (
                        py,
                        "scripts/evaluate.py",
                        "--ckpt",
                        CKPT,
                        "--config-id",
                        "dp_published_n200",
                        "--episodes",
                        "200",
                        "--seed-start",
                        "1000",
                        "--steps",
                        "10",
                        "--workers",
                        "4",
                        "--out",
                        str(rows / "e0_published_seeds.jsonl"),
                    ),
                ),
                12.0,
                gpu=True,
            ),
            Stage(
                "e3",
                "E3: the DDIM step ladder, 6 x 50 episodes on seeds 20000+",
                tuple(
                    (
                        py,
                        "scripts/evaluate.py",
                        "--ckpt",
                        CKPT,
                        "--config-id",
                        f"e3_ddim_n{n}",
                        "--episodes",
                        "50",
                        "--seed-start",
                        "20000",
                        "--steps",
                        str(n),
                        "--workers",
                        "4",
                        "--out",
                        str(rows / "e3.jsonl"),
                    )
                    for n in (1, 2, 4, 8, 10, 16)
                ),
                10.0,
                gpu=True,
            ),
            Stage(
                "latency",
                "E9: batch-1 chunk latency, eager and CUDA graph, at N=4 and N=10",
                tuple(
                    (
                        py,
                        "scripts/latency.py",
                        "--ckpt",
                        CKPT,
                        "--mode",
                        mode,
                        "--steps",
                        str(n),
                        "--out",
                        str(rows / "latency.jsonl"),
                        "--config-id",
                        f"e9_{mode}_n{n}",
                    )
                    for mode in ("eager", "cudagraph")
                    for n in (4, 10)
                ),
                4.0,
                gpu=True,
            ),
            Stage(
                "deploy",
                "E10: the fixed-rate control loop at 0/100/200 ms injected latency",
                (
                    (
                        py,
                        "scripts/deploy_eval.py",
                        "--ckpts",
                        CKPT,
                        "--latencies",
                        "0",
                        "100",
                        "200",
                        "--seeds",
                        "10000",
                        "10001",
                        "10002",
                        "--episodes-per-seed",
                        "5",
                        "--steps",
                        "10",
                        "--strategy",
                        "naive",
                        # Stock, not the fast sampler: the committed deploy block (E10/E12) was
                        # measured with `diffusers`' loop, and only its rows can be paired with
                        # `results/deploy.jsonl`. The two loops agree to 1e-5 on the action, which
                        # is enough for PushT's contact dynamics to flip episodes, so swapping it
                        # here changes the block rather than reproducing it.
                        "--sampler",
                        "stock",
                        "--dtype",
                        "float32",
                        "--out",
                        str(rows / "deploy.jsonl"),
                    ),
                ),
                24.0,
                gpu=True,
            ),
        ]

    stages += [
        Stage(
            "vqbet",
            f"E15: a second published family through the same harness, {vq_episodes} episodes",
            (
                (
                    py,
                    "scripts/evaluate.py",
                    "--ckpt",
                    VQBET,
                    "--config-id",
                    f"vqbet_pusht_n{vq_episodes}",
                    "--episodes",
                    str(vq_episodes),
                    "--seed-start",
                    "100000",
                    "--steps",
                    "1",
                    "--action-horizon",
                    "5",
                    "--workers",
                    "1",
                    "--out",
                    str(rows / "vqbet.jsonl"),
                ),
            ),
            1.5 if profile == "full" else 0.3,
            gpu=True,
        ),
        Stage(
            "tables",
            "regenerate tables, README and the power audit; fail on any diff",
            (
                (py, "scripts/make_tables.py"),
                (py, "scripts/make_readme.py"),
                (py, "scripts/power_audit.py"),
                (
                    "git",
                    "diff",
                    "--exit-code",
                    "--",
                    "results/tables",
                    "README.md",
                    "results/power_audit.json",
                ),
            ),
            1.0,
        ),
    ]
    return stages


def run_stage(stage: Stage, run: Path, log: RunLog) -> StageResult:
    """Run every command in a stage into one log file, stopping at the first failure."""
    if stage.requires is not None and not stage.requires.exists():
        log.progress("stage_skipped", stage=stage.name, missing=str(stage.requires))
        return StageResult(stage.name, "skipped", 0.0, detail=f"missing {stage.requires}")
    started = time.perf_counter()
    log_path = run / f"{stage.name}.log"
    log.progress("stage_start", stage=stage.name, minutes=stage.minutes, log=str(log_path))
    with log_path.open("w", encoding="utf-8", newline="\n") as fh:
        for command in stage.commands:
            fh.write("$ " + " ".join(command) + "\n")
            fh.flush()
            proc = subprocess.run(
                command, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=False
            )
            fh.write(f"[exit {proc.returncode}]\n")
            fh.flush()
            if proc.returncode != 0:
                head = " ".join(command[:3])
                return StageResult(
                    stage.name,
                    "failed",
                    time.perf_counter() - started,
                    returncode=proc.returncode,
                    detail=f"`{head}` exited {proc.returncode}; see {log_path.name}",
                )
    return StageResult(stage.name, "ok", time.perf_counter() - started)


def _opt_int(value: Any) -> int | None:
    """A row field that may be absent or unusable, as `None` (unknown) rather than 0."""
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _opt_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: Conditions a block varies *within* one file, so they have to agree to be the same episode.
#: `sampler` and `dtype` are here because they are *configurations*, not just labels: on
#: 2026-11-03 the deploy stage ran the fast DDIM loop while the committed block was the stock one,
#: and pairing them reported a 13 pp "reproducibility failure" that was two configurations. PushT
#: is chaotic enough that a 1e-5 sampler difference flips episodes
#: (`tests/gpu/test_pretrained_sampler_equivalence.py`), so a recorded mismatch must not pair.
CONDITIONS = ("steps", "latency", "sampler", "dtype")


def _key(row: dict[str, Any]) -> dict[str, Any] | None:
    """What makes two rows the same episode, plus the conditions that must agree.

    ``seed``/``episode``/``strategy`` identify the episode; ``steps`` and ``latency`` are
    conditions (E3's six N values, E10's latency levels), so pairing across them would compare
    two different measurements.

    A condition a row does not record is `None` -- unknown, not zero -- and `_same_episode`
    treats it as a wildcard. Reading an absent `sampler_steps` as 0 is what turned a 45-episode
    overlap into "no shared seeds" on 2026-11-03: `results/deploy.jsonl` predates the deploy
    script recording that field. It cannot cause an E3 pairing across N, because those rows all
    record their step count.

    `None` (the whole key) is for a row that is not an episode at all: the E9 latency ladder
    writes one row per (mode, N) benchmark, and pairing it by seed crashed the comparison of the
    first full run with `KeyError: 'eval_seed'`, after all twelve stages had succeeded.
    """
    if "eval_seed" not in row or "success" not in row:
        return None
    return {
        "seed": int(row["eval_seed"]),
        "episode": int(row["episode"]) if row.get("episode") is not None else -1,
        "strategy": str(row.get("strategy") or ""),
        "steps": _opt_int(row.get("sampler_steps")),
        "latency": _opt_float(row.get("latency_ms")),
        "sampler": _opt_text(row.get("sampler")),
        "dtype": _opt_text(row.get("dtype")),
    }


def _episode_id(key: dict[str, Any]) -> tuple[int, int, str]:
    return (key["seed"], key["episode"], key["strategy"])


def _same_episode(fresh: dict[str, Any], committed: dict[str, Any]) -> bool:
    """Unknown conditions are wildcards; recorded ones must agree."""
    return all(
        fresh[name] is None or committed[name] is None or fresh[name] == committed[name]
        for name in CONDITIONS
    )


def _pair_rows(
    new_rows: list[dict[str, Any]], old_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair fresh rows with committed ones on the same episode under the same conditions."""
    index: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in old_rows:
        key = _key(row)
        if key is not None:
            index.setdefault(_episode_id(key), []).append(row)
    used: set[int] = set()
    shared: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in new_rows:
        key = _key(row)
        if key is None:
            continue
        for candidate in index.get(_episode_id(key), []):
            if id(candidate) in used:
                continue
            other = _key(candidate)
            if other is not None and _same_episode(key, other):
                used.add(id(candidate))
                shared.append((row, candidate))
                break
    return shared


def compare_stage(fresh: Path, committed: Path) -> dict[str, Any]:
    """Pair a fresh block with its committed counterpart and state the agreement.

    Pairing is by episode and condition, so a fresh block over a *subset* of the committed seeds
    (the quick profile) compares on the seeds it actually ran rather than reporting a rate
    mismatch that is only a different `n`.
    """
    raw_new = read_rows(fresh)
    raw_old = read_rows(committed)
    if not raw_new or not raw_old:
        return {"stage": fresh.stem, "n": 0, "status": "not compared"}
    new_rows = [row for row in raw_new if _key(row) is not None]
    old_rows = [row for row in raw_old if _key(row) is not None]
    if not new_rows or not old_rows:
        return {"stage": fresh.stem, "n": 0, "status": "not an episode block"}
    shared = _pair_rows(new_rows, old_rows)
    if not shared:
        return {"stage": fresh.stem, "n": 0, "status": "no shared seeds"}
    new_ok = sum(1 for a, _ in shared if a["success"])
    old_ok = sum(1 for _, b in shared if b["success"])
    only_new = sum(1 for a, b in shared if a["success"] and not b["success"])
    only_old = sum(1 for a, b in shared if not a["success"] and b["success"])
    p = mcnemar_exact(only_new, only_old)
    discordance = (only_new + only_old) / len(shared)
    # A reproducibility row is a null whenever the run agrees with the evidence, and a null is only
    # worth the paper it is on if it says what it could have seen. The resolution is per block (n
    # and the discordance actually observed), not a global "45 episodes is fine".
    resolves = resolvable_effect(len(shared), discordance) if discordance > 0 else None
    return {
        "stage": fresh.stem,
        "n": len(shared),
        "fresh_success": round(new_ok / len(shared), 4),
        "committed_success": round(old_ok / len(shared), 4),
        "agree": len(shared) - only_new - only_old,
        "fresh_only": only_new,
        "committed_only": only_old,
        "discordance": round(discordance, 3),
        "delta_pp": round(100 * (new_ok - old_ok) / len(shared), 1),
        "resolves_pp": None if resolves is None else round(100 * resolves, 1),
        "p": round(p, 4),
        "summary": f"{only_new}/{only_old} discordant, exact McNemar {p_text(p)}",
    }


def compare_all(run: Path) -> list[dict[str, Any]]:
    rows = run / "rows"
    return [
        compare_stage(rows / Path(target).name, REPO / target)
        for stage, target in COMMITTED.items()
        if (rows / Path(target).name).exists()
    ]


def render(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return "no fresh rows to compare"
    lines = [
        f"{'stage':<14}{'n':>5}{'fresh':>9}{'committed':>11}{'agree':>7}{'disc':>6}"
        f"{'delta':>8}{'resolves':>10}{'p':>9}",
    ]
    for c in comparisons:
        if not c.get("n"):
            lines.append(f"{c['stage']:<14}{'-':>5}  {c.get('status', '')}")
            continue
        disc = c["fresh_only"] + c["committed_only"]
        resolves = c.get("resolves_pp")
        lines.append(
            f"{c['stage']:<14}{c['n']:>5}{c['fresh_success']:>9.1%}"
            f"{c['committed_success']:>11.1%}{c['agree']:>7}{disc:>6}"
            f"{c.get('delta_pp', 0.0):>+7.1f} "
            f"{'-' if resolves is None else f'{resolves:.1f} pp':>9}{c['p']:>9.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full", choices=["quick", "full"])
    parser.add_argument("--out-dir", type=Path, default=None, help="fresh rows and stage logs")
    parser.add_argument("--only", default="", help="comma-separated stage names")
    parser.add_argument("--skip", default="", help="comma-separated stage names")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failed stage")
    parser.add_argument(
        "--compare-only",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="skip every stage and only rebuild the reproducibility block of an existing run",
    )
    args = parser.parse_args()

    if args.compare_only is not None:
        comparisons = compare_existing(args.compare_only)
        print(render(comparisons))
        print(f"\nrun dir: {args.compare_only}")
        return

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = args.out_dir or Path("results/logs") / f"pipeline_{args.profile}_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    stages = build_stages(run, args.profile)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {stage.name for stage in stages}
        if unknown:
            parser.error(f"unknown stage(s): {sorted(unknown)}")
        stages = [stage for stage in stages if stage.name in wanted]
    if args.skip:
        unwanted = {name.strip() for name in args.skip.split(",") if name.strip()}
        stages = [stage for stage in stages if stage.name not in unwanted]
    if not stages:
        parser.error("no stages selected")

    if args.dry_run:
        for index, stage in enumerate(stages, 1):
            print(f"{index}/{len(stages)} {stage.name:<18} ~{stage.minutes:5.1f} min")
            print(f"    {stage.description}")
            for command in stage.commands:
                print("      $ " + " ".join(command))
        print(f"\nrows and logs -> {run}")
        return

    log = RunLog(default_log_path(f"pipeline_{args.profile}"), run_id=f"pipeline_{args.profile}")
    log.progress(
        "pipeline_start",
        profile=args.profile,
        stages=len(stages),
        planned_min=round(sum(stage.minutes for stage in stages), 1),
        run_dir=str(run),
        log=str(log.path),
    )
    results: list[StageResult] = []
    started = time.perf_counter()
    for index, stage in enumerate(stages, 1):
        print(f"[{index}/{len(stages)}] {stage.name} (~{stage.minutes:.0f} min) ...", flush=True)
        result = run_stage(stage, run, log)
        results.append(result)
        print(f"    {result.status} in {result.seconds / 60:.1f} min {result.detail}", flush=True)
        _write_summary(
            run,
            args.profile,
            [row.as_dict() for row in results],
            comparisons=[],
            seconds=time.perf_counter() - started,
        )
        if result.status == "failed" and args.fail_fast:
            break

    comparisons = compare_all(run)
    _write_summary(
        run,
        args.profile,
        [row.as_dict() for row in results],
        comparisons=comparisons,
        seconds=time.perf_counter() - started,
    )
    print("\n" + render(comparisons))
    for row in results:
        log.event("stage_result", **row.as_dict())
    failed = [row.name for row in results if row.status == "failed"]
    log.progress(
        "pipeline_end",
        ok=sum(1 for row in results if row.status == "ok"),
        failed=len(failed),
        skipped=sum(1 for row in results if row.status == "skipped"),
        wall_min=round((time.perf_counter() - started) / 60, 1),
        run_dir=str(run),
    )
    log.close()
    print(f"\nrun dir: {run}")
    print(f"run log: {log.path}")
    if failed:
        print(f"FAILED stages: {', '.join(failed)}")
        raise SystemExit(1)


def _write_summary(
    run: Path,
    profile: str,
    stages: list[dict[str, Any]],
    *,
    comparisons: list[dict[str, Any]],
    seconds: float,
) -> None:
    payload = {
        "profile": profile,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "stages": stages,
        "reproducibility": comparisons,
    }
    (run / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def compare_existing(run: Path) -> list[dict[str, Any]]:
    """Re-run only the comparison for a finished run directory and rewrite its `summary.json`.

    The stages are the expensive part and the comparison is the part that can be wrong on its own
    (it crashed on 2026-11-03, after 49 minutes of green stages). Recomputing it from the rows on
    disk is how that gets repaired without re-running a single episode: the stage records and
    wall-clock come from the existing summary, the reproducibility block is replaced.
    """
    summary_path = run / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found -- --compare-only needs a finished run")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    comparisons = compare_all(run)
    _write_summary(
        run,
        str(previous.get("profile", "full")),
        list(previous.get("stages") or []),
        comparisons=comparisons,
        seconds=float(previous.get("seconds") or 0.0),
    )
    return comparisons


if __name__ == "__main__":
    main()
