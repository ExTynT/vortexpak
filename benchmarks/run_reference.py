"""Execute both complete v2 validation paths sequentially; never recycle timings."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import protocol as bp  # noqa: E402

CORPUS_PATHS = {
    "minetest": "legacy_games/minetest_game",
    "kenney": "legacy_games/kenney_3d_platformer",
    "browserquest": "legacy_games/browserquest",
    "silesia": "silesia",
    "kodak": "kodak",
}


def commands(corpus_root, output, receipt):
    values = []
    for path in ("reference", "optimized"):
        for case, relative in CORPUS_PATHS.items():
            quality = case == "kodak"
            script = "rigorous_quality.py" if quality else "rigorous.py"
            args = [
                sys.executable,
                "-B",
                str(ROOT / "benchmarks" / script),
                "--corpus",
                str(corpus_root / relative),
                "--output",
                str(output / path / case),
                "--receipt",
                str(receipt),
                "--profile",
                "reference",
                "--validation-path",
                path,
                "--threads",
                "1",
                "--repeats",
                "7",
                "--warmups",
                "1",
            ]
            args += (
                ["--require-kodak"]
                if quality
                else ["--requests", "100" if case == "silesia" else "10000"]
            )
            values.append({"path": path, "case": case, "argv": args})
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root, out, receipt = (
        args.corpus_root.resolve(),
        args.output.resolve(),
        args.receipt.resolve(),
    )
    schedule = commands(root, out, receipt)
    if args.dry_run:
        print(json.dumps(schedule, indent=2))
        return
    if out.exists():
        ap.error("suite output exists; evidence is immutable")
    if root == out or root in out.parents:
        ap.error("output must be outside corpus")
    bp.verify_receipt(json.loads(receipt.read_text(encoding="utf-8")))
    for case, rel in CORPUS_PATHS.items():
        if not (root / rel).is_dir():
            ap.error("missing registered corpus: " + case)
    out.mkdir(parents=True)
    suite = {
        "schema": 2,
        "status": "running",
        "schedule": schedule,
        "runs": [],
        "equivalence": {},
        "protocol": bp.identity(),
        "runtime_scope": "subprocess start through exit, includes worker preparation and evidence persistence",
    }
    bp.atomic_json(out / "suite-plan.json", suite)
    lock = ROOT / "benchmarks/reports/v2-measurement.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            "another reference suite or stale lock exists; do not run measured suites concurrently"
        )
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "output": str(out)}, f)
    started = time.perf_counter_ns()
    try:
        for index, item in enumerate(schedule):
            print(
                f"[{index + 1}/{len(schedule)}] {item['path']}/{item['case']}",
                flush=True,
            )
            start = time.perf_counter_ns()
            log = out / f"{item['path']}-{item['case']}.log"
            with log.open("xb") as stream:
                result = subprocess.run(
                    item["argv"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=dict(
                        os.environ,
                        PYTHONDONTWRITEBYTECODE="1",
                        PYTHONIOENCODING="utf-8",
                    ),
                )
            elapsed = time.perf_counter_ns() - start
            record = {
                "path": item["path"],
                "case": item["case"],
                "returncode": result.returncode,
                "elapsed_wall_ns": elapsed,
                "log_sha256": bp.sha(log.read_bytes()),
            }
            suite["runs"].append(record)
            bp.atomic_json(out / f"stage-{index:02d}.json", record)
            if result.returncode:
                raise RuntimeError("experiment failed; inspect " + str(log))
            bp.load_report(out / item["path"] / item["case"])
        for case in CORPUS_PATHS:
            baseline = bp.load_report(out / "reference" / case)
            candidate = bp.load_report(out / "optimized" / case)
            suite["equivalence"][case] = bp.equivalent_results(baseline, candidate)
        for validation in ("reference", "optimized"):
            folder = out / validation
            for script, tail in (
                (
                    "report_benchmarks.py",
                    [str(folder), "--output", str(folder / "index.html")],
                ),
                (
                    "report_image.py",
                    [
                        "--root",
                        str(folder),
                        "--output",
                        str(folder / "benchmark-report.png"),
                    ],
                ),
            ):
                subprocess.run(
                    [sys.executable, "-B", str(ROOT / "benchmarks" / script), *tail],
                    check=True,
                )
        totals = {
            path: sum(
                row["elapsed_wall_ns"] for row in suite["runs"] if row["path"] == path
            )
            for path in ("reference", "optimized")
        }
        suite.update(
            status="complete",
            full_process_wall_ns=totals,
            observed_wall_change_pct=100
            * (totals["optimized"] / totals["reference"] - 1),
            elapsed_wall_ns=time.perf_counter_ns() - started,
            interpretation="harness wall-time comparison only; not a codec speedup or statistical population claim",
        )
        bp.atomic_json(out / "suite.json", suite)
        bp.atomic_json(
            out / "REFERENCE.json",
            {
                "schema": 2,
                "suite_sha256": bp.sha((out / "suite.json").read_bytes()),
                "selected": "optimized",
                "selection_rule": "first predeclared complete qualifying suite, not best-of",
                "runs": {
                    case: bp.load_report(out / "optimized" / case)["evidence_head"]
                    for case in CORPUS_PATHS
                },
            },
        )
        print(out / "REFERENCE.json", flush=True)
    except BaseException as exc:
        suite.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            elapsed_wall_ns=time.perf_counter_ns() - started,
        )
        bp.atomic_json(out / "suite-failure.json", suite)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
