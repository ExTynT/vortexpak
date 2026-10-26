"""Versioned experiment contract. Raw evidence, never a rendered chart, is authority."""

import argparse
import gzip
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import json
import math
from pathlib import Path
import random
import statistics
import uuid

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def spec():
    value = json.loads((ROOT / "benchmarks/protocol.json").read_text(encoding="utf-8"))
    if value.get("schema") != 2 or value.get("version") != "2.0.0":
        raise ValueError("unsupported protocol")
    if len(value["archive_configs"]) != len(set(value["archive_configs"])):
        raise ValueError("duplicate registered configurations")
    return value


def harness_fingerprint():
    names = [
        "protocol.json",
        "protocol.py",
        "measurement.py",
        "rigorous.py",
        "rigorous_quality.py",
        "run_reference.py",
    ]
    return {
        name: sha((ROOT / "benchmarks" / name).read_bytes())
        for name in names
        if (ROOT / "benchmarks" / name).is_file()
    }


def identity():
    return {"spec": spec(), "harness": harness_fingerprint()}


def case_name(manifest):
    for name, case in spec()["corpora"].items():
        if (
            manifest["sha256"] == case["sha256"]
            and len(manifest["files"]) == case["files"]
            and manifest["bytes"] == case["bytes"]
        ):
            return name
    return None


def settings(
    profile, manifest, track, configs, threads, repeats, warmups, requests, seed
):
    s = spec()
    if profile not in s["profiles"]:
        raise ValueError("unknown profile")
    case = case_name(manifest)
    registered = (
        list(s["quality_configs"]) if track == "quality" else s["archive_configs"]
    )
    value = dict(
        profile=profile,
        case=case,
        track=track,
        configs=list(configs),
        threads=threads,
        repeats=repeats,
        warmups=warmups,
        requests=requests,
        seed=seed,
        cache_modes=(
            ["codec_reset", "repeat_after_one_prime"] if track == "archive" else []
        ),
        query_sampling="uniform with replacement over manifest asset_id",
        validation="every output",
    )
    if len(configs) != len(set(configs)) or not set(configs).issubset(registered):
        raise ValueError("unregistered or duplicate configuration")
    if (
        min(threads, repeats) < 1
        or requests < (0 if track == "quality" else 1)
        or warmups < 0
        or not 0 <= seed < 2**32
    ):
        raise ValueError("invalid numeric settings")
    if profile != "diagnostic":
        expected = s["profiles"][profile]
        query_n = (
            expected["requests_silesia"]
            if case == "silesia"
            else expected["requests_assets"]
        )
        if track == "quality":
            query_n = 0
        if (
            case is None
            or (track == "quality") != (case == "kodak")
            or set(configs) != set(registered)
            or threads != expected["threads"]
            or repeats != expected["repeats"]
            or warmups != expected["warmups"]
            or seed != s["seed"]
            or requests != query_n
        ):
            raise ValueError(
                "reference profile requires the exact registered whole experiment"
            )
    return value


def make_plan(configs, repeats, warmups, seed):
    rng = random.Random(seed)
    return [
        {"round": rnd, "order": rng.sample(list(configs), len(configs))}
        for rnd in range(-warmups, repeats)
    ]


def query_ids(manifest, requests, seed):
    return random.Random(seed).choices(range(len(manifest["files"])), k=requests)


def trial_id(rnd, config):
    return f"round-{rnd:03d}-{config}"


def comparison_key(report):
    settings_ = {
        k: v for k, v in report["settings"].items() if k not in ("validation_path",)
    }
    return sha(
        canonical(
            {
                "protocol": report["protocol_id"],
                "settings": settings_,
                "input": report["manifest"]["sha256"],
            }
        )
    )


def initialize(manifest, configuration, track):
    experiment = identity()
    return {
        "schema": 2,
        "run_id": uuid.uuid4().hex,
        "session_id": uuid.uuid4().hex,
        "status": "running",
        "track": track,
        "protocol": experiment,
        "protocol_id": sha(canonical(experiment)),
        "manifest": manifest,
        "settings": configuration,
        "schedule": make_plan(
            configuration["configs"],
            configuration["repeats"],
            configuration["warmups"],
            configuration["seed"],
        ),
        "trials": [],
        "warmup_trials": [],
        "errors": [],
    }


def exact_metric(values):
    if not values:
        raise ValueError("no observations")
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "deterministic": len(set(values)) == 1,
        "uncertainty": "exact observed values, no sampling CI",
    }


def validate_run(report):
    if report.get("schema") != 2 or report.get("status") != "complete":
        raise ValueError("incomplete or unsupported experiment")
    if sha(canonical(report["protocol"])) != report["protocol_id"]:
        raise ValueError("protocol fingerprint mismatch")
    if report.get("errors"):
        raise ValueError("failed attempt in completed run")
    cfg = report["settings"]
    if set(report["summary"]) != set(cfg["configs"]):
        raise ValueError("summary configuration coverage differs")
    if cfg["profile"] != "diagnostic" and report.get("receipt") is None:
        raise ValueError("unattested reference")
    contract = report["protocol"]["spec"]
    if cfg["profile"] != "diagnostic":
        ref = contract["profiles"][cfg["profile"]]
        case = contract["corpora"].get(cfg["case"])
        ids = (
            list(contract["quality_configs"])
            if report["track"] == "quality"
            else contract["archive_configs"]
        )
        queries = (
            0
            if report["track"] == "quality"
            else (
                ref["requests_silesia"]
                if cfg["case"] == "silesia"
                else ref["requests_assets"]
            )
        )
        if (
            case is None
            or report["manifest"]["sha256"] != case["sha256"]
            or len(report["manifest"]["files"]) != case["files"]
            or report["manifest"]["bytes"] != case["bytes"]
            or set(cfg["configs"]) != set(ids)
            or cfg["seed"] != contract["seed"]
            or cfg["requests"] != queries
            or any(cfg[k] != ref[k] for k in ("threads", "repeats", "warmups"))
        ):
            raise ValueError("registered reference contract differs")
    expected = {(r["round"], c) for r in report["schedule"] for c in r["order"]}
    observed = [
        (t["round"], t["config"]) for t in report["trials"] + report["warmup_trials"]
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("duplicate or missing trial")
    planned = make_plan(cfg["configs"], cfg["repeats"], cfg["warmups"], cfg["seed"])
    if report["schedule"] != planned:
        raise ValueError("experiment schedule differs")
    if any(t["round"] < 0 for t in report["trials"]) or any(
        t["round"] >= 0 for t in report["warmup_trials"]
    ):
        raise ValueError("warmup/measurement mixture")
    for t in report["trials"] + report["warmup_trials"]:
        if report["track"] == "archive":
            if t["verified_files"] != len(report["manifest"]["files"]):
                raise ValueError("incomplete integrity check")
            for mode in cfg["cache_modes"]:
                lat = t.get("latency", {}).get(mode)
                if (
                    t["config"]
                    in {
                        "vpak",
                        "zip-store",
                        "zip-deflate",
                        "indexed-zstd3",
                        "perfile-zstd3",
                    }
                    and lat is None
                ):
                    raise ValueError("missing seekable query evidence")
                if lat is not None:
                    ids = query_ids(
                        report["manifest"],
                        cfg["requests"],
                        cfg["seed"] + max(t["round"], 0),
                    )
                    if (
                        lat["asset_ids"] != ids
                        or len(lat["raw_ns"]) != len(ids)
                        or lat["verified_queries"] != len(ids)
                        or any(type(ns) is not int or ns < 0 for ns in lat["raw_ns"])
                        or (
                            mode == "repeat_after_one_prime"
                            and lat["verified_primes"] != len(ids)
                        )
                    ):
                        raise ValueError("invalid query evidence")
        else:
            if len(t["images"]) != len(report["manifest"]["files"]) - len(
                report.get("excluded", [])
            ) or set(i["path"] for i in t["images"]) != set(report["image_contracts"]):
                raise ValueError("missing quality image")
    for name, summary in report["summary"].items():
        rows = [t for t in report["trials"] if t["config"] == name]
        key = "archive_bytes" if report["track"] == "archive" else "bytes"
        if summary[key]["median"] != statistics.median(t[key] for t in rows):
            raise ValueError("summary disagrees with raw observations")

        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite data")
            if isinstance(value, dict):
                for v in value.values():
                    finite(v)
            elif isinstance(value, list):
                for v in value:
                    finite(v)

        finite(summary)
    if report["track"] == "quality":
        count = sum(
            len(t["images"]) for t in report["trials"] + report["warmup_trials"]
        )
        if report["oracle"]["validated_encodes"] != count:
            raise ValueError("incomplete repeated encode validation")
        oracle_calls = report["oracle"]["reference_calls"]
        expected_calls = (
            count
            if cfg["validation_path"] == "reference"
            else report["oracle"]["unique_contracts"]
        )
        if oracle_calls != expected_calls:
            raise ValueError("oracle coverage differs")
    return report


def source_files(base):
    base = Path(base)
    paths = list((base / "vortexpak").rglob("*.py"))
    paths += list((base / "vortexpak_native/src").rglob("*.rs"))
    paths += [
        base / n
        for n in (
            "vortexpak_native/Cargo.toml",
            "vortexpak_native/Cargo.lock",
            "vortexpak_native/vortexpak.h",
            "pyproject.toml",
            "setup.py",
            "rust-toolchain.toml",
            ".cargo/config.toml",
        )
    ]
    return {
        p.relative_to(base).as_posix(): sha(p.read_bytes())
        for p in paths
        if p.is_file()
    }


def atomic_json(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    if path.exists() and not replace:
        tmp.unlink()
        raise FileExistsError(path)
    tmp.replace(path)


def build_context():
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in (
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_TARGET",
            "CARGO_TARGET_DIR",
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "AR",
        )
        or k.startswith(("CARGO_PROFILE_RELEASE_", "CARGO_TARGET_", "ZSTD_"))
    }
    paths = []
    for parent in [ROOT, *ROOT.parents]:
        paths += [parent / ".cargo/config", parent / ".cargo/config.toml"]
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    paths += [cargo_home / "config", cargo_home / "config.toml"]
    return {
        "environment": env,
        "cargo_config_hashes": {
            str(p): sha(p.read_bytes()) for p in paths if p.is_file()
        },
        "credential_policy": "credentials files never read; external configs hashed, not published in plaintext",
    }


def build_receipt(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("new immutable receipt directory required")
    before = source_files(ROOT)
    context = build_context()
    flags = {
        k: v
        for k, v in os.environ.items()
        if k in ("RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR")
        or k.startswith("CARGO_PROFILE_RELEASE_")
    }
    command = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        str(ROOT / "vortexpak_native/Cargo.toml"),
        "--message-format=json",
    ]
    built = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    )
    artifacts = []
    for line in built.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (
            row.get("reason") == "compiler-artifact"
            and row.get("target", {}).get("name") == "vortexpak_native"
        ):
            artifacts.extend(
                Path(f)
                for f in row.get("filenames", [])
                if Path(f).suffix in (".dll", ".so", ".dylib", ".pyd")
            )
    if not artifacts or source_files(ROOT) != before or build_context() != context:
        raise RuntimeError("build produced no library, or source changed during build")
    native = artifacts[-1].resolve()
    snapshot = output / "snapshot"
    for name in before:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    native_path = snapshot / "vortexpak" / native.name
    shutil.copy2(native, native_path)
    receipt = {
        "schema": 2,
        "build_context": context,
        "command": command,
        "flags": flags,
        "source_hashes": before,
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True),
        "snapshot_root": str(snapshot),
        "native_relpath": native_path.relative_to(snapshot).as_posix(),
        "native_sha256": sha(native_path.read_bytes()),
        "build_stderr": built.stderr,
        "snapshot_files": {
            p.relative_to(snapshot).as_posix(): sha(p.read_bytes())
            for p in snapshot.rglob("*")
            if p.is_file()
        },
    }
    receipt["receipt_id"] = sha(canonical(receipt))
    atomic_json(output / "receipt.json", receipt)
    verify_receipt(receipt)
    return receipt


def verify_receipt(receipt, root_override=None):
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_id"}
    if sha(canonical(unsigned)) != receipt.get("receipt_id"):
        raise ValueError("build receipt fingerprint differs")
    root = Path(root_override or receipt["snapshot_root"]).resolve()
    observed = {
        p.relative_to(root).as_posix(): sha(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if (
        observed != receipt["snapshot_files"]
        or source_files(root) != receipt["source_hashes"]
    ):
        raise ValueError("immutable implementation snapshot changed")
    if sha((root / receipt["native_relpath"]).read_bytes()) != receipt["native_sha256"]:
        raise ValueError("wrong native library")
    return root


def activate(receipt):
    if receipt is None:
        return
    root = verify_receipt(receipt)
    sys.dont_write_bytecode = True
    if "vortexpak" in sys.modules and not Path(
        sys.modules["vortexpak"].__file__
    ).resolve().is_relative_to(root):
        raise RuntimeError(
            "another implementation already imported; use a fresh process"
        )
    sys.path.insert(0, str(root))


def verify_loaded(receipt):
    if receipt is None:
        return
    from vortexpak import native

    root = verify_receipt(receipt)
    if (
        native._lib is None
        or Path(native._lib._name).resolve()
        != (root / receipt["native_relpath"]).resolve()
    ):
        raise RuntimeError("loaded DLL is not the receipted artifact")
    if sha(Path(native._lib._name).read_bytes()) != receipt["native_sha256"]:
        raise RuntimeError("loaded DLL hash differs")


def runtime_fingerprint(receipt=None):
    base = verify_receipt(receipt) if receipt else ROOT
    value = {
        "sources": source_files(base),
        "harness": harness_fingerprint(),
        "receipt_id": receipt["receipt_id"] if receipt else None,
    }
    from vortexpak import native

    value["native"] = sha(Path(native._lib._name).read_bytes()) if native._lib else None
    return value


def dependencies():
    result = {}
    for name in (
        "numpy",
        "pillow",
        "zstandard",
        "lz4",
        "xxhash",
        "psutil",
        "flip-evaluator",
    ):
        dist = importlib.metadata.distribution(name)
        binaries = {}
        for name_ in dist.files or []:
            path = Path(dist.locate_file(name_))
            if path.is_file() and path.suffix.lower() in (
                ".dll",
                ".pyd",
                ".so",
                ".dylib",
            ):
                binaries[str(name_)] = sha(path.read_bytes())
        result[name] = {"version": dist.version, "native_files": binaries}
    return result


def freeze_inputs(manifest, output):
    dest = ROOT.parent / "vortex_corpus" / ".benchmark-snapshots" / manifest["sha256"]
    if dest.exists():
        for row in manifest["files"]:
            data = (dest / row["path"]).read_bytes()
            if len(data) != row["bytes"] or sha(data) != row["sha256"]:
                raise ValueError("existing input snapshot differs")
        return {
            **manifest,
            "source_root": manifest["root"],
            "root": str(dest),
            "snapshot_policy": "full checked byte copy",
        }
    if dest.exists():
        raise FileExistsError("input snapshot exists")
    dest.mkdir(parents=True)
    for row in manifest["files"]:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or path.drive:
            raise ValueError("unsafe manifest path")
        source = Path(manifest["root"]) / path
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        data = target.read_bytes()
        if len(data) != row["bytes"] or sha(data) != row["sha256"]:
            raise ValueError("input changed while freezing: " + row["path"])
    return {
        **manifest,
        "source_root": manifest["root"],
        "root": str(dest),
        "snapshot_policy": "full checked byte copy",
    }


class IncompleteReportError(ValueError):
    pass


class EventStore:
    def __init__(self, root, report):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        if self.path.exists() or (self.root / "report.json").exists():
            raise FileExistsError("immutable run already exists")
        self.head = "0" * 64
        self.count = 0
        self.append(
            "run_started",
            {
                "run_id": report["run_id"],
                "pid": os.getpid(),
                "session_id": report["session_id"],
                "protocol_id": report["protocol_id"],
            },
        )

    def append(self, event, payload):
        row = {
            "index": self.count,
            "previous": self.head,
            "event": event,
            "payload": payload,
        }
        row["hash"] = sha(canonical(row))
        with self.path.open("ab") as f:
            f.write(canonical(row) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        self.head, self.count = row["hash"], self.count + 1
        return row

    def trial(self, value):
        name = "trials/" + trial_id(value["round"], value["config"]) + ".json"
        path = self.root / name
        atomic_json(path, value)
        self.append(
            "trial_completed",
            {
                "artifact": name,
                "sha256": sha(path.read_bytes()),
                "round": value["round"],
                "config": value["config"],
            },
        )


def ledger(root):
    path = Path(root) / "events.jsonl"
    rows, head = [], "0" * 64
    with path.open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                raise ValueError("truncated ledger")
            row = json.loads(line)
            unsigned = {k: v for k, v in row.items() if k != "hash"}
            if (
                row["index"] != len(rows)
                or row["previous"] != head
                or sha(canonical(unsigned)) != row["hash"]
            ):
                raise ValueError("ledger chain differs")
            head = row["hash"]
            rows.append(row)
    return rows, head


def capture_receipt(receipt, root):
    if receipt is None:
        return None
    source = verify_receipt(receipt)
    target = Path(root).resolve() / "implementation"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    copied = {
        **receipt,
        "snapshot_root": str(target),
        "upstream_receipt_id": receipt["receipt_id"],
    }
    copied.pop("receipt_id")
    copied["receipt_id"] = sha(canonical(copied))
    verify_receipt(copied)
    atomic_json(Path(root) / "receipt.json", copied)
    return copied


def capture_harness(root, report):
    for name, expected in report["protocol"]["harness"].items():
        data = (ROOT / "benchmarks" / name).read_bytes()
        if sha(data) != expected:
            raise ValueError("harness changed while snapshotting")
        path = Path(root) / "harness" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)
    atomic_json(Path(root) / "protocol.json", report["protocol"])
    atomic_json(Path(root) / "manifest.json", report["manifest"])


def finalize(root, report, store):
    root = Path(root)
    report["status"] = "complete"
    store.append(
        "measurements_complete",
        {"run_id": report["run_id"], "trials": len(report["trials"])},
    )
    report["evidence_head"] = store.head
    report["comparison_key"] = comparison_key(report)
    validate_run(report)
    atomic_json(root / "report.json", report)
    raw = (root / "report.json").read_bytes()
    with (root / "report.json.gz").open("xb") as f:
        f.write(gzip.compress(raw, mtime=0))
    paths = [
        "report.json",
        "report.json.gz",
        "events.jsonl",
        "protocol.json",
        "manifest.json",
    ]
    if (root / "receipt.json").exists():
        paths.append("receipt.json")
    for p in (root / "harness").glob("*"):
        paths.append(p.relative_to(root).as_posix())
    atomic_json(
        root / "bundle.json",
        {
            "schema": 2,
            "run_id": report["run_id"],
            "head": store.head,
            "files": {p: sha((root / p).read_bytes()) for p in paths},
        },
    )
    load_report(root)


def load_report(root, legacy=False):
    root = Path(root).resolve()
    if not (root / "report.json").exists():
        raise IncompleteReportError("incomplete experiment: no final report")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise IncompleteReportError(
            "incomplete experiment: " + str(report.get("status"))
        )
    if report.get("schema") != 2:
        if legacy:
            return report
        raise ValueError("legacy report requires explicit --legacy")
    validate_run(report)
    if report.get("receipt"):
        verify_receipt(report["receipt"], root / "implementation")
    rows, head = ledger(root)
    if (
        report["evidence_head"] != head
        or not rows
        or rows[-1]["event"] != "measurements_complete"
    ):
        raise ValueError("final evidence head differs")
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if bundle["run_id"] != report["run_id"] or bundle["head"] != head:
        raise ValueError("bundle identity differs")
    for name, expected in bundle["files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.drive
            or sha((root / path).read_bytes()) != expected
        ):
            raise ValueError("bundle artifact differs: " + name)
    by_id = {
        (t["round"], t["config"]): t for t in report["trials"] + report["warmup_trials"]
    }
    completed = []
    started = []
    for row in rows:
        if row["event"] == "trial_started":
            started.append((row["payload"]["round"], row["payload"]["config"]))
        if row["event"] == "trial_completed":
            info = row["payload"]
            key = (info["round"], info["config"])
            name = Path(info["artifact"])
            if name.is_absolute() or ".." in name.parts or name.drive:
                raise ValueError("unsafe trial artifact")
            raw = (root / name).read_bytes()
            if sha(raw) != info["sha256"] or canonical(json.loads(raw)) != canonical(
                by_id.get(key)
            ):
                raise ValueError("trial artifact disagrees with raw report")
            completed.append(key)
    if (
        set(started) != set(by_id)
        or len(started) != len(by_id)
        or set(completed) != set(by_id)
        or len(completed) != len(by_id)
    ):
        raise ValueError("unaccounted attempt")
    return report


def seal_interrupted(root):
    import psutil

    root = Path(root)
    rows, head = ledger(root)
    if not rows or any(r["event"] == "measurements_complete" for r in rows):
        raise ValueError("completed or empty session cannot be resumed")
    pid = rows[0]["payload"]["pid"]
    if psutil.pid_exists(pid):
        raise ValueError("old worker/session PID still exists; cannot recover safely")
    unsigned = {
        "index": len(rows),
        "previous": head,
        "event": "session_sealed_interrupted",
        "payload": {"reason": "new session requested; no historical attempts reused"},
    }
    unsigned["hash"] = sha(canonical(unsigned))
    with (root / "events.jsonl").open("ab") as f:
        f.write(canonical(unsigned) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return {"root": str(root.resolve()), "head": unsigned["hash"], "resumed_trials": 0}


def equivalent_results(left, right):
    validate_run(left)
    validate_run(right)
    if comparison_key(left) != comparison_key(right):
        raise ValueError("incompatible protocol/input/resource/sampling experiment")
    a, b = left.get("receipt"), right.get("receipt")
    if (a is None) != (b is None) or (
        a
        and any(
            a[k] != b[k]
            for k in ("source_hashes", "native_sha256", "rustc", "cargo", "flags")
        )
    ):
        raise ValueError("implementation differs; cannot attribute change to harness")
    if left.get("dependencies") != right.get("dependencies"):
        raise ValueError("dependency/oracle versions differ")
    pairs = zip(
        sorted(
            left["trials"] + left["warmup_trials"],
            key=lambda t: (t["round"], t["config"]),
        ),
        sorted(
            right["trials"] + right["warmup_trials"],
            key=lambda t: (t["round"], t["config"]),
        ),
    )
    checks = 0
    for a, b in pairs:
        if left["track"] == "archive":
            if any(
                a[k] != b[k]
                for k in (
                    "config",
                    "round",
                    "archive_bytes",
                    "archive_sha256",
                    "verified_files",
                )
            ):
                raise ValueError("archive/integrity output differs")
            for mode in ("codec_reset", "repeat_after_one_prime"):
                if mode in a["latency"]:
                    for key in ("asset_ids", "verified_queries", "verified_primes"):
                        if a["latency"][mode][key] != b["latency"][mode][key]:
                            raise ValueError("query validation coverage differs")
        else:
            ai = {
                v["path"]: {k: x for k, x in v.items() if k != "encode_ns"}
                for v in a["images"]
            }
            bi = {
                v["path"]: {k: x for k, x in v.items() if k != "encode_ns"}
                for v in b["images"]
            }
            if ai != bi:
                raise ValueError("encoded blocks or per-image quality differs")
        checks += 1
    if checks != len(left["trials"]) + len(left["warmup_trials"]) or checks != len(
        right["trials"]
    ) + len(right["warmup_trials"]):
        raise ValueError("different trial coverage")
    return {
        "equivalent": True,
        "compared_measured_trials": len(left["trials"]),
        "compared_warmup_trials": len(left["warmup_trials"]),
        "scope": "exact outputs/quality/coverage; no timing equality or codec-speed claim",
    }


def compare_runs(baseline, candidate):
    validate_run(baseline)
    validate_run(candidate)
    if comparison_key(baseline) != comparison_key(candidate):
        raise ValueError("incompatible experiment")
    if (
        baseline["settings"]["validation_path"]
        != candidate["settings"]["validation_path"]
    ):
        raise ValueError(
            "different harness validation paths; no codec timing attribution"
        )
    if baseline["settings"]["profile"] == "diagnostic":
        return {"status": "diagnostic_only", "reason": "not a reference experiment"}
    for key in ("cpu_model", "os", "power_scheme", "physical_cpus", "logical_cpus"):
        if (
            baseline["environment"]["host"][key]
            != candidate["environment"]["host"][key]
        ):
            raise ValueError("different host conditions")
    if baseline["dependencies"] != candidate["dependencies"]:
        raise ValueError("different measurement dependencies")
    from benchmarks.rigorous import median_ci

    differences, violations, inconclusive = [], [], False
    tolerance = baseline["protocol"]["spec"]["regression"]["timing_tolerance_percent"]
    for cfg in baseline["settings"]["configs"]:
        left = {t["round"]: t for t in baseline["trials"] if t["config"] == cfg}
        right = {t["round"]: t for t in candidate["trials"] if t["config"] == cfg}
        if set(left) != set(right):
            raise ValueError("unpaired rounds")
        sections = (
            ("pack", "restore") if baseline["track"] == "archive" else ("encode",)
        )
        if (
            baseline["track"] == "archive"
            and "codec_reset" in left[next(iter(left))]["latency"]
        ):
            if baseline["adequacy"]["p99_at_least_100_tail_observations_per_trial"]:
                sections += ("latency",)
        for section in sections:
            values = []
            for rnd in left:
                a = (
                    left[rnd]["latency"]["codec_reset"]["summary_ms"]["p99"]
                    if section == "latency"
                    else (
                        left[rnd][section]["wall_ns"]
                        if section != "encode"
                        else left[rnd]["encode_ns"]
                    )
                )
                b = (
                    right[rnd]["latency"]["codec_reset"]["summary_ms"]["p99"]
                    if section == "latency"
                    else (
                        right[rnd][section]["wall_ns"]
                        if section != "encode"
                        else right[rnd]["encode_ns"]
                    )
                )
                values.append(100 * (b / a - 1))
            stat = median_ci(values)
            lo, hi = stat["bootstrap_95ci"]
            differences.append(
                {"config": cfg, "metric": section, "paired_change_pct": stat}
            )
            inconclusive |= lo <= tolerance < hi
            if lo > tolerance:
                violations.append(
                    {
                        "config": cfg,
                        "metric": section,
                        "kind": "potential_timing_regression",
                    }
                )
        for rnd in left:
            a, b = left[rnd], right[rnd]
            if a["affinity"] != b["affinity"]:
                raise ValueError("actual CPU affinity differs")
            if baseline["track"] == "archive":
                if b["archive_bytes"] > a["archive_bytes"]:
                    violations.append(
                        {"config": cfg, "round": rnd, "kind": "size_regression"}
                    )
            else:
                old = {r["path"]: r for r in a["images"]}
                for row in b["images"]:
                    for metric in (
                        "rgb_mse_srgb8",
                        "rgb_mse_linear",
                        "alpha_mse",
                        "rgb_max_abs_error",
                        "flip_rgb_ppd67",
                    ):
                        if row[metric] > old[row["path"]][metric]:
                            violations.append(
                                {
                                    "config": cfg,
                                    "image": row["path"],
                                    "metric": metric,
                                    "kind": "quality_regression",
                                }
                            )
    return {
        "status": (
            "potential_regression"
            if violations
            else "inconclusive" if inconclusive else "no_material_regression_observed"
        ),
        "differences": differences,
        "violations": violations,
        "scope": "descriptive same-host paired rounds; no family-wise significance or production SLO claim",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build", "verify"])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args()
    if args.command == "build":
        if args.output is None:
            ap.error("--output required")
        result = build_receipt(args.output)
        print(result["receipt_id"])
    else:
        if args.receipt is None:
            ap.error("--receipt required")
        print(verify_receipt(json.loads(args.receipt.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
