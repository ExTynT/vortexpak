"""Reproducible, byte-exact archive benchmark. See docs/BENCHMARKING.md."""

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = (
    "vpak",
    "tar-zstd3",
    "tar-zstd9",
    "tar-zstd19",
    "tar-lz4",
    "zip-store",
    "zip-deflate",
    "indexed-zstd3",
    "perfile-zstd3",
)
SEEKABLE = {"vpak", "zip-store", "zip-deflate", "indexed-zstd3", "perfile-zstd3"}
BLOCK_BYTES = 2 * 1024 * 1024


from benchmarks import protocol as bp
from benchmarks import measurement as bm

ACTIVE_RECEIPT = None
CPU_QUANTUM_MS = None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def manifest(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("corpus must be a directory")
    rows = []
    excluded = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if ".git" in p.relative_to(root).parts:
            if p.is_file():
                excluded.append(rel)
            continue
        if p.is_symlink():
            raise ValueError("symlinks are not supported: " + rel)
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for part in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(part)
            rows.append(
                {"path": rel, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
            )
    if not rows:
        raise ValueError("empty corpus")
    canonical = json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode()
    return {
        "root": str(root),
        "files": rows,
        "bytes": sum(r["bytes"] for r in rows),
        "sha256": digest(canonical),
        "excluded_git_files": excluded,
    }


def load_inputs(m):
    files = [
        (r["path"], (Path(m["root"]) / r["path"]).read_bytes()) for r in m["files"]
    ]
    verify(dict(files), m)
    return files


def verify(decoded, m):
    if set(decoded) != {r["path"] for r in m["files"]}:
        raise AssertionError("restored path set differs")
    for row in m["files"]:
        value = decoded[row["path"]]
        if len(value) != row["bytes"] or digest(value) != row["sha256"]:
            raise AssertionError("byte integrity failure: " + row["path"])


def tar_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as ar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            ar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def pack(config, files, threads):
    import zstandard as zstd

    if config == "vpak":
        from vortexpak.codec import VortexPakCodec

        codec = VortexPakCodec(mode="strict", max_workers=threads)
        try:
            return codec.encode_files(files).to_bytes()
        finally:
            codec._pool.shutdown()
    if config.startswith("tar-"):
        raw = tar_bytes(files)
        if config == "tar-lz4":
            import lz4.frame

            return lz4.frame.compress(raw, compression_level=0, content_checksum=True)
        level = int(config.removeprefix("tar-zstd"))
        return zstd.ZstdCompressor(
            level=level, threads=0 if threads == 1 else threads - 1, write_checksum=True
        ).compress(raw)
    if config.startswith("zip-"):
        out = io.BytesIO()
        method = zipfile.ZIP_STORED if config == "zip-store" else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(out, "w", compression=method, compresslevel=6) as ar:
            for name, data in files:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = method
                info.external_attr = 0o100644 << 16
                ar.writestr(info, data, compresslevel=6)
        return out.getvalue()
    if config not in {"indexed-zstd3", "perfile-zstd3"}:
        raise ValueError(config)
    raw = b"".join(data for _, data in files)
    entries, offset = [], 0
    for name, data in files:
        entries.append([name, offset, len(data)])
        offset += len(data)
    if config == "perfile-zstd3":
        chunks = [(off, raw[off : off + size]) for _, off, size in entries if size]
    else:
        chunks = [
            (i, raw[i : i + BLOCK_BYTES]) for i in range(0, len(raw), BLOCK_BYTES)
        ]

    def compress(chunk):
        return zstd.ZstdCompressor(level=3, write_checksum=True).compress(chunk[1])

    with ThreadPoolExecutor(max_workers=threads) as pool:
        payloads = list(pool.map(compress, chunks))
    blocks, offset = [], 0
    for (raw_offset, data), payload in zip(chunks, payloads):
        blocks.append([raw_offset, len(data), offset, len(payload)])
        offset += len(payload)
    meta = json.dumps(
        {"files": entries, "blocks": blocks}, separators=(",", ":")
    ).encode()
    meta = zstd.ZstdCompressor(level=3, write_checksum=True).compress(meta)
    return b"VPBENCH1" + struct.pack("<Q", len(meta)) + meta + b"".join(payloads)


class IndexedReader:
    def __init__(self, blob):
        import zstandard as zstd

        import mmap

        self.handle = self.mapping = None
        if isinstance(blob, Path):
            self.handle = blob.open("rb")
            self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
            blob = self.mapping
        if blob[:8] != b"VPBENCH1":
            raise ValueError("not an indexed benchmark archive")
        n = struct.unpack_from("<Q", blob, 8)[0]
        meta = json.loads(zstd.ZstdDecompressor().decompress(blob[16 : 16 + n]))
        self.entries = {name: (offset, size) for name, offset, size in meta["files"]}
        self.blocks = meta["blocks"]
        self.starts = [b[0] for b in self.blocks]
        self.payload = memoryview(blob)[16 + n :]
        self.cache = OrderedDict()
        self.decompressed_bytes = self.compressed_bytes = 0

    def clear(self):
        self.cache.clear()

    def read(self, name):
        from bisect import bisect_right
        import zstandard as zstd

        offset, size = self.entries[name]
        if not size:
            return b""
        parts = []
        i = max(0, bisect_right(self.starts, offset) - 1)
        end = offset + size
        while i < len(self.blocks) and self.blocks[i][0] < end:
            start, raw_size, pos, length = self.blocks[i]
            if i not in self.cache:
                self.cache[i] = zstd.ZstdDecompressor().decompress(
                    self.payload[pos : pos + length]
                )
                self.decompressed_bytes += raw_size
                self.compressed_bytes += length
                if len(self.cache) > 16:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(i)
            parts.append(
                self.cache[i][max(0, offset - start) : min(raw_size, end - start)]
            )
            i += 1
        return b"".join(parts)

    def close(self):
        self.cache.clear()
        self.payload.release()
        if self.mapping is not None:
            self.mapping.close()
            self.handle.close()


class VpakReader:
    def __init__(self, path):
        from vortexpak.vfs import VortexVFS

        self.vfs = VortexVFS.open(path, cache_blocks=16)

    def clear(self):
        self.vfs._cache.clear()

    def read(self, name):
        return self.vfs.read_bytes(name, materialize_full_rgba=True)

    def close(self):
        self.vfs.close()


class ZipReader:
    def __init__(self, blob):
        self.ar = zipfile.ZipFile(blob if isinstance(blob, Path) else io.BytesIO(blob))

    def clear(self):
        pass

    def read(self, name):
        return self.ar.read(name)

    def close(self):
        self.ar.close()


def restore(config, blob, threads):
    if config == "vpak":
        from vortexpak.codec import VortexArchive, VortexPakCodec

        codec = VortexPakCodec(mode="strict", max_workers=threads)
        try:
            decoded = codec.decode_archive(
                VortexArchive.from_bytes(blob), materialize_full_rgba=True
            )
            return {name: bytes(data) for name, data in decoded.items()}
        finally:
            codec._pool.shutdown()
    if config.startswith("tar-"):
        if config == "tar-lz4":
            import lz4.frame

            raw = lz4.frame.decompress(blob)
        else:
            import zstandard as zstd

            raw = zstd.ZstdDecompressor().decompress(blob)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as ar:
            return {m.name: ar.extractfile(m).read() for m in ar if m.isfile()}
    reader = ZipReader(blob) if config.startswith("zip-") else IndexedReader(blob)
    try:
        names = reader.ar.namelist() if config.startswith("zip-") else reader.entries
        return {name: reader.read(name) for name in names}
    finally:
        reader.close()


@contextmanager
def phase(out):
    with bm.measure_phase(out, cpu_quantum_ms=CPU_QUANTUM_MS):
        yield out


def distribution(values):
    import numpy as np

    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("nonempty finite observations required")
    out = {
        "n": len(a),
        "mean": float(a.mean()),
        "stddev": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }
    for label, q in (
        ("p50", 0.5),
        ("p90", 0.9),
        ("p95", 0.95),
        ("p99", 0.99),
        ("p999", 0.999),
    ):
        out[label] = float(np.quantile(a, q, method="inverted_cdf"))
    out["tail_support_p99"] = len(a) * 0.01
    return out


def latency(reader, m, requests, seed, *, expected=None, validation_path="optimized"):
    if validation_path not in ("reference", "optimized"):
        raise ValueError("unknown validation path")
    expected = dict(load_inputs(m)) if expected is None else expected
    ids = bp.query_ids(m, requests, seed)
    result = {}
    for mode in ("codec_reset", "repeat_after_one_prime"):
        raw, prime_ns, verify_ns, clear_ns = [], 0, 0, 0
        buckets = {"0-4KiB": [], "4-64KiB": [], "64KiB-1MiB": [], "1MiB+": []}
        for asset_id in ids:
            row = m["files"][asset_id]
            name = row["path"]
            original = expected[name]
            start = time.perf_counter_ns()
            reader.clear()
            prime = None
            clear_ns += time.perf_counter_ns() - start
            if mode == "repeat_after_one_prime":
                start = time.perf_counter_ns()
                prime = reader.read(name)
                prime_ns += time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            data = reader.read(name)
            ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            for value in (data,) if prime is None else (prime, data):
                if type(value) is not bytes or len(value) != row["bytes"]:
                    raise AssertionError("owned byte output contract failed: " + name)
                valid = (
                    digest(value) == row["sha256"]
                    if validation_path == "reference"
                    else value == original
                )
                if not valid:
                    raise AssertionError("random read integrity failure: " + name)
            verify_ns += time.perf_counter_ns() - start
            raw.append(ns)
            size = row["bytes"]
            bucket = (
                "0-4KiB"
                if size <= 4096
                else (
                    "4-64KiB"
                    if size <= 65536
                    else "64KiB-1MiB" if size <= 1048576 else "1MiB+"
                )
            )
            buckets[bucket].append(ns / 1e6)
        ms = [ns / 1e6 for ns in raw]
        result[mode] = {
            "asset_ids": ids,
            "raw_ns": raw,
            "verified_queries": len(ids),
            "verified_primes": len(ids) if mode == "repeat_after_one_prime" else 0,
            "validation_path": validation_path,
            "prime_ns": prime_ns,
            "verification_ns": verify_ns,
            "clear_ns": clear_ns,
            "summary_ms": distribution(ms),
            "by_size_ms": {k: distribution(v) for k, v in buckets.items() if v},
            "deadline_miss_pct": {
                str(t): 100 * sum(x > t for x in ms) / len(ms)
                for t in (1, 4, 8.33, 16.67)
            },
        }
    return result


def worker(job):
    global ACTIVE_RECEIPT, CPU_QUANTUM_MS
    started = time.perf_counter_ns()
    ACTIVE_RECEIPT = job.get("receipt")
    CPU_QUANTUM_MS = job.get("calibration", {}).get("cpu_quantum_ms")
    bp.activate(ACTIVE_RECEIPT)
    affinity = bm.bind_cpu_budget(job["threads"])
    from vortexpak.native import IS_NATIVE_AVAILABLE

    if job["config"] == "vpak" and not IS_NATIVE_AVAILABLE:
        raise RuntimeError(
            "native VortexPak unavailable; fallback is a separate profile"
        )
    bp.verify_loaded(ACTIVE_RECEIPT)
    if source_fingerprint() != job["source_fingerprint"]:
        raise RuntimeError("implementation or harness changed before trial")
    m, cfg = job["manifest"], job["config"]
    load_start = time.perf_counter_ns()
    files = load_inputs(m)
    result = {
        "config": cfg,
        "round": job["round"],
        "affinity": affinity,
        "pack": {},
        "restore": {},
        "input_load_verify_ns": time.perf_counter_ns() - load_start,
        "setup_ns": time.perf_counter_ns() - started,
        "validation_path": job.get("validation_path", "optimized"),
    }
    with phase(result["pack"]):
        blob = pack(cfg, files, job["threads"])
    result.update(
        archive_bytes=len(blob),
        archive_sha256=digest(blob),
        ratio=m["bytes"] / len(blob),
    )
    del files
    with phase(result["restore"]):
        decoded = restore(cfg, blob, job["threads"])
    validation_start = time.perf_counter_ns()
    verify(decoded, m)
    result["restore_verification_ns"] = time.perf_counter_ns() - validation_start
    for stage in ("pack", "restore"):
        result[stage]["throughput_mib_s"] = (
            m["bytes"] / 2**20 / (result[stage]["wall_ms"] / 1000)
        )
    del decoded
    result["verified_files"] = len(m["files"])
    if cfg in SEEKABLE:
        load_start = time.perf_counter_ns()
        expected = dict(load_inputs(m))
        result["query_reference_load_verify_ns"] = time.perf_counter_ns() - load_start
        result["query_reference_bytes"] = sum(len(x) for x in expected.values())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "archive.vpak"
            persist_start = time.perf_counter_ns()
            path.write_bytes(blob)
            result["artifact_write_ns"] = time.perf_counter_ns() - persist_start
            result["open_index"] = {}
            with phase(result["open_index"]):
                reader = (
                    VpakReader(path)
                    if cfg == "vpak"
                    else (
                        ZipReader(path)
                        if cfg.startswith("zip-")
                        else IndexedReader(path)
                    )
                )
            try:
                lat_start = time.perf_counter_ns()
                result["latency"] = latency(
                    reader,
                    m,
                    job["requests"],
                    job["seed"],
                    expected=expected,
                    validation_path=result["validation_path"],
                )
                result["latency_total_ns"] = time.perf_counter_ns() - lat_start
            finally:
                reader.close()
    else:
        result["latency"] = {
            "status": "not_applicable",
            "reason": "solid stream; full restore required",
        }
    if source_fingerprint() != job["source_fingerprint"]:
        raise RuntimeError("implementation or harness changed during trial")
    result["worker_wall_ns"] = time.perf_counter_ns() - started
    return result


def median_ci(values, seed=17, samples=5000):
    import numpy as np

    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("nonempty finite observations required")
    if len(a) == 1:
        return {
            "n": 1,
            "median": float(a[0]),
            "min": float(a[0]),
            "max": float(a[0]),
            "mad": 0.0,
            "relative_mad": None,
            "bootstrap_95ci": [None, None],
            "reason": "one round cannot establish timing uncertainty",
        }
    boot = np.median(
        np.random.default_rng(seed).choice(a, (samples, len(a)), replace=True), axis=1
    )
    lo, hi = np.quantile(boot, [0.025, 0.975])
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {
        "n": len(a),
        "median": med,
        "bootstrap_95ci": [float(lo), float(hi)],
        "min": float(a.min()),
        "max": float(a.max()),
        "mad": mad,
        "relative_mad": mad / abs(med) if med else None,
    }


def nullable_stat(values):
    valid = [v for v in values if v is not None]
    if len(valid) != len(values):
        return {
            "n": len(valid),
            "n_total": len(values),
            "n_unresolved": len(values) - len(valid),
            "median": None,
            "bootstrap_95ci": [None, None],
            "reason": "unresolved observations; no biased conditional estimate",
        }
    result = median_ci(valid)
    result.update(n_total=len(values), n_unresolved=0)
    return result


def summarize(trials):
    output = {}
    numeric = (
        "wall_ns",
        "wall_ms",
        "cpu_ms",
        "cpu_raw_ns",
        "cpu_raw_ms",
        "rss_before_bytes",
        "rss_after_bytes",
        "rss_delta_bytes",
        "os_process_lifetime_peak_working_set_bytes",
        "throughput_mib_s",
    )
    for cfg in dict.fromkeys(t["config"] for t in trials):
        rows = [t for t in trials if t["config"] == cfg]
        metrics = {"archive_bytes": bp.exact_metric([t["archive_bytes"] for t in rows])}
        for section in ("pack", "restore", "open_index"):
            if section in rows[0]:
                metrics[section] = {
                    k: nullable_stat([t[section].get(k) for t in rows])
                    for k in numeric
                    if k in rows[0][section]
                }
                metrics[section]["cpu_status_counts"] = {
                    status: sum(t[section]["cpu_status"] == status for t in rows)
                    for status in set(t[section]["cpu_status"] for t in rows)
                }
        if cfg in SEEKABLE:
            metrics["latency"] = {
                mode: {
                    q: median_ci([t["latency"][mode]["summary_ms"][q] for t in rows])
                    for q in ("p50", "p95", "p99", "p999", "max")
                }
                for mode in ("codec_reset", "repeat_after_one_prime")
            }
            metrics["latency_estimand"] = (
                "median of per-round nearest-rank quantiles; not population SLO"
            )
        metrics["deterministic_archive"] = len({t["archive_sha256"] for t in rows}) == 1
        output[cfg] = metrics
    return output


def source_fingerprint():
    return bp.runtime_fingerprint(ACTIVE_RECEIPT)


def environment():
    import psutil

    versions = {}
    for name in (
        "numpy",
        "pillow",
        "zstandard",
        "lz4",
        "xxhash",
        "psutil",
        "flip-evaluator",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    def git(*args):
        r = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return r.stdout.strip() if not r.returncode else None

    freq = psutil.cpu_freq()
    return {
        "host": bm.host_info(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "cpu_frequency_mhz": freq._asdict() if freq else None,
        "ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "versions": versions,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "source_fingerprint": source_fingerprint(),
        "psutil_initial_cpu_percent": psutil.cpu_percent(interval=0.2),
    }


def render_report(report):
    lines = [
        "# Byte-exact archive benchmark v2",
        "",
        "Run: " + report["run_id"],
        "Protocol: " + report["protocol_id"],
        "Evidence: " + report["evidence_head"],
        "Profile: " + report["settings"]["profile"],
        "",
        "| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cfg, s in sorted(report["summary"].items()):
        p99 = s.get("latency", {}).get("codec_reset", {}).get("p99", {}).get("median")
        peak = s["pack"]["os_process_lifetime_peak_working_set_bytes"]["median"]
        peak_text = f"{peak / 2**20:.1f}" if peak is not None else "N/A"
        tail = f"{p99:.4f}" if p99 is not None else "N/A (solid)"
        if (
            p99 is not None
            and not report["adequacy"]["p99_at_least_100_tail_observations_per_trial"]
        ):
            tail += " (exploratory)"
        lines.append(
            f"| {cfg} | {s['archive_bytes']['median'] / 2**20:.3f} | "
            f"{s['pack']['wall_ms']['median']:.3f} | {s['restore']['wall_ms']['median']:.3f} | {peak_text} | {tail} |"
        )
    lines += [
        "",
        "Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.",
        "Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.",
        "OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.",
        "Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.",
        "Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.",
    ]
    return "\n".join(lines) + "\n"


def main():
    global ACTIVE_RECEIPT, CPU_QUANTUM_MS
    started = time.perf_counter_ns()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    ap.add_argument(
        "--profile", choices=list(bp.spec()["profiles"]), default="diagnostic"
    )
    ap.add_argument("--receipt", type=Path)
    ap.add_argument(
        "--validation-path", choices=["reference", "optimized"], default="optimized"
    )
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--requests", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--history", type=Path, nargs="*", default=[])
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    if args.worker:
        job = json.loads(Path(args.worker[0]).read_text(encoding="utf-8"))
        bp.atomic_json(args.worker[1], worker(job))
        return
    if args.corpus is None:
        ap.error("--corpus is required")
    original = manifest(args.corpus)
    cfg = bp.settings(
        args.profile,
        original,
        "archive",
        args.configs,
        args.threads,
        args.repeats,
        args.warmups,
        args.requests,
        args.seed,
    )
    cfg["validation_path"] = args.validation_path
    if args.estimate:
        history = [json.loads(p.read_text(encoding="utf-8")) for p in args.history]
        print(json.dumps(bm.estimate_runtime(history, cfg), indent=2))
        return
    if args.output is None or args.timeout < 1:
        ap.error("--output and a positive timeout required")
    root, out = args.corpus.resolve(), args.output.resolve()
    if root == out or root in out.parents:
        ap.error("output must be outside corpus")
    if out.exists() and any(out.iterdir()):
        ap.error("new empty output directory required; existing evidence is immutable")
    if args.profile != "diagnostic" and args.receipt is None:
        ap.error("reference requires a controlled build --receipt")
    resumed = bp.seal_interrupted(args.resume_from) if args.resume_from else None
    out.mkdir(parents=True, exist_ok=True)
    if args.receipt:
        ACTIVE_RECEIPT = bp.capture_receipt(
            json.loads(args.receipt.read_text(encoding="utf-8")), out
        )
        bp.activate(ACTIVE_RECEIPT)
        bp.verify_loaded(ACTIVE_RECEIPT)
    env = environment()
    if original["bytes"] * 7 > env["available_ram_bytes"]:
        ap.error("insufficient memory for the complete preloaded protocol")
    m = bp.freeze_inputs(original, out)
    report = bp.initialize(m, cfg, "archive")
    report.update(
        environment=env,
        receipt=ACTIVE_RECEIPT,
        resumed_from=resumed,
        dependencies=bp.dependencies() if args.profile != "diagnostic" else {},
        calibration=bm.calibrate(),
        scope="descriptive single-host experiment, not population SLO",
    )
    CPU_QUANTUM_MS = report["calibration"]["cpu_quantum_ms"]
    bp.capture_harness(out, report)
    store = bp.EventStore(out, report)
    child_env = dict(
        os.environ,
        RAYON_NUM_THREADS=str(args.threads),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED=str(args.seed),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
    )
    try:
        with tempfile.TemporaryDirectory() as td:
            for round_ in report["schedule"]:
                rnd = round_["round"]
                for config in round_["order"]:
                    store.append("trial_started", {"round": rnd, "config": config})
                    job = {
                        "manifest": m,
                        "config": config,
                        "round": rnd,
                        "threads": args.threads,
                        "requests": args.requests,
                        "seed": args.seed + max(rnd, 0),
                        "receipt": ACTIVE_RECEIPT,
                        "calibration": report["calibration"],
                        "validation_path": args.validation_path,
                        "source_fingerprint": env["source_fingerprint"],
                    }
                    job_path = Path(td) / (bp.trial_id(rnd, config) + "-job.json")
                    result_path = Path(td) / (bp.trial_id(rnd, config) + "-result.json")
                    bp.atomic_json(job_path, job)
                    print(f"round {rnd}: {config}", flush=True)
                    begin = time.perf_counter_ns()
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            str(job_path),
                            str(result_path),
                        ],
                        env=child_env,
                        capture_output=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=args.timeout,
                    )
                    if process.returncode:
                        raise RuntimeError(
                            process.stderr[-6000:]
                            or f"worker exit {process.returncode}"
                        )
                    trial = json.loads(result_path.read_text(encoding="utf-8"))
                    trial["subprocess_total_ns"] = time.perf_counter_ns() - begin
                    store.trial(trial)
                    report["trials" if rnd >= 0 else "warmup_trials"].append(trial)
        if source_fingerprint() != env["source_fingerprint"]:
            raise RuntimeError("implementation/harness changed")
        report["host_after"] = bm.host_info()
        before, after = env["host"], report["host_after"]
        if before["power_scheme"] != after["power_scheme"]:
            raise RuntimeError("power scheme changed during experiment")
        stats_begin = time.perf_counter_ns()
        report["summary"] = summarize(report["trials"])
        report["statistics_ns"] = time.perf_counter_ns() - stats_begin
        report["paired_vs_tar_zstd3"] = {}
        base = {t["round"]: t for t in report["trials"] if t["config"] == "tar-zstd3"}
        if base:
            for config in args.configs:
                report["paired_vs_tar_zstd3"][config] = {
                    stage
                    + "_wall_pct": median_ci(
                        [
                            100
                            * (
                                t[stage]["wall_ms"] / base[t["round"]][stage]["wall_ms"]
                                - 1
                            )
                            for t in report["trials"]
                            if t["config"] == config
                        ]
                    )
                    for stage in ("pack", "restore")
                }
        report["adequacy"] = {
            "at_least_7_rounds": args.repeats >= 7,
            "warmup_per_config": args.warmups >= 1,
            "p99_at_least_100_tail_observations_per_trial": args.requests >= 10000,
            "p999_at_least_100_tail_observations_per_trial": args.requests >= 100000,
            "tail_flags_are_coverage_not_precision_guarantees": True,
            "independent_hardware_replication": False,
            "os_cold_io_measured": False,
            "qualified_reference": args.profile != "diagnostic",
        }
        report["calibration_after"] = bm.calibrate()
        report["elapsed_wall_ns"] = time.perf_counter_ns() - started
        report["elapsed_scope"] = (
            "entry through statistics, excludes final evidence serialization; suite measures child exit"
        )
        bp.finalize(out, report, store)
        with (out / "report.md").open("x", encoding="utf-8") as f:
            f.write(render_report(report))
        print(out / "report.md", flush=True)
    except BaseException as exc:
        report["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        report["errors"].append({"type": type(exc).__name__, "error": str(exc)})
        store.append("run_" + report["status"], report["errors"][-1])
        bp.atomic_json(out / "failure.json", report, replace=False)
        raise


if __name__ == "__main__":
    main()
