"""Benchmark-v2 instruments; stdlib and psutil only.

Call calibrate outside measured workloads. No sampler threads, sleeps, codec
imports or protocol imports. CPU/RSS observations cover the whole process.
Only explicit bind_cpu_budget calls change affinity, never system settings.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

import psutil

__all__ = [
    "calibrate", "measure_phase", "bind_cpu_budget", "host_info",
    "estimate_runtime",
]

_EMPTY_PAIRS = 2048
_CPU_BUSY_BUDGET_NS = 250_000_000
_CPU_MAX_PROBES = 16_384
_CPU_CALIBRATION = None
_AFFINITY_BASELINE = None


def _clock_details(name):
    info = time.get_clock_info(name)
    return {
        "implementation": info.implementation,
        "monotonic": info.monotonic,
        "adjustable": info.adjustable,
        "resolution_s": info.resolution,
        "resolution_ns": info.resolution * 1e9,
    }


def _ns_distribution(samples):
    ordered = sorted(samples)
    result = {"n": len(ordered), "zero_count": ordered.count(0)}
    for name, fraction in (
        ("min_ns", 0), ("p50_ns", .50), ("p95_ns", .95),
        ("p99_ns", .99), ("max_ns", 1),
    ):
        index = max(0, math.ceil(len(ordered) * fraction) - 1)
        result[name] = ordered[index] if ordered else None
    result["percentile_method"] = "nearest rank"
    return result


def calibrate() -> dict:
    """Observe clock overhead and Windows CPU steps outside timed workloads.

    The CPU probe has a 250 ms wall budget and a fixed iteration cap. Scheduling
    can delay its return beyond that budget. A quantum is the minimum positive
    process_time_ns step seen in at least three transitions, NOT an OS guarantee
    or a confidence bound. Nothing is subtracted from subsequent measurements.
    The observation is cached for this PID only; no implicit recalibration.
    """
    global _CPU_CALIBRATION
    gc.collect()
    pairs = []
    for _ in range(_EMPTY_PAIRS):
        start = time.perf_counter_ns()
        stop = time.perf_counter_ns()
        pairs.append(stop - start)
    cpu = {
        "supported": False,
        "source": "time.process_time_ns (all process threads)",
        "reason": "Windows CPU-step calibration is unsupported on this OS",
        "cpu_quantum_ms": None,
        "observed_quantum_ns": None,
        "wall_budget_ns": _CPU_BUSY_BUDGET_NS,
        "max_probes": _CPU_MAX_PROBES,
        "elapsed_wall_ns": None,
        "positive_transitions": 0,
        "step_distribution_ns": _ns_distribution([]),
        "observation_only": True,
    }
    if sys.platform == "win32":
        steps = []
        token = 0
        started = time.perf_counter_ns()
        previous = time.process_time_ns()
        deadline = started + _CPU_BUSY_BUDGET_NS
        while len(steps) < _CPU_MAX_PROBES:
            if time.perf_counter_ns() >= deadline:
                break
            for _ in range(128):
                token = (token * 1664525 + 1013904223) & 0xffffffff
            current = time.process_time_ns()
            steps.append(current - previous)
            previous = current
        positive = [step for step in steps if step > 0]
        established = len(positive) >= 3 and all(s >= 0 for s in steps)
        quantum = min(positive) if established else None
        cpu.update(
            supported=established,
            reason=None if established else "Insufficient nonregressing steps",
            cpu_quantum_ms=quantum / 1e6 if quantum is not None else None,
            observed_quantum_ns=quantum,
            elapsed_wall_ns=time.perf_counter_ns() - started,
            positive_transitions=len(positive),
            step_distribution_ns=_ns_distribution(steps),
        )
    _CPU_CALIBRATION = (os.getpid(), cpu["cpu_quantum_ms"])
    return {
        "perf_counter": _clock_details("perf_counter"),
        "process_time": _clock_details("process_time"),
        "empty_perf_counter_pairs_ns": {
            "samples_ns": pairs,
            "distribution": _ns_distribution(pairs),
            "method": "adjacent perf_counter_ns calls; no subtraction",
        },
        "windows_process_cpu": cpu,
        "cpu_quantum_ms": cpu["cpu_quantum_ms"],
    }


def _memory_snapshot(process):
    try:
        info = process.memory_info()
    except (psutil.Error, OSError, NotImplementedError) as exc:
        return None, None, str(exc)
    peak = getattr(info, "peak_wset", None) if sys.platform == "win32" else None
    return int(info.rss), int(peak) if peak is not None else None, None


def _cpu_quantum(value):
    if value is None:
        if _CPU_CALIBRATION and _CPU_CALIBRATION[0] == os.getpid():
            return _CPU_CALIBRATION[1]
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("cpu_quantum_ms must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("cpu_quantum_ms must be a finite positive number")
    return float(value)


@contextmanager
def measure_phase(out, cpu_quantum_ms=None):
    """Fill a mutable mapping, including on workload exceptions.

    GC and RSS snapshots are outside the wall/CPU interval. cpu_raw_ms is always
    retained; cpu_ms is None for zero, sub-quantum or uncalibrated observations.
    An omitted quantum uses calibrate's cached value, never implicit busy work.
    Peak working set is OS process-lifetime evidence, NEVER a phase RSS peak.
    """
    quantum = _cpu_quantum(cpu_quantum_ms)
    process = psutil.Process()
    gc.collect()
    rss_before, _, before_error = _memory_snapshot(process)
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    try:
        yield
    finally:
        wall_stop = time.perf_counter_ns()
        cpu_stop = time.process_time_ns()
        rss_after, lifetime_peak, after_error = _memory_snapshot(process)
        wall_ns = int(wall_stop - wall_start)
        cpu_ns = int(cpu_stop - cpu_start)
        if cpu_ns < 0:
            status = "clock_regressed"
        elif cpu_ns == 0:
            status = "zero_unresolved"
        elif quantum is None:
            status = "uncalibrated"
        elif cpu_ns < math.ceil(quantum * 1e6):
            status = "subresolution"
        else:
            status = "qualified"
        rss_supported = rss_before is not None and rss_after is not None
        out.update({
            "wall_ns": wall_ns,
            "wall_ms": wall_ns / 1e6,
            "cpu_raw_ns": cpu_ns,
            "cpu_raw_ms": cpu_ns / 1e6,
            "cpu_ms": cpu_ns / 1e6 if status == "qualified" else None,
            "cpu_quantum_ms": quantum,
            "cpu_status": status,
            "cpu_scope": "all process threads; not wall time or utilization",
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": rss_after - rss_before if rss_supported else None,
            "rss_supported": rss_supported,
            "rss_before_error": before_error,
            "rss_after_error": after_error,
            "os_process_lifetime_peak_working_set_bytes": lifetime_peak,
            "os_process_lifetime_peak_working_set_supported": (
                lifetime_peak is not None
            ),
            "os_process_lifetime_peak_working_set_source": (
                "psutil.Process.memory_info().peak_wset"
                if lifetime_peak is not None else None
            ),
            "os_process_lifetime_peak_working_set_unsupported_reason": (
                "Not exposed by this OS/psutil, or snapshot failed"
                if lifetime_peak is None else None
            ),
        })


def _parse_windows_cores(data):
    """Parse native 64-bit RelationProcessorCore records, including group IDs."""
    cores = []
    seen = set()
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("Truncated Windows topology header")
        relationship, size = struct.unpack_from("<II", data, offset)
        if relationship != 0 or size < 48 or offset + size > len(data):
            raise ValueError("Invalid RelationProcessorCore record")
        flags, efficiency = struct.unpack_from("<BB", data, offset + 8)
        group_count = struct.unpack_from("<H", data, offset + 30)[0]
        if group_count < 1 or 32 + group_count * 16 > size:
            raise ValueError("Invalid Windows GROUP_AFFINITY count")
        logical = []
        for index in range(group_count):
            mask, group = struct.unpack_from(
                "<QH", data, offset + 32 + index * 16,
            )
            if not mask:
                raise ValueError("Empty Windows processor-core mask")
            for number in range(64):
                if mask & (1 << number):
                    identity = (group, number)
                    if identity in seen:
                        raise ValueError("Duplicate Windows logical processor")
                    seen.add(identity)
                    logical.append({"group": group, "number": number})
        cores.append({
            "core_index": len(cores),
            "smt": bool(flags & 1),
            "efficiency_class": efficiency,
            "logical_processors": logical,
        })
        offset += size
    if not cores:
        raise ValueError("No physical processor cores reported")
    return cores


def _windows_topology():
    result = {
        "supported": False,
        "source": "GetLogicalProcessorInformationEx(RelationProcessorCore)",
        "unsupported_reason": None,
        "affinity_supported": False,
        "affinity_unsupported_reason": None,
        "cores": [],
        "processor_groups": [],
        "physical_cores": None,
        "logical_processors": None,
    }
    if sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
        reason = "Physical-core affinity requires native 64-bit Windows"
        result.update(unsupported_reason=reason,
                      affinity_unsupported_reason=reason)
        return result
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel.GetLogicalProcessorInformationEx
        query.argtypes = [ctypes.c_int, ctypes.c_void_p,
                          ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_int
        size = ctypes.c_uint32()
        query(0, None, ctypes.byref(size))
        if ctypes.get_last_error() != 122 or not 0 < size.value <= 16_777_216:
            raise OSError("Windows topology sizing failed")
        buffer = ctypes.create_string_buffer(size.value)
        if not query(0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        cores = _parse_windows_cores(buffer.raw[:size.value])
        groups = sorted({
            cpu["group"] for core in cores for cpu in core["logical_processors"]
        })
        affinity_ok = groups == [0]
        result.update(
            supported=True, cores=cores, processor_groups=groups,
            physical_cores=len(cores),
            logical_processors=sum(len(c["logical_processors"]) for c in cores),
            affinity_supported=affinity_ok,
            affinity_unsupported_reason=(
                None if affinity_ok else
                "Multiple/nonzero processor groups are not safely bindable"
            ),
        )
    except (OSError, AttributeError, ValueError, struct.error) as exc:
        result.update(unsupported_reason=str(exc),
                      affinity_unsupported_reason=str(exc))
    return result


def _read_affinity(process):
    try:
        allowed = sorted(set(process.cpu_affinity()))
    except (AttributeError, NotImplementedError, psutil.Error, OSError) as exc:
        raise NotImplementedError("Process affinity unavailable: " + str(exc))
    if not allowed:
        raise NotImplementedError("OS returned an empty process affinity")
    return allowed


def bind_cpu_budget(threads) -> list:
    """Return verified process affinity; never modify system-wide settings.

    The first call captures the initial allowed CPUs for this PID. Single-CPU
    calls always reuse its first CPU, including after a multi-core call. Multi-
    CPU calls require Windows topology and choose one allowed logical processor
    per physical core. Unsupported topology/groups fail, with no SMT fallback.
    The caller owns affinity restoration; later calls can reuse the baseline.
    """
    global _AFFINITY_BASELINE
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    process = psutil.Process()
    previous = _read_affinity(process)
    pid = os.getpid()
    if _AFFINITY_BASELINE is None or _AFFINITY_BASELINE[0] != pid:
        _AFFINITY_BASELINE = (pid, previous[:])
    allowed = _AFFINITY_BASELINE[1]
    if threads == 1:
        selected = allowed[:1]
    else:
        topology = _windows_topology()
        if not topology["supported"] or not topology["affinity_supported"]:
            raise NotImplementedError(topology["affinity_unsupported_reason"])
        known = {
            cpu["number"] for core in topology["cores"]
            for cpu in core["logical_processors"]
        }
        if not set(allowed) <= known:
            raise NotImplementedError("Affinity CPU IDs do not match topology")
        representatives = []
        for core in topology["cores"]:
            available = sorted(
                cpu["number"] for cpu in core["logical_processors"]
                if cpu["number"] in allowed
            )
            if available:
                representatives.append(available[0])
        selected = sorted(representatives)[:threads]
        if len(selected) != threads:
            raise ValueError("Requested budget exceeds allowed physical cores")
    try:
        process.cpu_affinity(selected)
        actual = _read_affinity(process)
        if actual != selected:
            raise RuntimeError("Requested process affinity was not enforced")
    except (psutil.Error, OSError, NotImplementedError, RuntimeError):
        try:
            process.cpu_affinity(previous)
        except (psutil.Error, OSError, NotImplementedError) as exc:
            raise RuntimeError(
                "Affinity failed and restoration also failed"
            ) from exc
        raise
    return actual


def _cpu_model():
    try:
        if sys.platform == "win32":
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            source = "Windows registry ProcessorNameString (read-only)"
        elif sys.platform.startswith("linux"):
            model = None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                name, _, value = line.partition(":")
                if name.strip() in ("model name", "Hardware") and value.strip():
                    model = value.strip()
                    break
            source = "/proc/cpuinfo"
        else:
            model, source = None, None
        return {"value": model or None, "supported": bool(model),
                "source": source, "unsupported_reason": (
                    None if model else "No real CPU model source implemented"
                )}
    except (OSError, ValueError, ImportError) as exc:
        return {"value": None, "supported": False, "source": None,
                "unsupported_reason": str(exc)}


def _power_scheme_snapshot():
    result = {
        "supported": False, "read_only": True, "atomic_snapshot": False,
        "snapshot": [], "sha256": None, "unsupported_reason": None,
        "hash_scope": "length-prefixed command, stdout and stderr raw bytes",
    }
    if sys.platform != "win32":
        result["unsupported_reason"] = "Windows powercfg is unavailable"
        return result
    digest = hashlib.sha256()
    try:
        for arguments in (("/getactivescheme",), ("/query", "SCHEME_CURRENT")):
            command = ["powercfg", *arguments]
            completed = subprocess.run(
                command, capture_output=True, check=False, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for raw in (" ".join(command).encode("ascii"), completed.stdout,
                        completed.stderr):
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
            result["snapshot"].append({
                "command": command, "returncode": completed.returncode,
                "stdout": completed.stdout.decode("mbcs", errors="replace"),
                "stderr": completed.stderr.decode("mbcs", errors="replace"),
                "text_encoding": "mbcs; replacement on decoding errors",
            })
            if completed.returncode:
                raise OSError("Read-only powercfg query failed: "
                              + str(completed.returncode))
        result.update(supported=True, sha256=digest.hexdigest())
    except (OSError, subprocess.SubprocessError) as exc:
        result["unsupported_reason"] = str(exc)
    return result


def host_info() -> dict:
    """Capture real host evidence, with explicit missing-capability flags.

    Powercfg only reads the active scheme and its settings. Its two queries are
    not atomic and do not promise that the scheme stayed unchanged. Topology
    describes the whole host; affinity describes this process at capture time.
    """
    model = _cpu_model()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    try:
        memory = psutil.virtual_memory()
        ram, available_ram, ram_error = memory.total, memory.available, None
    except (psutil.Error, OSError, NotImplementedError) as exc:
        ram, available_ram, ram_error = None, None, str(exc)
    try:
        affinity = _read_affinity(psutil.Process())
        affinity_error = None
    except NotImplementedError as exc:
        affinity, affinity_error = None, str(exc)
    return {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_model": model["value"],
        "cpu_model_supported": model["supported"],
        "cpu_model_source": model["source"],
        "cpu_model_unsupported_reason": model["unsupported_reason"],
        "physical_cpus": physical,
        "physical_cpus_supported": physical is not None,
        "logical_cpus": logical,
        "logical_cpus_supported": logical is not None,
        "cpu_count_source": "psutil.cpu_count (whole host)",
        "os": {
            "system": platform.system(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": sys.version,
        "psutil_version": psutil.__version__,
        "ram_bytes": ram, "available_ram_bytes": available_ram,
        "ram_supported": ram is not None, "ram_unsupported_reason": ram_error,
        "affinity": affinity, "affinity_supported": affinity is not None,
        "affinity_unsupported_reason": affinity_error,
        "topology": _windows_topology(),
        "power_scheme": _power_scheme_snapshot(),
    }


def _settings_key(settings):
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a JSON-compatible mapping")
    return json.dumps(dict(settings), sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":"))


def estimate_runtime(history, settings) -> dict:
    """Estimate only from completed whole runs with exactly matching settings.

    history is None, one mapping, or an iterable of mappings containing
    {"settings": <mapping>, "elapsed_wall_ns": <positive integer>} for completed
    WHOLE runs. The caller must include corpus/software/host/affinity/cache and
    job-plan identity in settings when those distinctions matter. No scaling,
    future completion timestamp or predictive interval is invented. Historical
    range is descriptive, not a bound on the next run. This performs no timing.
    """
    key = _settings_key(settings)
    rows = [history] if isinstance(history, Mapping) else history
    observations = []
    ignored = 0
    for row in (() if rows is None else rows):
        if not isinstance(row, Mapping):
            ignored += 1
            continue
        duration = row.get("elapsed_wall_ns")
        try:
            matches = _settings_key(row.get("settings")) == key
        except (TypeError, ValueError):
            matches = False
        if (not matches or isinstance(duration, bool)
                or not isinstance(duration, int) or duration <= 0):
            ignored += 1
            continue
        observations.append(duration)
    observations.sort()
    count = len(observations)
    estimate = None
    if count:
        estimate = (observations[(count - 1) // 2]
                    + observations[count // 2] + 1) // 2
    return {
        "status": "calibrated" if count else "uncalibrated",
        "kind": "historical_runtime_estimate",
        "is_measured_eta": False,
        "settings_sha256": hashlib.sha256(key.encode("ascii")).hexdigest(),
        "matching_history_runs": count,
        "ignored_history_rows": ignored,
        "estimated_wall_ns": estimate,
        "estimated_seconds": estimate / 1e9 if estimate is not None else None,
        "historical_range_ns": (
            [observations[0], observations[-1]] if count else None
        ),
        "confidence_interval": None,
        "basis": "Median whole-run wall time; exact supplied-settings match",
        "reason": None if count else "No compatible completed whole-run history",
        "uncertainty": (
            "Not measured ETA; no predictive guarantee. Historical min/max "
            "are not bounds on the next run. Host equivalence is caller-supplied."
        ),
    }


def _self_check():
    """Standalone instrument checks only: no codec, corpus or benchmark run."""
    from types import SimpleNamespace
    from unittest.mock import patch

    calibration = calibrate()
    assert len(calibration["empty_perf_counter_pairs_ns"]["samples_ns"]) == 2048
    phase = {}
    with measure_phase(phase):
        pass
    assert isinstance(phase["wall_ns"], int) and phase["wall_ns"] >= 0
    assert phase["wall_ms"] == phase["wall_ns"] / 1e6
    for delta, expected in ((0, None), (1_000_000, None),
                            (15_625_000, 15.625)):
        events = []
        fake = SimpleNamespace(memory_info=lambda: (
            events.append("rss") or SimpleNamespace(rss=123, peak_wset=456)
        ))
        with patch.object(psutil, "Process", return_value=fake), \
                patch.object(gc, "collect",
                             side_effect=lambda: events.append("gc")), \
                patch.object(time, "perf_counter_ns", side_effect=[10, 110]), \
                patch.object(time, "process_time_ns", side_effect=[0, delta]):
            result = {}
            with measure_phase(result, cpu_quantum_ms=15.625):
                assert events == ["gc", "rss"]
            assert result["cpu_ms"] == expected
            assert result["cpu_raw_ms"] == delta / 1e6
            assert result["wall_ns"] == 100
    failed = {}
    try:
        with measure_phase(failed):
            raise LookupError("expected self-check exception")
    except LookupError:
        assert "wall_ns" in failed
    else:
        raise AssertionError("Workload exception was suppressed")
    record = {"settings": {"plan": 1}, "elapsed_wall_ns": 1_000_000_000}
    assert estimate_runtime([], record["settings"])["status"] == "uncalibrated"
    estimate = estimate_runtime(
        [record, dict(record, elapsed_wall_ns=3_000_000_000)], record["settings"],
    )
    assert estimate["estimated_seconds"] == 2 and not estimate["is_measured_eta"]
    assert estimate_runtime([record], {"plan": 2})["estimated_wall_ns"] is None
    data = b""
    for mask in (3, 12):
        data += (struct.pack("<IIBB20xH", 0, 48, 1, 0, 1)
                 + struct.pack("<QH6x", mask, 0))
    assert [c["logical_processors"][0]["number"]
            for c in _parse_windows_cores(data)] == [0, 2]
    info = host_info()
    binding = "unsupported"
    if info["affinity_supported"]:
        process = psutil.Process()
        original = _read_affinity(process)
        try:
            first = bind_cpu_budget(1)
            assert first == original[:1]
            topology = info["topology"]
            if topology["affinity_supported"]:
                cores = [
                    {c["number"] for c in core["logical_processors"]}
                    for core in topology["cores"]
                ]
                count = sum(bool(set(original) & core) for core in cores)
                actual = bind_cpu_budget(min(2, count))
                assert actual[0] == first[0]
                assert all(len(set(actual) & core) <= 1 for core in cores)
                assert bind_cpu_budget(1) == first
                binding = "single and physical-core budgets verified"
            else:
                try:
                    bind_cpu_budget(2)
                except NotImplementedError:
                    binding = "single verified; multi explicitly unsupported"
                else:
                    raise AssertionError("Unsupported topology was accepted")
        finally:
            process.cpu_affinity(original)
            assert _read_affinity(process) == original
    json.dumps([calibration, phase, info, estimate], allow_nan=False)
    print(json.dumps({
        "self_check": "passed", "cpu_model": info["cpu_model"],
        "physical_cpus": info["physical_cpus"],
        "logical_cpus": info["logical_cpus"],
        "cpu_quantum_ms": calibration["cpu_quantum_ms"],
        "topology_supported": info["topology"]["supported"],
        "power_scheme_supported": info["power_scheme"]["supported"],
        "affinity_check": binding,
    }, allow_nan=False))


if __name__ == "__main__":
    _self_check()
