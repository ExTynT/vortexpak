import copy
from pathlib import Path
import unittest

from benchmarks import protocol


class TestMeasurement(unittest.TestCase):
    def test_no_sampling_thread_and_unresolved_cpu(self):
        from benchmarks import measurement
        from unittest.mock import patch

        result = {}
        with patch.object(measurement.time, "process_time_ns", side_effect=[0, 0]):
            with measurement.measure_phase(result, cpu_quantum_ms=15.625):
                sum(range(10))
        self.assertIsNone(result["cpu_ms"])
        self.assertEqual("zero_unresolved", result["cpu_status"])
        self.assertIsInstance(result["wall_ns"], int)
        self.assertNotIn("rss_peak_sampled_bytes", result)

    def test_estimates_never_claim_to_be_measured(self):
        from benchmarks import measurement

        unknown = measurement.estimate_runtime([], {"case": "unknown"})
        self.assertFalse(unknown["is_measured_eta"])
        self.assertIsNone(unknown["estimated_wall_ns"])


class TestWindowsLedgerPublication(unittest.TestCase):
    @unittest.skipUnless(
        __import__("sys").platform == "win32", "Windows sharing semantics"
    )
    def test_live_reader_cannot_block_authoritative_event_commit(self):
        import ctypes
        from ctypes import wintypes
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = protocol.EventStore(
                td, {"run_id": "a", "session_id": "b", "protocol_id": "c"}
            )
            state = Path(td) / "state.json"
            state.write_bytes(b'{"non_authoritative":"viewer"}')
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            kernel.CreateFileW.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            handle = kernel.CreateFileW(str(state), 0x80000000, 3, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                store.append("checkpoint", {"value": 1})
            finally:
                kernel.CloseHandle(handle)
            rows, head = protocol.ledger(td)
            self.assertEqual(2, len(rows))
            self.assertEqual("checkpoint", rows[-1]["event"])
            self.assertEqual(store.head, head)
            self.assertEqual(b'{"non_authoritative":"viewer"}', state.read_bytes())


class TestLedger(unittest.TestCase):
    def test_chain_detects_truncation_and_rewriting(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = protocol.EventStore(
                td, {"run_id": "a", "session_id": "b", "protocol_id": "c"}
            )
            store.append("test", {"x": 1})
            rows, head = protocol.ledger(td)
            self.assertEqual(2, len(rows))
            self.assertEqual(store.head, head)
            p = Path(td) / "events.jsonl"
            raw = p.read_bytes()
            p.write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                protocol.ledger(td)
            p.write_bytes(raw.replace(b'"x":1', b'"x":2'))
            with self.assertRaisesRegex(ValueError, "chain"):
                protocol.ledger(td)

    def test_incomplete_never_loads_as_reference(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(protocol.IncompleteReportError):
                protocol.load_report(td)
            protocol.atomic_json(Path(td) / "report.json", {"status": "running"})
            with self.assertRaises(protocol.IncompleteReportError):
                protocol.load_report(td)
            with self.assertRaises(FileExistsError):
                protocol.atomic_json(Path(td) / "report.json", {})


class TestStatisticsPolicy(unittest.TestCase):
    def test_censored_cpu_never_becomes_conditional_median(self):
        from benchmarks.rigorous import nullable_stat

        value = nullable_stat([None, 15.625, 31.25])
        self.assertIsNone(value["median"])
        self.assertEqual(1, value["n_unresolved"])

    def test_diagnostic_comparison_never_passes_as_reference(self):
        m = {"sha256": "x", "files": [], "bytes": 0}
        cfg = protocol.settings(
            "diagnostic", m, "archive", ["tar-zstd3"], 1, 1, 0, 1, 17
        )
        cfg["validation_path"] = "optimized"
        r = protocol.initialize(m, cfg, "archive")
        r.update(
            status="complete",
            summary={"tar-zstd3": {"archive_bytes": {"median": 3}}},
            trials=[
                {
                    "round": 0,
                    "config": "tar-zstd3",
                    "verified_files": 0,
                    "archive_bytes": 3,
                }
            ],
        )
        self.assertEqual(
            "diagnostic_only", protocol.compare_runs(r, copy.deepcopy(r))["status"]
        )
        other = copy.deepcopy(r)
        other["settings"]["validation_path"] = "reference"
        with self.assertRaisesRegex(ValueError, "harness"):
            protocol.compare_runs(r, other)


class TestReceipt(unittest.TestCase):
    def test_snapshot_corruption_and_wrong_library(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "vortexpak").mkdir()
            (root / "vortexpak/a.py").write_bytes(b"a=1")
            (root / "vortexpak/native.dll").write_bytes(b"not executable")
            r = {
                "snapshot_root": str(root),
                "source_hashes": protocol.source_files(root),
                "native_relpath": "vortexpak/native.dll",
                "native_sha256": protocol.sha(b"not executable"),
                "snapshot_files": {
                    "vortexpak/a.py": protocol.sha(b"a=1"),
                    "vortexpak/native.dll": protocol.sha(b"not executable"),
                },
            }
            r["receipt_id"] = protocol.sha(protocol.canonical(r))
            self.assertEqual(root, protocol.verify_receipt(r))
            bad = copy.deepcopy(r)
            bad["native_sha256"] = "bad"
            with self.assertRaises(ValueError):
                protocol.verify_receipt(bad)
            (root / "vortexpak/a.py").write_bytes(b"a=2")
            with self.assertRaises(ValueError):
                protocol.verify_receipt(r)


class TestProtocolContract(unittest.TestCase):
    def test_registered_full_profile_is_exact(self):
        m = {**protocol.spec()["corpora"]["minetest"], "files": [None] * 1434}
        c = protocol.settings(
            "reference",
            m,
            "archive",
            protocol.spec()["archive_configs"],
            1,
            7,
            1,
            10000,
            20261025,
        )
        self.assertEqual("minetest", c["case"])
        for changed in ({"repeats": 6}, {"requests": 9999}, {"threads": 2}):
            values = dict(
                profile="reference",
                manifest=m,
                track="archive",
                configs=c["configs"],
                threads=1,
                repeats=7,
                warmups=1,
                requests=10000,
                seed=20261025,
            )
            values.update(changed)
            with self.assertRaises(ValueError):
                protocol.settings(**values)

    def test_plan_contains_every_configuration_and_round(self):
        p = protocol.make_plan(["a", "b", "c"], 7, 1, 17)
        self.assertEqual(list(range(-1, 7)), [r["round"] for r in p])
        self.assertTrue(all(set(r["order"]) == {"a", "b", "c"} for r in p))
        self.assertEqual(p, protocol.make_plan(["a", "b", "c"], 7, 1, 17))

    def test_raw_integrity_and_missing_trials(self):
        m = {"sha256": "x", "files": [], "bytes": 0}
        cfg = protocol.settings(
            "diagnostic", m, "archive", ["tar-zstd3"], 1, 1, 0, 1, 17
        )
        report = protocol.initialize(m, cfg, "archive")
        report.update(
            status="complete",
            summary={"tar-zstd3": {"archive_bytes": {"median": 3}}},
            trials=[
                {
                    "round": 0,
                    "config": "tar-zstd3",
                    "verified_files": 0,
                    "archive_bytes": 3,
                }
            ],
        )
        protocol.validate_run(report)
        for key in ("trials", "protocol_id"):
            bad = copy.deepcopy(report)
            bad[key] = [] if key == "trials" else "wrong"
            with self.assertRaises(ValueError):
                protocol.validate_run(bad)
        bad = copy.deepcopy(report)
        bad["trials"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            protocol.validate_run(bad)

    def test_nonfinite_canonical_and_unknown_profiles_fail(self):
        with self.assertRaises(ValueError):
            protocol.canonical({"a": float("nan")})
        with self.assertRaises(ValueError):
            protocol.settings("unknown", {}, "archive", [], 1, 1, 0, 1, 1)
        self.assertTrue((Path(protocol.ROOT) / "benchmarks/protocol.json").exists())


if __name__ == "__main__":
    unittest.main()
