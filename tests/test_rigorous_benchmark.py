"""The benchmark is tested as strictly as the codecs it compares."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rigorous", ROOT / "benchmarks/rigorous.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class TestSilesiaFetch(unittest.TestCase):
    def test_official_identity_and_unsafe_member_rejection(self):
        import hashlib
        import io
        import zipfile

        spec = importlib.util.spec_from_file_location(
            "fetch_silesia", ROOT / "benchmarks/fetch_silesia.py"
        )
        fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch)
        expected = {"sample": (3, hashlib.md5(b"abc").hexdigest())}
        for member, data, valid in (
            ("sample", b"abc", True),
            ("sample", b"bad", False),
            ("../escape", b"abc", False),
        ):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as z:
                z.writestr(member, data)
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / "corpus"
                with (
                    patch.object(fetch, "OFFICIAL", expected),
                    patch.object(
                        fetch.urllib.request,
                        "urlopen",
                        return_value=io.BytesIO(buffer.getvalue()),
                    ),
                    patch.object(
                        sys, "argv", ["fetch_silesia.py", "--output", str(output)]
                    ),
                ):
                    if valid:
                        fetch.main()
                        self.assertEqual(b"abc", (output / "sample").read_bytes())
                    else:
                        with self.assertRaises(ValueError):
                            fetch.main()
                        self.assertFalse(output.exists())
                self.assertFalse((Path(td) / "escape").exists())


class TestBenchmarkReport(unittest.TestCase):
    def test_html_escaping_and_complete_report_guard(self):
        spec = importlib.util.spec_from_file_location(
            "report_benchmarks", ROOT / "benchmarks/report_benchmarks.py"
        )
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        fixture = {
            "schema": 2,
            "track": "archive",
            "summary": {},
            "manifest": {"files": [], "bytes": 0, "sha256": "x"},
            "settings": {"profile": "diagnostic"},
            "environment": {"python": "<script>"},
        }
        html = report.html_document([(Path("bad<script>"), fixture)])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("diagnostic", html.lower())
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "incomplete"
            folder.mkdir()
            (folder / "report.json").write_text('{"status":"running"}')
            with patch.object(
                sys,
                "argv",
                ["report.py", str(folder), "--output", str(Path(td) / "index.html")],
            ):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    report.main()


class TestValidationEquivalence(unittest.TestCase):
    def test_every_read_and_prime_checked_without_changing_sequence(self):
        from unittest.mock import patch

        expected = {"asset": b"abcdefgh"}
        manifest = {
            "files": [
                {"path": "asset", "bytes": 8, "sha256": bench.digest(expected["asset"])}
            ]
        }

        class Reader:
            def __init__(self, bad=False):
                self.events = []
                self.bad = bad

            def clear(self):
                self.events.append("clear")

            def read(self, name):
                self.events.append("read")
                return b"abcdxfgh" if self.bad else expected[name]

        snapshots = []
        for validation in ("reference", "optimized"):
            reader = Reader()
            result = bench.latency(
                reader, manifest, 3, 7, expected=expected, validation_path=validation
            )
            snapshots.append(reader.events)
            self.assertEqual(3, result["codec_reset"]["verified_queries"])
            self.assertEqual(3, result["repeat_after_one_prime"]["verified_primes"])
            with self.assertRaises(AssertionError):
                bench.latency(
                    Reader(True),
                    manifest,
                    3,
                    7,
                    expected=expected,
                    validation_path=validation,
                )
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(
            ["clear", "read"] * 3 + ["clear", "read", "read"] * 3, snapshots[0]
        )
        with patch.object(
            bench, "digest", side_effect=AssertionError("must not hash repeats")
        ):
            bench.latency(Reader(), manifest, 3, 7, expected=expected)


class TestRigorousBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        (self.corpus / "empty").write_bytes(b"")
        (self.corpus / "nested").mkdir()
        (self.corpus / "nested/žltý.txt").write_bytes(b"hello\x00" * 120)
        (self.corpus / "noise.bin").write_bytes(bytes(range(256)) * 20)
        self.m = bench.manifest(self.corpus)
        self.files = bench.load_inputs(self.m)

    def test_all_containers_restore_exact_paths_and_bytes(self):
        with patch.object(bench, "BLOCK_BYTES", 128):
            for config in bench.CONFIGS:
                with self.subTest(config=config):
                    blob = bench.pack(config, self.files, 1)
                    decoded = bench.restore(config, blob, 1)
                    bench.verify(decoded, self.m)
                    self.assertEqual(dict(self.files), decoded)
                    self.assertEqual(blob, bench.pack(config, self.files, 1))

    def test_file_backed_seekable_readers(self):
        with patch.object(bench, "BLOCK_BYTES", 128):
            for config in sorted(bench.SEEKABLE):
                with self.subTest(config=config):
                    path = self.root / (config + ".archive")
                    path.write_bytes(bench.pack(config, self.files, 1))
                    reader = (
                        bench.VpakReader(path)
                        if config == "vpak"
                        else (
                            bench.ZipReader(path)
                            if config.startswith("zip-")
                            else bench.IndexedReader(path)
                        )
                    )
                    try:
                        for _ in range(2):
                            for name, data in self.files:
                                self.assertEqual(data, reader.read(name))
                            reader.clear()
                        obs = bench.latency(reader, self.m, 8, 1)
                        self.assertEqual(8, obs["codec_reset"]["summary_ms"]["n"])
                    finally:
                        reader.close()

    def test_manifest_detects_source_mutation(self):
        (self.corpus / "empty").write_bytes(b"changed")
        with self.assertRaisesRegex(AssertionError, "integrity"):
            bench.load_inputs(self.m)
        self.assertNotEqual(self.m["sha256"], bench.manifest(self.corpus)["sha256"])

    def test_verification_rejects_missing_extra_and_wrong_files(self):
        files = dict(self.files)
        for altered in ({}, {**files, "extra": b""}, {**files, "empty": b"x"}):
            with self.assertRaises(AssertionError):
                bench.verify(altered, self.m)

    def test_nearest_rank_percentiles_and_tail_support(self):
        out = bench.distribution(list(range(1, 101)))
        self.assertEqual(99, out["p99"])
        self.assertEqual(1, out["tail_support_p99"])
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                bench.distribution(values)

    def test_bootstrap_is_deterministic_and_does_not_drop_outliers(self):
        a = bench.median_ci([1, 1, 1, 1, 100])
        self.assertEqual(a, bench.median_ci([1, 1, 1, 1, 100]))
        self.assertEqual(100, a["max"])
        self.assertEqual(5, a["n"])
        self.assertEqual([3.0, 3.0], bench.median_ci([3] * 7)["bootstrap_95ci"])

    @unittest.skipUnless(importlib.util.find_spec("psutil"), "install benchmark extra")
    def test_subprocess_protocol_and_raw_report(self):
        out = self.root / "report"
        cmd = [
            sys.executable,
            str(ROOT / "benchmarks/rigorous.py"),
            "--corpus",
            str(self.corpus),
            "--output",
            str(out),
            "--configs",
            "zip-store",
            "indexed-zstd3",
            "tar-zstd3",
            "--repeats",
            "2",
            "--warmups",
            "0",
            "--requests",
            "8",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        self.assertEqual(0, r.returncode, r.stderr)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual("complete", report["status"])
        self.assertEqual(6, len(report["trials"]))
        self.assertFalse(report["adequacy"]["at_least_7_rounds"])
        self.assertFalse(
            report["adequacy"]["p99_at_least_100_tail_observations_per_trial"]
        )
        self.assertEqual(self.m["sha256"], report["manifest"]["sha256"])
        self.assertIn("paired_vs_tar_zstd3", report)
        self.assertEqual([], report["errors"])
        self.assertNotEqual(
            0, subprocess.run(cmd, capture_output=True, timeout=30).returncode
        )


if __name__ == "__main__":
    unittest.main()
