import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("psutil") is None:
    raise unittest.SkipTest("benchmark harness tests need the benchmark extra (psutil)")
SPEC = importlib.util.spec_from_file_location(
    "rigorous_quality", ROOT / "benchmarks/rigorous_quality.py"
)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


class TestOracleMemoization(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("flip_evaluator"), "install quality extra"
    )
    def test_cache_and_control_are_exact_and_reject_changed_bytes(self):
        from vortexpak.bcn import encode_bcn

        arr = np.random.default_rng(33).integers(0, 256, (8, 8, 4), dtype=np.uint8)
        blocks = encode_bcn(arr.tobytes(), 8, 8, 7)
        oracles = [quality.MetricOracle("reference"), quality.MetricOracle("optimized")]
        values = []
        for oracle in oracles:
            for _ in range(3):
                values.append(
                    oracle.evaluate({"fmt": 7, "ppd": 67}, arr, blocks, 8, 8, 7)
                )
            self.assertEqual(3, oracle.validated_encodes)
            changed = bytes([blocks[0] ^ 1]) + blocks[1:]
            with self.assertRaisesRegex(AssertionError, "nondeterministic"):
                oracle.evaluate({"fmt": 7, "ppd": 67}, arr, changed, 8, 8, 7)
        self.assertTrue(all(v == values[0] for v in values))
        self.assertEqual(3, oracles[0].reference_calls)
        self.assertEqual(1, oracles[1].reference_calls)
        oracles[1].evaluate({"fmt": 7, "ppd": 68}, arr, blocks, 8, 8, 7)
        self.assertEqual(2, oracles[1].reference_calls)


class TestQualityProcess(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("flip_evaluator"), "install quality extra"
    )
    def test_complete_evidence_and_renderers(self):
        import json
        import subprocess
        import sys
        import tempfile
        from benchmarks import protocol
        from vortexpak.native import IS_NATIVE_AVAILABLE

        if not IS_NATIVE_AVAILABLE:
            self.skipTest("explicit fallback profile has no native quality reference")
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            corpus, output = base / "images", base / "quality"
            corpus.mkdir()
            arr = np.random.default_rng(98).integers(0, 256, (8, 8, 4), dtype=np.uint8)
            Image.fromarray(arr).save(corpus / "image.png")
            cmd = [
                sys.executable,
                str(ROOT / "benchmarks/rigorous_quality.py"),
                "--corpus",
                str(corpus),
                "--output",
                str(output),
                "--repeats",
                "2",
                "--warmups",
                "1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            self.assertEqual(0, result.returncode, result.stderr)
            report = protocol.load_report(output)
            self.assertEqual(10, len(report["trials"]))
            self.assertEqual(5, report["oracle"]["reference_calls"])
            self.assertEqual(15, report["oracle"]["validated_encodes"])
            self.assertFalse(report["adequacy"]["qualified_reference"])
            self.assertEqual(json.loads((output / "report.json").read_text()), report)
            from benchmarks.report_benchmarks import html_document
            from benchmarks.report_image import make_image

            html = html_document([(output, report)])
            self.assertIn(report["evidence_head"], html)
            image = make_image([(output, report)])
            self.assertGreater(image.width, 1000)
            self.assertGreater(image.height, 500)


class TestRigorousQuality(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("flip_evaluator"), "install quality extra"
    )
    def test_rgb_psnr_cannot_be_inflated_by_opaque_alpha(self):
        a = np.zeros((16, 16, 4), dtype=np.uint8)
        a[..., 3] = 255
        b = a.copy()
        b[..., :3] = 10
        met = quality.metrics(a, b)
        self.assertEqual(100, met["rgb_mse_srgb8"])
        self.assertAlmostEqual(10 * math.log10(255**2 / 100), met["rgb_psnr_srgb_db"])
        self.assertTrue(met["alpha_exact"])
        self.assertIsNone(met["alpha_psnr_db"])

    @unittest.skipUnless(
        importlib.util.find_spec("flip_evaluator"), "install quality extra"
    )
    def test_identity_control(self):
        a = np.random.default_rng(6).integers(0, 256, (16, 16, 4), dtype=np.uint8)
        met = quality.metrics(a, a)
        self.assertEqual(0, met["flip_rgb_ppd67"])
        self.assertIsNone(met["rgb_psnr_srgb_db"])
        self.assertTrue(met["rgb_exact"])

    def test_every_texture_format_against_independent_reference(self):
        from vortexpak.bcn import BCN_BC1, BCN_BC7, BCN_BCX, decode_bcn, encode_bcn

        a = np.random.default_rng(7).integers(0, 256, (16, 16, 4), dtype=np.uint8)
        for fmt in (BCN_BC1, BCN_BC7, BCN_BCX):
            with self.subTest(fmt=fmt):
                data = encode_bcn(a.tobytes(), 16, 16, fmt)
                actual = np.frombuffer(decode_bcn(data, 16, 16, fmt), np.uint8).reshape(
                    a.shape
                )
                np.testing.assert_array_equal(
                    actual, quality.reference_decode(data, 16, 16, fmt)
                )

    def test_dds_is_a_real_independent_container(self):
        blocks = bytes(8)
        import io

        with Image.open(io.BytesIO(quality.dds(blocks, 4, 4, 1))) as im:
            self.assertEqual((4, 4), im.size)
            im.load()


if __name__ == "__main__":
    unittest.main()
