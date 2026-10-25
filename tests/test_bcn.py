"""VXTX-GPU BC1/BC4/BC5 block-compression tests: quality, alpha, decoder
interop between native and NumPy paths, and archive/VFS/CLI integration."""

import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from vortexpak import VortexArchive, VortexPakCodec, VortexVFS
from vortexpak.bcn import (
    BCN_BC1,
    BCN_BC4,
    BCN_BC5,
    BCN_BC7,
    _fallback_decode,
    _fallback_encode,
    decode_bcn,
    encode_bcn,
)
from vortexpak.codec import TX_BCN


def _sample_rgba(w: int = 64, h: int = 64) -> bytes:
    x = np.linspace(0, 255, w, dtype=np.uint16)
    y = np.linspace(0, 255, h, dtype=np.uint16)
    r = np.tile(x, (h, 1)).astype(np.uint8)
    g = np.repeat(y[:, None], w, axis=1).astype(np.uint8)
    b = ((r.astype(np.int32) + g) // 2).astype(np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)
    rng = np.random.default_rng(42)
    noise = (rng.integers(0, 64, size=(h, w)) - 32).astype(np.int32)
    b = np.clip(b.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    return rgba.tobytes()


def _psnr(a: bytes, b: bytes) -> float:
    va = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    vb = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    mse = float(((va - vb) ** 2).mean())
    return 100.0 if mse < 1e-12 else 10.0 * np.log10(255.0**2 / mse)


class TestBcnCore(unittest.TestCase):
    W, H = 64, 64

    def test_bc1_roundtrip_quality_and_ratio(self):
        src = _sample_rgba(self.W, self.H)
        blocks = encode_bcn(src, self.W, self.H, BCN_BC1)
        self.assertEqual(len(blocks), (self.W // 4) * (self.H // 4) * 8)
        dec = decode_bcn(blocks, self.W, self.H, BCN_BC1)
        self.assertGreater(_psnr(src, dec), 28.0)
        self.assertGreater(len(src) / len(blocks), 4.0)

    def test_bc1_flat_block_exact(self):
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., :3] = (41, 203, 107)
        rgba[..., 3] = 255
        src = rgba.tobytes()
        dec = decode_bcn(encode_bcn(src, 32, 32, BCN_BC1), 32, 32, BCN_BC1)
        self.assertGreater(_psnr(src, dec), 45.0)

    def test_bc1_alpha_1bit_exact(self):
        rgba = (
            np.frombuffer(_sample_rgba(self.W, self.H), dtype=np.uint8)
            .copy()
            .reshape(self.W, self.H, 4)
        )
        mask = (np.arange(self.W * self.H) % 3 == 0).reshape(self.W, self.H)
        rgba[..., 3] = np.where(mask, 0, 255)
        src = rgba.tobytes()
        dec = decode_bcn(
            encode_bcn(src, self.W, self.H, BCN_BC1), self.W, self.H, BCN_BC1
        )
        self.assertTrue((np.frombuffer(dec, dtype=np.uint8)[3::4] < 128).sum() > 0)
        got = np.frombuffer(dec, dtype=np.uint8).reshape(self.W, self.H, 4)[..., 3]
        want = np.where(mask, 0, 255)
        self.assertTrue((got == want).all())

    def test_bc4_bc5_roundtrip(self):
        y = np.repeat(np.arange(self.H, dtype=np.uint8)[:, None] * 4, self.W, axis=1)
        data = np.minimum(y + np.arange(self.W, dtype=np.uint8)[None, :], 255)
        src = data.tobytes()
        dec = decode_bcn(
            encode_bcn(src, self.W, self.H, BCN_BC4), self.W, self.H, BCN_BC4
        )
        self.assertGreater(_psnr(src, dec), 40.0)
        rg = np.empty((self.H, self.W, 2), dtype=np.uint8)
        rg[..., 0] = data
        rg[..., 1] = 255 - data
        src5 = rg.tobytes()
        blocks5 = encode_bcn(src5, self.W, self.H, BCN_BC5)
        self.assertEqual(len(blocks5), (self.W // 4) * (self.H // 4) * 16)
        dec5 = decode_bcn(blocks5, self.W, self.H, BCN_BC5)
        self.assertGreater(_psnr(src5, dec5), 40.0)

    def test_decoder_interop_native_vs_numpy(self):
        src = _sample_rgba(32, 32)
        blocks = encode_bcn(src, 32, 32, BCN_BC1)
        dec_fallback = _fallback_decode(blocks, 32, 32, BCN_BC1)
        self.assertGreater(_psnr(src, dec_fallback), 25.0)
        blocks_fb = _fallback_encode(src, 32, 32, BCN_BC1)
        dec_native = decode_bcn(blocks_fb, 32, 32, BCN_BC1)
        self.assertGreater(_psnr(src, dec_native), 25.0)

    def test_quality_tiers(self):
        src = _sample_rgba(32, 32)
        fast = encode_bcn(src, 32, 32, BCN_BC1, quality="fast")
        std = encode_bcn(src, 32, 32, BCN_BC1, quality="standard")
        mx = encode_bcn(src, 32, 32, BCN_BC1, quality="max")
        self.assertEqual(len(fast), len(std))
        self.assertEqual(len(std), len(mx))
        with self.assertRaises(ValueError):
            encode_bcn(src, 32, 32, BCN_BC1, quality="ultra")
        pf = _psnr(src, decode_bcn(fast, 32, 32, BCN_BC1))
        pm = _psnr(src, decode_bcn(mx, 32, 32, BCN_BC1))
        self.assertGreaterEqual(pm, pf - 0.05)

    def test_bc7_roundtrip(self):
        src = _sample_rgba(32, 32)
        blocks = encode_bcn(src, 32, 32, BCN_BC7)
        self.assertEqual(len(blocks), 8 * 8 * 16)
        dec = decode_bcn(blocks, 32, 32, BCN_BC7)
        self.assertGreater(_psnr(src, dec), 30.0)
        # decoder interop: native stream decoded by the NumPy path
        dec_fb = _fallback_decode(blocks, 32, 32, BCN_BC7)
        self.assertGreater(_psnr(src, dec_fb), 30.0)
        blocks_fb = _fallback_encode(src, 32, 32, BCN_BC7)
        dec2 = decode_bcn(blocks_fb, 32, 32, BCN_BC7)
        self.assertGreater(_psnr(src, dec2), 28.0)

    def test_bc1_rdo_native(self):
        src = _sample_rgba(32, 32)
        base = decode_bcn(encode_bcn(src, 32, 32, BCN_BC1), 32, 32, BCN_BC1)
        rdo = decode_bcn(encode_bcn(src, 32, 32, BCN_BC1, rdo=1.0), 32, 32, BCN_BC1)
        pb, pr = _psnr(src, base), _psnr(src, rdo)
        self.assertGreaterEqual(pr, pb - 0.05)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            encode_bcn(b"\x00" * 64, 0, 0, BCN_BC1)
        with self.assertRaises(ValueError):
            encode_bcn(b"\x00" * 64, 7, 4, BCN_BC1)
        with self.assertRaises(ValueError):
            decode_bcn(b"\x00" * 7, 4, 4, BCN_BC1)
        with self.assertRaises(ValueError):
            encode_bcn(b"\x00" * 64, 4, 4, 42)


class TestGpuArchiveIntegration(unittest.TestCase):
    def _make_corpus(self, root):
        root.mkdir(parents=True, exist_ok=True)
        tex = root / "textures"
        tex.mkdir(exist_ok=True)
        for i in range(3):
            arr = np.zeros((32, 32, 4), dtype=np.uint8)
            arr[..., 0] = i * 70
            arr[..., 3] = 255
            Image.fromarray(arr).save(tex / f"t{i}.png")
        (root / "data.json").write_text(json.dumps({"n": 1}))

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = __import__("pathlib").Path(self._tmp.name)
        self._make_corpus(self.root / "game")

    def tearDown(self):
        self._tmp.cleanup()

    def test_gpu_pack_decode_verify(self):
        codec = VortexPakCodec(mode="engine", gpu="bc1")
        arch = codec.encode_files(
            [
                (p.relative_to(self.root / "game").as_posix(), p.read_bytes())
                for p in sorted((self.root / "game").rglob("*"))
                if p.is_file()
            ]
        )
        blob = arch.to_bytes()
        arch2 = VortexArchive.from_bytes(blob)
        self.assertTrue(
            all(e.tx_flags & TX_BCN for e in arch2.entries if e.ext == ".png")
        )
        ok, count = codec.verify_archive(arch2)
        self.assertTrue(ok)
        self.assertEqual(count, 4)
        m = codec.last_rdo_metrics
        self.assertEqual(m.get("gpu_format"), "bc1")
        self.assertGreaterEqual(m.get("gpu_avg_compression_ratio", 0), 3.0)
        decoded = codec.decode_archive(arch2, materialize_full_rgba=True)
        arr = decoded["textures/t0.png"]
        self.assertEqual(arr.shape, (32, 32, 4))

    def test_vfs_gpu_reads(self):
        codec = VortexPakCodec(mode="engine", gpu="bc1")
        vpak = self.root / "gpu.vpak"
        vpak.write_bytes(codec.encode_directory(self.root / "game").to_bytes())
        with VortexVFS.open(vpak) as vfs:
            raw = vfs.read_bytes("textures/t0.png", materialize_full_rgba=False)
            self.assertEqual(len(raw), 32 * 32 // 2)
            rgba = vfs.read_bytes("textures/t0.png", materialize_full_rgba=True)
            self.assertEqual(len(rgba), 32 * 32 * 4)
            img = vfs.read_texture("textures/t0.png", as_pil=True)
            self.assertEqual(img.size, (32, 32))

    def test_gpu_auto_selection(self):
        codec = VortexPakCodec(mode="engine", gpu="auto")
        files = [
            (p.relative_to(self.root / "game").as_posix(), p.read_bytes())
            for p in sorted((self.root / "game").rglob("*"))
            if p.is_file()
        ]
        arch = codec.encode_files(files)
        modes = {e.tx_mode for e in arch.entries if e.tx_flags & TX_BCN}
        self.assertIn("BC1", modes)  # opaque corpus content stays BC1 (4bpp)
        blob = arch.to_bytes()
        arch2 = VortexArchive.from_bytes(blob)
        ok, _ = codec.verify_archive(arch2)
        self.assertTrue(ok)

    def test_gpu_bc7_explicit(self):
        codec = VortexPakCodec(mode="engine", gpu="bc7")
        files = [
            (p.relative_to(self.root / "game").as_posix(), p.read_bytes())
            for p in sorted((self.root / "game").rglob("*"))
            if p.is_file() and p.suffix == ".png"
        ]
        arch = codec.encode_files(files)
        self.assertTrue(all(e.tx_mode == "BC7" for e in arch.entries))
        ok, _ = codec.verify_archive(arch)
        self.assertTrue(ok)

    def test_cli_gpu_pack(self):
        vpak = self.root / "cli_gpu.vpak"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "vortexpak.cli",
                "pack",
                str(self.root / "game"),
                str(vpak),
                "--gpu",
                "bc1",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        res = subprocess.run(
            [sys.executable, "-m", "vortexpak.cli", "info", str(vpak)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("XXH3-64 Verified:  True", res.stdout)


if __name__ == "__main__":
    unittest.main()
