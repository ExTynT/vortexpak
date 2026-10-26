"""Engine-ready GPU textures: mip chains, sRGB tagging, DDS export, C-ABI info."""

import ctypes
import io
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vortexpak import VortexPakCodec, VortexVFS
from vortexpak.bcn import (
    BCN_BC1,
    BCN_BC5,
    chain_bytes,
    downsample_2x,
    encode_bcn,
    level_bytes,
    mip_dims,
    mip_sources,
)
from vortexpak.codec import TX_MIPS, TX_SRGB
from vortexpak.native import IS_NATIVE_AVAILABLE, _lib


def _gradient(w=64, h=64, alpha=255):
    y, x = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 0] = x * 255 // (w - 1)
    a[..., 1] = y * 255 // (h - 1)
    a[..., 2] = 255 - a[..., 0] // 2
    a[..., 3] = alpha
    return a


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class TestMipChain(unittest.TestCase):
    def test_dims_and_srgb_box_filter(self):
        self.assertEqual(mip_dims(64, 32), [(64, 32), (32, 16), (16, 8), (8, 4)])
        self.assertEqual(mip_dims(12, 12), [(12, 12)])
        a = np.zeros((8, 8, 4), np.uint8)
        a[::2, ::2] = 255
        # 25 % coverage in linear light is sRGB 137, not 64; alpha stays linear.
        self.assertEqual(downsample_2x(a, True)[0, 0].tolist(), [137, 137, 137, 64])
        self.assertEqual(downsample_2x(a, False)[0, 0].tolist(), [64, 64, 64, 64])


class TestEngineTextures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        game = self.root / "game"
        game.mkdir()
        self.src = _gradient()
        (game / "albedo.png").write_bytes(_png(self.src))
        (game / "odd.png").write_bytes(_png(_gradient(12, 12)))

    def tearDown(self):
        self._tmp.cleanup()

    def _archive(self, name, **kw):
        path = self.root / name
        codec = VortexPakCodec(mode="engine", **kw)
        arch = codec.encode_directory(self.root / "game")
        ok, _ = codec.verify_archive(arch)
        self.assertTrue(ok)
        path.write_bytes(arch.to_bytes())
        return path, {e.rel_path: e for e in arch.entries}

    def test_mips_are_default_and_level0_is_unchanged(self):
        with_mips, entries = self._archive("m.vpak", gpu="bc1")
        without, entries_flat = self._archive("f.vpak", gpu="bc1", mips=False)
        e = entries["albedo.png"]
        self.assertTrue(e.tx_flags & TX_MIPS and e.tx_flags & TX_SRGB)
        self.assertFalse(entries_flat["albedo.png"].tx_flags & TX_MIPS)
        self.assertFalse(entries["odd.png"].tx_flags & TX_MIPS)  # 12x12: no block-aligned mips
        with VortexVFS.open(with_mips) as a, VortexVFS.open(without) as b:
            raw = a.read_bytes("albedo.png", materialize_full_rgba=False)
            self.assertEqual(len(raw), chain_bytes(64, 64, BCN_BC1, len(mip_dims(64, 64))))
            flat = b.read_bytes("albedo.png", materialize_full_rgba=False)
            self.assertEqual(raw[: level_bytes(64, 64, BCN_BC1)], flat)
            self.assertEqual(a.read_bytes("albedo.png"), b.read_bytes("albedo.png"))
            info = a.texture_info("albedo.png")
            self.assertEqual([lv["width"] for lv in info["levels"]], [64, 32, 16, 8, 4])
            self.assertEqual(info["levels"][-1]["offset"] + info["levels"][-1]["size"], len(raw))
            # Every level is exactly the encoding of the sRGB box-filtered source.
            for lv, ref in zip(info["levels"][1:], mip_sources(self.src, True)):
                blocks = raw[lv["offset"] : lv["offset"] + lv["size"]]
                self.assertEqual(blocks, encode_bcn(ref.tobytes(), lv["width"], lv["height"], BCN_BC1))
                self.assertEqual(ref.shape[:2], (lv["height"], lv["width"]))

    def test_floor_gates_level0_and_reports_worst_mip(self):
        # Level 0 BC1 reaches ~37.7 dB; its 4x4 mip is one block holding a full
        # 2-D gradient that no one-subset BC mode can represent (~14 dB).
        from vortexpak.codec import _encode_texture_gpu

        res = _encode_texture_gpu(_png(self.src), "auto", 35.0, True)
        self.assertEqual(res[2], "BC1")
        q = res[6]
        self.assertGreaterEqual(q["color_psnr_db"], 35.0)
        self.assertLess(q["mip_min_color_psnr_db"], 20.0)
        self.assertEqual(_encode_texture_gpu(_png(self.src), "auto", 40.0, True), "below_floor")
        codec = VortexPakCodec(mode="engine", gpu="auto")
        codec.encode_files([("a.png", _png(self.src))])
        m = codec.last_rdo_metrics
        self.assertAlmostEqual(m["gpu_min_mip_color_psnr_db"], round(q["mip_min_color_psnr_db"], 2))

    def test_bc5_is_linear_and_bcx_has_no_dds(self):
        path, entries = self._archive("n.vpak", gpu="bc5")
        e = entries["albedo.png"]
        self.assertTrue(e.tx_flags & TX_MIPS)
        self.assertFalse(e.tx_flags & TX_SRGB)
        with VortexVFS.open(path) as vfs:
            raw = vfs.read_bytes("albedo.png", materialize_full_rgba=False)
            self.assertEqual(len(raw), chain_bytes(64, 64, BCN_BC5, 5))
        path, entries = self._archive("x.vpak", gpu="bcx")
        self.assertFalse(entries["albedo.png"].tx_flags & TX_MIPS)
        with VortexVFS.open(path) as vfs:
            self.assertFalse(vfs.texture_info("albedo.png")["gpu_native"])
            with self.assertRaises(ValueError):
                vfs.read_dds("albedo.png")

    def test_dds_is_read_by_an_independent_decoder(self):
        path, _ = self._archive("d.vpak", gpu="bc7")
        with VortexVFS.open(path) as vfs:
            dds = vfs.read_dds("albedo.png")
            ours = np.frombuffer(vfs.read_bytes("albedo.png"), np.uint8).reshape(64, 64, 4)
        self.assertEqual(struct.unpack_from("<I", dds, 28)[0], 5)  # dwMipMapCount
        self.assertEqual(struct.unpack_from("<I", dds, 128)[0], 99)  # BC7_UNORM_SRGB
        with Image.open(io.BytesIO(dds)) as im:
            self.assertEqual(im.size, (64, 64))
            np.testing.assert_array_equal(np.array(im.convert("RGBA")), ours)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE and _lib is not None, "native core required")
    def test_c_abi_texture_info_and_level0_decode(self):
        class Info(ctypes.Structure):
            _fields_ = [(n, t) for n, t in (("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                                            ("bcn_format", ctypes.c_int32), ("block_bytes", ctypes.c_uint32),
                                            ("mip_levels", ctypes.c_uint32), ("srgb", ctypes.c_uint32))]

        _lib.vfs_open_c.argtypes = [ctypes.c_char_p]
        _lib.vfs_open_c.restype = ctypes.c_void_p
        _lib.vfs_close_c.argtypes = [ctypes.c_void_p]
        _lib.vfs_get_texture_info_c.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(Info)]
        _lib.vfs_get_texture_info_c.restype = ctypes.c_int32
        _lib.vfs_read_asset_c.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                          ctypes.c_size_t, ctypes.c_int32]
        _lib.vfs_read_asset_c.restype = ctypes.c_int64
        path, _ = self._archive("c.vpak", gpu="bc1")
        h = _lib.vfs_open_c(str(path).encode())
        self.assertTrue(h)
        try:
            info = Info()
            self.assertEqual(_lib.vfs_get_texture_info_c(h, b"albedo.png", ctypes.byref(info)), 0)
            self.assertEqual((info.width, info.height, info.bcn_format, info.block_bytes,
                              info.mip_levels, info.srgb), (64, 64, BCN_BC1, 8, 5, 1))
            self.assertEqual(_lib.vfs_get_texture_info_c(h, b"missing.png", ctypes.byref(info)), -2)
            buf = ctypes.create_string_buffer(64 * 64 * 4)
            n = _lib.vfs_read_asset_c(h, b"albedo.png", buf, len(buf), 1)
            self.assertEqual(n, 64 * 64 * 4)
        finally:
            _lib.vfs_close_c(h)
        with VortexVFS.open(path) as vfs:
            self.assertEqual(buf.raw, vfs.read_bytes("albedo.png"))

    def test_cli_no_mips_and_dds_export(self):
        vpak, out = self.root / "cli.vpak", self.root / "albedo.dds"
        cli = [sys.executable, "-m", "vortexpak.cli"]
        res = subprocess.run(cli + ["pack", str(self.root / "game"), str(vpak), "--gpu", "bc1"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        res = subprocess.run(cli + ["extract", str(vpak), "albedo.png", "--dds", "-o", str(out)],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("BC1_SRGB 64x64, 5 mip level(s)", res.stdout)
        dds = out.read_bytes()
        # Pillow cannot decode BC1_UNORM_SRGB (72); the header is checked instead.
        self.assertEqual(struct.unpack_from("<4sIIII", dds), (b"DDS ", 124, 0xA1007, 64, 64))
        self.assertEqual(struct.unpack_from("<I", dds, 128)[0], 72)
        self.assertEqual(len(dds), 148 + chain_bytes(64, 64, BCN_BC1, 5))
        res = subprocess.run(cli + ["pack", str(self.root / "game"), str(vpak), "--gpu", "bc1", "--no-mips"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        with VortexVFS.open(vpak) as vfs:
            self.assertEqual(len(vfs.texture_info("albedo.png")["levels"]), 1)


if __name__ == "__main__":
    unittest.main()
