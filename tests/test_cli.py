"""End-to-end CLI tests: pack -> info -> ls -> extract -> vfs-bench, plus the
corrupt-archive error path (must exit 2 with a clean message, no traceback)."""

import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vortexpak import IS_NATIVE_AVAILABLE, VortexArchive


def _make_corpus(root: Path) -> None:
    tex = root / "textures"
    tex.mkdir(parents=True)
    for i in range(3):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[..., 0] = (i * 80) % 256
        arr[..., 1] = (i * 40 + 10) % 256
        arr[..., 3] = 255
        Image.fromarray(arr).save(tex / f"t{i}.png", format="PNG")
    data = root / "data"
    data.mkdir(parents=True)
    (data / "config.json").write_text(json.dumps({"name": "demo", "level": 3}))


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "vortexpak.cli", *args],
        capture_output=True,
        text=True,
    )


class TestCliEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.game_dir = cls.tmp / "game"
        cls.game_dir.mkdir()
        _make_corpus(cls.game_dir)
        cls.vpak = cls.tmp / "demo.vpak"
        cls.strict_vpak = cls.tmp / "demo_strict.vpak"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_01_pack_engine(self):
        res = _run_cli("pack", str(self.game_dir), str(self.vpak))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Packed", res.stdout)
        self.assertIn("assets", res.stdout)
        self.assertTrue(self.vpak.is_file())

    def test_02_pack_strict(self):
        res = _run_cli(
            "pack", str(self.game_dir), str(self.strict_vpak), "--mode", "strict"
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.strict_vpak.is_file())

    def test_03_info(self):
        res = _run_cli("info", str(self.vpak))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Archive Mode:", res.stdout)
        self.assertIn("XXH3-64 Verified:  True", res.stdout)

    def test_04_ls(self):
        res = _run_cli("ls", str(self.vpak), "--filter", "*.png")
        self.assertEqual(res.returncode, 0, res.stderr)
        for i in range(3):
            self.assertIn(f"t{i}.png", res.stdout)

    def test_05_extract_data_byte_exact(self):
        out = self.tmp / "config_copy.json"
        res = _run_cli("extract", str(self.vpak), "data/config.json", "-o", str(out))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(
            out.read_text(), (self.game_dir / "data" / "config.json").read_text()
        )

    def test_06_extract_texture_as_png(self):
        out = self.tmp / "t0_copy.png"
        res = _run_cli("extract", str(self.vpak), "textures/t0.png", "-o", str(out))
        self.assertEqual(res.returncode, 0, res.stderr)
        with Image.open(out) as img:
            img.verify()
        self.assertGreater(out.stat().st_size, 0)

    def test_07_vfs_bench(self):
        res = _run_cli("vfs-bench", str(self.vpak), "--queries", "20")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Effective Speed", res.stdout)

    def test_08_corrupt_archive_exits_2(self):
        blob = bytearray(self.vpak.read_bytes())
        (meta_len,) = struct.unpack_from("<I", blob, 6)

        # 1) Corruption inside the meta section: every command must refuse cleanly.
        meta_corrupt = bytearray(blob)
        meta_corrupt[10 + meta_len // 2] ^= 0xFF
        corrupt_meta = self.tmp / "corrupt_meta.vpak"
        corrupt_meta.write_bytes(bytes(meta_corrupt))
        for cmd in (
            ["info", str(corrupt_meta)],
            ["ls", str(corrupt_meta)],
            ["extract", str(corrupt_meta), "textures/t0.png"],
        ):
            res = _run_cli(*cmd)
            self.assertEqual(res.returncode, 2, f"{cmd}: {res.stderr}")
            self.assertIn("Corrupt archive", res.stderr)
            self.assertNotIn("Traceback", res.stderr)

        # 2) Corruption inside the bytes of one specific asset (meta intact):
        # info/extract of that archive fail with a clean error, ls is
        # metadata-only and still lists.
        arch = VortexArchive.from_bytes(bytes(blob))
        entry = next(e for e in arch.entries if e.rel_path == "textures/t0.png")
        block_offset = 10 + meta_len
        target_block = None
        for b in arch.blocks:
            if b.block_id == entry.block_id:
                target_block = b
                break
            block_offset += len(b.compressed_payload)
        flip_at = block_offset + len(target_block.compressed_payload) // 2
        payload_corrupt = bytearray(blob)
        payload_corrupt[flip_at] ^= 0xFF
        corrupt_payload = self.tmp / "corrupt_payload.vpak"
        corrupt_payload.write_bytes(bytes(payload_corrupt))
        res = _run_cli("info", str(corrupt_payload))
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("Corrupt archive", res.stderr)
        self.assertNotIn("Traceback", res.stderr)
        res = _run_cli("ls", str(corrupt_payload))
        self.assertEqual(res.returncode, 0, f"ls: {res.stderr}")
        self.assertNotIn("Traceback", res.stderr)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native AVX2 core unavailable")
    def test_08b_extract_from_payload_corrupt_archive(self):
        # The native VXLZ path detects payload corruption during decompression;
        # the pure-Python fallback stores raw-bypass blocks, so there the XXH3
        # verification inside the info command is the integrity gate.
        blob = bytearray(self.vpak.read_bytes())
        (meta_len,) = struct.unpack_from("<I", blob, 6)
        arch = VortexArchive.from_bytes(bytes(blob))
        entry = next(e for e in arch.entries if e.rel_path == "textures/t0.png")
        block_offset = 10 + meta_len
        target_block = None
        for b in arch.blocks:
            if b.block_id == entry.block_id:
                target_block = b
                break
            block_offset += len(b.compressed_payload)
        flip_at = block_offset + len(target_block.compressed_payload) // 2
        payload_corrupt = bytearray(blob)
        payload_corrupt[flip_at] ^= 0xFF
        corrupt_payload = self.tmp / "corrupt_payload_extract.vpak"
        corrupt_payload.write_bytes(bytes(payload_corrupt))
        res = _run_cli("extract", str(corrupt_payload), "textures/t0.png")
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("Corrupt archive", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_09_bad_magic_exits_2(self):
        junk = self.tmp / "junk.vpak"
        junk.write_bytes(b"NOTAVXPK" + b"\x00" * 64)
        res = _run_cli("info", str(junk))
        self.assertEqual(res.returncode, 2)
        self.assertIn("Corrupt archive", res.stderr)


if __name__ == "__main__":
    unittest.main()
