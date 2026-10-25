"""Robustness: corrupt/truncated .vpak inputs must raise VXPKError — never a
bare struct.error, KeyError or a hard crash. The fuzz loop mutates a valid
archive; from_bytes must either parse it or raise VXPKError, nothing else."""

import io
import json
import random
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import vortexpak.codec as codec_mod
from vortexpak import VortexArchive, VortexPakCodec
from vortexpak.codec import VXPKError


def _make_archive() -> VortexArchive:
    codec = VortexPakCodec(mode="engine")
    file_tuples = []
    for i in range(2):
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        arr[..., :3] = (i * 61 + 17) % 256
        arr[..., 3] = 255
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        file_tuples.append((f"textures/t{i}.png", buf.getvalue()))
    file_tuples.append(
        ("data/config.json", json.dumps({"hello": "world", "n": 42}).encode("utf-8"))
    )
    return codec.encode_files(file_tuples)


class TestMalformedContainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = _make_archive()
        cls.valid = cls.arch.to_bytes()

    def test_empty_input(self):
        with self.assertRaises(VXPKError):
            VortexArchive.from_bytes(b"")

    def test_short_header(self):
        for n in range(1, 10):
            with self.assertRaises(VXPKError, msg=f"len={n}"):
                VortexArchive.from_bytes(self.valid[:n])

    def test_bad_magic(self):
        bad = b"XXXXX" + self.valid[5:]
        with self.assertRaises(VXPKError):
            VortexArchive.from_bytes(bad)

    def test_truncated_variants(self):
        for frac in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 0.999):
            cut = int(len(self.valid) * frac)
            with self.assertRaises(VXPKError, msg=f"cut={cut}"):
                VortexArchive.from_bytes(self.valid[:cut])

    def test_trailing_junk_is_tolerated(self):
        arch = VortexArchive.from_bytes(self.valid + b"\x00" * 16)
        self.assertEqual(len(arch.entries), len(self.arch.entries))

    def test_lane_dictionary_roundtrip(self):
        """Lane-dictionary path: helper replaces a weak lane with dict-seeded
        blocks; decode via codec AND VFS is bit-exact; selection never grows
        the archive."""
        import vortexpak.codec as codec_mod
        from vortexpak.codec import (
            MicroBlock,
            _apply_lane_dictionaries,
            _decompress_block,
        )
        from unittest import mock

        raws = []
        for i in range(48):
            doc = {"entity": f"unit_{i % 6}", "tiles": [i % 16, 0, 12, 0] * 24}
            raws.append((f"data/t{i:03d}.json", json.dumps(doc).encode()))
        # Fabricate a lane whose current blocks are weak (stored raw), so the
        # dictionary path must win deterministically.
        weak = [
            MicroBlock(
                bid, f"LANE_STRUCTURED_{bid}", codec_mod.STRAT_RAW_BYPASS, len(d), d
            )
            for bid, (_, d) in enumerate(raws)
        ]
        groups = {"LANE_STRUCTURED": [(b.block_id, b.compressed_payload) for b in weak]}
        with (
            mock.patch.object(codec_mod, "DICT_MIN_LANE_BYTES", 1),
            mock.patch.object(codec_mod, "DICT_MIN_ITEMS", 4),
            mock.patch.object(codec_mod, "DICT_SIZE_BYTES", 4096),
        ):
            out = _apply_lane_dictionaries(weak, groups)
        dict_blocks = [b for b in out if b.strategy_id in (9, 10)]
        self.assertTrue(dict_blocks, "dictionary path must engage on a weak lane")
        self.assertLess(
            sum(len(b.compressed_payload) for b in out),
            sum(len(b.compressed_payload) for b in weak),
        )
        by_id = {b.block_id: b for b in out}
        cache = {}

        def resolver(did):
            if did not in cache:
                cache[did] = _decompress_block(by_id[did])
            return cache[did]

        for b in out:
            if b.strategy_id in (9, 10):
                data = _decompress_block(b, dict_resolver=resolver)
                self.assertEqual(data, b"") if b.raw_size == 0 else None
                self.assertEqual(len(data), b.raw_size)
        # End-to-end: full archive with dict blocks round-trips through VFS.
        entries = []
        lane = []
        for idx, (name, raw) in enumerate(raws):
            from vortexpak.codec import AssetEntry

            e = AssetEntry(rel_path=name, ext=".json", raw_size=len(raw), xxh64=0)
            e.block_id = idx
            e.offset_in_block = 0
            e.length_in_block = len(raw)
            entries.append(e)
            lane.append((idx, raw))
        with (
            mock.patch.object(codec_mod, "DICT_MIN_LANE_BYTES", 1),
            mock.patch.object(codec_mod, "DICT_MIN_ITEMS", 4),
            mock.patch.object(codec_mod, "DICT_SIZE_BYTES", 4096),
        ):
            blocks = _apply_lane_dictionaries(weak, {"LANE_STRUCTURED": lane})
        import xxhash

        for e, (name, raw) in zip(entries, raws):
            e.xxh64 = xxhash.xxh3_64_intdigest(raw)
        arch = codec_mod.VortexArchive(
            mode="strict",
            entries=entries,
            blocks=blocks,
            total_raw_bytes=sum(len(r) for _, r in raws),
        )
        blob = arch.to_bytes()
        arch2 = VortexArchive.from_bytes(blob)
        ok, count = VortexPakCodec(mode="strict").verify_archive(arch2)
        self.assertTrue(ok, "dict archive must verify 100%")
        self.assertEqual(count, len(raws))
        import tempfile
        from pathlib import Path

        from vortexpak import VortexVFS

        with tempfile.TemporaryDirectory() as td:
            vp = Path(td) / "d.vpak"
            vp.write_bytes(blob)
            with VortexVFS.open(vp) as vfs:
                for name, raw in raws[:8]:
                    self.assertEqual(vfs.read_bytes(name), raw)

    def test_zstd_meta_branch_roundtrip(self):
        with mock.patch.object(codec_mod, "IS_NATIVE_AVAILABLE", False):
            blob = self.arch.to_bytes()
        arch = VortexArchive.from_bytes(blob)
        self.assertEqual(arch.mode, "engine")
        self.assertEqual(len(arch.entries), len(self.arch.entries))

    def test_fuzz_bit_flips(self):
        rng = random.Random(20250101)
        n = len(self.valid)
        parsed = errored = 0
        for _ in range(300):
            i = rng.randrange(n)
            bad = bytearray(self.valid)
            bad[i] ^= 1 << rng.randrange(8)
            try:
                VortexArchive.from_bytes(bytes(bad))
                parsed += 1
            except VXPKError:
                errored += 1
            except Exception as exc:
                self.fail(f"flip at byte {i} raised {type(exc).__name__}: {exc}")
        self.assertGreater(parsed, 0)
        self.assertGreater(errored, 0)


if __name__ == "__main__":
    unittest.main()
