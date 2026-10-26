"""Tests for VortexPakCodec block-planning APIs and benchmarks/xon_iter.py helpers.

Covers the staged encode pipeline (plan_blocks / compress_blocks / finish_archive)
against encode_files, the xon_iter block cache, fingerprint stability, and the
extension-stratified quick_sample selector.
"""

from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Tuple

# Dependency fix for this interpreter: its user site-packages carries every
# third-party wheel but also an unrelated empty ``tests`` package that shadows
# this directory under ``python -m unittest tests.test_xon_iter``. Run that
# command with PYTHONNOUSERSITE=1; the wheels are re-added below, after the
# repo's ``tests`` package has already resolved to this directory.
import site
import sys

_user_site = site.getusersitepackages()
if _user_site and _user_site not in sys.path:
    sys.path.append(_user_site)

from vortexpak.codec import MicroBlock, VortexPakCodec  # noqa: E402
from vortexpak.native import IS_NATIVE_AVAILABLE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
XON_ITER_PATH = REPO_ROOT / "benchmarks" / "xon_iter.py"

FileList = List[Tuple[str, bytes]]
BlockKey = Tuple[int, int, int, bytes]

WORD_LETTERS = "abcdefghijklmnopqrstuvwxyz"
COMBOS: List[Tuple[str, str]] = [
    ("strict", "compact"),
    ("strict", "stream"),
    ("engine", "stream"),
]


def _load_xon_iter() -> Any:
    spec = importlib.util.spec_from_file_location("xon_iter", XON_ITER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load a module spec from {XON_ITER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_codec(mode: str, layout: str) -> VortexPakCodec:
    return VortexPakCodec(mode=mode, layout=layout, max_workers=4)


def _make_dds(rng: random.Random, fourcc: bytes, block_size: int) -> bytes:
    # DDS magic + dwSize=124 + 120 header bytes; fourCC sits at absolute offset 84.
    # 80x40 pixels = 20x10 = 200 BC blocks, matching the payload appended below.
    header = bytearray(128)
    header[0:4] = b"DDS "
    header[4:8] = (124).to_bytes(4, "little")
    header[12:16] = (40).to_bytes(4, "little")  # dwHeight
    header[16:20] = (80).to_bytes(4, "little")  # dwWidth
    header[80:84] = (4).to_bytes(4, "little")  # DDPF_FOURCC
    header[84:88] = fourcc
    return bytes(header) + rng.randbytes(200 * block_size)


def _make_words_text(rng: random.Random, size: int) -> bytes:
    parts: List[str] = []
    total = 0
    while total < size:
        word = "".join(rng.choice(WORD_LETTERS) for _ in range(rng.randint(2, 9)))
        parts.append(word)
        total += len(word) + 1
    return " ".join(parts).encode("ascii")


def _make_corpus() -> FileList:
    rng = random.Random(99)
    dxt1 = _make_dds(rng, b"DXT1", 8)
    dxt5 = _make_dds(rng, b"DXT5", 16)
    text = _make_words_text(rng, 150 * 1024)
    blob = rng.randbytes(200 * 1024)
    pattern = rng.randbytes(4096)
    big = pattern * (3 * 1024 * 1024 // 4096)
    return [
        ("tex/hero_albedo.dds", dxt1),
        ("tex/hero_normal.dds", dxt5),
        ("tex/hero_albedo_backup.dds", dxt1),
        ("notes/lore.txt", text),
        ("data/blob.bin", blob),
        ("data/empty.dat", b""),
        ("data/pattern.big", big),
    ]


def _make_group(rng: random.Random, suffix: str, count: int, size: int) -> FileList:
    return [
        (f"group/{suffix}/file{i:03d}.{suffix}", rng.randbytes(size))
        for i in range(count)
    ]


def _block_key(block: MicroBlock) -> BlockKey:
    return (block.block_id, block.strategy_id, block.raw_size, block.compressed_payload)


class TestEncodeCompose(unittest.TestCase):
    def test_encode_files_equals_plan_compress_finish(self) -> None:
        files = _make_corpus()
        for mode, layout in COMBOS:
            with self.subTest(mode=mode, layout=layout):
                direct = _make_codec(mode, layout).encode_files(files).to_bytes()
                codec = _make_codec(mode, layout)
                plan = codec.plan_blocks(files)
                blocks = codec.compress_blocks(plan.raw_blocks)
                composed = codec.finish_archive(plan, blocks).to_bytes()
                self.assertEqual(composed, direct)

    def test_compress_blocks_one_sorted_block_per_raw_id(self) -> None:
        files = _make_corpus()
        codec = _make_codec("strict", "compact")
        plan = codec.plan_blocks(files)
        raw_ids = [block_id for block_id, _lane, _payload in plan.raw_blocks]
        blocks = codec.compress_blocks(plan.raw_blocks)
        out_ids = [block.block_id for block in blocks]
        self.assertEqual(len(blocks), len(plan.raw_blocks))
        self.assertEqual(out_ids, sorted(out_ids))
        self.assertEqual(len(set(out_ids)), len(out_ids))
        self.assertEqual(sorted(out_ids), sorted(raw_ids))


@unittest.skipUnless(IS_NATIVE_AVAILABLE, "native AVX2 core unavailable")
class TestCachedBlocks(unittest.TestCase):
    MODE = "strict"
    LAYOUT = "compact"

    def test_cache_roundtrip_fingerprint_isolation_and_truncation(self) -> None:
        xon = _load_xon_iter()
        files = _make_corpus()
        codec = _make_codec(self.MODE, self.LAYOUT)
        plan = codec.plan_blocks(files)
        raw_blocks = plan.raw_blocks
        n = len(raw_blocks)
        fp = xon.fingerprint(codec)

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            blocks1, hits1, recompressed1 = xon.cached_blocks(
                codec, raw_blocks, cache_dir, fp
            )
            self.assertEqual((hits1, recompressed1), (0, n))

            blocks2, hits2, recompressed2 = xon.cached_blocks(
                codec, raw_blocks, cache_dir, fp
            )
            self.assertEqual((hits2, recompressed2), (n, 0))
            self.assertEqual(
                [_block_key(b) for b in blocks2], [_block_key(b) for b in blocks1]
            )
            ids2 = [block.block_id for block in blocks2]
            self.assertEqual(ids2, sorted(ids2))

            direct = _make_codec(self.MODE, self.LAYOUT).encode_files(files).to_bytes()
            rebuilt = (
                _make_codec(self.MODE, self.LAYOUT)
                .finish_archive(plan, blocks2)
                .to_bytes()
            )
            self.assertEqual(rebuilt, direct)

            other_fp = ("0" if fp[-1] != "0" else "1") + fp[1:]
            _blocks3, _hits3, recompressed3 = xon.cached_blocks(
                codec, raw_blocks, cache_dir, other_fp
            )
            self.assertEqual(recompressed3, n)

        # Fresh cache dir so that exactly the n files for this fingerprint exist:
        # truncating any single *.blk must invalidate exactly one block.
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            fresh1, fresh_hits1, fresh_rec1 = xon.cached_blocks(
                codec, raw_blocks, cache_dir, fp
            )
            self.assertEqual((fresh_hits1, fresh_rec1), (0, n))
            _fresh2, fresh_hits2, fresh_rec2 = xon.cached_blocks(
                codec, raw_blocks, cache_dir, fp
            )
            self.assertEqual((fresh_hits2, fresh_rec2), (n, 0))
            blk_files = sorted(cache_dir.rglob("*.blk"))
            self.assertTrue(blk_files)
            victim = blk_files[0]
            victim.write_bytes(victim.read_bytes()[:5])
            after, after_hits, after_rec = xon.cached_blocks(
                codec, raw_blocks, cache_dir, fp
            )
            self.assertEqual(after_rec, 1)
            self.assertEqual(after_hits, n - 1)
            self.assertEqual(
                [_block_key(b) for b in after], [_block_key(b) for b in fresh1]
            )


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_is_64_hex_chars_and_stable(self) -> None:
        xon = _load_xon_iter()
        codec = _make_codec("strict", "compact")
        fp1 = xon.fingerprint(codec)
        fp2 = xon.fingerprint(codec)
        self.assertEqual(len(fp1), 64)
        self.assertTrue(set(fp1) <= set("0123456789abcdef"))
        self.assertEqual(fp1, fp2)


class TestQuickSample(unittest.TestCase):
    def test_deterministic_and_order_independent(self) -> None:
        xon = _load_xon_iter()
        files = _make_corpus()
        first = xon.quick_sample(files)
        self.assertEqual(xon.quick_sample(files), first)
        shuffled = list(files)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(xon.quick_sample(shuffled), first)
        paths = [path for path, _data in first]
        self.assertEqual(paths, sorted(paths))

    def test_small_group_whole_and_every_group_represented(self) -> None:
        xon = _load_xon_iter()
        rng = random.Random(5)
        files = (
            _make_group(rng, "huge", 30, 100 * 1024)
            + _make_group(rng, "tiny", 4, 10 * 1024)
            + _make_group(rng, "mid", 12, 100 * 1024)
        )
        sample_paths = {path for path, _data in xon.quick_sample(files)}
        tiny_paths = {path for path, _data in files if path.endswith(".tiny")}
        self.assertTrue(tiny_paths <= sample_paths)
        for suffix in ("huge", "tiny", "mid"):
            self.assertTrue(
                any(path.endswith("." + suffix) for path in sample_paths),
                f"suffix group .{suffix} not represented",
            )

    def test_large_group_gets_about_six_percent(self) -> None:
        xon = _load_xon_iter()
        rng = random.Random(11)
        files = _make_group(rng, "big2", 40, 100 * 1024)
        taken = sum(
            len(data)
            for path, data in xon.quick_sample(files)
            if path.endswith(".big2")
        )
        total = 40 * 100 * 1024
        low = total * 6 // 100
        self.assertGreaterEqual(taken, low)
        self.assertLessEqual(taken, low + 100 * 1024)


if __name__ == "__main__":
    unittest.main()
