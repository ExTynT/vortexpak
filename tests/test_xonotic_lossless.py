"""Lossless Xonotic corpus: DXT split records, dedup, VXCM entropy backend, VFS reads."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vortexpak.codec import (
    STRAT_VXCM,
    TX_DXT_SPLIT,
    VXPKError,
    VortexArchive,
    VortexPakCodec,
)
from vortexpak.native import (
    IS_NATIVE_AVAILABLE,
    _py_dxt_merge,
    _py_dxt_split,
    native_dxt_merge,
    native_dxt_split,
)
from vortexpak.vfs import VortexVFS

_KB = 1024
_COPY_PATHS = ("c/t1_copy.dds", "d/text_copy.cfg")


def _make_dds(rng: random.Random, fourcc: bytes, n_blocks: int, tail_len: int) -> bytes:
    """b'DDS ' + u32le(124) + 120 random header bytes (fourCC at absolute offset 84)
    + n_blocks*bs random bytes + tail_len random bytes; bs = 8 for DXT1 else 16."""
    header = bytearray(rng.randbytes(120))
    header[76:80] = fourcc  # absolute file offset 84 (8 + 76)
    bs = 8 if fourcc == b"DXT1" else 16
    return (
        b"DDS "
        + (124).to_bytes(4, "little")
        + bytes(header)
        + rng.randbytes(n_blocks * bs)
        + rng.randbytes(tail_len)
    )


def _make_text(rng: random.Random, size: int) -> bytes:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words: List[str] = []
    total = 0
    while total < size:
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 10)))
        words.append(word)
        total += len(word) + 1
    return " ".join(words).encode("ascii")


def _build_corpus() -> List[Tuple[str, bytes]]:
    rng = random.Random(1234)
    t1 = _make_dds(rng, b"DXT1", 300, 0)
    text = _make_text(rng, 200 * _KB)
    pattern = rng.randbytes(4 * _KB)
    return [
        ("a/t1.dds", t1),
        ("a/t2.dds", _make_dds(rng, b"DXT1", 64, 3)),
        ("b/t3.dds", _make_dds(rng, b"DXT5", 200, 0)),
        ("b/t4.dds", _make_dds(rng, b"DXT3", 50, 0)),
        ("c/n.dds", _make_dds(rng, b"DX10", 64, 0)),
        ("c/t1_copy.dds", t1),
        ("d/text.txt", text),
        ("d/text_copy.cfg", text),
        ("e/rand.bin", rng.randbytes(300 * _KB)),
        ("e/empty.dat", b""),
        ("f/big.bin", pattern * (9 * 1024 * 1024 // (4 * _KB))),
    ]


_CORPUS: Optional[List[Tuple[str, bytes]]] = None
_CODECS: Dict[str, VortexPakCodec] = {}
_ARCHIVES: Dict[str, VortexArchive] = {}
_BLOBS: Dict[str, bytes] = {}


def _corpus() -> List[Tuple[str, bytes]]:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _build_corpus()
    return _CORPUS


def _get_codec(layout: str) -> VortexPakCodec:
    if layout not in _CODECS:
        _CODECS[layout] = VortexPakCodec(
            mode="strict", layout=layout, entropy_backend="auto"
        )
    return _CODECS[layout]


def _get_archive(layout: str) -> VortexArchive:
    if layout not in _ARCHIVES:
        _ARCHIVES[layout] = _get_codec(layout).encode_files(_corpus())
    return _ARCHIVES[layout]


def _get_blob(layout: str) -> bytes:
    if layout not in _BLOBS:
        _BLOBS[layout] = _get_archive(layout).to_bytes()
    return _BLOBS[layout]


class TestXonoticLossless(unittest.TestCase):
    def test_roundtrip_strict_both_layouts(self) -> None:
        for layout in ("compact", "stream"):
            with self.subTest(layout=layout):
                codec = _get_codec(layout)
                parsed = VortexArchive.from_bytes(_get_blob(layout))
                decoded = codec.decode_archive(parsed, materialize_full_rgba=True)
                for path, data in _corpus():
                    self.assertEqual(bytes(decoded[path]), data, f"{layout}:{path}")
                self.assertIs(codec.verify_archive(parsed)[0], True)

    def test_vfs_read_bytes_and_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vpak = Path(tmp) / "corpus.vpak"
            vpak.write_bytes(_get_blob("compact"))
            with VortexVFS.open(vpak) as vfs:
                for path, data in _corpus():
                    self.assertEqual(vfs.read_bytes(path), data, path)
                    self.assertEqual(vfs.stat(path)["raw_size"], len(data), path)

    def test_dxt_split_flag_set_exactly_on_dxt_dds(self) -> None:
        self.assertEqual(TX_DXT_SPLIT, 1 << 6)
        flagged = {"a/t1.dds", "a/t2.dds", "b/t3.dds", "b/t4.dds", "c/t1_copy.dds"}
        entries = {e.rel_path: e for e in _get_archive("compact").entries}
        self.assertEqual(set(entries), {p for p, _ in _corpus()})
        for path, entry in entries.items():
            has_flag = bool(entry.tx_flags & TX_DXT_SPLIT)
            self.assertEqual(has_flag, path in flagged, path)

    def test_duplicate_files_share_block_slice(self) -> None:
        entries = {e.rel_path: e for e in _get_archive("compact").entries}
        for first, copy in (
            ("a/t1.dds", "c/t1_copy.dds"),
            ("d/text.txt", "d/text_copy.cfg"),
        ):
            a, c = entries[first], entries[copy]
            self.assertEqual(
                (c.block_id, c.offset_in_block, c.length_in_block),
                (a.block_id, a.offset_in_block, a.length_in_block),
                copy,
            )
        subset = [item for item in _corpus() if item[0] not in _COPY_PATHS]
        slim = len(_get_codec("compact").encode_files(subset).to_bytes())
        self.assertLess(len(_get_blob("compact")) - slim, 1024)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native core unavailable")
    def test_vxcm_strategy_used_only_in_compact(self) -> None:
        self.assertEqual(STRAT_VXCM, 11)
        compact = [b.strategy_id for b in _get_archive("compact").blocks]
        stream = [b.strategy_id for b in _get_archive("stream").blocks]
        self.assertGreaterEqual(compact.count(STRAT_VXCM), 1)
        self.assertNotIn(STRAT_VXCM, stream)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native core unavailable")
    def test_corrupt_vxcm_payload_detected(self) -> None:
        codec = _get_codec("compact")
        parsed = VortexArchive.from_bytes(_get_blob("compact"))
        vxcm = [b for b in parsed.blocks if b.strategy_id == STRAT_VXCM]
        self.assertGreaterEqual(len(vxcm), 1)
        target = max(vxcm, key=lambda b: b.raw_size)
        payload = bytearray(target.compressed_payload)
        self.assertGreater(len(payload), 0)
        payload[len(payload) // 2] ^= 0xFF
        target.compressed_payload = bytes(payload)
        corrupt = VortexArchive.from_bytes(parsed.to_bytes())
        try:
            codec.decode_archive(corrupt, materialize_full_rgba=True)
            self.assertIs(codec.verify_archive(corrupt)[0], False)
        except VXPKError:
            pass

    def test_dxt_split_merge_roundtrip(self) -> None:
        data_by_path = dict(_corpus())
        for path in [p for p, _ in _corpus() if p.endswith(".dds")]:
            with self.subTest(path=path):
                data = data_by_path[path]
                rec = native_dxt_split(data)
                py_rec = _py_dxt_split(data)
                if path == "c/n.dds":
                    self.assertIsNone(rec)
                    self.assertIsNone(py_rec)
                    continue
                self.assertIsNotNone(rec)
                self.assertIsNotNone(py_rec)
                assert rec is not None and py_rec is not None
                self.assertEqual(rec, py_rec)
                self.assertEqual(native_dxt_merge(rec), data)
                self.assertEqual(_py_dxt_merge(py_rec), data)

    def test_py_dxt_merge_rejects_truncated_record(self) -> None:
        rec = _py_dxt_split(dict(_corpus())["a/t1.dds"])
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertGreater(len(rec), 4)
        with self.assertRaises(ValueError):
            _py_dxt_merge(rec[: len(rec) // 2])


if __name__ == "__main__":
    unittest.main()
