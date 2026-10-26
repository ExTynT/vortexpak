"""
VXJ JPEG Record Container Tests
===============================
VXJ records (native JPEG repackaging, 'VXJ1' magic) inside VortexPak archives:
compact vs stream layout flag policy, byte-exact roundtrips through
to_bytes/from_bytes and the VFS, native record roundtrip, and corruption
detection when a VXJ-carrying block is damaged.
"""

import io
import site
import sys
import tempfile
import unittest
from pathlib import Path

# This host ships an empty `tests` package in the user site which shadows the
# repo's tests/ directory (PEP 420), so the suite is normally launched with
# PYTHONNOUSERSITE=1. That hides the user site entirely -- re-append it AFTER
# `tests` has resolved so third-party wheels living there still import.
if site.getusersitepackages() not in sys.path:
    sys.path.append(site.getusersitepackages())

import numpy as np
from PIL import Image

from vortexpak.codec import (
    TX_JPEG_VXJ,
    VortexArchive,
    VortexPakCodec,
    VXPKError,
)
from vortexpak.native import (
    IS_NATIVE_AVAILABLE,
    native_vxj_decode,
    native_vxj_encode,
)
from vortexpak.vfs import VortexVFS


def _gradient_image(width, height, mode):
    """Smooth gradient + N(0, 10) noise + random rectangles, uint8 (seed 5)."""
    rng = np.random.default_rng(5)
    gx = np.arange(width, dtype=np.float32) * (255.0 / max(width - 1, 1))
    gy = np.arange(height, dtype=np.float32) * (255.0 / max(height - 1, 1))
    if mode == "L":
        arr = np.add.outer(gy * 0.5, gx * 0.5)
    else:
        arr = np.dstack(
            [
                np.tile(gx, (height, 1)),
                np.tile(gy.reshape(height, 1), (1, width)),
                np.add.outer(gy * 0.5, gx * 0.5),
            ]
        )
    arr = arr + rng.normal(0.0, 10.0, size=arr.shape)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    for _ in range(4):
        x0 = int(rng.integers(0, width - 4))
        y0 = int(rng.integers(0, height - 4))
        rw = int(rng.integers(4, width - x0))
        rh = int(rng.integers(4, height - y0))
        if arr.ndim == 2:
            arr[y0 : y0 + rh, x0 : x0 + rw] = int(rng.integers(0, 256))
        else:
            arr[y0 : y0 + rh, x0 : x0 + rw] = (
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
            )
    return Image.fromarray(arr, mode)


def _jpeg_bytes(img, **kwargs):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def _random_words(total_bytes):
    rng = np.random.default_rng(5)
    vocab = ["vortex", "pak", "jpeg", "stream", "compact", "record", "block", "entry"]
    chunks = []
    remaining = total_bytes
    while remaining > 0:
        piece = vocab[int(rng.integers(0, len(vocab)))] + " "
        chunks.append(piece)
        remaining -= len(piece)
    return "".join(chunks)[:total_bytes].encode("ascii")


def _as_bytes(data):
    if isinstance(data, bytes):
        return data
    return bytes(data)


class VXJContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = _jpeg_bytes(_gradient_image(96, 64, "RGB"), quality=90)
        b = _jpeg_bytes(
            _gradient_image(80, 80, "RGB"),
            quality=75,
            subsampling=2,
            optimize=True,
        )
        g = _jpeg_bytes(_gradient_image(64, 40, "L"), quality=85)
        g_big = _jpeg_bytes(_gradient_image(256, 192, "L"), quality=85)
        p = _jpeg_bytes(_gradient_image(96, 64, "RGB"), progressive=True)
        cls.corpus = [
            ("j/a.jpg", a),
            ("j/b.jpg", b),
            ("j/g.jpg", g),
            ("j/g_big.jpg", g_big),
            ("j/p.jpg", p),
            ("j/fake.jpg", b"not a jpeg at all" * 3),
            ("j/a_copy.jpg", a),
            ("t/x.txt", _random_words(50 * 1024)),
        ]

    def _entry_for(self, arch, rel_path):
        for entry in arch.entries:
            if entry.rel_path == rel_path:
                return entry
        self.fail("no archive entry for %r" % rel_path)

    def _block_for(self, arch, block_id):
        for block in arch.blocks:
            if block.block_id == block_id:
                return block
        self.fail("no archive block with id %r" % block_id)

    def test_compact_and_stream_roundtrip_byte_exact(self):
        for layout in ("compact", "stream"):
            with self.subTest(layout=layout):
                codec = VortexPakCodec(mode="strict", layout=layout)
                arch = codec.encode_files(list(self.corpus))
                parsed = VortexArchive.from_bytes(arch.to_bytes())
                ok, _ = codec.verify_archive(parsed)
                self.assertTrue(ok, "verify_archive must report ok for %s" % layout)
                decoded = codec.decode_archive(parsed, materialize_full_rgba=True)
                for rel, blob in self.corpus:
                    self.assertIn(rel, decoded)
                    self.assertEqual(_as_bytes(decoded[rel]), blob)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native VXJ core unavailable")
    def test_tx_jpeg_vxj_flags(self):
        blobs = dict(self.corpus)
        codec = VortexPakCodec(mode="strict", layout="compact")
        arch = codec.encode_files(list(self.corpus))
        flags = {e.rel_path: e.tx_flags for e in arch.entries}
        self.assertTrue(flags["j/a.jpg"] & TX_JPEG_VXJ, "baseline a.jpg must use VXJ")
        rec_g = native_vxj_encode(blobs["j/g.jpg"])
        should_use_vxj = rec_g is not None and len(rec_g) < len(blobs["j/g.jpg"])
        self.assertEqual(
            bool(flags["j/g.jpg"] & TX_JPEG_VXJ),
            should_use_vxj,
            "g.jpg flag must follow the smaller-record policy",
        )
        self.assertTrue(
            flags["j/a_copy.jpg"] & TX_JPEG_VXJ,
            "duplicate of a.jpg must use VXJ as well",
        )
        rec_b = native_vxj_encode(blobs["j/b.jpg"])
        should_use_vxj = rec_b is not None and len(rec_b) < len(blobs["j/b.jpg"])
        self.assertEqual(
            bool(flags["j/b.jpg"] & TX_JPEG_VXJ),
            should_use_vxj,
            "b.jpg flag must follow the smaller-record policy",
        )
        for never in ("j/p.jpg", "j/fake.jpg", "t/x.txt"):
            self.assertFalse(
                flags[never] & TX_JPEG_VXJ, "%s must never use VXJ" % never
            )
        stream_codec = VortexPakCodec(mode="strict", layout="stream")
        stream_arch = stream_codec.encode_files(list(self.corpus))
        for entry in stream_arch.entries:
            self.assertFalse(
                entry.tx_flags & TX_JPEG_VXJ,
                "stream layout must never set TX_JPEG_VXJ (got on %s)" % entry.rel_path,
            )

    def test_vfs_roundtrip_compact(self):
        codec = VortexPakCodec(mode="strict", layout="compact")
        arch = codec.encode_files(list(self.corpus))
        with tempfile.TemporaryDirectory() as tmp:
            vpak_path = Path(tmp) / "vxj_corpus.vpak"
            vpak_path.write_bytes(arch.to_bytes())
            vfs = VortexVFS.open(vpak_path)
            try:
                for rel, blob in self.corpus:
                    self.assertEqual(_as_bytes(vfs.read_bytes(rel)), blob)
            finally:
                vfs.close()

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native VXJ core unavailable")
    def test_native_vxj_record_roundtrip(self):
        blobs = dict(self.corpus)
        rec = native_vxj_encode(blobs["j/a.jpg"])
        self.assertIsNotNone(rec, "a baseline JPEG must produce a VXJ record")
        self.assertTrue(rec.startswith(b"VXJ1"))
        self.assertEqual(
            native_vxj_decode(rec, len(blobs["j/a.jpg"])), blobs["j/a.jpg"]
        )
        self.assertIsNone(native_vxj_encode(blobs["j/fake.jpg"]))

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native VXJ core unavailable")
    def test_native_vxj_grayscale_record_roundtrip(self):
        raw = dict(self.corpus)["j/g_big.jpg"]
        rec = native_vxj_encode(raw)
        self.assertIsNotNone(rec, "a grayscale JPEG must produce a VXJ record")
        self.assertTrue(rec.startswith(b"VXJ1"))
        self.assertEqual(native_vxj_decode(rec, len(raw)), raw)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native VXJ core unavailable")
    def test_compact_corruption_detected(self):
        blobs = dict(self.corpus)
        codec = VortexPakCodec(mode="strict", layout="compact")
        arch = codec.encode_files(list(self.corpus))
        entry = self._entry_for(arch, "j/a.jpg")
        block = self._block_for(arch, entry.block_id)
        payload = _as_bytes(block.compressed_payload)
        data = bytearray(arch.to_bytes())
        base = data.find(payload)
        if base < 0:
            self.fail("serialized archive does not contain the block payload verbatim")
        if block.strategy_id == 0:
            inner = entry.offset_in_block + entry.length_in_block // 2
        else:
            inner = len(payload) // 2
        inner = min(inner, len(payload) - 1)
        data[base + inner] ^= 0xFF
        try:
            parsed = VortexArchive.from_bytes(bytes(data))
            decoded = codec.decode_archive(parsed, materialize_full_rgba=True)
        except VXPKError:
            return  # corruption surfaced as a container error: acceptable
        ok, _ = codec.verify_archive(parsed)
        if not ok:
            return  # corruption flagged by verification: acceptable
        self.assertIn("j/a.jpg", decoded)
        self.assertNotEqual(
            _as_bytes(decoded["j/a.jpg"]),
            blobs["j/a.jpg"],
            "corrupted block decoded back to the original bytes undetected",
        )


if __name__ == "__main__":
    unittest.main()
