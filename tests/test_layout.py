"""Block layout: layout="stream" bounds the bytes decoded per random read."""

import random
import tempfile
import unittest
from pathlib import Path

from vortexpak import VortexArchive, VortexPakCodec, VortexVFS
from vortexpak.codec import STREAM_BLOCK_BYTES, STREAM_ISOLATE_BYTES, TARGET_BLOCK_BYTES


def _corpus():
    rng = random.Random(5)
    files = [("big.bin", rng.randbytes(300_000)), ("huge.bin", rng.randbytes(3 * TARGET_BLOCK_BYTES // 2))]
    words = [b"local", b"function", b"end", b"return", b"minetest", b"node", b"pos"]
    for i in range(400):
        text = b" ".join(rng.choice(words) for _ in range(rng.randint(100, 600)))
        files.append((f"mods/m{i % 17}/f{i}.lua", text))
    files.append(("empty.txt", b""))
    return files


class TestStreamLayout(unittest.TestCase):
    def _blocks_by_entry(self, arch):
        raw = {b.block_id: b.raw_size for b in arch.blocks}
        users = {}
        for e in arch.entries:
            bids = [s[0] for s in e.stripe_spans] if e.stripe_spans else [e.block_id]
            for bid in bids:
                users.setdefault(bid, []).append(e.rel_path)
        return raw, users

    def test_small_reads_never_decode_unrelated_bulk(self):
        files = _corpus()
        arch = VortexPakCodec(mode="strict", layout="stream").encode_files(files)
        raw, users = self._blocks_by_entry(arch)
        sizes = dict(files)
        for e in arch.entries:
            bids = [s[0] for s in e.stripe_spans] if e.stripe_spans else [e.block_id]
            if len(sizes[e.rel_path]) >= STREAM_ISOLATE_BYTES:
                self.assertTrue(all(users[b] == [e.rel_path] for b in bids), e.rel_path)
            elif sizes[e.rel_path]:
                self.assertLessEqual(raw[e.block_id], STREAM_BLOCK_BYTES + STREAM_ISOLATE_BYTES)
        huge = next(e for e in arch.entries if e.rel_path == "huge.bin")
        self.assertEqual(len(huge.stripe_spans), 2)

    def test_both_layouts_restore_every_byte(self):
        files = _corpus()
        for layout in ("stream", "compact"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as td:
                codec = VortexPakCodec(mode="strict", layout=layout)
                blob = codec.encode_files(files).to_bytes()
                decoded = codec.decode_archive(VortexArchive.from_bytes(blob), materialize_full_rgba=True)
                self.assertEqual({k: bytes(v) for k, v in decoded.items()}, dict(files))
                path = Path(td) / "a.vpak"
                path.write_bytes(blob)
                with VortexVFS.open(path) as vfs:
                    for name, data in files:
                        self.assertEqual(vfs.read_bytes(name), data)

    def test_unknown_layout_rejected(self):
        with self.assertRaises(ValueError):
            VortexPakCodec(layout="fast")


if __name__ == "__main__":
    unittest.main()
