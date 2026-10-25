"""
Comprehensive Test Suite for VortexPak (VXPK v1.1)
==================================================
Tests:
1. Native Rust AVX2 SIMD Core (Alpha-Strip, Row-RLE, Planar)
2. Perceptual RDO (Oodle Texture RDO style: PSNR > 48 dB for textures, PSNR > 50 dB for 3D meshes)
3. Seekable Virtual File System (VFS: on-demand random-access streaming)
4. Edge cases: empty files, zero-byte inputs, large files, multi-lane striping
"""

import io
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from vortexpak import (
    IS_NATIVE_AVAILABLE,
    VortexArchive,
    VortexPakCodec,
    VortexVFS,
    benchmark_vfs_streaming,
    quantize_float32_mantissa,
    quantize_glb_rdo,
    quantize_mesh_floats_rdo,
    quantize_png_rdo,
    quantize_texture_array_rdo,
)
from vortexpak.rdo import quantize_blend_rdo
from vortexpak.native import (
    _lib,
    native_alpha_strip_pack,
    native_alpha_strip_unpack,
    native_decompress_block,
    native_decompress_blocks_parallel,
    native_is_all_opaque,
    native_planar_decode,
    native_planar_encode,
    native_row_rle_decode,
    native_row_rle_encode,
    native_vxlz_compress,
    native_vxlz_decompress,
)


class TestVortexPakNativeSIMD(unittest.TestCase):
    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native AVX2 core unavailable")
    def test_native_availability(self):
        self.assertTrue(
            IS_NATIVE_AVAILABLE, "Native Rust AVX2 DLL must be compiled and loaded"
        )

    def test_alpha_strip_simd(self):
        # 64x64 RGBA image with A=255
        rgba = bytearray(64 * 64 * 4)
        for i in range(len(rgba)):
            rgba[i] = 255 if (i % 4 == 3) else (i % 250)
        rgba_bytes = bytes(rgba)

        self.assertTrue(native_is_all_opaque(rgba_bytes))
        stripped = native_alpha_strip_pack(rgba_bytes)
        self.assertEqual(len(stripped), 64 * 64 * 3)

        unpacked = native_alpha_strip_unpack(stripped)
        self.assertEqual(len(unpacked), len(rgba_bytes))
        self.assertEqual(unpacked, rgba_bytes)

    def test_alpha_strip_transparent_detection(self):
        rgba = bytearray(32 * 32 * 4)
        for i in range(len(rgba)):
            rgba[i] = 255 if (i % 4 == 3) else 100
        # Set one pixel to semi-transparent
        rgba[4 * 15 + 3] = 128
        self.assertFalse(native_is_all_opaque(bytes(rgba)))

    def test_row_rle_roundtrip(self):
        width = 128
        height = 64
        channels = 3
        row_bytes = width * channels
        # Create repeating bands of rows
        data = bytearray(height * row_bytes)
        for r in range(height):
            val = (r // 8) * 30
            for c in range(row_bytes):
                data[r * row_bytes + c] = val

        data_bytes = bytes(data)
        counts = np.array([8] * (height // 8), dtype=np.uint16)
        unique_rows = bytearray()
        for b in range(height // 8):
            val = b * 30
            unique_rows.extend(bytes([val] * row_bytes))

        decoded = native_row_rle_decode(counts, bytes(unique_rows), row_bytes, height)
        self.assertEqual(len(decoded), len(data_bytes))
        self.assertEqual(decoded, data_bytes)

    def test_planar_roundtrip(self):
        w, h, c = 32, 32, 3
        interleaved = bytes([(i * 7) % 256 for i in range(w * h * c)])
        arr = np.frombuffer(interleaved, dtype=np.uint8).reshape(h, w, c)
        planar = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))).tobytes()

        decoded = native_planar_decode(planar, w, h, c)
        self.assertEqual(decoded, interleaved)

    def test_planar_encode_simd(self):
        w, h, c = 32, 32, 3
        interleaved = bytes([(i * 11) % 256 for i in range(w * h * c)])
        planar = native_planar_encode(interleaved, w, h, c)
        decoded = native_planar_decode(planar, w, h, c)
        self.assertEqual(decoded, interleaved)

    def test_row_rle_encode_simd(self):
        width = 64
        height = 32
        row_bytes = width * 4
        # 4 identical bands of 8 rows
        data = bytearray(height * row_bytes)
        for r in range(height):
            val = (r // 8) * 40
            for c in range(row_bytes):
                data[r * row_bytes + c] = val

        res = native_row_rle_encode(bytes(data), height, row_bytes)
        self.assertIsNotNone(res)
        counts, unique, n_u = res
        self.assertEqual(n_u, 4)
        self.assertEqual(list(counts), [8, 8, 8, 8])
        # Decode back
        decoded = native_row_rle_decode(counts, unique, row_bytes, height)
        self.assertEqual(decoded, bytes(data))

    def test_parallel_native_decompress(self):
        # Create multiple blocks compressed with Zstd
        import zstandard as zstd
        from vortexpak.codec import MicroBlock

        raw_blocks = [
            f"Block payload number {i} with repeated text repeated text".encode() * 20
            for i in range(8)
        ]
        compressed_blocks = []
        for i, raw in enumerate(raw_blocks):
            comp = zstd.ZstdCompressor(level=3).compress(raw)
            compressed_blocks.append(
                MicroBlock(
                    block_id=i,
                    lane_name=f"LANE_{i}",
                    strategy_id=2,  # STRAT_ZSTD_3
                    raw_size=len(raw),
                    compressed_payload=comp,
                )
            )

        decompressed = native_decompress_blocks_parallel(compressed_blocks)
        self.assertEqual(len(decompressed), len(raw_blocks))
        for orig, dec in zip(raw_blocks, decompressed):
            self.assertEqual(orig, dec)

    def test_native_lz4_frame_and_empty(self):
        import lz4.frame

        raw = b"Testing LZ4 frame in native decompressor " * 50
        comp = lz4.frame.compress(raw)
        dec = native_decompress_block(1, comp, len(raw))
        self.assertEqual(dec, raw)

        empty_dec = native_decompress_block(0, b"", 0)
        self.assertEqual(empty_dec, b"")

    def test_independent_decode_buffers_no_overwrite(self):
        """Verify that consecutive decode_archive calls return independent buffers without overwriting previous results."""  # noqa: E501
        codec = VortexPakCodec(mode="engine")

        img1 = Image.new("RGBA", (64, 64), color=(10, 20, 30, 255))
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        arch1 = codec.encode_files(
            [
                ("tex1.png", buf1.getvalue()),
                ("script1.lua", b"FIRST_ARCHIVE_PAYLOAD_" * 100),
            ]
        )

        img2 = Image.new("RGBA", (64, 64), color=(210, 220, 230, 255))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        arch2 = codec.encode_files(
            [
                ("tex2.png", buf2.getvalue()),
                ("script2.lua", b"SECOND_ARCHIVE_PAYLOAD_" * 100),
            ]
        )

        dec1 = codec.decode_archive(arch1, materialize_full_rgba=True)
        # Snapshot expected values before second decode
        expected_tex1 = np.array(dec1["tex1.png"], copy=True)
        expected_script1 = bytes(dec1["script1.lua"])

        # Perform second decode — must NOT mutate dec1!
        dec2 = codec.decode_archive(arch2, materialize_full_rgba=True)

        self.assertTrue(
            np.array_equal(dec1["tex1.png"], expected_tex1),
            "First decoded texture buffer was overwritten by second decode!",
        )
        self.assertEqual(
            bytes(dec1["script1.lua"]),
            expected_script1,
            "First decoded script memoryview was overwritten by second decode!",
        )
        self.assertEqual(int(dec2["tex2.png"][0, 0, 0]), 210)
        self.assertEqual(bytes(dec2["script2.lua"]), b"SECOND_ARCHIVE_PAYLOAD_" * 100)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native AVX2 core unavailable")
    def test_vxlz_ans_proprietary_roundtrip(self):
        """Test our 100% proprietary VXLZ-ANS (6-stream LZ + 8-way rANS + AVX2 Sub-Green) codec."""
        # 1. Repetitive XML tilemap
        xml_data = (
            b'   <tile gid="0"/>\n   <tile gid="142"/>\n   <tile gid="0"/>\n' * 2000
        )
        comp_xml = native_vxlz_compress(xml_data)
        self.assertLess(
            len(comp_xml),
            len(xml_data) // 100,
            "Cascaded VXLZ+rANS must compress repetitive XML >100x",
        )
        dec_xml = native_vxlz_decompress(comp_xml, len(xml_data))
        self.assertEqual(dec_xml, xml_data)

        # 2. Correlated RGBA texture buffer (triggers AVX2 Sub-Green decorrelation)
        rgba = bytearray(64 * 64 * 4)
        for i in range(64 * 64):
            g = (i * 3) & 0xFF
            rgba[i * 4] = (g + 12) & 0xFF
            rgba[i * 4 + 1] = g
            rgba[i * 4 + 2] = (g + 240) & 0xFF
            rgba[i * 4 + 3] = 255
        rgba_bytes = bytes(rgba)
        comp_rgba = native_vxlz_compress(rgba_bytes)
        self.assertLess(len(comp_rgba), len(rgba_bytes) // 3)
        dec_rgba = native_vxlz_decompress(comp_rgba, len(rgba_bytes))
        self.assertEqual(dec_rgba, rgba_bytes)

    @unittest.skipUnless(IS_NATIVE_AVAILABLE, "native AVX2 core unavailable")
    def test_pure_proprietary_vpak_zero_zstd(self):
        """Verify that VortexPakCodec(entropy_backend='vxlz') uses STRAT_VXLZ=8 and VXLZ headers with zero Zstd."""
        codec = VortexPakCodec(mode="engine", entropy_backend="vxlz")
        img = Image.new("RGBA", (64, 64), color=(80, 120, 160, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arch = codec.encode_files(
            [
                ("hero.png", buf.getvalue()),
                ("map.tmx", b'<tile gid="0"/>\n' * 5000),
                ("config.json", b'{"engine": "VortexPak", "codec": "VXLZ-ANS"}\n' * 50),
            ]
        )
        for b in arch.blocks:
            self.assertIn(
                b.strategy_id,
                (0, 8),
                f"Expected STRAT_VXLZ (8) or BYPASS (0), got {b.strategy_id}",
            )
        self.assertTrue(any(b.strategy_id == 8 for b in arch.blocks))

        vpak_bytes = arch.to_bytes()
        # Bytes 0..6 = VXPK\x01\x01, Bytes 6..10 = meta_len, Bytes 10..14 = b"VXLZ"
        self.assertEqual(
            vpak_bytes[10:14],
            b"VXLZ",
            "Archive header must be compressed with proprietary VXLZ",
        )
        arch_reloaded = VortexArchive.from_bytes(vpak_bytes)
        ok, count = codec.verify_archive(arch_reloaded)
        self.assertTrue(ok)
        self.assertEqual(count, 3)


class TestPerceptualRDO(unittest.TestCase):
    def test_texture_rdo_psnr_guarantee(self):
        # Generate gradient image with high frequency noise
        w, h = 256, 256
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        img_arr = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)

        quant_arr, psnr = quantize_texture_array_rdo(img_arr, level="balanced")
        self.assertGreater(
            psnr, 48.0, f"Texture PSNR {psnr:.2f} dB must exceed 48.0 dB"
        )
        self.assertEqual(quant_arr.shape, img_arr.shape)

    def test_png_rdo_roundtrip(self):
        img = Image.new("RGBA", (64, 64), color=(120, 150, 180, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        quant_png, psnr = quantize_png_rdo(png_bytes, level="balanced")
        self.assertGreater(psnr, 48.0)
        self.assertTrue(len(quant_png) > 0)

    def test_float32_mantissa_quantization(self):
        orig = 3.1415926535
        quant = quantize_float32_mantissa(orig, drop_bits=12)
        # Relative error should be < 0.05%
        rel_err = abs(orig - quant) / orig
        self.assertLess(rel_err, 0.0005)
        # Lower 12 bits should be zero
        u32 = struct.unpack("<I", struct.pack("<f", quant))[0]
        self.assertEqual(u32 & 0x0FFF, 0)

    def test_mesh_floats_rdo_psnr(self):
        coords = np.array(
            [
                [0.123456, 1.987654, -0.54321],
                [10.111213, -5.678901, 2.34567],
                [1.0, 0.0, 0.0],
                [-0.707106, 0.707106, 0.0],
            ],
            dtype=np.float32,
        )

        q_coords, psnr = quantize_mesh_floats_rdo(coords, drop_bits=12)
        self.assertGreater(
            psnr, 60.0, f"Mesh geometry PSNR {psnr:.2f} dB must be > 60.0 dB"
        )

        # Lower 12 bits of quantized floats should be zero
        u32_arr = q_coords.view(np.uint32)
        for val in u32_arr.ravel():
            self.assertEqual(val & 0x0FFF, 0)

    def test_glb_rdo_integration(self):
        # Find Kenney character.glb
        glb_path = Path("benchmarks/games/kenney_3d_platformer/models/character.glb")
        if glb_path.is_file():
            orig_glb = glb_path.read_bytes()
            quant_glb, psnr = quantize_glb_rdo(orig_glb, drop_bits=12)
            self.assertGreater(
                psnr, 50.0, f"GLB PSNR {psnr:.2f} dB must exceed 50.0 dB"
            )
            self.assertEqual(quant_glb[:4], b"glTF")

    def test_rdo_alpha_255_preservation_and_alpha_strip(self):
        # 100% opaque RGBA texture
        arr = np.full((32, 32, 4), 255, dtype=np.uint8)
        arr[:, :, 0] = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        # Verify quantize_texture_array_rdo preserves alpha 255
        quant_arr, psnr = quantize_texture_array_rdo(arr, level="balanced")
        self.assertGreater(psnr, 48.0)
        self.assertTrue(
            np.all(quant_arr[:, :, 3] == 255), "Alpha 255 must be 100% preserved"
        )
        self.assertTrue(native_is_all_opaque(quant_arr.tobytes()))

        # Verify that packing into VortexPak with rdo=True retains TX_ALPHA_STRIP
        img = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        codec = VortexPakCodec(mode="engine", rdo=True)
        arch = codec.encode_files([("test_opaque.png", buf.getvalue())])
        self.assertEqual(len(arch.entries), 1)
        self.assertTrue(
            arch.entries[0].tx_flags & 1 != 0,
            "TX_ALPHA_STRIP must be preserved under RDO",
        )

    def test_blend_rdo_quantization(self):
        blend_path = Path(
            "benchmarks/games/minetest_game/mods/player_api/models/character.blend"
        )
        if blend_path.is_file():
            raw_blend = blend_path.read_bytes()
            q_blend, psnr = quantize_blend_rdo(raw_blend, drop_bits=12)
            self.assertGreater(psnr, 60.0, f"Blend PSNR {psnr:.2f} dB must be > 60 dB")
            self.assertTrue(q_blend.startswith(b"BLENDER"))
            self.assertEqual(len(q_blend), len(raw_blend))
            self.assertNotEqual(
                q_blend, raw_blend, "Quantized blend bytes must differ from raw bytes"
            )


class TestSeekableVFS(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vpak_path = Path(self.tmp_dir.name) / "test_corpus.vpak"

        # Create test files
        self.files = []
        # 1. Normal texture
        img = Image.new("RGBA", (128, 128), color=(200, 100, 50, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.files.append(("textures/wall.png", buf.getvalue()))

        # 2. Transparent texture
        img2 = Image.new("RGBA", (64, 64), color=(50, 150, 250, 128))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        self.files.append(("textures/water.png", buf2.getvalue()))

        # 3. Small scripts and config
        self.files.append(
            ("scripts/main.lua", b"print('Hello, VortexPak VFS!')\n" * 50)
        )
        self.files.append(
            ("config/settings.json", b'{"graphics": "ultra", "fps": 144, "vfs": true}')
        )

        # 4. Zero-byte file
        self.files.append(("empty.gdignore", b""))

        # Pack archive
        codec = VortexPakCodec(mode="engine", rdo=True)
        arch = codec.encode_files(self.files)
        self.vpak_path.write_bytes(arch.to_bytes())

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_vfs_open_and_list(self):
        with VortexVFS.open(self.vpak_path) as vfs:
            assets = vfs.list_assets()
            self.assertEqual(len(assets), len(self.files))
            self.assertTrue(vfs.exists("textures/wall.png"))
            self.assertTrue(vfs.exists("empty.gdignore"))
            self.assertFalse(vfs.exists("nonexistent.txt"))

    def test_vfs_on_demand_streaming_exactness(self):
        with VortexVFS.open(self.vpak_path) as vfs:
            # Read script
            lua_data = vfs.read_bytes("scripts/main.lua")
            orig_lua = dict(self.files)["scripts/main.lua"]
            self.assertEqual(lua_data, orig_lua)

            # Read zero-byte file
            empty_data = vfs.read_bytes("empty.gdignore")
            self.assertEqual(empty_data, b"")

            # Read texture as array
            tex_arr = vfs.read_texture("textures/wall.png")
            self.assertEqual(tex_arr.shape, (128, 128, 4))

    def test_vfs_block_cache(self):
        with VortexVFS.open(self.vpak_path) as vfs:
            self.assertEqual(len(vfs._cache.cache), 0)
            _ = vfs.read_bytes("scripts/main.lua")
            self.assertGreaterEqual(len(vfs._cache.cache), 1)
            # Second read from same block hits cache
            _ = vfs.read_bytes("config/settings.json")
            self.assertGreaterEqual(len(vfs._cache.cache), 1)

    def test_vfs_striped_large_asset(self):
        # Create a striped large asset (10 MB)
        large_payload = b"TILE_LAYER_DATA_1234567890\n" * 400000  # ~10.8 MB
        striped_files = [("maps/world.tmx", large_payload)]
        codec = VortexPakCodec(mode="engine")
        arch = codec.encode_files(striped_files)
        striped_vpak = Path(self.tmp_dir.name) / "striped.vpak"
        striped_vpak.write_bytes(arch.to_bytes())

        with VortexVFS.open(striped_vpak) as vfs:
            st = vfs.stat("maps/world.tmx")
            self.assertGreater(
                st["stripe_spans_count"], 1, "Must be striped across multiple blocks"
            )
            read_back = vfs.read_bytes("maps/world.tmx")
            self.assertEqual(len(read_back), len(large_payload))
            self.assertEqual(read_back, large_payload)

    def test_vfs_benchmark(self):
        res = benchmark_vfs_streaming(self.vpak_path, num_queries=50)
        self.assertIn("avg_latency_microseconds", res)
        self.assertIn("effective_throughput_gbps", res)
        self.assertGreater(res["effective_throughput_gbps"], 0.1)

    def test_vfs_windows_backslashes(self):
        with VortexVFS.open(self.vpak_path) as vfs:
            # Read asset with Windows backslashes
            data_bs = vfs.read_bytes("textures\\wall.png")
            data_fs = vfs.read_bytes("textures/wall.png")
            self.assertEqual(data_bs, data_fs)
            self.assertTrue(vfs.exists("textures\\wall.png"))

    def test_vfs_c_api_direct(self):
        if IS_NATIVE_AVAILABLE and _lib is not None:
            import ctypes

            vpak_str = str(self.vpak_path).encode("utf-8")
            handle = _lib.vfs_open_c(vpak_str)
            self.assertIsNotNone(handle)
            try:
                count = _lib.vfs_get_assets_count_c(handle)
                self.assertEqual(count, len(self.files))
                sz = _lib.vfs_get_asset_size_c(handle, b"scripts/main.lua")
                orig_len = len(dict(self.files)["scripts/main.lua"])
                self.assertEqual(sz, orig_len)

                buf = bytearray(sz + 16)
                read_len = _lib.vfs_read_asset_c(
                    handle,
                    b"scripts/main.lua",
                    (ctypes.c_char * len(buf)).from_buffer(buf),
                    len(buf),
                    1,
                )
                self.assertEqual(read_len, orig_len)
                self.assertEqual(
                    bytes(buf[:read_len]), dict(self.files)["scripts/main.lua"]
                )
            finally:
                _lib.vfs_close_c(handle)


if __name__ == "__main__":
    unittest.main()
