"""
VortexPak Native Acceleration Bridge (Rust + AVX2 SIMD)
======================================================
Loads vortexpak_native.dll for hardware-accelerated transforms and parallel decoding.
Gracefully falls back to pure Python / NumPy implementations if native DLL is unavailable.
"""

from __future__ import annotations

import ctypes
import math
import os
import sysconfig
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# Find native library. Search order: cargo target dir (dev builds), the package
# dir itself, then site-packages (where `maturin build` / cdylib binding ships it).
# Set VORTEXPAK_DISABLE_NATIVE=1 to force the pure Python / NumPy fallback
# (used by CI on hosts without AVX2, e.g. arm64 macOS runners).
_DISABLE_NATIVE = os.environ.get("VORTEXPAK_DISABLE_NATIVE", "") == "1"

_pkg_dir = Path(__file__).resolve().parent
_candidate_dirs = [
    _pkg_dir.parent / "vortexpak_native" / "target" / "release",
    _pkg_dir,
    Path(sysconfig.get_paths()["purelib"]),
]
_name_patterns = [
    "vortexpak_native.dll",
    "vortexpak_native.pyd",
    "vortexpak_native.*.pyd",
    "vortexpak_native.so",
    "vortexpak_native.*.so",
    "libvortexpak_native.so",
    "vortexpak_native.dylib",
    "libvortexpak_native.dylib",
]
_DLL_LOCATIONS: List[Path] = []
for _d in _candidate_dirs:
    for _pat in _name_patterns:
        _DLL_LOCATIONS.extend(sorted(_d.glob(_pat)))

_lib: Optional[ctypes.CDLL] = None
_native_load_error: Optional[str] = None
if _DISABLE_NATIVE:
    _native_load_error = "VORTEXPAK_DISABLE_NATIVE=1 (fallback forced)"
else:
    for loc in _DLL_LOCATIONS:
        try:
            _lib = ctypes.CDLL(str(loc))
            break
        except OSError as exc:
            _native_load_error = f"{loc.name}: {exc}"
            _lib = None

IS_NATIVE_AVAILABLE = _lib is not None
if not IS_NATIVE_AVAILABLE and _native_load_error is not None:
    warnings.warn(
        f"VortexPak native AVX2 core unavailable ({_native_load_error}); "
        "using the pure Python/NumPy fallback.",
        ImportWarning,
        stacklevel=2,
    )

if IS_NATIVE_AVAILABLE and _lib is not None:
    # 1. vxtx_is_all_opaque
    _lib.vxtx_is_all_opaque.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _lib.vxtx_is_all_opaque.restype = ctypes.c_int

    # 2. vxtx_alpha_strip_pack
    _lib.vxtx_alpha_strip_pack.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    _lib.vxtx_alpha_strip_pack.restype = ctypes.c_int

    # 3. vxtx_alpha_strip_unpack
    _lib.vxtx_alpha_strip_unpack.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    _lib.vxtx_alpha_strip_unpack.restype = ctypes.c_int

    # 4. vxtx_row_rle_decode
    _lib.vxtx_row_rle_decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.vxtx_row_rle_decode.restype = ctypes.c_int

    # 5. vxtx_planar_decode
    _lib.vxtx_planar_decode.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _lib.vxtx_planar_decode.restype = ctypes.c_int

    # 6. vxtx_planar_encode
    _lib.vxtx_planar_encode.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _lib.vxtx_planar_encode.restype = ctypes.c_int

    # 7. rdo_quantize_image
    _lib.rdo_quantize_image.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    _lib.rdo_quantize_image.restype = ctypes.c_int

    # 8. rdo_quantize_floats
    _lib.rdo_quantize_floats.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    _lib.rdo_quantize_floats.restype = ctypes.c_int

    # 9. calculate_image_psnr_c
    _lib.calculate_image_psnr_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.calculate_image_psnr_c.restype = ctypes.c_double

    # 10. calculate_float_psnr_c
    _lib.calculate_float_psnr_c.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    _lib.calculate_float_psnr_c.restype = ctypes.c_double

    # 11. vxtx_row_rle_encode
    _lib.vxtx_row_rle_encode.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.vxtx_row_rle_encode.restype = ctypes.c_int

    # 12. vpak_decompress_block_c
    _lib.vpak_decompress_block_c.argtypes = [
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint8,
    ]
    _lib.vpak_decompress_block_c.restype = ctypes.c_int64

    # 13. CMicroBlockDesc for parallel decompress
    class CMicroBlockDesc(ctypes.Structure):
        _fields_ = [
            ("block_id", ctypes.c_uint32),
            ("strategy_id", ctypes.c_uint32),
            ("raw_size", ctypes.c_size_t),
            ("comp_ptr", ctypes.c_char_p),
            ("comp_len", ctypes.c_size_t),
            ("dst_ptr", ctypes.c_char_p),
            ("dst_cap", ctypes.c_size_t),
            ("pat_ptr", ctypes.c_char_p),
            ("pat_len", ctypes.c_size_t),
            ("tok_byte", ctypes.c_uint8),
            ("out_len", ctypes.c_size_t),
            ("status", ctypes.c_int32),
        ]

    _lib.vpak_decompress_blocks_parallel_c.argtypes = [
        ctypes.POINTER(CMicroBlockDesc),
        ctypes.c_size_t,
    ]
    _lib.vpak_decompress_blocks_parallel_c.restype = ctypes.c_int

    # 13b. CTextureDecodeDesc for parallel batch VXTX texture reconstruction
    class CTextureDecodeDesc(ctypes.Structure):
        _fields_ = [
            ("src_ptr", ctypes.c_char_p),
            ("src_len", ctypes.c_size_t),
            ("dst_ptr", ctypes.c_char_p),
            ("dst_len", ctypes.c_size_t),
            ("width", ctypes.c_uint32),
            ("height", ctypes.c_uint32),
            ("orig_channels", ctypes.c_uint32),
            ("tx_flags", ctypes.c_uint32),
            ("rle_unique_rows", ctypes.c_uint32),
            ("materialize_rgba", ctypes.c_uint32),
            ("status", ctypes.c_int32),
        ]

    _lib.vxtx_decode_textures_batch_parallel_c.argtypes = [
        ctypes.POINTER(CTextureDecodeDesc),
        ctypes.c_size_t,
    ]
    _lib.vxtx_decode_textures_batch_parallel_c.restype = ctypes.c_int

    # 14. rdo_quantize_glb_c
    _lib.rdo_quantize_glb_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.rdo_quantize_glb_c.restype = ctypes.c_int64

    # 15. VFS C-ABI
    _lib.vfs_open_c.argtypes = [ctypes.c_char_p]
    _lib.vfs_open_c.restype = ctypes.c_void_p

    _lib.vfs_close_c.argtypes = [ctypes.c_void_p]
    _lib.vfs_close_c.restype = None

    _lib.vfs_get_asset_size_c.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _lib.vfs_get_asset_size_c.restype = ctypes.c_int64

    _lib.vfs_read_asset_c.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _lib.vfs_read_asset_c.restype = ctypes.c_int64

    _lib.vfs_get_assets_count_c.argtypes = [ctypes.c_void_p]
    _lib.vfs_get_assets_count_c.restype = ctypes.c_size_t

    # 16. Proprietary VXLZ-ANS C-ABI
    _lib.vxlz_compress_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.vxlz_compress_c.restype = ctypes.c_int64

    _lib.vxlz_decompress_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib.vxlz_decompress_c.restype = ctypes.c_int64

    class CVxlzCompressDesc(ctypes.Structure):
        _fields_ = [
            ("src_ptr", ctypes.c_char_p),
            ("src_len", ctypes.c_size_t),
            ("dst_ptr", ctypes.c_char_p),
            ("dst_cap", ctypes.c_size_t),
            ("out_len", ctypes.c_size_t),
            ("out_strategy", ctypes.c_uint32),
            ("allow_bypass", ctypes.c_uint32),
            ("status", ctypes.c_int32),
        ]

    _lib.vxlz_compress_blocks_parallel_c.argtypes = [
        ctypes.POINTER(CVxlzCompressDesc),
        ctypes.c_size_t,
    ]
    _lib.vxlz_compress_blocks_parallel_c.restype = ctypes.c_int


def native_is_all_opaque(rgba_bytes: bytes) -> bool:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        return bool(_lib.vxtx_is_all_opaque(rgba_bytes, len(rgba_bytes)))
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(-1, 4)
    return bool(np.all(arr[:, 3] == 255))


def native_alpha_strip_pack(rgba_bytes: bytes) -> bytes:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        out_len = len(rgba_bytes) // 4 * 3
        out_buf = np.empty(out_len, dtype=np.uint8)
        res = _lib.vxtx_alpha_strip_pack(
            rgba_bytes,
            len(rgba_bytes),
            out_buf.ctypes.data_as(ctypes.c_char_p),
        )
        if res == 0:
            return out_buf.tobytes()
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(-1, 4)
    return arr[:, :3].tobytes()


def native_alpha_strip_unpack(rgb_bytes: bytes) -> bytes:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        out_len = len(rgb_bytes) // 3 * 4
        out_buf = np.empty(out_len, dtype=np.uint8)
        res = _lib.vxtx_alpha_strip_unpack(
            rgb_bytes,
            len(rgb_bytes),
            out_buf.ctypes.data_as(ctypes.c_char_p),
        )
        if res == 0:
            return out_buf.tobytes()
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(-1, 3)
    out = np.empty((arr.shape[0], 4), dtype=np.uint8)
    out[:, :3] = arr
    out[:, 3] = 255
    return out.tobytes()


def native_row_rle_decode(
    counts: np.ndarray, unique_rows: bytes, row_bytes: int, height: int
) -> bytes:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        dst_len = height * row_bytes
        out_buf = np.empty(dst_len, dtype=np.uint8)
        c_counts = counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        res = _lib.vxtx_row_rle_decode(
            c_counts,
            len(counts),
            unique_rows,
            row_bytes,
            out_buf.ctypes.data_as(ctypes.c_char_p),
            dst_len,
        )
        if res == 0:
            return out_buf.tobytes()
    u_arr = np.frombuffer(unique_rows, dtype=np.uint8).reshape(-1, row_bytes)
    return np.repeat(u_arr, counts, axis=0).tobytes()


def native_planar_encode(
    interleaved_bytes: bytes, width: int, height: int, channels: int
) -> bytes:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        total_len = width * height * channels
        out_buf = np.empty(total_len, dtype=np.uint8)
        res = _lib.vxtx_planar_encode(
            interleaved_bytes,
            out_buf.ctypes.data_as(ctypes.c_char_p),
            width,
            height,
            channels,
        )
        if res == 0:
            return out_buf.tobytes()
    arr = np.frombuffer(interleaved_bytes, dtype=np.uint8).reshape(
        height, width, channels
    )
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))).tobytes()


def native_planar_decode(
    planar_bytes: bytes, width: int, height: int, channels: int
) -> bytes:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        total_len = width * height * channels
        out_buf = np.empty(total_len, dtype=np.uint8)
        res = _lib.vxtx_planar_decode(
            planar_bytes,
            out_buf.ctypes.data_as(ctypes.c_char_p),
            width,
            height,
            channels,
        )
        if res == 0:
            return out_buf.tobytes()
    arr = np.frombuffer(planar_bytes, dtype=np.uint8).reshape(channels, height, width)
    return np.transpose(arr, (1, 2, 0)).tobytes()


def native_decode_single_texture_vxtx(
    payload_mv: memoryview,
    flags: int,
    orig_shape: Tuple[int, ...],
    rle_unique: int,
    materialize_rgba: bool = True,
) -> Optional[np.ndarray]:
    if not IS_NATIVE_AVAILABLE or _lib is None:
        return None
    h = orig_shape[0]
    w = orig_shape[1]
    orig_c = orig_shape[2] if len(orig_shape) > 2 else 1
    out_c = orig_c if materialize_rgba else (3 if (flags & 1) else orig_c)
    out_shape = (h, w, out_c) if len(orig_shape) > 2 else (h, w)
    dst_len = h * w * out_c
    out_buf = np.empty(max(dst_len, 1), dtype=np.uint8)

    # Get raw pointer to memoryview
    src_arr = np.frombuffer(payload_mv, dtype=np.uint8)
    td = CTextureDecodeDesc()
    td.src_ptr = ctypes.cast(src_arr.ctypes.data, ctypes.c_char_p)
    td.src_len = len(src_arr)
    td.dst_ptr = ctypes.cast(out_buf.ctypes.data, ctypes.c_char_p)
    td.dst_len = dst_len
    td.width = w
    td.height = h
    td.orig_channels = orig_c
    td.tx_flags = flags
    td.rle_unique_rows = rle_unique
    td.materialize_rgba = 1 if materialize_rgba else 0
    td.status = 0

    res = _lib.vxtx_decode_textures_batch_parallel_c(ctypes.byref(td), 1)
    if res == 0 and td.status == 0:
        return out_buf[:dst_len].reshape(out_shape)
    return None


def native_row_rle_encode(
    data: bytes, height: int, row_bytes: int
) -> Optional[Tuple[np.ndarray, bytes, int]]:
    """
    AVX2 hardware-accelerated scanline deduplication.
    Returns (counts_u16_array, unique_rows_bytes, unique_count) if beneficial (< 70% unique), else None.
    """
    if IS_NATIVE_AVAILABLE and _lib is not None:
        n_unique = _lib.vxtx_row_rle_encode(data, height, row_bytes, None, None, 0)
        if 0 < n_unique < height * 0.70:
            counts = np.empty(n_unique, dtype=np.uint16)
            unique_buf = bytearray(n_unique * row_bytes)
            c_counts = counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            res = _lib.vxtx_row_rle_encode(
                data,
                height,
                row_bytes,
                c_counts,
                (ctypes.c_char * len(unique_buf)).from_buffer(unique_buf),
                n_unique,
            )
            if res > 0:
                return counts, bytes(unique_buf), n_unique

    # Fallback in NumPy
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, row_bytes)
    diff = np.any(arr[1:] != arr[:-1], axis=1)
    keep_idx = np.concatenate(([0], np.nonzero(diff)[0] + 1))
    if len(keep_idx) < height * 0.70:
        counts = np.diff(np.append(keep_idx, height)).astype(np.uint16)
        reduced = np.ascontiguousarray(arr[keep_idx])
        return counts, reduced.tobytes(), int(len(keep_idx))
    return None


def native_decompress_block(
    strategy_id: int,
    comp_payload: bytes,
    raw_size: int,
    token_pattern: bytes = b"",
    token_byte: bytes = b"",
) -> memoryview:
    """Decompress a single block using pure Rust native decoder without zero-fill or double copy."""
    if IS_NATIVE_AVAILABLE and _lib is not None:
        if raw_size == 0:
            return memoryview(b"")
        out_buf = np.empty(raw_size, dtype=np.uint8)
        t_byte = token_byte[0] if token_byte else 0
        written = _lib.vpak_decompress_block_c(
            strategy_id,
            comp_payload,
            len(comp_payload),
            raw_size,
            out_buf.ctypes.data_as(ctypes.c_char_p),
            token_pattern,
            len(token_pattern),
            t_byte,
        )
        if written >= 0:
            return memoryview(out_buf)[:written]

    # Fallback
    import lz4.frame
    import zstandard as zstd

    if strategy_id == 0:
        return memoryview(comp_payload)
    elif strategy_id == 1:
        return memoryview(lz4.frame.decompress(comp_payload))
    elif strategy_id in (2, 3, 4, 5):
        return memoryview(zstd.ZstdDecompressor().decompress(comp_payload))
    elif strategy_id in (6, 7):
        folded = zstd.ZstdDecompressor().decompress(comp_payload)
        return memoryview(folded.replace(token_byte, token_pattern))
    elif strategy_id == 8:
        return memoryview(native_vxlz_decompress(comp_payload, raw_size))
    raise ValueError(f"Unknown block strategy: {strategy_id}")


def native_decompress_blocks_parallel(blocks: list) -> List[Any]:
    """
    GIL-free zero-copy parallel decompression of all archive micro-blocks using Rust Rayon thread pool.
    Allocates a fresh independent buffer per call so returned memoryviews are never overwritten by subsequent calls.
    """
    if not blocks:
        return []

    if IS_NATIVE_AVAILABLE and _lib is not None:
        c_descs = (CMicroBlockDesc * len(blocks))()
        total_raw = sum(b.raw_size for b in blocks)
        big_buf = np.empty(max(total_raw, 1), dtype=np.uint8)
        big_ptr = big_buf.ctypes.data
        offsets = []
        cur_off = 0

        for idx, b in enumerate(blocks):
            c_desc = c_descs[idx]
            c_desc.block_id = b.block_id
            c_desc.strategy_id = b.strategy_id
            c_desc.raw_size = b.raw_size
            c_desc.comp_ptr = b.compressed_payload
            c_desc.comp_len = len(b.compressed_payload)
            c_desc.dst_ptr = ctypes.cast(big_ptr + cur_off, ctypes.c_char_p)
            c_desc.dst_cap = b.raw_size
            c_desc.pat_ptr = b.token_pattern if b.token_pattern else None
            c_desc.pat_len = len(b.token_pattern) if b.token_pattern else 0
            c_desc.tok_byte = b.token_byte[0] if b.token_byte else 0
            c_desc.out_len = 0
            c_desc.status = 0
            offsets.append((cur_off, b.raw_size))
            cur_off += b.raw_size

        res = _lib.vpak_decompress_blocks_parallel_c(c_descs, len(blocks))
        if res == 0:
            all_ok = True
            for idx in range(len(blocks)):
                if c_descs[idx].status != 0:
                    all_ok = False
                    break
            if all_ok:
                mv = memoryview(big_buf)
                return [
                    mv[off : off + c_descs[idx].out_len]
                    for idx, (off, _) in enumerate(offsets)
                ]

    # Fallback to single block decompress
    return [
        memoryview(
            native_decompress_block(
                b.strategy_id,
                b.compressed_payload,
                b.raw_size,
                b.token_pattern,
                b.token_byte,
            )
        )
        for b in blocks
    ]


def native_decode_archive_parallel(
    blocks: list,
    entries: list,
    materialize_full_rgba: bool = True,
) -> Optional[dict]:
    """
    Full end-to-end parallel archive decompression + batch parallel AVX2 VXTX texture materialization.
    Allocates fresh per-call output buffers so returned arrays/memoryviews remain completely independent across calls.
    """
    if not IS_NATIVE_AVAILABLE or _lib is None or not blocks:
        return None

    c_descs = (CMicroBlockDesc * len(blocks))()
    total_raw = sum(b.raw_size for b in blocks)
    big_buf = np.empty(max(total_raw, 1), dtype=np.uint8)
    big_ptr = big_buf.ctypes.data
    block_offsets = [0] * len(blocks)
    cur_off = 0

    for idx, b in enumerate(blocks):
        c_desc = c_descs[idx]
        c_desc.block_id = b.block_id
        c_desc.strategy_id = b.strategy_id
        c_desc.raw_size = b.raw_size
        c_desc.comp_ptr = b.compressed_payload
        c_desc.comp_len = len(b.compressed_payload)
        c_desc.dst_ptr = ctypes.cast(big_ptr + cur_off, ctypes.c_char_p)
        c_desc.dst_cap = b.raw_size
        c_desc.pat_ptr = b.token_pattern if b.token_pattern else None
        c_desc.pat_len = len(b.token_pattern) if b.token_pattern else 0
        c_desc.tok_byte = b.token_byte[0] if b.token_byte else 0
        c_desc.out_len = 0
        c_desc.status = 0
        block_offsets[b.block_id] = cur_off
        cur_off += b.raw_size

    res = _lib.vpak_decompress_blocks_parallel_c(c_descs, len(blocks))
    if res != 0:
        return None
    for idx in range(len(blocks)):
        if c_descs[idx].status != 0:
            return None

    mv_big = memoryview(big_buf)
    results = {}

    if materialize_full_rgba:
        tex_entries = [
            e for e in entries if e.is_texture_tensor and not (e.tx_flags & 8)
        ]  # TX_BCN tensors decode via the Python bridge, not the VXTX batch
    else:
        tex_entries = []

    if tex_entries:
        n_tex = len(tex_entries)
        c_tex_descs = (CTextureDecodeDesc * n_tex)()
        total_tex_bytes = 0
        tex_meta = []
        for e in tex_entries:
            sh = e.tx_shape
            h = sh[0]
            w = sh[1]
            c = sh[2] if len(sh) > 2 else 1
            n_b = h * w * c
            tex_meta.append((e, h, w, c, total_tex_bytes, n_b))
            total_tex_bytes += n_b

        tex_out_buf = np.empty(max(total_tex_bytes, 1), dtype=np.uint8)
        tex_out_ptr = tex_out_buf.ctypes.data

        for idx, (e, h, w, c, t_off, n_b) in enumerate(tex_meta):
            td = c_tex_descs[idx]
            src_start = block_offsets[e.block_id] + e.offset_in_block
            td.src_ptr = ctypes.cast(big_ptr + src_start, ctypes.c_char_p)
            td.src_len = e.length_in_block
            td.dst_ptr = ctypes.cast(tex_out_ptr + t_off, ctypes.c_char_p)
            td.dst_len = n_b
            td.width = w
            td.height = h
            td.orig_channels = c
            td.tx_flags = e.tx_flags
            td.rle_unique_rows = e.tx_rle_unique_rows
            td.materialize_rgba = 1
            td.status = 0

        tres = _lib.vxtx_decode_textures_batch_parallel_c(c_tex_descs, n_tex)
        if tres != 0:
            return None
        for idx, (e, h, w, c, t_off, n_b) in enumerate(tex_meta):
            if c_tex_descs[idx].status != 0:
                return None
            results[e.rel_path] = tex_out_buf[t_off : t_off + n_b].reshape(e.tx_shape)

    for e in entries:
        if e.is_texture_tensor:
            if e.tx_flags & 8:  # TX_BCN: VXTX-GPU assets decode via Python bridge
                from .bcn import GPU_FORMAT_IDS, decode_bcn, gpu_level0

                fmt = GPU_FORMAT_IDS.get(e.tx_mode)
                if fmt is None:
                    return None
                if e.stripe_spans:
                    payload = b"".join(
                        mv_big[block_offsets[b] + o : block_offsets[b] + o + l]
                        for b, o, l in e.stripe_spans
                    )
                else:
                    src_start = block_offsets[e.block_id] + e.offset_in_block
                    payload = bytes(mv_big[src_start : src_start + e.length_in_block])
                if materialize_full_rgba:
                    level0 = gpu_level0(
                        payload, e.tx_shape[1], e.tx_shape[0], fmt, bool(e.tx_flags & 16)
                    )
                    rgba = decode_bcn(level0, e.tx_shape[1], e.tx_shape[0], fmt)
                    results[e.rel_path] = np.frombuffer(rgba, dtype=np.uint8).reshape(
                        e.tx_shape
                    )
                else:
                    results[e.rel_path] = np.frombuffer(payload, dtype=np.uint8)
                continue
            if materialize_full_rgba:
                continue
            # Unmaterialized zero-copy tensor view
            src_start = block_offsets[e.block_id] + e.offset_in_block
            payload = mv_big[src_start : src_start + e.length_in_block]
            work_shape = (
                (e.tx_shape[0], e.tx_shape[1], 3) if (e.tx_flags & 1) else e.tx_shape
            )
            if e.tx_flags & 2:  # TX_ROW_RLE
                counts_bytes = e.tx_rle_unique_rows * 2
                row_shape = (e.tx_rle_unique_rows,) + work_shape[1:]
                results[e.rel_path] = np.frombuffer(
                    payload[counts_bytes:], dtype=np.uint8
                ).reshape(row_shape)
            elif e.tx_flags & 4:  # TX_PLANAR_CHW
                planar_shape = (work_shape[2], work_shape[0], work_shape[1])
                results[e.rel_path] = np.frombuffer(payload, dtype=np.uint8).reshape(
                    planar_shape
                )
            else:
                results[e.rel_path] = np.frombuffer(payload, dtype=np.uint8).reshape(
                    work_shape
                )
        elif e.stripe_spans:
            # Consecutive stripes in `big_buf` are already contiguous in memory!
            b0, off0, _ = e.stripe_spans[0]
            total_len = sum(ln for _, _, ln in e.stripe_spans)
            start_off = block_offsets[b0] + off0
            results[e.rel_path] = mv_big[  # type: ignore[assignment]
                start_off : start_off + total_len
            ]
        elif e.aux_block_id >= 0:
            return None
        else:
            start_off = block_offsets[e.block_id] + e.offset_in_block
            results[e.rel_path] = mv_big[start_off : start_off + e.length_in_block]

    return results


def native_quantize_glb(glb_bytes: bytes, drop_bits: int = 12) -> bytes:
    """Format-aware glTF 2.0 binary vertex quantization using pure Rust."""
    if IS_NATIVE_AVAILABLE and _lib is not None:
        out_buf = np.empty(len(glb_bytes) + 4096, dtype=np.uint8)
        written = _lib.rdo_quantize_glb_c(
            glb_bytes,
            len(glb_bytes),
            drop_bits,
            out_buf.ctypes.data_as(ctypes.c_char_p),
            len(out_buf),
        )
        if written > 0:
            return out_buf[:written].tobytes()
    return glb_bytes


def native_rdo_quantize_image(
    src_bytes: bytes,
    width: int,
    height: int,
    channels: int,
    level: str = "balanced",
) -> bytes:
    lvl_code = 2 if level == "balanced" else (1 if level == "subtle" else 0)
    if lvl_code == 0:
        return src_bytes
    if IS_NATIVE_AVAILABLE and _lib is not None:
        total = width * height * channels
        out_buf = np.empty(total, dtype=np.uint8)
        res = _lib.rdo_quantize_image(
            src_bytes,
            width,
            height,
            channels,
            lvl_code,
            out_buf.ctypes.data_as(ctypes.c_char_p),
        )
        if res == 0:
            return out_buf.tobytes()

    # Fallback in Python/Numpy
    arr = np.frombuffer(src_bytes, dtype=np.uint8).copy()
    rem4 = arr % 4
    # Deadzone rounding to multiple of 4 or 2
    arr = np.where(rem4 == 1, arr - 1, arr)
    arr = np.where((rem4 == 3) & (arr < 255), arr + 1, arr)
    if channels == 4:
        orig = np.frombuffer(src_bytes, dtype=np.uint8)
        alpha_orig = orig[3::4]
        # Strictly preserve 0 and 255 in alpha channel!
        arr[3::4] = np.where(
            (alpha_orig == 255) | (alpha_orig == 0), alpha_orig, arr[3::4]
        )
    return arr.tobytes()


def native_calculate_image_psnr(orig: bytes, quant: bytes) -> float:
    if IS_NATIVE_AVAILABLE and _lib is not None:
        return float(_lib.calculate_image_psnr_c(orig, quant, len(orig)))
    a = np.frombuffer(orig, dtype=np.uint8).astype(np.float64)
    b = np.frombuffer(quant, dtype=np.uint8).astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def native_vxlz_compress(raw_bytes: bytes) -> bytes:
    """Compress a byte buffer using the proprietary Rust VXLZ-ANS engine."""
    if not raw_bytes:
        return b"\x00"
    if IS_NATIVE_AVAILABLE and _lib is not None:
        dst_cap = len(raw_bytes) + 64
        out_buf = np.empty(dst_cap, dtype=np.uint8)
        written = _lib.vxlz_compress_c(
            raw_bytes,
            len(raw_bytes),
            out_buf.ctypes.data_as(ctypes.c_char_p),
            dst_cap,
        )
        if written > 0:
            return out_buf[:written].tobytes()
    return b"\x00" + raw_bytes


def native_vxlz_decompress(comp_bytes: bytes, raw_size: int) -> bytes:
    """Decompress a proprietary VXLZ-ANS buffer into exact raw_size bytes."""
    if raw_size == 0:
        return b""
    if IS_NATIVE_AVAILABLE and _lib is not None:
        out_buf = np.empty(raw_size, dtype=np.uint8)
        written = _lib.vxlz_decompress_c(
            comp_bytes,
            len(comp_bytes),
            out_buf.ctypes.data_as(ctypes.c_char_p),
            raw_size,
        )
        if written == raw_size:
            return out_buf.tobytes()
    raise RuntimeError("VXLZ decompression failed")


def native_vxlz_compress_blocks_parallel(
    raw_blocks: List[Tuple[int, str, bytes]],
    allow_bypass: bool = True,
) -> Optional[List[Tuple[int, str, int, int, bytes]]]:
    """
    Compress all micro-blocks concurrently in Rust Rayon using proprietary VXLZ-ANS.
    Returns list of (block_id, lane_name, strategy_id, raw_size, compressed_payload).
    """
    if not IS_NATIVE_AVAILABLE or _lib is None or not raw_blocks:
        return None

    n = len(raw_blocks)
    c_descs = (CVxlzCompressDesc * n)()
    total_cap = sum(len(b) + 64 for _, _, b in raw_blocks)
    out_arena = np.empty(max(total_cap, 1), dtype=np.uint8)
    out_ptr = out_arena.ctypes.data
    offsets = []
    cur_off = 0

    for idx, (bid, lname, raw_b) in enumerate(raw_blocks):
        cap = len(raw_b) + 64
        d = c_descs[idx]
        d.src_ptr = raw_b
        d.src_len = len(raw_b)
        d.dst_ptr = ctypes.cast(out_ptr + cur_off, ctypes.c_char_p)
        d.dst_cap = cap
        d.out_len = 0
        d.out_strategy = 0
        d.allow_bypass = 1 if allow_bypass else 0
        d.status = 0
        offsets.append(cur_off)
        cur_off += cap

    res = _lib.vxlz_compress_blocks_parallel_c(c_descs, n)
    if res != 0:
        return None

    mv = memoryview(out_arena)
    out_list = []
    for idx, (bid, lname, raw_b) in enumerate(raw_blocks):
        d = c_descs[idx]
        if d.status != 0:
            return None
        off = offsets[idx]
        comp_bytes = bytes(mv[off : off + d.out_len])
        out_list.append((bid, lname, int(d.out_strategy), len(raw_b), comp_bytes))
    return out_list
