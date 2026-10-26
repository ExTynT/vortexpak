"""VXTX-GPU: proprietary BC1/BC4/BC5 block-compression bridge.

Native path: AVX2-targeted Rust core (Rayon-parallel) via ctypes. Fallback:
pure NumPy reference implementation (slower, same wire format) so archives
stay portable on hosts without the native core.

Wire format is the standard GPU block layout consumed directly by
Direct3D/Vulkan/Metal, so decoded payloads upload without CPU-side
recompression:
  BC1: 8 B per 4x4 block  (RGB, 1-bit alpha via 3-color+black mode)
  BC4: 8 B per 4x4 block  (single channel)
  BC5: 16 B per 4x4 block (two BC4 planes, plane-major RG)

Stored GPU textures may carry a mip chain: levels 0..n-1 concatenated,
largest first, each a whole number of 4x4 blocks (see mip_dims).
"""

from __future__ import annotations

import ctypes
import struct
from typing import List, Tuple

import numpy as np

from .native import _lib, IS_NATIVE_AVAILABLE

BCN_BC1 = 1
BCN_BC4 = 4
BCN_BC5 = 5
BCN_BC7 = 7
BCN_BCX = 11

GPU_FORMAT_IDS = {
    "BC1": BCN_BC1,
    "BC4": BCN_BC4,
    "BC5": BCN_BC5,
    "BC7": BCN_BC7,
    "BCX": BCN_BCX,
}
GPU_CHANNELS = {BCN_BC1: 4, BCN_BC4: 1, BCN_BC5: 2, BCN_BC7: 4, BCN_BCX: 4}
GPU_BLOCK_BYTES = {BCN_BC1: 8, BCN_BC4: 8, BCN_BC5: 16, BCN_BC7: 16, BCN_BCX: 16}

QUALITY_IDS = {"fast": 0, "standard": 1, "max": 2}

if IS_NATIVE_AVAILABLE and _lib is not None:
    assert _lib is not None  # narrowed for mypy module-level bindings
    for _fn in ("bcn_encode_bc7_c", "bcn_decode_bc7_c"):
        _f = getattr(_lib, _fn)
        _f.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        _f.restype = ctypes.c_int64
    _rdo = _lib.bcn_encode_bc1_rdo_c
    _rdo.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _rdo.restype = ctypes.c_int64
    _bcx_enc = _lib.bcn_encode_bcx_c
    _bcx_enc.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _bcx_enc.restype = ctypes.c_int64
    _bcx_dec = _lib.bcn_decode_bcx_c
    _bcx_dec.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _bcx_dec.restype = ctypes.c_int64
    _dec = _lib.bcn_decode_c
    _dec.argtypes = [
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _dec.restype = ctypes.c_int64
    _enc = _lib.bcn_encode_ex_c
    _enc.argtypes = [
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _enc.restype = ctypes.c_int64


def _validate(w: int, h: int, length: int, channels: int) -> None:
    if w <= 0 or h <= 0 or w % 4 or h % 4:
        raise ValueError(f"BCN dimensions must be positive multiples of 4: {w}x{h}")
    if length != w * h * channels:
        raise ValueError(
            f"input length {length} does not match {w}x{h}x{channels} = {w * h * channels}"
        )


def _native_ready() -> bool:
    return IS_NATIVE_AVAILABLE and _lib is not None


def encode_bcn(
    src: bytes,
    w: int,
    h: int,
    fmt: int,
    quality: str = "max",
    rdo: float = 0.0,
) -> bytes:
    """Encode raw pixels into GPU block data. rdo (BC1 only, native only)
    enables closed-loop rate-distortion feedback in [0, 1]."""
    """Encode raw pixels (interleaved, row-major) into GPU block data.

    quality: "fast" (trial-fit), "standard" (cluster fit + 1 LSQ pass),
    "max" (multi-axis seeding + alternating LSQ + 565 snap polish).
    Wire format is identical across tiers."""
    if fmt not in GPU_CHANNELS:
        raise ValueError(f"unknown BCN format id {fmt}")
    if not isinstance(src, (bytes, bytearray, memoryview)):
        if hasattr(src, "tobytes"):
            src = src.tobytes()
        else:
            raise ValueError("BCN source must be bytes-like")
    src = bytes(src)
    _validate(w, h, len(src), GPU_CHANNELS[fmt])
    if quality not in QUALITY_IDS:
        raise ValueError(f"unknown BCN quality {quality!r}")
    if quality not in QUALITY_IDS:
        raise ValueError(f"unknown BCN quality {quality!r}")
    q = QUALITY_IDS[quality]
    if fmt == BCN_BC7 and _native_ready():
        out = np.empty(_blocks_len(w, h, fmt), dtype=np.uint8)
        n = _lib.bcn_encode_bc7_c(
            src,
            len(src),
            w,
            h,
            out.ctypes.data_as(ctypes.c_char_p),
            out.size,
        )
        if n >= 0:
            return out[:n].tobytes()
        raise ValueError("BC7 encode failed")
    if fmt == BCN_BCX and _native_ready():
        nb = (w // 4) * (h // 4)
        out = np.empty(4 + (nb + 7) // 8 + nb * 16, dtype=np.uint8)
        n = _bcx_enc(
            q, 15000, src, len(src), w, h, out.ctypes.data_as(ctypes.c_char_p), out.size
        )
        if n >= 0:
            return out[:n].tobytes()
        raise ValueError("BCX encode failed")
    if fmt == BCN_BC1 and rdo > 0.0 and _native_ready():
        out = np.empty(_blocks_len(w, h, fmt), dtype=np.uint8)
        n = _rdo(
            q,
            int(round(rdo * 100)),
            2,
            src,
            len(src),
            w,
            h,
            out.ctypes.data_as(ctypes.c_char_p),
            out.size,
        )
        if n >= 0:
            return out[:n].tobytes()
    if _native_ready():
        out = np.empty(_blocks_len(w, h, fmt), dtype=np.uint8)
        n = _enc(
            fmt,
            q,
            src,
            len(src),
            w,
            h,
            out.ctypes.data_as(ctypes.c_char_p),
            out.size,
        )
        if n >= 0:
            return out[:n].tobytes()
    return _fallback_encode(src, w, h, fmt)


def decode_bcn(blocks: bytes, w: int, h: int, fmt: int) -> bytes:
    """Decode GPU block data back into raw pixels."""
    if fmt not in GPU_CHANNELS:
        raise ValueError(f"unknown BCN format id {fmt}")
    if not isinstance(blocks, (bytes, bytearray, memoryview)):
        if hasattr(blocks, "tobytes"):
            blocks = blocks.tobytes()
        else:
            raise ValueError("BCN payload must be bytes-like")
    blocks = bytes(blocks)
    if fmt != BCN_BCX and len(blocks) != _blocks_len(w, h, fmt):
        raise ValueError(
            f"BCN payload length {len(blocks)} != expected {_blocks_len(w, h, fmt)} for {w}x{h}"
        )
    if fmt == BCN_BCX and _native_ready():
        out = np.empty(w * h * 4, dtype=np.uint8)
        n = _bcx_dec(
            blocks, len(blocks), w, h, out.ctypes.data_as(ctypes.c_char_p), out.size
        )
        if n >= 0:
            return out[:n].tobytes()
    if fmt == BCN_BC7 and _native_ready():
        out = np.empty(w * h * GPU_CHANNELS[fmt], dtype=np.uint8)
        n = _lib.bcn_decode_bc7_c(
            blocks,
            len(blocks),
            w,
            h,
            out.ctypes.data_as(ctypes.c_char_p),
            out.size,
        )
        if n >= 0:
            return out[:n].tobytes()
    if _native_ready():
        out = np.empty(w * h * GPU_CHANNELS[fmt], dtype=np.uint8)
        n = _lib.bcn_decode_c(
            fmt,
            blocks,
            len(blocks),
            w,
            h,
            out.ctypes.data_as(ctypes.c_char_p),
            out.size,
        )
        if n >= 0:
            return out[:n].tobytes()
    return _fallback_decode(blocks, w, h, fmt)


def _blocks_len(w: int, h: int, fmt: int) -> int:
    return (w // 4) * (h // 4) * GPU_BLOCK_BYTES[fmt]


# ====================== NumPy reference implementations ======================

_W = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _pack565(r: int, g: int, b: int) -> int:
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _expand565(c: int) -> np.ndarray:
    r5 = (c >> 11) & 31
    g6 = (c >> 5) & 63
    b5 = c & 31
    return np.array(
        [(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)],
        dtype=np.uint8,
    )


_BC7_W4 = np.array(
    [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int64
)


def _bc4_palette(e0: int, e1: int) -> np.ndarray:
    """D3D/Vulkan BC4 order: 0 = e0, 1 = e1, 2..7 = interpolants (or 0/255)."""
    a, b = int(e0), int(e1)
    if a > b:
        mids = [((7 - i) * a + i * b) // 7 for i in range(1, 7)]
        return np.array([a, b, *mids], dtype=np.uint8)
    mids = [((5 - i) * a + i * b) // 5 for i in range(1, 5)]
    return np.array([a, b, *mids, 0, 255], dtype=np.uint8)


def _bc7_palette(e0, e1) -> np.ndarray:
    w = _BC7_W4[:, None]
    a = np.asarray(e0, np.int64)[None, :]
    b = np.asarray(e1, np.int64)[None, :]
    return ((64 - w) * a + w * b + 32) >> 6


def _encode_bc1_np(rgba: bytes, w: int, h: int) -> bytes:
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    nb = (w // 4) * (h // 4)
    out = np.zeros((nb, 8), dtype=np.uint8)
    bi = 0
    for by in range(h // 4):
        for bx in range(w // 4):
            blk = arr[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4].reshape(16, 4)
            if (blk[:, 3] < 255).any():
                out[bi] = _encode_bc1_alpha_block(blk)
            else:
                out[bi] = _encode_bc1_color_block(blk)
            bi += 1
    return out.tobytes()


def _u8(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8)


def _encode_bc1_color_block(blk: np.ndarray) -> np.ndarray:
    out = np.zeros(8, dtype=np.uint8)
    rgb = blk[:, :3]
    mn = rgb.min(axis=0)
    mx = rgb.max(axis=0)
    y = rgb.astype(np.float64) @ _W
    cands = [
        (mn, mx),
        (mx, mn),
        (rgb[int(y.argmin())], rgb[int(y.argmax())]),
        (rgb[int(y.argmax())], rgb[int(y.argmin())]),
    ]
    for shift in (3, 4):
        i0 = mn + ((mx - mn) >> shift)
        i1 = mx - ((mx - mn) >> shift)
        if (i0 != i1).any():
            cands += [(i0, i1), (i1, i0)]
    best_err = np.inf
    best = None
    for ea, eb in cands:
        c0 = _pack565(int(ea[0]), int(ea[1]), int(ea[2]))
        c1 = _pack565(int(eb[0]), int(eb[1]), int(eb[2]))
        if c0 <= c1:
            continue
        p0 = _expand565(c0).astype(np.int64)
        p1 = _expand565(c1).astype(np.int64)
        pal = np.stack([p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3])
        d = rgb.astype(np.int64)[:, None, :] - pal[None, :, :]
        err = (d.astype(np.float64) ** 2 * _W).sum(-1)
        total = float(err.sum())
        if total < best_err:
            best_err = total
            idx = err.argmin(1).astype(np.uint32)
            best = (c0, c1, sum(int(v) << (2 * k) for k, v in enumerate(idx)))
    if best is None:
        c0 = _pack565(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]))
        c1 = max(c0 - 1, 0) if c0 > 0 else (c0 + 1)
        if c1 > c0:
            c0, c1 = c1, c0
        out[0:2] = _u8(c0.to_bytes(2, "little"))
        out[2:4] = _u8(c1.to_bytes(2, "little"))
        return out
    c0, c1, indices = best
    out[0:2] = _u8(c0.to_bytes(2, "little"))
    out[2:4] = _u8(c1.to_bytes(2, "little"))
    out[4:8] = _u8(indices.to_bytes(4, "little"))
    return out


def _encode_bc1_alpha_block(blk: np.ndarray) -> np.ndarray:
    out = np.zeros(8, dtype=np.uint8)
    opaque = blk[blk[:, 3] >= 128]
    if len(opaque) == 0:
        out[4:8] = 0xFF
        return out
    d = opaque[:, :3].astype(np.int64)
    dists = ((d[:, None, :] - d[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(dists.argmax()), dists.shape)
    c0 = _pack565(*[int(v) for v in opaque[i][:3]])
    c1 = _pack565(*[int(v) for v in opaque[j][:3]])
    if c0 > c1:
        c0, c1 = c1, c0
    p0 = _expand565(c0).astype(np.int64)
    p1 = _expand565(c1).astype(np.int64)
    pal = np.stack([p0, p1, (p0 + p1) // 2])
    idx = np.full(16, 3, dtype=np.uint32)
    op = blk[:, 3] >= 128
    dd = blk[op, :3].astype(np.int64)[:, None, :] - pal[None, :, :]
    err = (dd.astype(np.float64) ** 2 * _W).sum(-1)
    idx[op] = err.argmin(1).astype(np.uint32)
    out[0:2] = _u8(c0.to_bytes(2, "little"))
    out[2:4] = _u8(c1.to_bytes(2, "little"))
    out[4:8] = _u8(
        int(sum(int(v) << (2 * k) for k, v in enumerate(idx))).to_bytes(4, "little")
    )
    return out


def _decode_bc1_np(blocks: bytes, w: int, h: int) -> bytes:
    b = np.frombuffer(blocks, dtype=np.uint8).reshape(-1, 8)
    c0 = b[:, 0].astype(np.uint16) | (b[:, 1].astype(np.uint16) << 8)
    c1 = b[:, 2].astype(np.uint16) | (b[:, 3].astype(np.uint16) << 8)

    def exp565(c: np.ndarray) -> np.ndarray:
        r5 = (c >> 11) & 31
        g6 = (c >> 5) & 63
        b5 = c & 31
        cols = np.stack(
            [(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)],
            axis=1,
        ).astype(np.int64)
        return cols

    p0 = exp565(c0)
    p1 = exp565(c1)
    three = c0 <= c1
    p2 = np.where(three[:, None], (p0 + p1) // 2, (2 * p0 + p1) // 3)
    p3 = np.where(three[:, None], 0, (p0 + 2 * p1) // 3)
    pal = np.stack([p0, p1, p2, p3], axis=1).astype(np.uint8)  # (nb, 4, 3)
    idx_u32 = np.frombuffer(blocks[: len(b) * 8], dtype=np.uint8).reshape(-1, 8)[:, 4:8]
    idx = (
        idx_u32[:, 0].astype(np.uint32)
        | (idx_u32[:, 1].astype(np.uint32) << 8)
        | (idx_u32[:, 2].astype(np.uint32) << 16)
        | (idx_u32[:, 3].astype(np.uint32) << 24)
    )
    bw, bh = w // 4, h // 4
    dst = np.zeros((h, w, 4), dtype=np.uint8)
    three_g = three.reshape(bh, bw)
    for row in range(4):
        for col in range(4):
            pid = ((idx >> (2 * (row * 4 + col))) & 3).reshape(bh, bw)
            yy = np.arange(bh)[:, None] * 4 + row
            xx = np.arange(bw)[None, :] * 4 + col
            dst[yy, xx, :3] = pal.reshape(bh, bw, 4, 3)[
                np.arange(bh)[:, None], np.arange(bw)[None, :], pid
            ]
            transparent = three_g & (pid == 3)
            dst[yy, xx, 3] = np.where(transparent, 0, 255)
    return dst.tobytes()


def _encode_bc4_np(data: bytes, w: int, h: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
    out = np.zeros(((w // 4) * (h // 4), 8), dtype=np.uint8)
    bi = 0
    for by in range(h // 4):
        for bx in range(w // 4):
            vals = arr[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4].reshape(16)
            mn, mx = int(vals.min()), int(vals.max())
            cands = [(mn, mx), (mx, mn)]
            if mn < mx:
                cands += [(mn + 1, mx), (mn, mx - 1), (mx, mn + 1), (mx - 1, mn)]
            best_err = None
            best = (mn, mx, 0)
            for e0, e1 in cands:
                pal = _bc4_palette(e0, e1).astype(np.int64)
                d = np.abs(vals.astype(np.int64)[:, None] - pal[None, :])
                idx = d.argmin(1).astype(np.uint64)
                err = int((d.min(1).astype(np.int64) ** 2).sum())
                packed = sum(int(v) << (3 * k) for k, v in enumerate(idx))
                if best_err is None or err < best_err:
                    best_err = err
                    best = (e0, e1, packed)
            e0, e1, packed = best
            out[bi, 0] = e0
            out[bi, 1] = e1
            out[bi, 2:8] = np.frombuffer(packed.to_bytes(8, "little"), dtype=np.uint8)[
                :6
            ]
            bi += 1
    return out.tobytes()


def _decode_bc4_np(blocks: bytes, w: int, h: int) -> bytes:
    b = np.frombuffer(blocks, dtype=np.uint8).reshape(-1, 8)
    bw = w // 4
    dst = np.zeros((h, w), dtype=np.uint8)
    for bi in range(b.shape[0]):
        pal = _bc4_palette(int(b[bi, 0]), int(b[bi, 1]))
        idx = int.from_bytes(b[bi, 2:8].tobytes(), "little")
        by, bx = bi // bw, bi % bw
        for i in range(16):
            dst[by * 4 + i // 4, bx * 4 + i % 4] = pal[(idx >> (3 * i)) & 7]
    return dst.tobytes()


def _encode_bc7_np(src: bytes, w: int, h: int) -> bytes:
    """Spec BC7 mode 6 (reference path: bbox endpoints, exact index search)."""
    arr = np.frombuffer(src, dtype=np.uint8).reshape(h, w, 4).astype(np.int64)
    out = np.zeros(((w // 4) * (h // 4), 16), dtype=np.uint8)
    bi = 0
    for by in range(h // 4):
        for bx in range(w // 4):
            blk = arr[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4].reshape(16, 4)
            best = None
            # Shared RGBA P-bit: opaque blocks need P=1 (alpha 255), fully
            # transparent ones P=0 (alpha 0); mirrors the native encoder.
            if blk[:, 3].min() == 255:
                pbits = [(1, 1)]
            elif blk[:, 3].max() == 0:
                pbits = [(0, 0)]
            else:
                pbits = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for pa, pb in pbits:
                q0 = (
                    np.clip(np.round((blk.min(0) - pa) / 2), 0, 127).astype(
                        np.int64
                    )
                    * 2
                    + pa
                )
                q1 = (
                    np.clip(np.round((blk.max(0) - pb) / 2), 0, 127).astype(
                        np.int64
                    )
                    * 2
                    + pb
                )
                d = ((blk[:, None, :] - _bc7_palette(q0, q1)[None]) ** 2).sum(-1)
                err = int(d.min(1).sum())
                if best is None or err < best[0]:
                    best = (err, q0, q1, d.argmin(1))
            assert best is not None
            _, q0, q1, idx = best
            if idx[0] >= 8:  # anchor rule: pixel 0 index MSB must be 0
                q0, q1, idx = q1, q0, 15 - idx
            bits = np.zeros(128, dtype=np.uint8)

            def put(pos: int, n: int, v: int) -> None:
                for k in range(n):
                    bits[pos + k] = (v >> k) & 1

            put(0, 7, 0x40)
            for c in range(4):
                put(7 + c * 14, 7, int(q0[c]) >> 1)
                put(14 + c * 14, 7, int(q1[c]) >> 1)
            bits[63] = int(q0[0]) & 1
            bits[64] = int(q1[0]) & 1
            pos = 65
            for i, v in enumerate(idx):
                n = 3 if i == 0 else 4
                put(pos, n, int(v))
                pos += n
            out[bi] = np.packbits(bits, bitorder="little")
            bi += 1
    return out.tobytes()


def _decode_bc7_np(blocks: bytes, w: int, h: int) -> bytes:
    """Spec BC7 mode 6 decoder; other modes decode to transparent black."""
    bl = np.frombuffer(blocks, dtype=np.uint8).reshape(-1, 16)
    dst = np.zeros((h, w, 4), dtype=np.uint8)
    bw = w // 4
    for bi in range(bl.shape[0]):
        if int(bl[bi, 0]) & 0x7F != 0x40:
            continue
        bits = np.unpackbits(bl[bi], bitorder="little")

        def getv(pos: int, n: int) -> int:
            return sum(int(bits[pos + k]) << k for k in range(n))

        p0, p1 = int(bits[63]), int(bits[64])
        e0 = [(getv(7 + c * 14, 7) << 1) | p0 for c in range(4)]
        e1 = [(getv(14 + c * 14, 7) << 1) | p1 for c in range(4)]
        pal = _bc7_palette(e0, e1).astype(np.uint8)
        by, bx = divmod(bi, bw)
        pos = 65
        for i in range(16):
            n = 3 if i == 0 else 4
            dst[by * 4 + i // 4, bx * 4 + i % 4] = pal[getv(pos, n)]
            pos += n
    return dst.tobytes()


def _encode_bcx_np(src: bytes, w: int, h: int) -> bytes:
    """Fallback BCX: all blocks as BC1 (no upgrade without native core)."""
    bc1 = _encode_bc1_np(src, w, h)
    nb = (w // 4) * (h // 4)
    bm = (nb + 7) // 8
    return struct.pack("<I", bm) + bytes(bm) + bc1


def _decode_bcx_np(blocks: bytes, w: int, h: int) -> bytes:
    bw, bh = w // 4, h // 4
    bm = struct.unpack_from("<I", blocks, 0)[0]
    if bm != (bw * bh + 7) // 8 or len(blocks) < 4 + bm:
        raise ValueError("BCX bitmap size mismatch")
    bitmap = blocks[4 : 4 + bm]
    pos = 4 + bm
    dst = np.zeros((h, w, 4), dtype=np.uint8)
    for by in range(bh):
        for bx in range(bw):
            bi = by * bw + bx
            is7 = (bitmap[bi // 8] >> (bi % 8)) & 1
            bl = 16 if is7 else 8
            if pos + bl > len(blocks):
                raise ValueError("BCX payload truncated")
            chunk = blocks[pos : pos + bl]
            if is7:
                px = np.frombuffer(_decode_bc7_np(chunk, 4, 4), np.uint8).reshape(
                    4, 4, 4
                )
            else:
                px = np.frombuffer(_decode_bc1_np(chunk, 4, 4), np.uint8).reshape(
                    4, 4, 4
                )
            dst[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4] = px
            pos += bl
    if pos != len(blocks):
        raise ValueError(f"BCX payload length {len(blocks)} != expected {pos}")
    return dst.tobytes()


def _fallback_encode(src: bytes, w: int, h: int, fmt: int) -> bytes:
    if fmt == BCN_BC1:
        return _encode_bc1_np(src, w, h)
    if fmt == BCN_BC4:
        return _encode_bc4_np(src, w, h)
    if fmt == BCN_BC7:
        return _encode_bc7_np(src, w, h)
    if fmt == BCN_BCX:
        return _encode_bcx_np(src, w, h)
    arr = np.frombuffer(src, dtype=np.uint8).reshape(h, w, 2)
    r = np.frombuffer(_encode_bc4_np(arr[..., 0].tobytes(), w, h), np.uint8).reshape(
        -1, 8
    )
    g = np.frombuffer(_encode_bc4_np(arr[..., 1].tobytes(), w, h), np.uint8).reshape(
        -1, 8
    )
    return np.concatenate([r, g], axis=1).tobytes()  # spec: per-block R then G


def _fallback_decode(blocks: bytes, w: int, h: int, fmt: int) -> bytes:
    if fmt == BCN_BC1:
        return _decode_bc1_np(blocks, w, h)
    if fmt == BCN_BC4:
        return _decode_bc4_np(blocks, w, h)
    if fmt == BCN_BC7:
        return _decode_bc7_np(blocks, w, h)
    if fmt == BCN_BCX:
        return _decode_bcx_np(blocks, w, h)
    pairs = np.frombuffer(blocks, dtype=np.uint8).reshape(-1, 16)
    r = np.frombuffer(_decode_bc4_np(pairs[:, :8].tobytes(), w, h), dtype=np.uint8)
    g = np.frombuffer(_decode_bc4_np(pairs[:, 8:].tobytes(), w, h), dtype=np.uint8)
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = r.reshape(h, w)
    out[..., 1] = g.reshape(h, w)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Mip chains and DDS export (engine interop)
# ---------------------------------------------------------------------------

# DXGI formats (UNORM, UNORM_SRGB) for the GPU-native block formats.
DXGI_FORMATS = {BCN_BC1: (71, 72), BCN_BC4: (80, 80), BCN_BC5: (83, 83), BCN_BC7: (98, 99)}


def mip_dims(w: int, h: int) -> List[Tuple[int, int]]:
    """Mip level sizes stored for a w x h texture: halve while the next level
    stays a whole number of 4x4 blocks. Level 0 is always included."""
    dims = [(w, h)]
    while w % 8 == 0 and h % 8 == 0:
        w, h = w // 2, h // 2
        dims.append((w, h))
    return dims


def level_bytes(w: int, h: int, fmt: int) -> int:
    return (w // 4) * (h // 4) * GPU_BLOCK_BYTES[fmt]


def chain_bytes(w: int, h: int, fmt: int, levels: int) -> int:
    return sum(level_bytes(lw, lh, fmt) for lw, lh in mip_dims(w, h)[:levels])


def gpu_level0(payload: bytes, w: int, h: int, fmt: int, has_mips: bool) -> bytes:
    """Level-0 blocks of a stored GPU payload; validates the full chain length."""
    if not has_mips or fmt == BCN_BCX:
        return payload
    expected = chain_bytes(w, h, fmt, len(mip_dims(w, h)))
    if len(payload) != expected:
        raise ValueError(f"mip chain is {len(payload)} B, expected {expected} B for {w}x{h}")
    return payload[: level_bytes(w, h, fmt)]


_SRGB_TO_LINEAR = np.where(
    np.arange(256) / 255.0 <= 0.04045,
    np.arange(256) / 255.0 / 12.92,
    ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4,
)
# Linear-light values (x255) of the sRGB code midpoints k + 0.5: a linear value
# rounds to sRGB code k exactly when it lies between midpoints k - 1 and k.
_SRGB_LINEAR_255 = (_SRGB_TO_LINEAR * 255.0).astype(np.float32)
_mid = (np.arange(255) + 0.5) / 255.0
_SRGB_MIDPOINTS_255 = (
    np.where(_mid <= 0.04045, _mid / 12.92, ((_mid + 0.055) / 1.055) ** 2.4) * 255.0
).astype(np.float32)
# Bucketed exact lookup: buckets of 1/16 are narrower than the smallest gap
# between midpoints (0.077), so each holds at most one; the code of x is the
# midpoint count at the bucket start plus one comparison.
_BUCKETS = 16
_bucket_start = np.arange(255 * _BUCKETS + 1, dtype=np.float64) / _BUCKETS
_BUCKET_BASE = np.searchsorted(
    _SRGB_MIDPOINTS_255.astype(np.float64), _bucket_start, side="right"
).astype(np.uint8)
_BUCKET_NEXT = np.append(_SRGB_MIDPOINTS_255, np.float32(np.inf))[_BUCKET_BASE.astype(np.int64)]


def _to_linear(arr: np.ndarray, srgb: bool) -> np.ndarray:
    """uint8 texels -> float32 in [0, 255]; sRGB color channels in linear light."""
    x = arr.astype(np.float32)
    if srgb and arr.shape[2] >= 3:
        x[..., :3] = _SRGB_LINEAR_255[arr[..., :3]]
    return x


def _to_uint8(x: np.ndarray, srgb: bool) -> np.ndarray:
    out = np.empty(x.shape, dtype=np.uint8)
    if srgb and x.shape[2] >= 3:
        c = x[..., :3]
        idx = np.minimum((c * _BUCKETS).astype(np.int32), 255 * _BUCKETS)
        out[..., :3] = _BUCKET_BASE[idx] + (c >= _BUCKET_NEXT[idx])
        out[..., 3:] = np.clip(np.rint(x[..., 3:]), 0, 255)
    else:
        out[:] = np.clip(np.rint(x), 0, 255)
    return out


def _half(x: np.ndarray) -> np.ndarray:
    return (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) * np.float32(0.25)


def downsample_2x(arr: np.ndarray, srgb: bool) -> np.ndarray:
    """2x2 box filter. With srgb, color channels are averaged in linear light;
    alpha and non-color data (BC4/BC5 sources) are averaged as stored."""
    return _to_uint8(_half(_to_linear(arr, srgb)), srgb)


def mip_sources(src: np.ndarray, srgb: bool) -> List[np.ndarray]:
    """Source pixels of mip levels 1..n (see mip_dims): level i is the box
    average of the corresponding 2^i x 2^i source texels (linear light for
    sRGB color), computed in float and rounded once, never from decoded
    blocks or from rounded intermediate levels."""
    h, w = src.shape[0], src.shape[1]
    x, out = _to_linear(src, srgb), []
    for _ in mip_dims(w, h)[1:]:
        x = _half(x)
        out.append(_to_uint8(x, srgb))
    return out


def to_dds(payload: bytes, w: int, h: int, fmt: int, levels: int, srgb: bool) -> bytes:
    """Wrap a stored GPU payload in a DDS (DX10 header) container that D3D,
    Vulkan/GL tooling, texconv, RenderDoc and engine importers read directly."""
    if fmt not in DXGI_FORMATS:
        raise ValueError("only GPU-native BC1/BC4/BC5/BC7 payloads have a DDS form")
    if len(payload) != chain_bytes(w, h, fmt, levels):
        raise ValueError("payload size does not match the declared mip chain")
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 | (0x20000 if levels > 1 else 0)
    caps = 0x1000 | ((0x8 | 0x400000) if levels > 1 else 0)
    dxgi = DXGI_FORMATS[fmt][1 if srgb else 0]
    header = (
        struct.pack("<4sI", b"DDS ", 124)
        + struct.pack("<IIIIII", flags, h, w, level_bytes(w, h, fmt), 0, levels)
        + bytes(44)
        + struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
        + struct.pack("<IIIII", caps, 0, 0, 0, 0)
        + struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
    )
    return header + payload
