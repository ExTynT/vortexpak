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
"""

from __future__ import annotations

import ctypes

import numpy as np

from .native import _lib, IS_NATIVE_AVAILABLE

BCN_BC1 = 1
BCN_BC4 = 4
BCN_BC5 = 5
BCN_BC7 = 7

GPU_FORMAT_IDS = {"BC1": BCN_BC1, "BC4": BCN_BC4, "BC5": BCN_BC5, "BC7": BCN_BC7}
GPU_CHANNELS = {BCN_BC1: 4, BCN_BC4: 1, BCN_BC5: 2, BCN_BC7: 4}
GPU_BLOCK_BYTES = {BCN_BC1: 8, BCN_BC4: 8, BCN_BC5: 16, BCN_BC7: 16}

QUALITY_IDS = {"fast": 0, "standard": 1, "max": 2}

if IS_NATIVE_AVAILABLE and _lib is not None:
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
    if len(blocks) != _blocks_len(w, h, fmt):
        raise ValueError(
            f"BCN payload length {len(blocks)} != expected {_blocks_len(w, h, fmt)} for {w}x{h}"
        )
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


def _bc4_palette(e0: int, e1: int) -> np.ndarray:
    a, b = int(e0), int(e1)
    if a > b:
        return np.array(
            [(a * (7 - i) + b * i + 3) // 7 for i in range(8)], dtype=np.uint8
        )
    pal = np.array([(a * (5 - i) + b * i + 2) // 5 for i in range(6)], dtype=np.uint8)
    return np.concatenate([pal, [0, 255]]).astype(np.uint8)


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
    return out


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
    nb, bw, bh = c0.shape[0], w // 4, h // 4
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
    nb, bw, bh = b.shape[0], w // 4, h // 4
    dst = np.zeros((h, w), dtype=np.uint8)
    for bi in range(nb):
        pal = _bc4_palette(int(b[bi, 0]), int(b[bi, 1]))
        idx = int.from_bytes(b[bi, 2:8].tobytes(), "little")
        by, bx = bi // bw, bi % bw
        for i in range(16):
            dst[by * 4 + i // 4, bx * 4 + i % 4] = pal[(idx >> (3 * i)) & 7]
    return dst.tobytes()


def _encode_bc7_np(src: bytes, w: int, h: int) -> bytes:
    arr = np.frombuffer(src, dtype=np.uint8).reshape(h, w, 4)
    out = np.zeros(((w // 4) * (h // 4), 16), dtype=np.uint8)
    bi = 0
    for by in range(h // 4):
        for bx in range(w // 4):
            blk = arr[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4].reshape(16, 4)
            mn = blk.min(axis=0).astype(int)
            mx = blk.max(axis=0).astype(int)
            # P-bit: keep even base + majority parity; components stay in 0..255.
            p0 = int((mn % 2).sum()) >= 2
            p1 = int((mx % 2).sum()) >= 2
            c0 = (mn & 0xFE) | int(p0)
            c1 = (mx & 0xFE) | int(p1)
            pal = np.stack([c0, (2 * c0 + c1 + 1) // 3, (c0 + 2 * c1 + 1) // 3, c1])
            d = blk.astype(np.int64)[:, None, :] - pal[None, :, :]
            idx = (d**2).sum(-1).argmin(1).astype(np.uint64)
            packed = sum(int(v) << (2 * k) for k, v in enumerate(idx))
            bits = np.zeros(128, dtype=np.uint8)
            bits[0] = 1
            for c in range(4):
                for k in range(7):
                    if (int(c0[c]) >> 1 >> k) & 1:
                        bits[4 + c * 7 + k] = 1
                    if (int(c1[c]) >> 1 >> k) & 1:
                        bits[32 + c * 7 + k] = 1
            bits[60] = 1 if p0 else 0
            bits[61] = 1 if p1 else 0
            for k in range(32):
                if (packed >> k) & 1:
                    bits[62 + k] = 1
            out[bi] = np.packbits(bits, bitorder="little")
            bi += 1
    return bytes(out.reshape(-1).tolist())


def _decode_bc7_np(blocks: bytes, w: int, h: int) -> bytes:
    bl = np.frombuffer(blocks, dtype=np.uint8).reshape(-1, 16)
    dst = np.zeros((h, w, 4), dtype=np.uint8)
    bh, bw = h // 4, w // 4
    for bi in range(bl.shape[0]):

        def bit(pos):
            return (int(bl[bi, pos // 8]) >> (pos % 8)) & 1

        def getv(pos, ln):
            return sum(bit(pos + k) << k for k in range(ln))

        p0, p1 = bit(60), bit(61)
        c0 = np.array(
            [((getv(4 + c * 7, 7) << 1) | p0) for c in range(4)], dtype=np.int64
        )
        c1 = np.array(
            [((getv(32 + c * 7, 7) << 1) | p1) for c in range(4)], dtype=np.int64
        )
        pal = np.stack([c0, (2 * c0 + c1 + 1) // 3, (c0 + 2 * c1 + 1) // 3, c1]).astype(
            np.uint8
        )
        by, bx = bi // bw, bi % bw
        for i in range(16):
            dst[by * 4 + i // 4, bx * 4 + i % 4] = pal[min(getv(62 + 2 * i, 2), 3)]
    return dst.tobytes()


def _fallback_encode(src: bytes, w: int, h: int, fmt: int) -> bytes:
    if fmt == BCN_BC1:
        return _encode_bc1_np(src, w, h).tobytes()
    if fmt == BCN_BC4:
        return _encode_bc4_np(src, w, h)
    if fmt == BCN_BC7:
        return _encode_bc7_np(src, w, h)
    arr = np.frombuffer(src, dtype=np.uint8).reshape(h, w, 2)
    r = _encode_bc4_np(arr[..., 0].tobytes(), w, h)
    g = _encode_bc4_np(arr[..., 1].tobytes(), w, h)
    return r + g


def _fallback_decode(blocks: bytes, w: int, h: int, fmt: int) -> bytes:
    if fmt == BCN_BC1:
        return _decode_bc1_np(blocks, w, h)
    if fmt == BCN_BC4:
        return _decode_bc4_np(blocks, w, h)
    if fmt == BCN_BC7:
        return _decode_bc7_np(blocks, w, h)
    half = len(blocks) // 2
    r = np.frombuffer(_decode_bc4_np(blocks[:half], w, h), dtype=np.uint8)
    g = np.frombuffer(_decode_bc4_np(blocks[half:], w, h), dtype=np.uint8)
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = r.reshape(h, w)
    out[..., 1] = g.reshape(h, w)
    return out.tobytes()
