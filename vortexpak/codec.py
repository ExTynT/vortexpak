"""
VortexPak (VXPK v1.0) — Format-Aware Multi-Stream Game Asset Codec
==================================================================

Combines three modern game-encoding paradigms:
1. Meta OpenZL-style Format-Aware Structural Decomposition:
   - Texture Tensor Reconditioning (VXTX): Unpacks legacy 1995 PNG/DEFLATE
     streams into lossless pixel tensors with:
       * Opaque-Alpha Channel Stripping (ALPHA_CONST_255)
       * SIMD Consecutive Row-Run-Length Deduplication (ROW_RLE)
       * Planar Channel Separation (PLANAR_CHW) for dense 4K textures
   - Bit-Exact Container Split (Strict Mode): Separates PNG metadata/palette
     chunks from IDAT deflate streams and glTF/GLB JSON schemas from binary
     vertex buffers.
   - Repeated Structural Token Folding (TOKEN_FOLD): Compresses massive
     repetitive XML/TMX/JSON tilemap patterns in C memory before entropy coding.
2. RAD Oodle Kraken-style Rate-Distortion-Speed Optimizer (RDSO):
   - Evaluates candidate entropy levels (Zstd-3, Zstd-6, Zstd-12, Zstd-19, LZ4)
     per semantic micro-block during asset cooking.
   - High-Entropy Zero-Copy Bypass (LANE_BYPASS): Automatically detects
     incompressible audio/video streams (.ogg, .mp3, .jpg) where entropy coding
     yields < 1.5% reduction and routes them to zero-copy slices (>15 GB/s decode).
3. DirectStorage-inspired Parallel Multi-Lane Micro-Block Streaming:
   - Packs semantic lanes and multi-megabyte map files into independent parallel
     chunks decoded concurrently across all CPU cores via GIL-free native C threads.
"""

from __future__ import annotations

import io
import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lz4.frame
import numpy as np
import xxhash
import zstandard as zstd
from PIL import Image

from .native import (
    IS_NATIVE_AVAILABLE,
    native_alpha_strip_pack,
    native_alpha_strip_unpack,
    native_decode_archive_parallel,
    native_decode_single_texture_vxtx,
    native_decompress_blocks_parallel,
    native_is_all_opaque,
    native_planar_decode,
    native_planar_encode,
    native_row_rle_decode,
    native_row_rle_encode,
    native_vxlz_compress,
    native_vxlz_compress_blocks_parallel,
    native_vxlz_decompress,
    native_vxcm_compress,
    native_vxcm_decompress,
    native_dxt_split,
    native_dxt_merge,
    native_vxj_encode,
    native_vxj_decode,
)
from .bcn import (
    GPU_FORMAT_IDS,
    BCN_BC1,
    BCN_BC5,
    BCN_BC7,
    BCN_BCX,
    decode_bcn,
    encode_bcn,
    gpu_level0,
    mip_sources,
)
from .rdo import (
    calculate_image_ssim,
    quantize_blend_rdo,
    quantize_glb_rdo,
    quantize_obj_rdo,
    quantize_texture_array_rdo,
)

VXPK_MAGIC = b"VXPK\x01\x01"

# Default quality floor for gpu="auto": color PSNR (RGB, alpha excluded) and
# alpha PSNR a GPU-native format must reach, else the texture is stored
# losslessly. A conservative product default, not a perceptual guarantee.
GPU_MIN_PSNR_DB = 35.0

# Upper sanity bound for the decompressed meta section (guards corrupt headers).
_MAX_META_BYTES = 1 << 30


class VXPKError(ValueError):
    """Raised when a .vpak container is malformed, truncated or corrupt."""


# Codec strategy IDs per micro-block
STRAT_RAW_BYPASS = 0
STRAT_LZ4_FAST = 1
STRAT_ZSTD_3 = 2
STRAT_ZSTD_6 = 3
STRAT_ZSTD_12 = 4
STRAT_ZSTD_19 = 5
STRAT_TOKEN_FOLD_ZSTD6 = 6
STRAT_TOKEN_FOLD_ZSTD19 = 7
STRAT_VXLZ = 8
STRAT_ZSTD6_DICT = 9  # zstd-6 frame, seeded with the lane's trained dictionary block
STRAT_ZSTD19_DICT = 10  # zstd-19 frame, seeded with the lane's trained dictionary block
_DICT_STRATEGIES = frozenset((STRAT_ZSTD6_DICT, STRAT_ZSTD19_DICT))
STRAT_VXCM = 11
_VALID_STRATEGY_IDS = frozenset(range(STRAT_VXCM + 1))

# Lane-dictionary RDSO: repetitive text lanes (JSON/XML/shaders/configs) get a
# dictionary trained on the lane's own content, stored ONCE as a block; every
# block then compresses seeded with it. Random access is preserved (the dict
# block is a normal block). Selection is per-buffer min(VXLZ, zstd6+dict,
# zstd19+dict) + dict overhead — so a lane never gets bigger than today.
DICT_LANES = ("LANE_STRUCTURED", "LANE_SCRIPTS")
DICT_MIN_LANE_BYTES = 8 * 1024 * 1024
DICT_MIN_ITEMS = 32
DICT_SIZE_BYTES = 110 * 1024
DICT_MAX_SAMPLES = 2000

# VFS seek granularity: one random read decompresses at most ~this much.
# Cold-read p99 budget fix: lanes/stipes cap block size instead of block count.
TARGET_BLOCK_BYTES = 2 * 1024 * 1024
COMPACT_BLOCK_BYTES = 16 * 1024 * 1024
# layout="stream": assets >= STREAM_ISOLATE_BYTES get their own block(s) and
# smaller ones share ~STREAM_BLOCK_BYTES blocks, bounding the bytes decoded per
# random read. Costs some ratio (cross-asset context); layout="compact" keeps it.
STREAM_ISOLATE_BYTES = 64 * 1024
STREAM_BLOCK_BYTES = 256 * 1024

# Texture transform flags (VXTX)
TX_RAW_DENSE = 0
TX_ALPHA_STRIP = 1 << 0
TX_ROW_RLE = 1 << 1
TX_PLANAR_CHW = 1 << 2
TX_BCN = 1 << 3  # payload is GPU block data (BC1/BC4/BC5/BC7/BCX), see vortexpak/bcn.py
TX_MIPS = 1 << 4  # TX_BCN payload holds a full mip chain (bcn.mip_dims), largest first
TX_SRGB = 1 << 5  # TX_BCN color texture: sample as sRGB (DXGI *_UNORM_SRGB)
TX_DXT_SPLIT = 1 << 6
TX_JPEG_VXJ = 1 << 7


@dataclass
class AssetEntry:
    rel_path: str
    ext: str
    raw_size: int
    xxh64: int
    is_texture_tensor: bool = False
    tx_flags: int = 0
    tx_shape: Tuple[int, ...] = ()
    tx_mode: str = ""
    tx_rle_unique_rows: int = 0
    block_id: int = 0
    offset_in_block: int = 0
    length_in_block: int = 0
    aux_block_id: int = -1
    aux_offset: int = 0
    aux_length: int = 0
    # Support multi-block striped large assets (e.g. 67 MB map.tmx striped across 4 parallel blocks)
    stripe_spans: List[Tuple[int, int, int]] = field(default_factory=list)


@dataclass
class MicroBlock:
    block_id: int
    lane_name: str
    strategy_id: int
    raw_size: int
    compressed_payload: bytes
    token_pattern: bytes = b""
    token_byte: bytes = b""
    dict_block_id: int = -1


@dataclass
class BlockPlan:
    entries: List[AssetEntry]
    raw_blocks: List[Tuple[int, str, bytes]]
    lane_groups: Dict[str, List[Tuple[int, bytes]]]
    lane_samples: Dict[str, List[bytes]]
    total_raw_bytes: int
    total_pixel_raw_bytes: int
    tex_psnrs: List[float]
    tex_ssims: List[float]
    geom_psnrs: List[float]
    total_tex_sse: float
    total_tex_samples: int
    max_tex_abs_err: int
    gpu_qualities: List[Dict[str, Any]]
    gpu_ratios: List[float]
    gpu_formats: Dict[str, int]
    gpu_fallbacks: Dict[str, int]
    dedup_files: int
    dedup_bytes: int


@dataclass
class VortexArchive:
    mode: str  # "strict" (100% bit-exact files) or "engine" (bit-exact data + lossless GPU pixel tensors)
    entries: List[AssetEntry]
    blocks: List[MicroBlock]
    total_raw_bytes: int
    total_pixel_raw_bytes: int = 0

    def to_bytes(self) -> bytes:
        """Serialize the entire VortexPak archive into a compact binary container."""
        header_dict = {
            "mode": self.mode,
            "total_raw_bytes": self.total_raw_bytes,
            "total_pixel_raw_bytes": self.total_pixel_raw_bytes,
            "entries": [
                [
                    e.rel_path,
                    e.raw_size,
                    int(e.is_texture_tensor),
                    e.tx_flags,
                    list(e.tx_shape),
                    e.tx_mode,
                    e.tx_rle_unique_rows,
                    e.block_id,
                    e.offset_in_block,
                    e.length_in_block,
                    e.aux_block_id,
                    e.aux_offset,
                    e.aux_length,
                    e.stripe_spans,
                ]
                for e in self.entries
            ],
            "blocks": [
                [
                    b.block_id,
                    b.lane_name,
                    b.strategy_id,
                    b.raw_size,
                    len(b.compressed_payload),
                    b.token_pattern.hex() if b.token_pattern else "",
                    b.token_byte.hex() if b.token_byte else "",
                    b.dict_block_id,
                ]
                for b in self.blocks
            ],
        }
        meta_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
        # Pack 64-bit XXH3 checksums in compact binary instead of ASCII JSON
        hashes_bin = struct.pack(
            f"<{len(self.entries)}Q", *(e.xxh64 for e in self.entries)
        )
        meta_combined = struct.pack("<I", len(meta_json)) + meta_json + hashes_bin
        if IS_NATIVE_AVAILABLE:
            meta_comp = (
                b"VXLZ"
                + struct.pack("<I", len(meta_combined))
                + native_vxlz_compress(meta_combined)
            )
        else:
            meta_comp = zstd.ZstdCompressor(level=19).compress(meta_combined)

        out = bytearray()
        out.extend(VXPK_MAGIC)
        out.extend(struct.pack("<I", len(meta_comp)))
        out.extend(meta_comp)
        for b in self.blocks:
            out.extend(b.compressed_payload)
        return bytes(out)

    @staticmethod
    @staticmethod
    def from_bytes(data: bytes) -> "VortexArchive":
        """Parse a .vpak container. Raises VXPKError on any malformed input."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise VXPKError("VXPK payload must be bytes-like")
        data = bytes(data)
        if len(data) < 10:
            raise VXPKError(f"truncated header: {len(data)} B, need at least 10 B")
        if data[:6] != VXPK_MAGIC:
            raise VXPKError("Invalid VortexPak magic header")
        (meta_len,) = struct.unpack_from("<I", data, 6)
        pos = 10
        if pos + meta_len > len(data):
            raise VXPKError(
                f"truncated meta section: header declares {meta_len} B at offset 10, archive has {len(data)} B"
            )
        header_dict, hashes = parse_vpk_meta(data[pos : pos + meta_len])
        pos += meta_len
        entries = [
            AssetEntry(
                rel_path=row[0],
                ext=Path(row[0]).suffix.lower(),
                raw_size=row[1],
                xxh64=hashes[idx],
                is_texture_tensor=bool(row[2]),
                tx_flags=row[3],
                tx_shape=tuple(row[4]),
                tx_mode=row[5],
                tx_rle_unique_rows=row[6],
                block_id=row[7],
                offset_in_block=row[8],
                length_in_block=row[9],
                aux_block_id=row[10],
                aux_offset=row[11],
                aux_length=row[12],
                stripe_spans=[tuple(s) for s in row[13]] if row[13] else [],
            )
            for idx, row in enumerate(header_dict["entries"])
        ]
        blocks = []
        for brow in header_dict["blocks"]:
            b_id, lane_name, strat_id, raw_sz, comp_sz, pat_hex, tok_hex = brow[:7]
            dict_id = brow[7] if len(brow) > 7 else -1
            if not isinstance(dict_id, int) or dict_id < -1:
                raise VXPKError(f"block #{b_id} has invalid dict_block_id {dict_id!r}")
            if pos + comp_sz > len(data):
                raise VXPKError(
                    f"block #{b_id} payload truncated: need {comp_sz} B at offset {pos}, archive ends at {len(data)} B"
                )
            payload = data[pos : pos + comp_sz]
            pos += comp_sz
            blocks.append(
                MicroBlock(
                    block_id=b_id,
                    lane_name=lane_name,
                    strategy_id=strat_id,
                    raw_size=raw_sz,
                    compressed_payload=payload,
                    token_pattern=bytes.fromhex(pat_hex) if pat_hex else b"",
                    token_byte=bytes.fromhex(tok_hex) if tok_hex else b"",
                    dict_block_id=dict_id,
                )
            )
        return VortexArchive(
            mode=header_dict["mode"],
            entries=entries,
            blocks=blocks,
            total_raw_bytes=header_dict["total_raw_bytes"],
            total_pixel_raw_bytes=header_dict.get("total_pixel_raw_bytes", 0),
        )


def parse_vpk_meta(meta_comp: bytes) -> Tuple[dict, tuple]:
    """Decompress and fully validate the meta section of a .vpak container.

    Returns (header_dict, hashes). Shared by VortexArchive.from_bytes and
    VortexVFS so both entry points enforce identical validation. Raises
    VXPKError on any malformed input.
    """
    if not isinstance(meta_comp, (bytes, bytearray)):
        raise VXPKError("meta section must be bytes-like")
    meta_comp = bytes(meta_comp)
    if len(meta_comp) >= 4 and meta_comp.startswith(b"VXLZ"):
        if len(meta_comp) < 8:
            raise VXPKError("truncated VXLZ meta header")
        (raw_meta_len,) = struct.unpack_from("<I", meta_comp, 4)
        if raw_meta_len > _MAX_META_BYTES:
            raise VXPKError(f"implausible meta size {raw_meta_len} B")
        try:
            meta_combined = native_vxlz_decompress(meta_comp[8:], raw_meta_len)
        except Exception as exc:
            raise VXPKError(f"VXLZ meta decompression failed: {exc}") from exc
    else:
        try:
            meta_combined = zstd.ZstdDecompressor().decompress(meta_comp)
        except Exception as exc:
            raise VXPKError(f"zstd meta decompression failed: {exc}") from exc
    if len(meta_combined) < 4:
        raise VXPKError("truncated meta payload")
    (json_len,) = struct.unpack_from("<I", meta_combined, 0)
    if 4 + json_len > len(meta_combined):
        raise VXPKError("meta JSON length exceeds meta payload")
    hashes_bin = meta_combined[4 + json_len :]
    try:
        header_dict = json.loads(meta_combined[4 : 4 + json_len].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VXPKError(f"meta JSON is corrupt: {exc}") from exc
    if not isinstance(header_dict, dict):
        raise VXPKError("meta JSON must be an object")
    for key in ("mode", "entries", "blocks", "total_raw_bytes"):
        if key not in header_dict:
            raise VXPKError(f"meta JSON is missing key '{key}'")
    if header_dict["mode"] not in ("strict", "engine"):
        raise VXPKError(f"unknown archive mode {header_dict['mode']!r}")
    if (
        not isinstance(header_dict["total_raw_bytes"], int)
        or header_dict["total_raw_bytes"] < 0
    ):
        raise VXPKError(
            f"implausible total_raw_bytes {header_dict['total_raw_bytes']!r}"
        )
    entry_rows = header_dict["entries"]
    if not isinstance(entry_rows, list) or not isinstance(header_dict["blocks"], list):
        raise VXPKError("meta 'entries' and 'blocks' must be lists")
    n_entries = len(entry_rows)
    if len(hashes_bin) != 8 * n_entries:
        raise VXPKError(
            f"XXH3 table size mismatch: expected {8 * n_entries} B for {n_entries} entries, have {len(hashes_bin)} B"
        )
    try:
        hashes = struct.unpack(f"<{n_entries}Q", hashes_bin)
    except struct.error as exc:
        raise VXPKError(f"XXH3 table is corrupt: {exc}") from exc
    n_blocks = len(header_dict["blocks"])
    for idx, row in enumerate(entry_rows):
        if not isinstance(row, list) or len(row) != 14:
            raise VXPKError(
                f"entry #{idx} is malformed: expected 14 fields, got {len(row) if isinstance(row, list) else type(row).__name__}"  # noqa: E501
            )
        block_id = row[7]
        aux_block_id = row[10]
        if not isinstance(row[0], str):
            raise VXPKError(f"entry #{idx} rel_path must be a string")
        if not isinstance(row[1], int) or row[1] < 0 or row[1] > _MAX_META_BYTES:
            raise VXPKError(f"entry #{idx} has implausible raw_size {row[1]!r}")
        if not isinstance(row[4], list) or not all(isinstance(v, int) for v in row[4]):
            raise VXPKError(f"entry #{idx} tx_shape must be a list of ints")
        if not isinstance(row[5], str):
            raise VXPKError(f"entry #{idx} tx_mode must be a string")
        if not isinstance(block_id, int) or block_id < 0:
            raise VXPKError(f"entry #{idx} has invalid block_id {block_id!r}")
        if block_id >= n_blocks:
            raise VXPKError(
                f"entry #{idx} references block {block_id}, archive has {n_blocks} blocks"
            )
        if aux_block_id != -1 and (
            not isinstance(aux_block_id, int) or aux_block_id < 0
        ):
            raise VXPKError(f"entry #{idx} has invalid aux_block_id {aux_block_id!r}")
        if aux_block_id != -1 and aux_block_id >= n_blocks:
            raise VXPKError(
                f"entry #{idx} references aux block {aux_block_id}, archive has {n_blocks} blocks"
            )
        for off, length, label in (
            (row[8], row[9], "block"),
            (row[11], row[12], "aux"),
        ):
            if (
                not isinstance(off, int)
                or off < 0
                or not isinstance(length, int)
                or length < 0
            ):
                raise VXPKError(f"entry #{idx} has invalid {label} offset/length")
        for span in row[13] or []:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 3
                or not all(isinstance(v, int) and v >= 0 for v in span)
            ):
                raise VXPKError(f"entry #{idx} has malformed stripe span {span!r}")
            if span[0] >= n_blocks:
                raise VXPKError(
                    f"entry #{idx} stripe references block {span[0]}, archive has {n_blocks} blocks"
                )
    for brow in header_dict["blocks"]:
        if not isinstance(brow, list) or len(brow) not in (7, 8):
            raise VXPKError(
                f"block header is malformed: expected 7 or 8 fields, got {len(brow) if isinstance(brow, list) else type(brow).__name__}"  # noqa: E501
            )
        b_id, lane_name, strat_id, raw_sz, comp_sz, pat_hex, tok_hex = brow[:7]
        if not isinstance(lane_name, str):
            raise VXPKError(f"block #{b_id} lane_name must be a string")
        if not isinstance(raw_sz, int) or raw_sz < 0 or raw_sz > _MAX_META_BYTES:
            raise VXPKError(f"block #{b_id} has implausible raw_size {raw_sz!r}")
        if not isinstance(comp_sz, int) or comp_sz < 0:
            raise VXPKError(f"block #{b_id} has invalid compressed size {comp_sz!r}")
        if strat_id not in _VALID_STRATEGY_IDS:
            raise VXPKError(f"block #{b_id} has unknown strategy_id {strat_id!r}")
        if len(brow) > 7:
            did = brow[7]
            if not isinstance(did, int) or did < -1 or did >= n_blocks:
                raise VXPKError(f"block #{b_id} has invalid dict_block_id {did!r}")
        try:
            if pat_hex:
                bytes.fromhex(pat_hex)
            if tok_hex:
                bytes.fromhex(tok_hex)
        except (ValueError, TypeError) as exc:
            raise VXPKError(f"block #{b_id} token fields are corrupt: {exc}") from exc
    strat_by_id = {brow[0]: brow[2] for brow in header_dict["blocks"]}
    for brow in header_dict["blocks"]:
        if len(brow) > 7 and brow[7] != -1:
            if strat_by_id.get(brow[7]) in _DICT_STRATEGIES:
                raise VXPKError(
                    f"block #{brow[0]} dictionary chains to another dictionary block"
                )
    return header_dict, hashes


def _compress_block_rdso(
    block_id: int,
    lane_name: str,
    raw_bytes: bytes,
    allow_bypass: bool = True,
    entropy_backend: str = "vxlz",
) -> MicroBlock:
    """
    Rate-Distortion-Speed Optimizer (RDSO):
    Selects the optimal transform + entropy strategy for a micro-block.
    Supports:
      - 'vxlz': 100% proprietary VortexLZ-ANS (zero external codec dependencies)
      - 'hybrid': evaluates proprietary VXLZ-ANS alongside Zstd candidates
      - 'zstd': evaluates Zstd (3, 6, 12, 19) candidates
    """
    if not raw_bytes and entropy_backend != "vxcm":
        return MicroBlock(block_id, lane_name, STRAT_RAW_BYPASS, 0, b"")

    if entropy_backend == "vxcm":
        comp = native_vxcm_compress(raw_bytes, 9)
        if allow_bypass and len(comp) >= len(raw_bytes) * 0.992:
            return MicroBlock(
                block_id, lane_name, STRAT_RAW_BYPASS, len(raw_bytes), raw_bytes
            )
        return MicroBlock(block_id, lane_name, STRAT_VXCM, len(raw_bytes), comp)

    if entropy_backend == "vxlz":
        comp_vxlz = native_vxlz_compress(raw_bytes)
        if allow_bypass and len(comp_vxlz) >= len(raw_bytes) * 0.992:
            return MicroBlock(
                block_id, lane_name, STRAT_RAW_BYPASS, len(raw_bytes), raw_bytes
            )
        return MicroBlock(
            block_id=block_id,
            lane_name=lane_name,
            strategy_id=STRAT_VXLZ,
            raw_size=len(raw_bytes),
            compressed_payload=comp_vxlz,
        )

    # 1. High-Entropy Media Bypass check
    if allow_bypass and lane_name.startswith("LANE_MEDIA"):
        c_probe = zstd.ZstdCompressor(level=3).compress(raw_bytes)
        if len(c_probe) >= len(raw_bytes) * 0.985:
            return MicroBlock(
                block_id, lane_name, STRAT_RAW_BYPASS, len(raw_bytes), raw_bytes
            )

    # 2. Evaluate candidate strategies
    candidates: List[Tuple[int, bytes]] = []
    if entropy_backend == "hybrid" and IS_NATIVE_AVAILABLE:
        candidates.append((STRAT_VXLZ, native_vxlz_compress(raw_bytes)))

    for strat_id, lvl in [
        (STRAT_ZSTD_3, 3),
        (STRAT_ZSTD_6, 6),
        (STRAT_ZSTD_12, 12),
    ]:
        comp = zstd.ZstdCompressor(level=lvl).compress(raw_bytes)
        candidates.append((strat_id, comp))

    if len(raw_bytes) <= 20_000_000:
        comp19 = zstd.ZstdCompressor(level=19).compress(raw_bytes)
        candidates.append((STRAT_ZSTD_19, comp19))

    best_strat, best_comp = min(
        candidates,
        key=lambda x: len(x[1]) * (1.002 if x[0] == STRAT_ZSTD_19 else 1.0),
    )

    if allow_bypass and len(best_comp) >= len(raw_bytes) * 0.992:
        return MicroBlock(
            block_id, lane_name, STRAT_RAW_BYPASS, len(raw_bytes), raw_bytes
        )

    return MicroBlock(
        block_id=block_id,
        lane_name=lane_name,
        strategy_id=best_strat,
        raw_size=len(raw_bytes),
        compressed_payload=best_comp,
    )


def _decompress_block(block: MicroBlock, dict_resolver=None) -> bytes:
    """Decompress a single MicroBlock according to its RDSO strategy tag."""
    try:
        if block.strategy_id in _DICT_STRATEGIES:
            if dict_resolver is None:
                raise VXPKError(
                    f"block #{block.block_id} needs its lane dictionary but no resolver is available"
                )
            d = zstd.ZstdDecompressor(
                dict_data=zstd.ZstdCompressionDict(dict_resolver(block.dict_block_id))
            )
            return d.decompress(
                block.compressed_payload, max_output_size=block.raw_size
            )
        st = block.strategy_id
        if st == STRAT_RAW_BYPASS:
            return block.compressed_payload
        elif st == STRAT_LZ4_FAST:
            return lz4.frame.decompress(block.compressed_payload)
        elif st in (STRAT_ZSTD_3, STRAT_ZSTD_6, STRAT_ZSTD_12, STRAT_ZSTD_19):
            return zstd.ZstdDecompressor().decompress(block.compressed_payload)
        elif st in (STRAT_TOKEN_FOLD_ZSTD6, STRAT_TOKEN_FOLD_ZSTD19):
            folded = zstd.ZstdDecompressor().decompress(block.compressed_payload)
            return folded.replace(block.token_byte, block.token_pattern)
        elif st == STRAT_VXLZ:
            return native_vxlz_decompress(block.compressed_payload, block.raw_size)
        elif st == STRAT_VXCM:
            if not IS_NATIVE_AVAILABLE:
                raise VXPKError(
                    f"block #{block.block_id} uses VXCM and needs the native core"
                )
            return native_vxcm_decompress(block.compressed_payload, block.raw_size)
        else:
            raise ValueError(f"Unknown block strategy: {st}")
    except VXPKError:
        raise
    except Exception as exc:
        raise VXPKError(f"block #{block.block_id} payload corrupt: {exc}") from exc


def _encode_texture_vxtx(
    png_bytes: bytes,
    rdo_level: str = "none",
) -> Tuple[
    int, Tuple[int, ...], str, int, bytes, np.ndarray, float, float, int, int, float
]:
    """
    Transform a PNG file into a VXTX Preconditioned Pixel Tensor:
    - RDO quantization (optional): perceptual LSB deadzone + scanline snapping (PSNR > 48 dB)
    - TX_ALPHA_STRIP: strips constant A==255 channel from RGBA images
    - TX_ROW_RLE: deduplicates consecutive identical scanlines with uint16 run-lengths
    - TX_PLANAR_CHW: separates (H, W, C) into planar (C, H, W) for large dense textures
    Returns:
      (flags, orig_shape, mode, rle_unique, payload, full_arr, psnr_db, sse, num_samples, max_abs_err, ssim)
    """
    img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(img, dtype=np.uint8)
    orig_shape = arr.shape
    orig_full = arr.copy()
    mode = img.mode
    flags = TX_RAW_DENSE
    rle_unique = 0
    psnr = 100.0
    sse = 0.0
    num_samples = int(orig_full.size)
    max_abs_err = 0
    ssim = 1.0

    # 1. Opaque-Alpha Stripping FIRST (RGBA -> RGB when A is 100% 255)
    if arr.ndim == 3 and arr.shape[2] == 4:
        raw_rgba = arr.tobytes()
        if native_is_all_opaque(raw_rgba):
            stripped_rgb = native_alpha_strip_pack(raw_rgba)
            arr = np.frombuffer(stripped_rgb, dtype=np.uint8).reshape(
                arr.shape[0], arr.shape[1], 3
            )
            flags |= TX_ALPHA_STRIP

    # 2. RDO Quantization on pixel tensor
    if rdo_level != "none":
        arr, psnr = quantize_texture_array_rdo(arr, level=rdo_level)

    # Reconstruct full_arr for verification hash and exact global MSE/PSNR
    if flags & TX_ALPHA_STRIP:
        unpacked_rgba = native_alpha_strip_unpack(arr.tobytes())
        full_arr = (
            np.frombuffer(unpacked_rgba, dtype=np.uint8).reshape(orig_shape).copy()
        )
    else:
        full_arr = arr.copy()

    if rdo_level != "none" and num_samples > 0:
        diff = np.abs(orig_full.astype(np.int32) - full_arr.astype(np.int32))
        max_abs_err = int(np.max(diff))  # all channels, alpha included
        # PSNR/SSIM over color only, so unchanged alpha cannot inflate them.
        if orig_full.ndim == 3 and orig_full.shape[2] in (2, 4):
            n_color = orig_full.shape[2] - 1
            diff = diff[..., :n_color]
            ssim = calculate_image_ssim(
                orig_full[..., :n_color], full_arr[..., :n_color]
            )
        else:
            ssim = calculate_image_ssim(orig_full, full_arr)
        num_samples = int(diff.size)
        sse = float(np.sum(diff.astype(np.float64) ** 2))
        mse = sse / num_samples
        psnr = 10.0 * math.log10((255.0**2) / mse) if mse > 1e-12 else 100.0

    # 3. Consecutive Row-RLE using AVX2 SIMD
    if arr.nbytes >= 65536 and arr.shape[0] > 4:
        row_bytes = arr.shape[1] * arr.shape[2] if arr.ndim == 3 else arr.shape[1]
        rle_res = native_row_rle_encode(arr.tobytes(), arr.shape[0], row_bytes)
        if rle_res is not None:
            counts, reduced_bytes, rle_unique = rle_res
            flags |= TX_ROW_RLE
            payload = counts.tobytes() + reduced_bytes
            return (
                flags,
                orig_shape,
                mode,
                rle_unique,
                payload,
                full_arr,
                psnr,
                sse,
                num_samples,
                max_abs_err,
                ssim,
            )

    # 4. Planar Channel Separation (C, H, W) using SIMD
    if arr.nbytes >= 4_000_000 and arr.ndim == 3:
        arr_planar_bytes = native_planar_encode(
            arr.tobytes(), arr.shape[1], arr.shape[0], arr.shape[2]
        )
        flags |= TX_PLANAR_CHW
        return (
            flags,
            orig_shape,
            mode,
            rle_unique,
            arr_planar_bytes,
            full_arr,
            psnr,
            sse,
            num_samples,
            max_abs_err,
            ssim,
        )

    return (
        flags,
        orig_shape,
        mode,
        rle_unique,
        arr.tobytes(),
        full_arr,
        psnr,
        sse,
        num_samples,
        max_abs_err,
        ssim,
    )


def _decode_texture_vxtx(
    payload: memoryview,
    flags: int,
    orig_shape: Tuple[int, ...],
    rle_unique: int,
    materialize_full_rgba: bool = False,
    tx_mode: str = "",
) -> np.ndarray:
    """
    Reconstruct the lossless pixel tensor from a VXTX preconditioned buffer
    using native AVX2 SIMD routines.
    """
    if flags & TX_BCN:
        from .bcn import decode_bcn

        fmt = GPU_FORMAT_IDS.get(tx_mode)
        if fmt is None:
            channels = orig_shape[2] if len(orig_shape) > 2 else 1
            fmt = {4: 1, 2: 5, 1: 4}.get(channels)
        if fmt is None:
            raise VXPKError(f"unsupported GPU tensor shape {orig_shape}")
        if not materialize_full_rgba:
            return np.frombuffer(payload, dtype=np.uint8)
        level0 = gpu_level0(
            bytes(payload), orig_shape[1], orig_shape[0], fmt, bool(flags & TX_MIPS)
        )
        rgba = decode_bcn(level0, orig_shape[1], orig_shape[0], fmt)
        return np.frombuffer(rgba, dtype=np.uint8).reshape(orig_shape)

    if materialize_full_rgba and IS_NATIVE_AVAILABLE:
        fast_arr = native_decode_single_texture_vxtx(
            payload, flags, orig_shape, rle_unique, materialize_rgba=True
        )
        if fast_arr is not None:
            return fast_arr

    work_shape = orig_shape
    if flags & TX_ALPHA_STRIP:
        work_shape = (orig_shape[0], orig_shape[1], 3)

    if flags & TX_ROW_RLE:
        counts_bytes = rle_unique * 2
        counts = np.frombuffer(payload[:counts_bytes], dtype=np.uint16)
        if not materialize_full_rgba:
            row_shape = (rle_unique,) + work_shape[1:]
            return np.frombuffer(payload[counts_bytes:], dtype=np.uint8).reshape(
                row_shape
            )
        # Use native AVX2 row-rle expansion
        row_bytes = (
            work_shape[1] * work_shape[2] if len(work_shape) > 2 else work_shape[1]
        )
        unique_bytes = bytes(payload[counts_bytes:])
        expanded = native_row_rle_decode(counts, unique_bytes, row_bytes, work_shape[0])
        arr = np.frombuffer(expanded, dtype=np.uint8).reshape(work_shape)
    elif flags & TX_PLANAR_CHW:
        c = work_shape[2]
        if not materialize_full_rgba:
            planar_shape = (c, work_shape[0], work_shape[1])
            return np.frombuffer(payload, dtype=np.uint8).reshape(planar_shape)
        # Use native planar decode
        interleaved_bytes = native_planar_decode(
            bytes(payload), work_shape[1], work_shape[0], c
        )
        arr = np.frombuffer(interleaved_bytes, dtype=np.uint8).reshape(work_shape)
    else:
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(work_shape)

    if materialize_full_rgba and (flags & TX_ALPHA_STRIP):
        unpacked_bytes = native_alpha_strip_unpack(arr.tobytes())
        return np.frombuffer(unpacked_bytes, dtype=np.uint8).reshape(orig_shape)

    return arr


def _texture_quality(src: np.ndarray, dec: np.ndarray) -> Dict[str, Any]:
    """Color fidelity (RGB, or RG for 2-channel BC5) and alpha fidelity, kept
    separate so unchanged opaque alpha can never inflate the color PSNR.
    A PSNR of None means bit-exact."""
    color = 3 if src.shape[2] == 4 else src.shape[2]
    diff = src[..., :color].astype(np.int64) - dec[..., :color].astype(np.int64)
    sse = int(np.sum(diff * diff))
    quality: Dict[str, Any] = {
        "color_sse": sse,
        "color_samples": int(diff.size),
        "color_psnr_db": (
            10.0 * math.log10(255.0**2 * diff.size / sse) if sse else None
        ),
        "color_max_abs_error": int(np.abs(diff).max()),
        "alpha_exact": True,
        "alpha_psnr_db": None,
    }
    if src.shape[2] == 4:
        a = src[..., 3].astype(np.int64) - dec[..., 3].astype(np.int64)
        a_sse = int(np.sum(a * a))
        quality["alpha_exact"] = a_sse == 0
        if a_sse:
            quality["alpha_psnr_db"] = 10.0 * math.log10(255.0**2 * a.size / a_sse)
    return quality


def _meets_floor(quality: Dict[str, Any], opaque: bool, floor_db: float) -> bool:
    color = quality["color_psnr_db"]
    if color is not None and color < floor_db:
        return False
    if opaque:
        return bool(quality["alpha_exact"])
    alpha = quality["alpha_psnr_db"]
    return alpha is None or alpha >= floor_db


def _encode_texture_gpu(
    png_bytes: bytes,
    gpu_format: str,
    min_psnr_db: float = GPU_MIN_PSNR_DB,
    mips: bool = True,
) -> Union[Tuple[int, Tuple[int, ...], str, bytes, np.ndarray, float, dict], str]:
    """VXTX-GPU path: PNG -> RGBA -> BC1/BC5/BC7/BCX block payload.

    An explicit format is always used. "auto" tries GPU-native formats
    cheapest first (BC1 4 bpp, then BC7 8 bpp) and takes the first whose color
    and alpha PSNR on level 0 (the authored resolution) reach min_psnr_db;
    opaque sources must stay exactly opaque. "auto" never ships a level 0 below
    the floor: it returns "below_floor" and the caller stores the texture
    losslessly. "ineligible" means the asset cannot use the GPU path at all
    (undecodable, or not a multiple of 4).

    With mips, GPU-native payloads (not BCX) carry the mip chain, box filtered
    from the source (in linear light for sRGB color). Small mips are not held
    to the floor: a single 4x4 block cannot represent a 2-D color gradient in
    any one-subset BC mode, and such levels only cover a few screen pixels.
    Their worst color PSNR is reported as "mip_min_color_psnr_db". Color
    formats (BC1/BC7/BCX) are tagged sRGB. Reported quality is level 0's.

    Returns (flags, tx_shape, tx_mode, payload, decoded_tensor, ratio, quality)."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return "ineligible"
    if arr.ndim != 3 or arr.shape[0] % 4 or arr.shape[1] % 4 or arr.shape[0] == 0:
        return "ineligible"
    h, w = int(arr.shape[0]), int(arr.shape[1])
    fmt_map = {"bc1": BCN_BC1, "bc5": BCN_BC5, "bc7": BCN_BC7, "bcx": BCN_BCX}
    if gpu_format == "auto":
        # GPU-native only; BCX is a storage format (needs a load-time decode).
        candidates = ["bc1", "bc7"]
    elif gpu_format in fmt_map:
        candidates = [gpu_format]
    else:
        return "ineligible"
    opaque = bool((arr[..., 3] == 255).all())
    for cand in candidates:
        fmt = fmt_map[cand]
        src = np.ascontiguousarray(arr[..., :2] if fmt == BCN_BC5 else arr)
        channels = src.shape[2]
        srgb = fmt in (BCN_BC1, BCN_BC7, BCN_BCX)
        levels = [src]
        if mips and fmt != BCN_BCX:
            levels += mip_sources(src, srgb)
        blocks = encode_bcn(src.tobytes(), w, h, fmt)
        dec = np.frombuffer(decode_bcn(blocks, w, h, fmt), dtype=np.uint8)
        dec = dec.reshape(h, w, channels)
        quality = _texture_quality(src, dec)
        if gpu_format == "auto" and not _meets_floor(quality, opaque, min_psnr_db):
            continue
        parts, mip_psnr = [blocks], []
        for level in levels[1:]:
            lh, lw = level.shape[0], level.shape[1]
            lb = encode_bcn(level.tobytes(), lw, lh, fmt)
            ld = np.frombuffer(decode_bcn(lb, lw, lh, fmt), dtype=np.uint8)
            mip_psnr.append(
                _texture_quality(level, ld.reshape(lh, lw, channels))["color_psnr_db"]
            )
            parts.append(lb)
        finite = [p for p in mip_psnr if p is not None]
        quality["mip_min_color_psnr_db"] = min(finite) if finite else None
        flags = TX_BCN | (TX_SRGB if srgb else 0) | (TX_MIPS if len(parts) > 1 else 0)
        ratio = round(src.nbytes / max(len(blocks), 1), 2)
        return (
            flags,
            (h, w, channels),
            cand.upper(),
            b"".join(parts),
            dec,
            ratio,
            quality,
        )
    return "below_floor"


def _apply_lane_dictionaries(
    blocks: List[MicroBlock],
    lane_groups: Dict[str, List[Tuple[int, bytes]]],
    lane_samples: Optional[Dict[str, List[bytes]]] = None,
) -> List[MicroBlock]:
    """Lane-dictionary RDSO: repetitive text lanes get a dictionary trained on
    the lane's own content, stored once as a normal block; every lane block is
    re-emitted as a zstd frame seeded with it. A lane is replaced only when
    the dictionary path (dictionary included) is strictly smaller — size never
    regresses."""
    if not lane_groups:
        return blocks
    blocks = list(blocks)
    next_id = max((b.block_id for b in blocks), default=-1) + 1
    by_id = {b.block_id: b for b in blocks}
    for prefix, items in lane_groups.items():
        if (
            sum(len(d) for _, d in items) < DICT_MIN_LANE_BYTES
            or len(items) < DICT_MIN_ITEMS
        ):
            continue
        samples = None
        if lane_samples:
            samples = lane_samples.get(prefix)
        if not samples:
            # Fall back to block buffers (works, but the trainer prefers many
            # small diverse samples over a few 2 MB buffers).
            samples = [d[:16384] for _, d in items[:DICT_MAX_SAMPLES]]
        try:
            dictionary = zstd.train_dictionary(DICT_SIZE_BYTES, samples)
            dict_raw = dictionary.as_bytes()
        except Exception:
            continue
        dict_payload = zstd.ZstdCompressor(level=19).compress(dict_raw)
        c3 = zstd.ZstdCompressor(level=3, dict_data=dictionary)
        c6 = zstd.ZstdCompressor(level=6, dict_data=dictionary)
        c19 = zstd.ZstdCompressor(level=19, dict_data=dictionary)
        dict_id = next_id
        next_id += 1
        new_blocks = [
            MicroBlock(
                block_id=dict_id,
                lane_name=prefix + "_DICT",
                strategy_id=STRAT_ZSTD_19,
                raw_size=len(dict_raw),
                compressed_payload=dict_payload,
            )
        ]
        # Sample gate: level-19+dict is expensive; only run the full pass
        # when it projects a win over the current blocks on a sample.
        sample = items[:8]
        if sample:
            proj = sum(len(c19.compress(d)) for _, d in sample)
            base = sum(len(by_id[bid].compressed_payload) for bid, _ in sample)
            if proj + len(dict_payload) >= base:
                continue
        total = len(dict_payload)
        for bid, data in items:
            best = c3.compress(data)
            strat = STRAT_ZSTD6_DICT
            p6 = c6.compress(data)
            if len(p6) < len(best):
                best, strat = p6, STRAT_ZSTD6_DICT
            p19 = c19.compress(data)
            if len(p19) < len(best):
                best, strat = p19, STRAT_ZSTD19_DICT
            total += len(best)
            new_blocks.append(
                MicroBlock(
                    block_id=bid,
                    lane_name=by_id[bid].lane_name,
                    strategy_id=strat,
                    raw_size=len(data),
                    compressed_payload=best,
                    dict_block_id=dict_id,
                )
            )
        old_total = sum(len(by_id[bid].compressed_payload) for bid, _ in items)
        if total < old_total:
            replaced = {bid for bid, _ in items}
            blocks = [b for b in blocks if b.block_id not in replaced] + new_blocks
            by_id.update({b.block_id: b for b in new_blocks})
    blocks.sort(key=lambda b: b.block_id)
    return blocks


def _split_into_parallel_buffers(
    items: List[Tuple[AssetEntry, bytes]],
    num_chunks: int,
    lane_prefix: str,
    start_block_id: int,
    isolate_bytes: Optional[int] = None,
    block_bytes: int = TARGET_BLOCK_BYTES,
) -> Tuple[List[Tuple[int, str, bytes]], int]:
    """Pack (AssetEntry, payload) pairs into balanced parallel micro-blocks.

    With isolate_bytes, every payload of at least that size gets its own
    block(s), striped above TARGET_BLOCK_BYTES, so reading a small asset never
    decodes unrelated bulk; the rest share balanced blocks of ~block_bytes."""
    big: List[Tuple[AssetEntry, bytes]] = []
    if isolate_bytes is not None:
        big = [it for it in items if len(it[1]) >= isolate_bytes]
        items = [it for it in items if len(it[1]) < isolate_bytes]
    raw_blocks: List[Tuple[int, str, bytes]] = []
    bid = start_block_id

    if items:
        total_len = sum(len(b) for _, b in items)
        actual_chunks = min(num_chunks, len(items)) if total_len >= 524288 else 1
        actual_chunks = min(
            len(items),
            max(actual_chunks, -(-total_len // block_bytes)),
        )
        bin_entries: List[List[Tuple[AssetEntry, bytes]]] = [
            [] for _ in range(actual_chunks)
        ]
        bin_sizes = [0] * actual_chunks

        target_per_chunk = max(1, total_len // actual_chunks)
        cur_bin = 0
        for entry, blob in items:
            if (
                cur_bin < actual_chunks - 1
                and bin_sizes[cur_bin] >= target_per_chunk
                and bin_sizes[cur_bin] > 0
            ):
                cur_bin += 1
            bin_entries[cur_bin].append((entry, blob))
            bin_sizes[cur_bin] += len(blob)

        for b_idx in range(actual_chunks):
            if not bin_entries[b_idx]:
                continue
            buf = bytearray()
            for entry, blob in bin_entries[b_idx]:
                entry.block_id = bid
                entry.offset_in_block = len(buf)
                entry.length_in_block = len(blob)
                buf.extend(blob)
            raw_blocks.append((bid, f"{lane_prefix}_{b_idx}", bytes(buf)))
            bid += 1

    for entry, blob in big:
        n_parts = -(-len(blob) // TARGET_BLOCK_BYTES)
        step = -(-len(blob) // n_parts)
        spans: List[Tuple[int, int, int]] = []
        for s_idx in range(n_parts):
            part = blob[s_idx * step : (s_idx + 1) * step]
            spans.append((bid, 0, len(part)))
            raw_blocks.append((bid, f"{lane_prefix}_{bid - start_block_id}", part))
            bid += 1
        entry.block_id = spans[0][0]
        entry.offset_in_block = 0
        entry.length_in_block = len(blob)
        entry.stripe_spans = spans if n_parts > 1 else []

    return raw_blocks, bid


class VortexPakCodec:
    """
    Main encoder/decoder interface for VortexPak (VXPK v1.0).
    """

    def __init__(
        self,
        mode: str = "engine",
        rdo: bool = False,
        rdo_level: str = "balanced",
        max_workers: int = 8,
        entropy_backend: str = "auto",
        gpu: str = "none",
        gpu_min_psnr_db: float = GPU_MIN_PSNR_DB,
        layout: str = "stream",
        mips: bool = True,
    ):
        if layout not in ("stream", "compact"):
            raise ValueError("layout must be 'stream' or 'compact'")
        if mode not in ("strict", "engine"):
            raise ValueError("mode must be 'strict' or 'engine'")
        if mode == "strict" and (rdo or gpu != "none"):
            raise ValueError(
                "strict mode is byte-exact; rdo and gpu require mode='engine'"
            )
        if not 0.0 < gpu_min_psnr_db <= 100.0:
            raise ValueError("gpu_min_psnr_db must be in (0, 100]")
        if entropy_backend not in ("auto", "vxcm", "vxlz", "hybrid", "zstd"):
            raise ValueError(
                "entropy_backend must be 'auto', 'vxcm', 'vxlz', 'hybrid', or 'zstd'"
            )
        if entropy_backend == "auto":
            entropy_backend = (
                "vxcm" if layout == "compact" and IS_NATIVE_AVAILABLE else "vxlz"
            )
        if gpu not in ("none", "bc1", "bc5", "bc7", "bcx", "auto"):
            raise ValueError("gpu must be 'none', 'bc1', 'bc5', 'bc7', or 'auto'")
        self.mode = mode
        self.rdo = rdo
        self.rdo_level = rdo_level if rdo else "none"
        self.max_workers = max_workers
        self.entropy_backend = entropy_backend
        self.gpu = gpu
        self.gpu_min_psnr_db = gpu_min_psnr_db
        self.layout = layout
        self.mips = mips
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self.last_rdo_metrics: Dict[str, Any] = {}

    def _lane_split(
        self,
        items: List[Tuple[AssetEntry, bytes]],
        n_chunks: int,
        prefix: str,
        bid: int,
    ) -> Tuple[List[Tuple[int, str, bytes]], int]:
        if self.layout == "stream":
            return _split_into_parallel_buffers(
                items,
                n_chunks,
                prefix,
                bid,
                isolate_bytes=STREAM_ISOLATE_BYTES,
                block_bytes=STREAM_BLOCK_BYTES,
            )
        if self.entropy_backend == "vxcm":
            return _split_into_parallel_buffers(
                items, 1, prefix, bid, block_bytes=COMPACT_BLOCK_BYTES
            )
        return _split_into_parallel_buffers(items, n_chunks, prefix, bid)

    def get_rdo_metrics(self) -> Dict[str, Any]:
        return self.last_rdo_metrics

    def encode_directory(self, root_dir: Path) -> VortexArchive:
        files = sorted(
            [p for p in root_dir.rglob("*") if p.is_file() and ".git" not in p.parts]
        )
        file_tuples = [
            (str(p.relative_to(root_dir)).replace("\\", "/"), p.read_bytes())
            for p in files
        ]
        return self.encode_files(file_tuples)

    def encode_files(self, file_tuples: List[Tuple[str, bytes]]) -> VortexArchive:
        plan = self.plan_blocks(file_tuples)
        return self.finish_archive(plan, self.compress_blocks(plan.raw_blocks))

    def plan_blocks(self, file_tuples: List[Tuple[str, bytes]]) -> BlockPlan:
        """Plan transforms, deduplication and block placement without compression."""
        entries: List[AssetEntry] = []
        seen_raw: Dict[Tuple[int, int], List[Tuple[bytes, AssetEntry]]] = {}
        duplicates: List[Tuple[AssetEntry, AssetEntry]] = []
        dedup_bytes = 0
        compact_vxcm = self.layout == "compact" and self.entropy_backend == "vxcm"
        stripe_bytes = COMPACT_BLOCK_BYTES if compact_vxcm else TARGET_BLOCK_BYTES

        # High-speed zero-copy lanes
        lane_textures: List[Tuple[AssetEntry, bytes]] = []
        lane_png_strict: List[Tuple[AssetEntry, bytes]] = []
        lane_maps_large: List[Tuple[AssetEntry, bytes]] = []
        lane_structured: List[Tuple[AssetEntry, bytes]] = []
        lane_scripts_text: List[Tuple[AssetEntry, bytes]] = []
        lane_3d_binary: List[Tuple[AssetEntry, bytes]] = []
        lane_media: List[Tuple[AssetEntry, bytes]] = []
        _lane_file_samples: Dict[str, List[bytes]] = {}

        total_raw_bytes = 0
        total_pixel_raw_bytes = 0
        tex_psnrs: List[float] = []
        tex_ssims: List[float] = []
        gpu_qualities: List[Dict[str, Any]] = []
        gpu_ratios: List[float] = []
        gpu_formats: Dict[str, int] = {}
        gpu_fallbacks = {"below_floor": 0, "ineligible": 0}
        geom_psnrs: List[float] = []
        total_tex_sse = 0.0
        total_tex_samples = 0
        max_tex_abs_err = 0

        # Sort input files by (extension, filename_stem, rel_path) to maximize cross-file LZ matches
        sorted_inputs = sorted(
            file_tuples,
            key=lambda x: (
                Path(x[0]).suffix.lower(),
                Path(x[0]).stem.split("_")[0].lower(),
                x[0],
            ),
        )

        # GPU block compression is per-texture independent: run it across the
        # pool up front (native encode, Pillow and NumPy release the GIL), keep
        # only what the archive needs, and consume results in input order.
        gpu_results: Dict[str, Any] = {}
        if self.mode == "engine" and self.gpu != "none":

            def _gpu_job(item: Tuple[str, bytes]) -> Tuple[str, Any]:
                rel, raw = item
                res = _encode_texture_gpu(
                    raw, self.gpu, self.gpu_min_psnr_db, self.mips
                )
                if isinstance(res, str):
                    return rel, res
                flags, shape, mode, payload, dec, ratio, quality = res
                digest = xxhash.xxh3_64_intdigest(dec)
                return rel, (
                    flags,
                    shape,
                    mode,
                    payload,
                    dec.nbytes,
                    digest,
                    ratio,
                    quality,
                )

            images = [
                item
                for item in sorted_inputs
                if Path(item[0]).suffix.lower() in (".png", ".tga", ".jpg", ".jpeg")
            ]
            gpu_results = dict(self._pool.map(_gpu_job, images))

        for rel_path, raw_bytes in sorted_inputs:
            ext = Path(rel_path).suffix.lower()
            total_raw_bytes += len(raw_bytes)

            _image_ok = True
            if ext in (".png", ".tga", ".jpg", ".jpeg"):
                # Real game trees contain .png/.tga files that are not actual
                # images (shaders, scripts, corrupt exports) — route them to
                # the data lanes instead of crashing the pack.
                try:
                    with Image.open(io.BytesIO(raw_bytes)) as _im:
                        _im.verify()
                except Exception:
                    _image_ok = False
            if (
                self.mode == "engine"
                and ext in (".png", ".tga", ".jpg", ".jpeg")
                and _image_ok
            ):
                if self.gpu != "none":
                    gpu_res = gpu_results[rel_path]
                    if isinstance(gpu_res, str):
                        gpu_fallbacks[gpu_res] += 1
                    else:
                        (
                            g_flags,
                            g_shape,
                            g_mode,
                            g_payload,
                            g_nbytes,
                            g_digest,
                            g_ratio,
                            g_quality,
                        ) = gpu_res
                        total_pixel_raw_bytes += g_nbytes
                        gpu_qualities.append(g_quality)
                        gpu_ratios.append(g_ratio)
                        gpu_formats[g_mode] = gpu_formats.get(g_mode, 0) + 1
                        entries.append(
                            AssetEntry(
                                rel_path=rel_path,
                                ext=ext,
                                raw_size=len(raw_bytes),
                                xxh64=g_digest,
                                is_texture_tensor=True,
                                tx_flags=g_flags,
                                tx_shape=g_shape,
                                tx_mode=g_mode,
                            )
                        )
                        lane_textures.append((entries[-1], g_payload))
                        continue
                (
                    tx_flags,
                    tx_shape,
                    tx_mode,
                    rle_u,
                    payload,
                    img_arr,
                    t_psnr,
                    t_sse,
                    t_samples,
                    t_max_err,
                    t_ssim,
                ) = _encode_texture_vxtx(raw_bytes, rdo_level=self.rdo_level)
                if self.rdo:
                    if t_psnr < 99.9:
                        tex_psnrs.append(t_psnr)
                        tex_ssims.append(t_ssim)
                    total_tex_sse += t_sse
                    total_tex_samples += t_samples
                    if t_max_err > max_tex_abs_err:
                        max_tex_abs_err = t_max_err
                total_pixel_raw_bytes += img_arr.nbytes
                pix_hash = xxhash.xxh3_64_intdigest(img_arr)
                entry = AssetEntry(
                    rel_path=rel_path,
                    ext=ext,
                    raw_size=len(raw_bytes),
                    xxh64=pix_hash,
                    is_texture_tensor=True,
                    tx_flags=tx_flags,
                    tx_shape=tx_shape,
                    tx_mode=tx_mode,
                    tx_rle_unique_rows=rle_u,
                )
                entries.append(entry)
                lane_textures.append((entry, payload))
                continue

            original_raw = raw_bytes
            dedup_key = (len(raw_bytes), xxhash.xxh3_128_intdigest(raw_bytes))
            first_copy = next(
                (
                    prior
                    for data, prior in seen_raw.get(dedup_key, [])
                    if data == raw_bytes
                ),
                None,
            )
            if first_copy is not None:
                entry = AssetEntry(rel_path, ext, first_copy.raw_size, first_copy.xxh64)
                entries.append(entry)
                duplicates.append((entry, first_copy))
                dedup_bytes += len(raw_bytes)
                continue

            # Perceptual RDO for 3D model geometry and strict mode PNGs
            if self.rdo:
                if ext == ".glb":
                    try:
                        raw_bytes, psnr = quantize_glb_rdo(raw_bytes, drop_bits=12)
                    except Exception:
                        psnr = 100.0
                    if psnr < 99.9:
                        geom_psnrs.append(psnr)
                elif ext == ".blend":
                    raw_bytes, psnr = quantize_blend_rdo(raw_bytes, drop_bits=12)
                    if psnr < 99.9:
                        geom_psnrs.append(psnr)
                elif ext == ".obj":
                    try:
                        raw_text, psnr = quantize_obj_rdo(
                            raw_bytes.decode("utf-8", errors="ignore")
                        )
                        raw_bytes = raw_text.encode("utf-8")
                        if psnr < 99.9:
                            geom_psnrs.append(psnr)
                    except Exception:
                        pass

            h64 = xxhash.xxh3_64_intdigest(raw_bytes)
            entry = AssetEntry(
                rel_path=rel_path,
                ext=ext,
                raw_size=len(raw_bytes),
                xxh64=h64,
                is_texture_tensor=False,
            )
            entries.append(entry)
            seen_raw.setdefault(dedup_key, []).append((original_raw, entry))
            if ext == ".dds":
                rec = native_dxt_split(raw_bytes)
                if rec is not None:
                    raw_bytes = rec
                    entry.tx_flags |= TX_DXT_SPLIT
            elif compact_vxcm and ext in (".jpg", ".jpeg"):
                rec = native_vxj_encode(raw_bytes)
                if rec is not None and len(rec) < len(raw_bytes):
                    raw_bytes = rec
                    entry.tx_flags |= TX_JPEG_VXJ

            if ext in (".ogg", ".mp3", ".jpg", ".jpeg", ".fla"):
                lane_media.append((entry, raw_bytes))
            elif ext == ".png":
                lane_png_strict.append((entry, raw_bytes))
            elif ext == ".tmx" or len(raw_bytes) >= 8_000_000:
                lane_maps_large.append((entry, raw_bytes))
            elif ext in (".json", ".tscn", ".tres", ".xml", ".yml", ".svg"):
                lane_structured.append((entry, raw_bytes))
                s = _lane_file_samples.setdefault("LANE_STRUCTURED", [])
                if len(s) < DICT_MAX_SAMPLES:
                    s.append(raw_bytes[:16384])
            elif ext in (".glb", ".blend", ".b3d", ".obj", ".mts", ".res", ".ttf"):
                lane_3d_binary.append((entry, raw_bytes))
            else:
                lane_scripts_text.append((entry, raw_bytes))
                s = _lane_file_samples.setdefault("LANE_SCRIPTS", [])
                if len(s) < DICT_MAX_SAMPLES:
                    s.append(raw_bytes[:16384])

        raw_blocks_to_compress: List[Tuple[int, str, bytes]] = []
        next_bid = 0

        if lane_textures:
            small_textures = []
            for entry, payload in lane_textures:
                if len(payload) >= 4_000_000:
                    # Stripe massive 4K texture payloads across parallel micro-blocks
                    n_stripes = max(
                        1 if compact_vxcm else 4, -(-len(payload) // stripe_bytes)
                    )
                    step = (len(payload) + n_stripes - 1) // n_stripes
                    spans: List[Tuple[int, int, int]] = []
                    first_bid = next_bid
                    for s_idx in range(n_stripes):
                        part = payload[s_idx * step : (s_idx + 1) * step]
                        if not part:
                            continue
                        spans.append((next_bid, 0, len(part)))
                        raw_blocks_to_compress.append(
                            (next_bid, f"LANE_VXTX_BIG_{s_idx}", part)
                        )
                        next_bid += 1
                    entry.block_id = first_bid
                    entry.offset_in_block = 0
                    entry.length_in_block = len(payload)
                    entry.stripe_spans = spans
                else:
                    small_textures.append((entry, payload))
            if small_textures:
                rbs, next_bid = self._lane_split(
                    small_textures, 32, "LANE_VXTX", next_bid
                )
                raw_blocks_to_compress.extend(rbs)

        if lane_png_strict:
            # Sort strict PNGs by size so small palette PNGs cluster together
            lane_png_strict.sort(key=lambda x: len(x[1]))
            rbs_p, next_bid = self._lane_split(
                lane_png_strict, 8, "LANE_PNG_STRICT", next_bid
            )
            raw_blocks_to_compress.extend(rbs_p)

        # Stripe ultra-large map files (>= 8 MB, e.g. 67 MB map.tmx) across 8 parallel blocks
        for entry, raw_b in lane_maps_large:
            n_stripes = max(1 if compact_vxcm else 8, -(-len(raw_b) // stripe_bytes))
            step = (len(raw_b) + n_stripes - 1) // n_stripes
            spans = []
            for s_idx in range(n_stripes):
                part = raw_b[s_idx * step : (s_idx + 1) * step]
                if not part:
                    continue
                spans.append((next_bid, 0, len(part)))
                raw_blocks_to_compress.append(
                    (next_bid, f"LANE_MAP_STRIPE_{s_idx}", part)
                )
                next_bid += 1
            entry.stripe_spans = spans

        # Split semantic lanes into balanced micro-blocks for fast VFS random seek & parallel decode
        for lane_items, lane_name, n_chk in [
            (lane_structured, "LANE_STRUCTURED", 2),
            (lane_scripts_text, "LANE_SCRIPTS", 4),
            (lane_3d_binary, "LANE_3D_GEOM", 2),
            (lane_media, "LANE_MEDIA", 4),
        ]:
            if lane_items:
                rbs, next_bid = self._lane_split(lane_items, n_chk, lane_name, next_bid)
                raw_blocks_to_compress.extend(rbs)

        for entry, first_copy in duplicates:
            entry.block_id = first_copy.block_id
            entry.offset_in_block = first_copy.offset_in_block
            entry.length_in_block = first_copy.length_in_block
            entry.stripe_spans = list(first_copy.stripe_spans)
            entry.tx_flags = first_copy.tx_flags

        lane_groups: Dict[str, List[Tuple[int, bytes]]] = {}
        if self.entropy_backend != "vxcm":
            for bid, lname, bdata in raw_blocks_to_compress:
                if lname.startswith(DICT_LANES):
                    lane_groups.setdefault(lname.rsplit("_", 1)[0], []).append(
                        (bid, bdata)
                    )

        return BlockPlan(
            entries=entries,
            raw_blocks=raw_blocks_to_compress,
            lane_groups=lane_groups,
            lane_samples=_lane_file_samples,
            total_raw_bytes=total_raw_bytes,
            total_pixel_raw_bytes=total_pixel_raw_bytes,
            tex_psnrs=tex_psnrs,
            tex_ssims=tex_ssims,
            geom_psnrs=geom_psnrs,
            total_tex_sse=total_tex_sse,
            total_tex_samples=total_tex_samples,
            max_tex_abs_err=max_tex_abs_err,
            gpu_qualities=gpu_qualities,
            gpu_ratios=gpu_ratios,
            gpu_formats=gpu_formats,
            gpu_fallbacks=gpu_fallbacks,
            dedup_files=len(duplicates),
            dedup_bytes=dedup_bytes,
        )

    def compress_blocks(
        self, raw_blocks: List[Tuple[int, str, bytes]]
    ) -> List[MicroBlock]:
        """Compress planned blocks, preserving the existing backend dispatch."""
        raw_blocks_to_compress = raw_blocks
        if self.entropy_backend == "vxlz" and IS_NATIVE_AVAILABLE:
            vxlz_res = native_vxlz_compress_blocks_parallel(
                raw_blocks_to_compress, allow_bypass=True
            )
            if vxlz_res is not None:
                blocks = [
                    MicroBlock(
                        block_id=bid,
                        lane_name=lname,
                        strategy_id=strat_id,
                        raw_size=raw_sz,
                        compressed_payload=comp_b,
                    )
                    for bid, lname, strat_id, raw_sz, comp_b in vxlz_res
                ]
            else:
                blocks = [
                    _compress_block_rdso(
                        bid, lname, bdata, entropy_backend=self.entropy_backend
                    )
                    for bid, lname, bdata in raw_blocks_to_compress
                ]
        else:

            def _compress_one(args: Tuple[int, str, bytes]) -> MicroBlock:
                bid, lname, bdata = args
                return _compress_block_rdso(
                    bid, lname, bdata, entropy_backend=self.entropy_backend
                )

            blocks = list(self._pool.map(_compress_one, raw_blocks_to_compress))

        blocks.sort(key=lambda b: b.block_id)
        return blocks

    def finish_archive(
        self, plan: BlockPlan, blocks: List[MicroBlock]
    ) -> VortexArchive:
        """Apply lane dictionaries and publish metrics for a completed plan."""
        if self.entropy_backend != "vxcm":
            blocks = _apply_lane_dictionaries(
                blocks, plan.lane_groups, plan.lane_samples
            )
        tex_psnrs = plan.tex_psnrs
        tex_ssims = plan.tex_ssims
        geom_psnrs = plan.geom_psnrs
        total_tex_sse = plan.total_tex_sse
        total_tex_samples = plan.total_tex_samples
        max_tex_abs_err = plan.max_tex_abs_err
        gpu_qualities = plan.gpu_qualities
        gpu_ratios = plan.gpu_ratios
        gpu_formats = plan.gpu_formats
        gpu_fallbacks = plan.gpu_fallbacks
        all_psnrs = tex_psnrs + geom_psnrs
        if self.rdo and (all_psnrs or total_tex_samples > 0):
            global_tex_mse = total_tex_sse / max(total_tex_samples, 1)
            global_tex_psnr = (
                10.0 * math.log10((255.0**2) / global_tex_mse)
                if global_tex_mse > 1e-12
                else (float(np.mean(tex_psnrs)) if tex_psnrs else 100.0)
            )
            self.last_rdo_metrics = {
                "global_texture_psnr_db": round(global_tex_psnr, 2),
                "avg_texture_psnr_db": (
                    round(float(np.mean(tex_psnrs)), 2) if tex_psnrs else 100.0
                ),
                "avg_geometry_psnr_db": (
                    round(float(np.mean(geom_psnrs)), 2) if geom_psnrs else 100.0
                ),
                "min_psnr_db": (
                    round(float(np.min(all_psnrs)), 2) if all_psnrs else 100.0
                ),
                "overall_psnr_db": round(global_tex_psnr, 2),
                "avg_texture_ssim": (
                    round(float(np.mean(tex_ssims)), 5) if tex_ssims else 1.0
                ),
                "min_texture_ssim": (
                    round(float(np.min(tex_ssims)), 5) if tex_ssims else 1.0
                ),
                "max_pixel_abs_error": max_tex_abs_err,
                "quantized_assets_count": len(all_psnrs),
            }
        else:
            self.last_rdo_metrics = {
                "global_texture_psnr_db": 100.0,
                "avg_texture_psnr_db": 100.0,
                "avg_geometry_psnr_db": 100.0,
                "min_psnr_db": 100.0,
                "overall_psnr_db": 100.0,
                "avg_texture_ssim": 1.0,
                "min_texture_ssim": 1.0,
                "max_pixel_abs_error": 0,
                "quantized_assets_count": 0,
            }

        if self.mode == "engine" and self.gpu != "none":
            sse = sum(q["color_sse"] for q in gpu_qualities)
            samples = sum(q["color_samples"] for q in gpu_qualities)
            finite = [
                q["color_psnr_db"]
                for q in gpu_qualities
                if q["color_psnr_db"] is not None
            ]
            self.last_rdo_metrics.update(
                {
                    "gpu_format": self.gpu,
                    "gpu_min_psnr_db": (
                        self.gpu_min_psnr_db if self.gpu == "auto" else None
                    ),
                    "gpu_asset_count": len(gpu_qualities),
                    "gpu_formats": dict(sorted(gpu_formats.items())),
                    "gpu_lossless_fallbacks": dict(gpu_fallbacks),
                    # Pooled over all GPU-coded texels (not a mean of dB);
                    # None = bit-exact. Alpha is reported separately.
                    "gpu_color_psnr_db": (
                        round(10.0 * math.log10(255.0**2 * samples / sse), 2)
                        if sse
                        else None
                    ),
                    "gpu_min_color_psnr_db": round(min(finite), 2) if finite else None,
                    # Worst mip level (not floor-gated; see _encode_texture_gpu).
                    "gpu_min_mip_color_psnr_db": min(
                        (
                            round(q["mip_min_color_psnr_db"], 2)
                            for q in gpu_qualities
                            if q.get("mip_min_color_psnr_db") is not None
                        ),
                        default=None,
                    ),
                    "gpu_alpha_exact_count": sum(
                        q["alpha_exact"] for q in gpu_qualities
                    ),
                    "gpu_avg_compression_ratio": (
                        round(float(np.mean(gpu_ratios)), 2) if gpu_ratios else None
                    ),
                }
            )

        self.last_rdo_metrics.update(
            dedup_files=plan.dedup_files, dedup_bytes=plan.dedup_bytes
        )

        return VortexArchive(
            mode=self.mode,
            entries=plan.entries,
            blocks=blocks,
            total_raw_bytes=plan.total_raw_bytes,
            total_pixel_raw_bytes=plan.total_pixel_raw_bytes,
        )

    def decode_archive(
        self,
        archive: VortexArchive,
        materialize_full_rgba: bool = False,
    ) -> Dict[str, Any]:
        """
        Decompress all micro-blocks in parallel across CPU cores and return a dictionary
        mapping `rel_path` -> `bytes`/`memoryview` or `np.ndarray` (for VXTX textures in engine mode).
        """
        if not IS_NATIVE_AVAILABLE:
            for entry in archive.entries:
                if entry.tx_flags & TX_JPEG_VXJ:
                    raise VXPKError(
                        f"entry {entry.rel_path} uses VXJ and needs the native core"
                    )
        block_by_id = {b.block_id: b for b in archive.blocks}
        _dict_cache: Dict[int, bytes] = {}

        def _dict_resolver(did: int) -> bytes:
            if did not in _dict_cache:
                _dict_cache[did] = _decompress_block(block_by_id[did])
            return _dict_cache[did]

        native_ok = IS_NATIVE_AVAILABLE and not any(
            b.strategy_id in _DICT_STRATEGIES for b in archive.blocks
        )
        try:
            if native_ok:
                fast_res = native_decode_archive_parallel(
                    archive.blocks,
                    archive.entries,
                    materialize_full_rgba=materialize_full_rgba,
                )
                if fast_res is not None:
                    return fast_res
                decompressed_blocks = native_decompress_blocks_parallel(archive.blocks)
            else:
                decompressed_blocks = list(
                    self._pool.map(
                        lambda b: _decompress_block(b, dict_resolver=_dict_resolver),
                        archive.blocks,
                    )
                )
        except VXPKError:
            raise
        except Exception as exc:
            raise VXPKError(f"archive payload corrupt: {exc}") from exc

        block_views = [memoryview(b) for b in decompressed_blocks]
        results: Dict[str, Any] = {}

        for e in archive.entries:
            if e.stripe_spans and not e.is_texture_tensor:
                if e.tx_flags & TX_DXT_SPLIT:
                    results[e.rel_path] = native_dxt_merge(
                        b"".join(
                            block_views[bid][off : off + ln]
                            for bid, off, ln in e.stripe_spans
                        )
                    )
                elif e.tx_flags & TX_JPEG_VXJ:
                    results[e.rel_path] = native_vxj_decode(
                        b"".join(
                            block_views[bid][off : off + ln]
                            for bid, off, ln in e.stripe_spans
                        ),
                        e.raw_size,
                    )
                elif not materialize_full_rgba:
                    results[e.rel_path] = tuple(
                        block_views[bid][off : off + ln]
                        for bid, off, ln in e.stripe_spans
                    )
                else:
                    results[e.rel_path] = b"".join(
                        block_views[bid][off : off + ln]
                        for bid, off, ln in e.stripe_spans
                    )
                continue

            if e.stripe_spans and e.is_texture_tensor:
                joined_payload = b"".join(
                    block_views[bid][off : off + ln] for bid, off, ln in e.stripe_spans
                )
                mv = memoryview(joined_payload)
            else:
                mv = block_views[e.block_id][
                    e.offset_in_block : e.offset_in_block + e.length_in_block
                ]
            if e.is_texture_tensor:
                results[e.rel_path] = _decode_texture_vxtx(
                    mv,
                    e.tx_flags,
                    e.tx_shape,
                    e.tx_rle_unique_rows,
                    materialize_full_rgba=materialize_full_rgba,
                    tx_mode=e.tx_mode,
                )
            elif e.tx_flags & TX_DXT_SPLIT:
                results[e.rel_path] = native_dxt_merge(bytes(mv))
            elif e.tx_flags & TX_JPEG_VXJ:
                results[e.rel_path] = native_vxj_decode(bytes(mv), e.raw_size)
            elif e.aux_block_id >= 0:
                meta_bytes = bytes(mv)
                idat_mv = block_views[e.aux_block_id][
                    e.aux_offset : e.aux_offset + e.aux_length
                ]
                out_png = bytearray(meta_bytes[:8])
                mpos = 8
                ipos = 0
                while mpos < len(meta_bytes):
                    ln = struct.unpack_from(">I", meta_bytes, mpos)[0]
                    ctype = meta_bytes[mpos + 4 : mpos + 8]
                    if ctype == b"IDAT":
                        out_png.extend(meta_bytes[mpos : mpos + 8])
                        out_png.extend(idat_mv[ipos : ipos + ln])
                        out_png.extend(meta_bytes[mpos + 8 : mpos + 12])
                        ipos += ln
                        mpos += 12
                    else:
                        out_png.extend(meta_bytes[mpos : mpos + 12 + ln])
                        mpos += 12 + ln
                results[e.rel_path] = bytes(out_png)
            else:
                results[e.rel_path] = mv if not materialize_full_rgba else bytes(mv)

        return results

    def verify_archive(self, archive: VortexArchive) -> Tuple[bool, int]:
        """
        Verify 100% lossless integrity (XXH3-64 hash match on every file and every pixel tensor).
        """
        decoded = self.decode_archive(archive, materialize_full_rgba=True)
        for e in archive.entries:
            val = decoded[e.rel_path]
            if e.is_texture_tensor:
                h = xxhash.xxh3_64_intdigest(np.ascontiguousarray(val))
            else:
                h = xxhash.xxh3_64_intdigest(val)
            if h != e.xxh64:
                return False, 0
        return True, len(archive.entries)
