"""
Seekable Virtual File System (VFS) for VortexPak (.vpak) Archives
==================================================================
Direct on-demand streaming API: loads individual game assets directly from
disk by seeking to specific micro-block byte offsets and decompressing ONLY
the targeted semantic block, without touching or decompressing the rest of
the multi-megabyte archive.
"""

from __future__ import annotations

import io
import struct
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import zstandard as zstd
from PIL import Image

from .codec import (
    VXPK_MAGIC,
    VXPKError,
    AssetEntry,
    MicroBlock,
    _decode_texture_vxtx,
    _decompress_block,
    parse_vpk_meta,
)
from .native import IS_NATIVE_AVAILABLE, native_decompress_block


class BlockLRUCache:
    """Least-Recently-Used cache for decompressed micro-blocks to accelerate co-located asset reads."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def get(self, block_id: int) -> Optional[bytes]:
        if block_id in self.cache:
            self.cache.move_to_end(block_id)
            return self.cache[block_id]
        return None

    def put(self, block_id: int, data: bytes) -> None:
        if block_id in self.cache:
            self.cache.move_to_end(block_id)
            self.cache[block_id] = data
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[block_id] = data

    def clear(self) -> None:
        self.cache.clear()


class VortexVFS:
    """
    On-Demand Seekable Virtual File System (VFS) Reader for .vpak archives.
    """

    def __init__(
        self,
        vpak_path_or_file: Union[str, Path, io.BufferedReader],
        cache_blocks: int = 16,
    ):
        if isinstance(vpak_path_or_file, (str, Path)):
            self.path = Path(vpak_path_or_file)
            self._file = open(self.path, "rb")
            self._owns_file = True
        else:
            self.path = Path(getattr(vpak_path_or_file, "name", "memory_vpak"))
            self._file = vpak_path_or_file
            self._owns_file = False

        self._cache = BlockLRUCache(capacity=cache_blocks)
        self._init_header_and_index()

    def _init_header_and_index(self) -> None:
        """Read and parse ONLY the compact header metadata at startup."""
        self._file.seek(0)
        header = self._file.read(10)
        if len(header) < 10 or header[:6] != VXPK_MAGIC:
            raise VXPKError(f"Invalid VortexPak magic header in {self.path}")
        (meta_len,) = struct.unpack("<I", header[6:10])
        try:
            self._file.seek(0, 2)
            file_size = self._file.tell()
        except (OSError, io.UnsupportedOperation):
            file_size = None
        if file_size is not None and 10 + meta_len > file_size:
            raise VXPKError(
                f"truncated meta section in {self.path}: declares {meta_len} B, file has {file_size} B"
            )
        self._file.seek(10)
        header_dict, hashes = parse_vpk_meta(self._file.read(meta_len))
        self.mode: str = header_dict["mode"]
        self.total_raw_bytes: int = header_dict["total_raw_bytes"]
        self.total_pixel_raw_bytes: int = header_dict.get("total_pixel_raw_bytes", 0)
        self.entries: List[AssetEntry] = [
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
        self._index: Dict[str, AssetEntry] = {e.rel_path: e for e in self.entries}
        self.block_descriptors: Dict[int, Dict[str, Any]] = {}
        cur_file_pos = 10 + meta_len
        for brow in header_dict["blocks"]:
            b_id, lane_name, strat_id, raw_sz, comp_sz, pat_hex, tok_hex = brow[:7]
            self.block_descriptors[b_id] = {
                "block_id": b_id,
                "lane_name": lane_name,
                "strategy_id": strat_id,
                "dict_id": brow[7] if len(brow) > 7 else -1,
                "raw_size": raw_sz,
                "comp_size": comp_sz,
                "file_offset": cur_file_pos,
                "token_pattern": bytes.fromhex(pat_hex) if pat_hex else b"",
                "token_byte": bytes.fromhex(tok_hex) if tok_hex else b"",
            }
            cur_file_pos += comp_sz

    @classmethod
    def open(cls, path: Union[str, Path], cache_blocks: int = 16) -> "VortexVFS":
        return cls(path, cache_blocks=cache_blocks)

    def close(self) -> None:
        if self._owns_file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "VortexVFS":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def list_assets(self, pattern: Optional[str] = None) -> List[str]:
        """List all asset relative paths in the archive, optionally filtered by pattern."""
        if not pattern:
            return list(self._index.keys())
        import fnmatch

        return [p for p in self._index.keys() if fnmatch.fnmatch(p, pattern)]

    def exists(self, rel_path: str) -> bool:
        """Check if asset exists in the virtual file system."""
        clean_path = rel_path.replace("\\", "/")
        return clean_path in self._index

    def stat(self, rel_path: str) -> Dict[str, Any]:
        """Get asset metadata."""
        clean_path = rel_path.replace("\\", "/")
        if clean_path not in self._index:
            raise FileNotFoundError(f"Asset not found in VFS: {rel_path}")
        e = self._index[clean_path]
        return {
            "rel_path": e.rel_path,
            "raw_size": e.raw_size,
            "is_texture_tensor": e.is_texture_tensor,
            "tx_shape": e.tx_shape,
            "tx_flags": e.tx_flags,
            "block_id": e.block_id,
            "length_in_block": e.length_in_block,
            "stripe_spans_count": len(e.stripe_spans),
        }

    def _load_block_decompressed(self, block_id: int) -> bytes:
        """Fetch and decompress ONLY the single micro-block with ID block_id."""
        cached = self._cache.get(block_id)
        if cached is not None:
            return cached

        b_desc = self.block_descriptors.get(block_id)
        if not b_desc:
            raise KeyError(f"Block ID {block_id} not found in archive directory")

        # Seek directly to disk offset of this block
        self._file.seek(b_desc["file_offset"])
        compressed_payload = self._file.read(b_desc["comp_size"])
        if len(compressed_payload) != b_desc["comp_size"]:
            raise VXPKError(
                f"block #{block_id} truncated on disk: expected {b_desc['comp_size']} B, read {len(compressed_payload)} B"  # noqa: E501
            )

        try:
            if b_desc["strategy_id"] in (9, 10):  # lane-dictionary zstd frame
                dict_raw = self._load_block_decompressed(b_desc["dict_id"])
                dctx = zstd.ZstdDecompressor(
                    dict_data=zstd.ZstdCompressionDict(dict_raw)
                )
                decompressed = dctx.decompress(
                    compressed_payload, max_output_size=b_desc["raw_size"]
                )
                self._cache.put(block_id, decompressed)
                return decompressed
            if IS_NATIVE_AVAILABLE:
                decompressed = native_decompress_block(
                    b_desc["strategy_id"],
                    compressed_payload,
                    b_desc["raw_size"],
                    b_desc["token_pattern"],
                    b_desc["token_byte"],
                )
            else:
                mb = MicroBlock(
                    block_id=b_desc["block_id"],
                    lane_name=b_desc["lane_name"],
                    strategy_id=b_desc["strategy_id"],
                    raw_size=b_desc["raw_size"],
                    compressed_payload=compressed_payload,
                    token_pattern=b_desc["token_pattern"],
                    token_byte=b_desc["token_byte"],
                )
                decompressed = _decompress_block(mb)
        except VXPKError:
            raise
        except Exception as exc:
            raise VXPKError(f"block #{block_id} payload corrupt: {exc}") from exc

        self._cache.put(block_id, decompressed)
        return decompressed

    def read_bytes(
        self,
        rel_path: str,
        materialize_full_rgba: bool = True,
    ) -> bytes:
        """
        Directly stream a single asset's raw bytes from disk without decompressing other lanes.
        """
        clean_path = rel_path.replace("\\", "/")
        if clean_path not in self._index:
            raise FileNotFoundError(f"Asset not found in VFS: {rel_path}")

        entry = self._index[clean_path]

        # Multi-block striped asset
        if entry.stripe_spans:
            parts = []
            for bid, off, ln in entry.stripe_spans:
                block_data = self._load_block_decompressed(bid)
                parts.append(block_data[off : off + ln])
            joined = b"".join(parts)
            if entry.is_texture_tensor:
                arr = _decode_texture_vxtx(
                    memoryview(joined),
                    entry.tx_flags,
                    entry.tx_shape,
                    entry.tx_rle_unique_rows,
                    materialize_full_rgba=materialize_full_rgba,
                    tx_mode=entry.tx_mode,
                )
                return bytes(arr)
            return joined

        # Single block asset
        block_data = self._load_block_decompressed(entry.block_id)
        slice_data = memoryview(block_data)[
            entry.offset_in_block : entry.offset_in_block + entry.length_in_block
        ]

        # Recondition texture or container
        if entry.is_texture_tensor:
            arr = _decode_texture_vxtx(
                slice_data,
                entry.tx_flags,
                entry.tx_shape,
                entry.tx_rle_unique_rows,
                materialize_full_rgba=materialize_full_rgba,
                tx_mode=entry.tx_mode,
            )
            return bytes(arr)

        if entry.aux_block_id >= 0:
            meta_bytes = bytes(slice_data)
            aux_block = self._load_block_decompressed(entry.aux_block_id)
            idat_mv = memoryview(aux_block)[
                entry.aux_offset : entry.aux_offset + entry.aux_length
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
            return bytes(out_png)

        return bytes(slice_data)

    def read_texture(
        self,
        rel_path: str,
        as_pil: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        """
        Directly stream a texture asset as a GPU-ready NumPy array or PIL Image.
        """
        clean_path = rel_path.replace("\\", "/")
        if clean_path not in self._index:
            raise FileNotFoundError(f"Texture not found in VFS: {rel_path}")

        entry = self._index[clean_path]
        if entry.is_texture_tensor:
            if entry.stripe_spans:
                parts = []
                for bid, off, ln in entry.stripe_spans:
                    block_data = self._load_block_decompressed(bid)
                    parts.append(block_data[off : off + ln])
                slice_mv = memoryview(b"".join(parts))
            else:
                block_data = self._load_block_decompressed(entry.block_id)
                slice_mv = memoryview(block_data)[
                    entry.offset_in_block : entry.offset_in_block
                    + entry.length_in_block
                ]
            arr = _decode_texture_vxtx(
                slice_mv,
                entry.tx_flags,
                entry.tx_shape,
                entry.tx_rle_unique_rows,
                materialize_full_rgba=True,
                tx_mode=entry.tx_mode,
            )
            if as_pil:
                if arr.ndim == 3 and arr.shape[2] == 2:
                    rgb = np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
                    rgb[..., 0] = arr[..., 0]
                    rgb[..., 1] = arr[..., 1]
                    arr = rgb
                if (
                    entry.tx_flags & 8
                ):  # TX_BCN: tx_mode is a format name, not a PIL mode
                    pil_mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
                else:
                    pil_mode = entry.tx_mode or (
                        "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
                    )
                return Image.fromarray(arr, mode=pil_mode)
            return arr
        else:
            raw_b = self.read_bytes(clean_path)
            img = Image.open(io.BytesIO(raw_b))
            if as_pil:
                return img
            return np.array(img)

    def open_stream(self, rel_path: str) -> io.BytesIO:
        """Open a file-like seekable stream for a specific asset."""
        return io.BytesIO(self.read_bytes(rel_path))


def benchmark_vfs_streaming(
    vpak_path: Union[str, Path],
    num_queries: int = 200,
) -> Dict[str, Any]:
    """
    Brutal & Realistic On-Demand VFS Streaming Benchmark:
    1. Cold-Cache Random Seek (0% cache hits — cache cleared before EVERY single query, seeded RNG across all assets).
    2. Warm-Cache Scene Locality Streaming (16-block LRU cache, realistic scene clusters of 8 assets).
    """
    import random

    with VortexVFS.open(vpak_path, cache_blocks=16) as vfs:
        assets = vfs.list_assets()
        if not assets:
            return {"error": "Archive is empty"}

        rng = random.Random(42)

        # 1. Brutal Cold-Cache Random Seek (0% cache hit rate: clear cache before EVERY query)
        cold_queries = [rng.choice(assets) for _ in range(num_queries)]
        cold_latencies_ms = []
        cold_bytes = 0
        for p in cold_queries:
            vfs._cache.clear()
            t0 = time.perf_counter()
            data = vfs.read_bytes(p)
            cold_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            cold_bytes += len(data)

        # 2. Realistic Scene/Level Streaming (clusters of 8 co-located assets with 16-block LRU cache)
        vfs._cache.clear()
        scene_queries: list = []
        while len(scene_queries) < num_queries:
            start_idx = rng.randint(0, max(0, len(assets) - 8))
            scene_queries.extend(assets[start_idx : start_idx + 8])
        scene_queries = scene_queries[:num_queries]

        warm_latencies_ms = []
        total_streamed_bytes = 0
        t_bench_start = time.perf_counter()
        for p in scene_queries:
            t0 = time.perf_counter()
            data = vfs.read_bytes(p)
            warm_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            total_streamed_bytes += len(data)
        total_time_sec = time.perf_counter() - t_bench_start

        avg_lat_ms = float(np.mean(warm_latencies_ms))
        med_lat_ms = float(np.median(warm_latencies_ms))
        p99_lat_ms = float(np.percentile(warm_latencies_ms, 99))
        throughput_gbps = (total_streamed_bytes / 1e9) / max(total_time_sec, 1e-6)

        cold_avg_ms = float(np.mean(cold_latencies_ms))
        cold_med_ms = float(np.median(cold_latencies_ms))
        cold_p95_ms = float(np.percentile(cold_latencies_ms, 95))

        return {
            "vpak_file": str(vpak_path),
            "total_assets_in_vpak": len(assets),
            "num_queries": num_queries,
            "total_streamed_mb": round(total_streamed_bytes / 1e6, 2),
            "total_time_ms": round(total_time_sec * 1000.0, 2),
            "avg_latency_microseconds": round(avg_lat_ms * 1000.0, 1),
            "median_latency_microseconds": round(med_lat_ms * 1000.0, 1),
            "p99_latency_microseconds": round(p99_lat_ms * 1000.0, 1),
            "effective_throughput_gbps": round(throughput_gbps, 2),
            "cold_cache_avg_ms": round(cold_avg_ms, 3),
            "cold_cache_median_ms": round(cold_med_ms, 3),
            "cold_cache_p95_ms": round(cold_p95_ms, 3),
        }
