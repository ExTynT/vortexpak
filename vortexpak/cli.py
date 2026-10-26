"""
CLI interface for VortexPak (VXPK v1.1)
=======================================
Supports:
- High-throughput packing with native AVX2 SIMD acceleration
- Perceptual RDO quantization for textures and 3D models (--rdo, --rdo-level)
- Seekable On-Demand Virtual File System (VFS) streaming (extract, vfs-bench, ls)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image
import time
from pathlib import Path

from .codec import VortexArchive, VortexPakCodec, VXPKError, GPU_MIN_PSNR_DB
from .native import IS_NATIVE_AVAILABLE
from .vfs import VortexVFS, benchmark_vfs_streaming


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VortexPak (VXPK v1.1) — Format-Aware Game Asset Codec & VFS Engine"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. Pack
    p_pack = sub.add_parser("pack", help="Pack a game directory into a .vpak archive")
    p_pack.add_argument("input_dir", type=Path, help="Input game directory")
    p_pack.add_argument("output_file", type=Path, help="Output .vpak file")
    p_pack.add_argument(
        "--mode",
        choices=["engine", "strict"],
        default="engine",
        help="'engine' (VXTX GPU-ready pixel tensors + bit-exact data) or 'strict' (100%% bit-exact containers)",
    )
    p_pack.add_argument(
        "--rdo",
        action="store_true",
        help="Enable Perceptual RDO quantization (Oodle Texture style) for textures and float32 3D models (PSNR > 48 dB)",  # noqa: E501
    )
    p_pack.add_argument(
        "--rdo-level",
        choices=["subtle", "balanced"],
        default="balanced",
        help="RDO quantization strength: 'subtle' (PSNR > 52 dB) or 'balanced' (PSNR > 48 dB, ~2x size reduction)",
    )
    p_pack.add_argument(
        "--gpu",
        choices=["none", "bc1", "bc5", "bc7", "bcx", "auto"],
        default="none",
        help="VXTX-GPU block compression for engine-mode textures (BC1: 4:1, BC5: 2:1 for normal maps)",
    )
    p_pack.add_argument(
        "--layout",
        choices=["stream", "compact"],
        default="stream",
        help="'stream' isolates large assets and keeps shared blocks small for predictable random reads; "
        "'compact' maximizes cross-asset context for the smallest archive",
    )
    p_pack.add_argument(
        "--no-mips",
        action="store_true",
        help="store GPU textures without a mip chain (UI / pixel-art sprites); "
        "by default BC1/BC4/BC5/BC7 textures carry full mips for engine upload",
    )
    p_pack.add_argument(
        "--gpu-min-psnr",
        type=float,
        default=GPU_MIN_PSNR_DB,
        help="--gpu auto only: minimum color and alpha PSNR (dB) a GPU format must reach; "
        "otherwise the texture is stored losslessly (default: %(default)s)",
    )

    # 2. Info
    p_info = sub.add_parser(
        "info", help="Inspect a .vpak archive and verify XXH3-64 integrity"
    )
    p_info.add_argument("archive_file", type=Path, help="Input .vpak file")

    # 3. Extract (On-Demand VFS read)
    p_ext = sub.add_parser(
        "extract",
        help="On-demand stream and extract a single asset from .vpak without full decompression",
    )
    p_ext.add_argument("archive_file", type=Path, help="Input .vpak file")
    p_ext.add_argument(
        "asset_path", type=str, help="Relative path of the asset inside the archive"
    )
    p_ext.add_argument(
        "--dds",
        action="store_true",
        help="write a GPU texture as a DDS file (DX10 header, full mip chain, sRGB-tagged) to --output",
    )
    p_ext.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output file destination",
    )

    # 4. List assets
    p_ls = sub.add_parser("ls", help="List all assets inside a .vpak archive")
    p_ls.add_argument("archive_file", type=Path, help="Input .vpak file")
    p_ls.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Glob filter (e.g. '*.png', 'models/*')",
    )

    # 5. VFS Benchmark
    p_vfs_bench = sub.add_parser(
        "vfs-bench",
        help="Benchmark Seekable VFS random-access streaming throughput & latency",
    )
    p_vfs_bench.add_argument("archive_file", type=Path, help="Input .vpak file")
    p_vfs_bench.add_argument(
        "--queries",
        type=int,
        default=200,
        help="Number of random asset streaming queries",
    )

    args = parser.parse_args()

    if args.cmd == "pack":
        if args.mode == "strict" and (args.rdo or args.gpu != "none"):
            parser.error("--mode strict is byte-exact; --rdo and --gpu require --mode engine")
        codec = VortexPakCodec(
            mode=args.mode,
            rdo=args.rdo,
            rdo_level=args.rdo_level,
            gpu=args.gpu,
            gpu_min_psnr_db=args.gpu_min_psnr,
            layout=args.layout,
            mips=not args.no_mips,
        )
        t0 = time.perf_counter()
        arch = codec.encode_directory(args.input_dir)
        blob = arch.to_bytes()
        t_enc = time.perf_counter() - t0
        args.output_file.write_bytes(blob)
        rdo_str = f" [RDO-{args.rdo_level.upper()}]" if args.rdo else ""
        native_str = " [AVX2 Native]" if IS_NATIVE_AVAILABLE else " [Py/NumPy]"
        print(
            f"[VortexPak]{rdo_str}{native_str} Packed {len(arch.entries):,d} assets ({arch.total_raw_bytes:,d} B) -> "
            f"{len(blob):,d} B ({100*len(blob)/arch.total_raw_bytes:.2f}%) in {t_enc*1000:.1f} ms"
        )

    elif args.cmd == "info":
        try:
            blob = args.archive_file.read_bytes()
            arch = VortexArchive.from_bytes(blob)
            codec = VortexPakCodec(mode=arch.mode)
            t0 = time.perf_counter()
            _ = codec.decode_archive(arch, materialize_full_rgba=False)
            t_dec = time.perf_counter() - t0
            ok, count = codec.verify_archive(arch)
            if not ok:
                raise VXPKError(
                    f"XXH3-64 verification failed ({count:,d}/{len(arch.entries):,d} assets intact)"
                )
        except VXPKError as exc:
            print(f"[VortexPak] Corrupt archive: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(f"Archive Mode:      {arch.mode}")
        print(f"Native AVX2 Core:  {IS_NATIVE_AVAILABLE}")
        print(f"Total Assets:      {len(arch.entries):,d}")
        print(f"Original Size:     {arch.total_raw_bytes:,d} B")
        print(
            f"Compressed Size:   {len(blob):,d} B ({100*len(blob)/arch.total_raw_bytes:.2f}%)"
        )
        print(f"Parallel Blocks:   {len(arch.blocks)}")
        print(f"Decompress Time:   {t_dec*1000:.2f} ms")
        print(
            f"XXH3-64 Verified:  {ok} ({count:,d}/{len(arch.entries):,d} assets 100% lossless)"
        )

    elif args.cmd == "extract":
        try:
            vfs_ctx = VortexVFS.open(args.archive_file)
        except VXPKError as exc:
            print(f"[VortexPak] Corrupt archive: {exc}", file=sys.stderr)
            raise SystemExit(2)
        with vfs_ctx as vfs:
            t0 = time.perf_counter()
            try:
                data = vfs.read_bytes(args.asset_path)
            except VXPKError as exc:
                print(f"[VortexPak] Corrupt archive: {exc}", file=sys.stderr)
                raise SystemExit(2)
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[VFS On-Demand] Streamed '{args.asset_path}' ({len(data):,d} B) in {t_ms:.2f} ms without full archive decompression!"  # noqa: E501
            )
            if args.dds:
                if not args.output:
                    parser.error("--dds requires --output")
                try:
                    dds = vfs.read_dds(args.asset_path)
                except ValueError as exc:
                    parser.error(str(exc))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(dds)
                info = vfs.texture_info(args.asset_path)
                print(
                    f"Saved DDS {info['format']}{'_SRGB' if info['srgb'] else ''} "
                    f"{info['width']}x{info['height']}, {len(info['levels'])} mip level(s) to {args.output}"
                )
            elif args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if (
                    vfs.stat(args.asset_path)["is_texture_tensor"]
                    and args.output.suffix.lower() == ".png"
                ):
                    img = vfs.read_texture(args.asset_path, as_pil=True)
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    img.save(args.output, format="PNG")
                else:
                    args.output.write_bytes(data)
                print(f"Saved asset to {args.output}")

    elif args.cmd == "ls":
        try:
            vfs_ctx = VortexVFS.open(args.archive_file)
        except VXPKError as exc:
            print(f"[VortexPak] Corrupt archive: {exc}", file=sys.stderr)
            raise SystemExit(2)
        with vfs_ctx as vfs:
            assets = vfs.list_assets(pattern=args.filter)
            print(f"Listing {len(assets):,d} assets in {args.archive_file}:")
            for a in assets:
                st = vfs.stat(a)
                print(
                    f"  {st['rel_path']:50s} | Size: {st['raw_size']:8,d} B | Block: {st['block_id']}"
                )

    elif args.cmd == "vfs-bench":
        try:
            res = benchmark_vfs_streaming(args.archive_file, num_queries=args.queries)
        except VXPKError as exc:
            print(f"[VortexPak] Corrupt archive: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print("=== Seekable Virtual File System (VFS) Benchmark ===")
        print(f"Archive:             {res['vpak_file']}")
        print(f"Total Assets:        {res['total_assets_in_vpak']:,d}")
        print(f"Random Queries:      {res['num_queries']:,d}")
        print(f"Total Streamed:      {res['total_streamed_mb']} MB")
        print(f"Total Time:          {res['total_time_ms']} ms")
        print(
            f"Average Latency:     {res['avg_latency_microseconds']} microseconds / asset"
        )
        print(
            f"P99 Latency:         {res['p99_latency_microseconds']} microseconds / asset"
        )
        print(f"Effective Speed:     {res['effective_throughput_gbps']} GB/s")


if __name__ == "__main__":
    main()
