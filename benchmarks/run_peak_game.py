"""Legacy exploratory runner, not a validated competitive scoreboard.
See docs/BENCHMARKING.md for the canonical measurement protocol.


Peak-game benchmark: pack a FULL open-source game directory with VortexPak
and compare against general-purpose codecs on the identical byte corpus.

  1. strict full-corpus pack (bit-exact, zip-class workload):
     size, pack/unpack time vs zstd-19 (MT) and lz4 on the same bytes
  2. GPU texture subset (engine mode, gpu=auto):
     chunked pack, size, bpp, post-codec PSNR sample, VFS random-read latency

Results land in benchmarks/peak_results.json (always overwritten = current).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import lz4.frame as lz4_frame
import numpy as np
import zstandard as zstd_mod
from PIL import Image

from vortexpak import VortexArchive, VortexPakCodec, VortexVFS, IS_NATIVE_AVAILABLE

TEXTURE_EXTS = {".png", ".tga", ".jpg", ".jpeg"}
CHUNK_BYTES = 250 * 1024 * 1024


def walk(game_dir: Path):
    files = []
    total = 0
    for p in sorted(game_dir.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            files.append((str(p.relative_to(game_dir)).replace("\\", "/"), p, sz))
            total += sz
    return files, total


def by_category(files):
    cats = {}
    for name, _p, sz in files:
        ext = Path(name).suffix.lower()
        if ext in TEXTURE_EXTS:
            k = "textures"
        elif ext in (".ogg", ".wav", ".flac", ".mp3"):
            k = "audio"
        elif ext in (".md3", ".iqm", ".dpm", ".obj", ".mdl", ".glb", ".blend"):
            k = "models"
        elif ext in (".map", ".bsp", ".mapinfo", ".shader", ".material"):
            k = "maps"
        else:
            k = "data"
        c = cats.setdefault(k, [0, 0])
        c[0] += 1
        c[1] += sz
    return cats


def strict_bench(files, out_dir: Path, res: dict) -> None:
    total = res["corpus"]["bytes"]
    codec = VortexPakCodec(mode="strict")
    t0 = time.perf_counter()
    arch = codec.encode_files([(n, p.read_bytes()) for n, p, _ in files])
    t1 = time.perf_counter()
    blob = arch.to_bytes()
    del arch
    t2 = time.perf_counter()
    vpak = out_dir / "peak_strict.vpak"
    vpak.write_bytes(blob)
    blob_len = len(blob)
    del blob

    # General-purpose baselines on the identical per-file corpus (memory-safe:
    # per-file compression, no 2.5 GB in-memory blob).
    cctx = zstd_mod.ZstdCompressor(level=19, threads=4)
    z19 = lz = 0
    t = time.perf_counter()
    for _n, p, _s in files:
        z19 += len(cctx.compress(p.read_bytes()))
    z19_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    for _n, p, _s in files:
        lz += len(lz4_frame.compress(p.read_bytes(), compression_level=0))
    lz_ms = (time.perf_counter() - t) * 1000

    t4 = time.perf_counter()
    arch2 = VortexArchive.from_bytes(vpak.read_bytes())
    codec2 = VortexPakCodec(mode="strict")
    decoded = codec2.decode_archive(arch2, materialize_full_rgba=False)
    t5 = time.perf_counter()
    del arch2, decoded

    res["strict_full"] = {
        "files": len(files),
        "raw_bytes": total,
        "vxpk_bytes": blob_len,
        "vxpk_ratio": round(total / max(blob_len, 1), 2),
        "pack_ms": round((t1 - t0) * 1000),
        "serialize_ms": round((t2 - t1) * 1000),
        "unpack_ms": round((t5 - t4) * 1000),
        "zstd19_bytes": z19,
        "zstd19_ratio": round(total / max(z19, 1), 2),
        "zstd19_ms": round(z19_ms),
        "lz4_bytes": lz,
        "lz4_ratio": round(total / max(lz, 1), 2),
        "lz4_ms": round(lz_ms),
    }

    # Frame-budget latency: cold shuffled pass + warm cache-resident pass.
    import gc

    def pct(samples):
        a = np.asarray(samples)
        p = np.percentile(a, [50, 95, 99, 99.9])
        return {
            "p50_ms": round(float(p[0]), 3),
            "p95_ms": round(float(p[1]), 3),
            "p99_ms": round(float(p[2]), 3),
            "p999_ms": round(float(p[3]), 3),
            "max_ms": round(float(a.max()), 3),
            "le_1ms_pct": round(float((a <= 1.0).mean() * 100), 2),
            "le_4ms_pct": round(float((a <= 4.0).mean() * 100), 2),
            "le_16ms_pct": round(float((a <= 16.6).mean() * 100), 2),
            "n": int(a.size),
        }

    with VortexVFS.open(vpak) as vfs:
        names = [n for n, _p, _s in files]
        rng = random.Random(7)
        cold_names = rng.sample(names, min(400, len(names)))
        gc.disable()
        cold_ms = []
        for n in cold_names:
            try:
                t0 = time.perf_counter_ns()
                vfs.read_bytes(n, materialize_full_rgba=False)
                cold_ms.append((time.perf_counter_ns() - t0) / 1e6)
            except Exception:
                pass
        warm_names = rng.sample(cold_names, min(12, len(cold_names)))
        for n in warm_names:
            try:
                vfs.read_bytes(n, materialize_full_rgba=False)
            except Exception:
                pass
        warm_ms = []
        for _ in range(40):
            for n in warm_names:
                try:
                    t0 = time.perf_counter_ns()
                    vfs.read_bytes(n, materialize_full_rgba=False)
                    warm_ms.append((time.perf_counter_ns() - t0) / 1e6)
                except Exception:
                    pass
        gc.enable()
    res["vfs_latency"] = {
        "cold_random_raw": pct(cold_ms),
        "warm_cache_raw": pct(warm_ms),
    }


def gpu_bench(files, out_dir: Path, cap_mb: float, quality: str, res: dict) -> None:
    tex = [(n, p, sz) for n, p, sz in files if Path(n).suffix.lower() in TEXTURE_EXTS]
    tex.sort(key=lambda x: -x[2])
    cap = cap_mb * 1024 * 1024
    chosen, acc = [], 0
    for n, p, sz in tex:
        if acc + sz > cap:
            continue
        chosen.append((n, p, sz))
        acc += sz
        if acc >= cap:
            break
    random.Random(3).shuffle(chosen)

    chunks, cur, cur_bytes = [], [], 0
    for item in chosen:
        cur.append(item)
        cur_bytes += item[2]
        if cur_bytes >= CHUNK_BYTES:
            chunks.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        chunks.append(cur)

    t0 = time.perf_counter()
    total_blocks = 0
    vpaks = []
    for ci, chunk in enumerate(chunks):
        try:
            codec = VortexPakCodec(mode="engine", gpu="auto")
            arch = codec.encode_files([(n, p.read_bytes()) for n, p, _ in chunk])
            blob = arch.to_bytes()
        except Exception as exc:
            print(f"chunk {ci} failed ({exc}) — skipping")
            continue
        total_blocks += len(blob)
        vp = out_dir / f"peak_gpu_{ci}.vpak"
        vp.write_bytes(blob)
        vpaks.append(vp)
    pack_ms = (time.perf_counter() - t0) * 1000

    chosen_bytes = sum(sz for _, _, sz in chosen)
    res["gpu_subset"] = {
        "texture_files": len(chosen),
        "raw_bytes": chosen_bytes,
        "chunks": len(chunks),
        "vxpk_bytes": total_blocks,
        "ratio": round(chosen_bytes / max(total_blocks, 1), 2),
        "bpp_avg": round(total_blocks * 8 / max(1, chosen_bytes / 4), 2),
        "pack_ms": round(pack_ms),
        "quality": quality,
    }

    psnrs = []
    # GPU-routed count on the first chunk via VFS stat
    gpu_count = 0
    if vpaks:
        with VortexVFS.open(vpaks[0]) as vfs:
            for n, _p, _s in chunks[0][:120]:
                try:
                    st = vfs.stat(n)
                except Exception:
                    continue
                if st.get("is_texture_tensor"):
                    gpu_count += 1
    res["gpu_subset"]["gpu_routed_sample"] = gpu_count
    # direct PSNR sample over up to 24 textures (matching auto fmt choice)
    rng = random.Random(11)
    picks = rng.sample(chosen, min(24, len(chosen)))
    codec = VortexPakCodec(mode="engine", gpu="auto")
    for n, p, _s in picks:
        try:
            img = Image.open(p)
            arr = np.array(img.convert("RGBA"), dtype=np.uint8)
        except Exception:
            continue
        if arr.shape[0] % 4 or arr.shape[1] % 4 or arr.shape[0] == 0:
            continue
        has_alpha = bool((arr[..., 3] < 255).any())
        from vortexpak.bcn import BCN_BC1, BCN_BC7, decode_bcn, encode_bcn

        fmt = BCN_BC7 if has_alpha else BCN_BC1
        src = arr.tobytes()
        blocks = encode_bcn(src, arr.shape[1], arr.shape[0], fmt, quality=quality)
        dec = decode_bcn(blocks, arr.shape[1], arr.shape[0], fmt)
        va = np.frombuffer(src, np.uint8).astype(np.float64)
        vb = np.frombuffer(dec, np.uint8).astype(np.float64)
        mse = float(((va - vb) ** 2).mean())
        psnrs.append(100.0 if mse < 1e-12 else 10 * np.log10(255.0**2 / mse))
    if psnrs:
        res["gpu_subset"]["psnr_sample_count"] = len(psnrs)
        res["gpu_subset"]["psnr_avg_db"] = round(float(np.mean(psnrs)), 2)
        res["gpu_subset"]["psnr_min_db"] = round(float(np.min(psnrs)), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_dir")
    ap.add_argument("--out-dir", default="benchmarks")
    ap.add_argument("--tex-cap-mb", type=float, default=600.0)
    ap.add_argument("--quality", default="max", choices=["fast", "standard", "max"])
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()
    game_dir = Path(args.game_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files, total = walk(game_dir)
    res = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game": game_dir.name,
        "native": IS_NATIVE_AVAILABLE,
        "corpus": {
            "files": len(files),
            "bytes": total,
            "categories": by_category(files),
        },
    }
    print(f"corpus: {len(files)} files, {total / 1e6:.1f} MB")
    print(
        "categories:",
        {
            k: f"{v[0]} files / {v[1] / 1e6:.1f} MB"
            for k, v in res["corpus"]["categories"].items()
        },
    )

    t = time.perf_counter()
    strict_bench(files, out_dir, res)
    print("strict done in %.0f s" % (time.perf_counter() - t))
    out_dir.joinpath("peak_results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.skip_gpu:
        try:
            t = time.perf_counter()
            gpu_bench(files, out_dir, args.tex_cap_mb, args.quality, res)
            print("gpu done in %.0f s" % (time.perf_counter() - t))
        except Exception as exc:
            res["gpu_error"] = f"{type(exc).__name__}: {exc}"

    out = out_dir / "peak_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"results -> {out}")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
