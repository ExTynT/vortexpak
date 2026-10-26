"""North-Star baseline: what the Xonotic release ships today, and what the best
lossless repack of the same files achieves. Exploratory single run; sizes are
deterministic, times are one observation on the recording host.

    python benchmarks/xonotic_baseline.py --release ../vortex_corpus/xonotic/Xonotic/data \
        --assets ../vortex_corpus/xonotic/xonotic_assets --output benchmarks/reports/v2/north-star
"""

import argparse
import collections
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MB = 1e6


def release_breakdown(release: Path) -> dict:
    pk3s = sorted(release.glob("*.pk3"))
    if not pk3s:
        raise SystemExit(f"no .pk3 files in {release}")
    by_type: dict = collections.defaultdict(lambda: {"files": 0, "raw_bytes": 0, "pk3_bytes": 0})
    for p in pk3s:
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if i.is_dir():
                    continue
                t = by_type[Path(i.filename).suffix.lower() or "(none)"]
                t["files"] += 1
                t["raw_bytes"] += i.file_size
                t["pk3_bytes"] += i.compress_size
    return {
        "pk3_files": {p.name: p.stat().st_size for p in pk3s},
        "download_bytes": sum(p.stat().st_size for p in pk3s),
        "entries": sum(t["files"] for t in by_type.values()),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1]["pk3_bytes"])),
    }


def dds_headroom(files: list) -> dict:
    blobs = [d for n, d in files if n.lower().endswith(".dds")]
    fourcc: collections.Counter = collections.Counter()
    for b in blobs:
        fourcc[b[84:88].decode("ascii", "replace") if b[:4] == b"DDS " and len(b) >= 128 else "other"] += len(b)
    t0 = time.perf_counter()
    solid = len(zstd.ZstdCompressor(level=19, threads=-1).compress(b"".join(blobs)))
    return {"files": len(blobs), "raw_bytes": sum(map(len, blobs)), "fourcc_bytes": dict(fourcc.most_common()),
            "solid_zstd19_bytes": solid, "solid_zstd19_s": round(time.perf_counter() - t0, 1)}


def vpak_strict(files: list, layout: str) -> dict:
    from vortexpak.codec import VortexArchive, VortexPakCodec
    codec = VortexPakCodec(mode="strict", layout=layout, max_workers=16)
    t0 = time.perf_counter()
    blob = codec.encode_files(files).to_bytes()
    pack_s = time.perf_counter() - t0
    decoded = codec.decode_archive(VortexArchive.from_bytes(blob), materialize_full_rgba=True)
    codec._pool.shutdown()
    exact = all(hashlib.sha256(bytes(decoded[n])).digest() == hashlib.sha256(d).digest() for n, d in files)
    if not exact or len(decoded) != len(files):
        raise SystemExit(f"strict {layout} archive is not byte-exact")
    return {"bytes": len(blob), "pack_s": round(pack_s, 1), "byte_exact_files": len(files)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True, help="directory with the shipped .pk3 files")
    ap.add_argument("--assets", type=Path, required=True, help="the same pk3 contents, extracted")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rel = release_breakdown(args.release)
    files = sorted(
        (p.relative_to(args.assets).as_posix(), p.read_bytes()) for p in args.assets.rglob("*") if p.is_file()
    )
    out = {"host": platform.processor(), "release": rel, "assets_files": len(files),
           "assets_bytes": sum(len(d) for _, d in files), "dds": dds_headroom(files),
           "vpak_strict": {layout: vpak_strict(files, layout) for layout in ("stream", "compact")}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    dl = rel["download_bytes"]
    lines = ["# North-Star baseline: Xonotic 0.8.6", "",
             f"Shipped download: **{dl / MB:,.1f} MB** in {len(rel['pk3_files'])} pk3 (ZIP/DEFLATE) files, "
             f"{rel['entries']:,} entries. Target ≤ 0.60× = **{0.6 * dl / MB:,.1f} MB**.", "",
             "| Type | Files | Raw MB | In pk3 MB | Share of download |", "|---|---:|---:|---:|---:|"]
    for ext, t in list(rel["by_type"].items())[:10]:
        lines.append(f"| {ext} | {t['files']:,} | {t['raw_bytes'] / MB:,.1f} | {t['pk3_bytes'] / MB:,.1f} | "
                     f"{100 * t['pk3_bytes'] / dl:.1f} % |")
    d = out["dds"]
    lines += ["", f"DXT textures (.dds): {d['raw_bytes'] / MB:,.1f} MB raw; solid zstd-19 (lossless) "
              f"{d['solid_zstd19_bytes'] / MB:,.1f} MB. Formats: "
              + ", ".join(f"{k} {v / MB:,.1f} MB" for k, v in d["fourcc_bytes"].items()) + ".", "",
              "Byte-exact VortexPak strict repack of the extracted files (current code, 16 workers):", ""]
    for layout, v in out["vpak_strict"].items():
        lines.append(f"- {layout}: **{v['bytes'] / MB:,.1f} MB** ({v['bytes'] / dl:.3f}× download), "
                     f"pack {v['pack_s']} s, {v['byte_exact_files']:,} files verified byte-exact")
    lines += ["", "Already-compressed .jpg and .ogg cannot shrink much losslessly, so the target",
              "requires lossy rate-distortion re-encoding of the DXT textures (quality-gated against the shipped",
              "texture) plus lossless JPEG recompression. Reproduce with the command in the script docstring.", ""]
    (args.output / "BASELINE.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "BASELINE.md")


if __name__ == "__main__":
    main()
