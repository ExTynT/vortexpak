"""Texture outcome of engine-mode gpu="auto" per quality floor (exploratory).

Packs every decodable image of each corpus, decodes the archive, and measures
RGB PSNR of every GPU-coded texture independently of the codec's own metrics.
Use --impl to point at a receipted implementation snapshot (e.g. the previous
build) to measure the old behaviour; omit --floors for its default.
"""
import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CASES = {"minetest": "legacy_games/minetest_game", "kenney": "legacy_games/kenney_3d_platformer",
         "browserquest": "legacy_games/browserquest", "xonotic": "xonotic"}


def load(root: Path):
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".png", ".tga", ".jpg", ".jpeg"):
            continue
        if ".git" in p.relative_to(root).parts:
            continue
        data = p.read_bytes()
        try:
            with Image.open(io.BytesIO(data)) as im:
                src = np.array(im.convert("RGBA"))
        except Exception:
            continue
        files.append((p.relative_to(root).as_posix(), data, src))
    return files


def measure(files, floor, mips=True):
    from vortexpak.codec import TX_BCN, VortexArchive, VortexPakCodec
    kw = {} if floor is None else {"gpu_min_psnr_db": floor}
    if not mips:
        kw["mips"] = False
    codec = VortexPakCodec(mode="engine", gpu="auto", max_workers=16, **kw)
    t0 = time.perf_counter()
    blob = codec.encode_files([(n, d) for n, d, _ in files]).to_bytes()
    pack_s = time.perf_counter() - t0
    arch = VortexArchive.from_bytes(blob)
    dec = codec.decode_archive(arch, materialize_full_rgba=True)
    codec._pool.shutdown()
    by = {e.rel_path: e for e in arch.entries}
    modes, psnr, alpha_broken, sse, n = Counter(), [], 0, 0.0, 0
    for name, data, src in files:
        e = by[name]
        mode = e.tx_mode if e.tx_flags & TX_BCN else "lossless"
        modes[mode] += 1
        if mode == "lossless":
            with Image.open(io.BytesIO(data)) as im:
                native = np.array(im)  # VXTX keeps the source pixel mode
            if not np.array_equal(np.asarray(dec[name]).reshape(native.shape), native):
                raise AssertionError("lossless fallback not exact: " + name)
            continue
        d = np.asarray(dec[name]).reshape(src.shape)
        diff = src[..., :3].astype(np.float64) - d[..., :3].astype(np.float64)
        e_sse = float(np.sum(diff * diff))
        sse, n = sse + e_sse, n + diff.size
        psnr.append(float("inf") if e_sse == 0 else 10 * np.log10(255.0**2 * diff.size / e_sse))
        alpha_broken += bool((src[..., 3] == 255).all() and not (d[..., 3] == 255).all())
    p = np.array(psnr) if psnr else np.array([float("inf")])
    return {"textures": len(files), "archive_bytes": len(blob), "pack_s": round(pack_s, 1), "modes": dict(modes),
            "gpu_pooled_rgb_psnr_db": round(10 * np.log10(255.0**2 * n / sse), 2) if sse else None,
            "gpu_min_rgb_psnr_db": round(float(p.min()), 2), "gpu_below_30": int((p < 30).sum()),
            "gpu_below_35": int((p < 35).sum()), "opaque_alpha_broken": alpha_broken}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--impl", type=Path, help="implementation root to import vortexpak from")
    ap.add_argument("--floors", type=float, nargs="*", default=None)
    ap.add_argument("--no-mips", action="store_true", help="store GPU textures without mip chains")
    args = ap.parse_args()
    if args.impl:
        sys.path.insert(0, str(args.impl.resolve()))
    import vortexpak
    out = {"implementation": vortexpak.__file__, "results": {}}
    for case, rel in CASES.items():
        files = load(args.corpus_root / rel)
        for floor in args.floors or [None]:
            row = measure(files, floor, mips=not args.no_mips)
            out["results"].setdefault(case, {})["default" if floor is None else str(floor)] = row
            print(case, floor, row, flush=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
