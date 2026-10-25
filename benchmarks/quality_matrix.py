"""
Dynamic quality/capability matrix for VortexPak GPU block codecs.
===============================================================

Measures every codec configuration on a deterministic synthetic corpus
(smooth gradients, noise, hard edges, alpha ramps, normal maps...) and
writes `benchmarks/quality_matrix.json`:

  - measured: PSNR / bpp / encode speed per scenario x configuration
  - capabilities: feature matrix vs published competitors (literature
    references, clearly separated from self-measured numbers)
  - gaps: auto-derived list of weaknesses with priority

Run `python benchmarks/quality_matrix.py` any time the codec changes —
the JSON (and the charts generated from it by generate_quality_charts.py)
always reflect the current state of the code.
"""

from __future__ import annotations

import gc
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vortexpak import IS_NATIVE_AVAILABLE, VortexPakCodec, VortexVFS
from vortexpak.bcn import BCN_BC1, BCN_BC5, BCN_BC7, decode_bcn, encode_bcn

W = H = 256
OUT = Path(__file__).resolve().parent / "quality_matrix.json"


# ---------------- deterministic synthetic corpus ----------------


def _grad_noise() -> np.ndarray:
    x = np.linspace(0, 255, W, dtype=np.uint16)
    r = np.tile(x, (H, 1)).astype(np.uint8)
    g = np.repeat(np.linspace(0, 255, H, dtype=np.uint16)[:, None], W, 1).astype(
        np.uint8
    )
    b = np.clip((r.astype(int) + g) // 2 + ((r * 7) % 24 - 12), 0, 255).astype(np.uint8)
    return np.stack([r, g, b, np.full((H, W), 255, np.uint8)], -1)


def _smooth() -> np.ndarray:
    x = np.linspace(0, 255, W, dtype=np.uint16)
    r = np.tile(x, (H, 1)).astype(np.uint8)
    g = np.repeat(np.linspace(0, 255, H, dtype=np.uint16)[:, None], W, 1).astype(
        np.uint8
    )
    b = np.clip((r.astype(int) + g) // 2, 0, 255).astype(np.uint8)
    return np.stack([r, g, b, np.full((H, W), 255, np.uint8)], -1)


def _structured() -> np.ndarray:
    img = np.zeros((H, W, 4), np.uint8)
    img[..., :3] = (40, 90, 160)
    img[..., 3] = 255
    img[32:96, 32:96] = (220, 60, 30, 255)
    img[32:96, 140:220] = (30, 200, 90, 255)
    img[150:230, 60:130] = (240, 220, 40, 255)
    img[120:150, 20 : W - 20] = (250, 250, 250, 255)
    return img


def _bilinear() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    r = (xx * 255 // (W - 1)).astype(np.uint8)
    g = (yy * 255 // (H - 1)).astype(np.uint8)
    b = ((r.astype(int) * (255 - g) + (255 - r.astype(int)) * g) // 255).astype(
        np.uint8
    )
    return np.stack([r, g, b, np.full((H, W), 255, np.uint8)], -1)


def _hard_edges() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 3] = 255
    img[(xx // 16 + yy // 16) % 2 == 0] = (250, 245, 240, 255)
    img[(xx // 16 + yy // 16) % 2 == 1] = (15, 20, 30, 255)
    img[(xx % 64 < 4) | (yy % 64 < 4)] = (255, 40, 40, 255)
    return img


def _noise50() -> np.ndarray:
    rng = np.random.default_rng(20240501)
    img = rng.integers(0, 256, (H, W, 4)).astype(np.uint8)
    img[..., 3] = 255
    return img


def _skin_ramp() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    t = (xx + yy) * 255 // (W + H - 2)
    r = np.clip(t + 90, 0, 255).astype(np.uint8)
    g = np.clip(t // 2 + 40, 0, 255).astype(np.uint8)
    b = np.clip(t // 4 + 60, 0, 255).astype(np.uint8)
    return np.stack([r, g, b, np.full((H, W), 255, np.uint8)], -1)


def _alpha_ramp() -> np.ndarray:
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 0] = 200
    img[..., 1] = 100
    img[..., 2] = 50
    img[..., 3] = np.tile(
        np.linspace(0, 255, W, dtype=np.uint16).astype(np.uint8), (H, 1)
    )
    return img


def _alpha_binary() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 4), np.uint8)
    img[..., :3] = (180, 60, 200)
    img[..., 3] = np.where((xx // 16 + yy // 16) % 2 == 0, 255, 0).astype(np.uint8)
    return img


def _normal_map() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    r = (128 + 120 * np.sin(xx / 9.0)).clip(0, 255).astype(np.uint8)
    g = (128 + 120 * np.cos(yy / 7.0)).clip(0, 255).astype(np.uint8)
    return np.stack([r, g], -1)  # RG only: BC5 stores two channels


def vfs_latency() -> dict:
    """Frame-budget read latency: pack a realistic multi-block archive, then
    measure per-asset read latency distribution cold (random order, LRU
    evicting) and warm (same 12 assets, block cache resident), raw payload
    and fully materialized. The metric that matters for games is p99 vs the
    1 ms / 4 ms budgets — the average is cosmetic."""
    imgs = [
        _smooth(),
        _grad_noise(),
        _structured(),
        _bilinear(),
        _hard_edges(),
        _skin_ramp(),
        _alpha_ramp(),
        _alpha_binary(),
        _noise50(),
    ]
    files = []
    for rep in range(24):
        for i, img in enumerate(imgs):
            files.append((f"textures/set{rep}/tex{i}.png", img.tobytes()))
    codec = VortexPakCodec(mode="engine", gpu="auto")
    arch = codec.encode_files(files)
    vpak = Path(__file__).resolve().parent / "latency_probe.vpak"
    vpak.write_bytes(arch.to_bytes())

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

    out = {"archive_mb": round(vpak.stat().st_size / 1e6, 1), "entries": len(files)}
    with VortexVFS.open(vpak) as vfs:
        entries = vfs.list_assets()
        rng = random.Random(5)
        gc.disable()
        order = entries[:]
        rng.shuffle(order)
        cold_raw, cold_mat = [], []
        for name in order:
            mat_first = rng.random() < 0.5
            for mat in (mat_first, not mat_first):
                t0 = time.perf_counter_ns()
                vfs.read_bytes(name, materialize_full_rgba=mat)
                ms = (time.perf_counter_ns() - t0) / 1e6
                (cold_mat if mat else cold_raw).append(ms)
        warm_set = rng.sample(entries, 12)
        for a in warm_set:
            vfs.read_bytes(a, materialize_full_rgba=False)
        warm_raw, warm_mat = [], []
        for _ in range(40):
            for a in warm_set:
                t0 = time.perf_counter_ns()
                vfs.read_bytes(a, materialize_full_rgba=False)
                warm_raw.append((time.perf_counter_ns() - t0) / 1e6)
                t0 = time.perf_counter_ns()
                vfs.read_bytes(a, materialize_full_rgba=True)
                warm_mat.append((time.perf_counter_ns() - t0) / 1e6)
        gc.enable()

    out["cold_random_raw"] = pct(cold_raw)
    out["cold_random_materialized"] = pct(cold_mat)
    out["warm_cache_raw"] = pct(warm_raw)
    out["warm_cache_materialized"] = pct(warm_mat)
    vpak.unlink()
    return out


KODAK_DIR = Path(__file__).resolve().parent.parent.parent / "vortex_corpus" / "kodak"


def _flip_score(a: np.ndarray, b: np.ndarray) -> float:
    """NVIDIA FLIP perceptual difference (0 = identical, 1 = max error).
    Lower is better; <0.10 is 'no visible artifact' territory for textures."""
    try:
        import flip_evaluator as fe

        ref = a.astype(np.float64) / 255.0
        test = b.astype(np.float64) / 255.0
        result = fe.evaluate(
            ref, test, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
        )
        return float(result[1])
    except Exception:
        return -1.0  # FLIP unavailable — caller reports it as missing


def _run_kodak() -> dict:
    """Kodak PhotoCD suite (24 images, 768x512, industry-standard for texture
    codec evaluation since the DXT era). Measures BC1/BC7/RDO quality against
    published ISPC/Basis/NVTT baselines."""
    from PIL import Image

    from vortexpak.bcn import BCN_BC1, BCN_BC7, decode_bcn, encode_bcn

    files = sorted(KODAK_DIR.glob("kodim*.png"))
    if not files or len(files) < 20:
        return {
            "status": "corpus_missing",
            "detail": f"Expected 24 Kodak images at {KODAK_DIR}; found {len(files)}. Download from lemire/kodakimagecollection (GitHub) + SCU-VIP-Lab/compression-eval-datasets (HF).",
        }

    results = {}
    configs = [
        ("bc1_max", BCN_BC1, "max", 0.0),
        ("bc1_max_rdo", BCN_BC1, "max", 1.0),
        ("bc7", BCN_BC7, "max", 0.0),
    ]
    for cfg_name, fmt, quality, rdo in configs:
        psnrs, flips = [], []
        t0 = time.perf_counter()
        for f in files:
            arr = np.array(Image.open(f).convert("RGBA"), dtype=np.uint8)
            h, w = arr.shape[0], arr.shape[1]
            src = arr.tobytes()
            blocks = encode_bcn(src, w, h, fmt, quality=quality, rdo=rdo)
            dec = np.frombuffer(decode_bcn(blocks, w, h, fmt), dtype=np.uint8).reshape(
                h, w, 4
            )
            psnrs.append(_psnr(src, dec.tobytes()))
            flips.append(_flip_score(arr, dec))
        enc_s = time.perf_counter() - t0
        results[cfg_name] = {
            "psnr_avg_db": round(float(np.mean(psnrs)), 2),
            "psnr_min_db": round(float(np.min(psnrs)), 2),
            "psnr_max_db": round(float(np.max(psnrs)), 2),
            "flip_avg": (
                round(float(np.mean([f for f in flips if f >= 0])), 4)
                if any(f >= 0 for f in flips)
                else None
            ),
            "flip_worst": (
                round(float(np.max([f for f in flips if f >= 0])), 4)
                if any(f >= 0 for f in flips)
                else None
            ),
            "encode_total_ms": round(enc_s * 1000, 0),
            "images": len(files),
        }
    results["status"] = "ok"
    results["corpus"] = "kodak_photoCD_24"
    results["image_size"] = "768x512"
    results["industry_reference"] = (
        "Results directly comparable with published ISPC/Basis/NVTT numbers on the same set."
    )
    return results


def _psnr(a: bytes, b: bytes) -> float:
    va = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    vb = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    mse = float(((va - vb) ** 2).mean())
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(255.0**2 / mse)


# ---------------- measurement ----------------


def _run_config(
    name: str, src: bytes, w: int, h: int, fmt: int, quality: str, rdo: float = 0.0
) -> dict:
    t0 = time.perf_counter()
    blocks = encode_bcn(src, w, h, fmt, quality=quality, rdo=rdo)
    t1 = time.perf_counter()
    dec = decode_bcn(blocks, w, h, fmt)
    channels = 2 if fmt == BCN_BC5 else 4
    dec = np.frombuffer(dec, dtype=np.uint8).reshape(h, w, channels)
    psnr = _psnr(src, dec)
    px = w * h
    return {
        "config": name,
        "psnr_db": round(psnr, 2),
        "bpp": round(len(blocks) * 8 / px, 2),
        "size_kb": round(len(blocks) / 1024, 1),
        "encode_ms": round((t1 - t0) * 1000, 1),
        "encode_mpx_s": round(px / 1e6 / max(t1 - t0, 1e-9), 1),
    }


def measure() -> dict:
    scenarios = {
        "smooth_gradient": (_smooth(), False),
        "gradient_noise": (_grad_noise(), False),
        "structured_shapes": (_structured(), False),
        "bilinear_patch": (_bilinear(), False),
        "hard_edge_mosaic": (_hard_edges(), False),
        "noise50": (_noise50(), False),
        "skin_ramp": (_skin_ramp(), False),
        "alpha_ramp": (_alpha_ramp(), True),
        "alpha_binary": (_alpha_binary(), True),
    }
    normal = _normal_map()
    out: dict = {}
    for name, (img, has_alpha) in scenarios.items():
        src = img.tobytes()
        entries = [
            _run_config("bc1_fast", src, W, H, BCN_BC1, "fast"),
            _run_config("bc1_standard", src, W, H, BCN_BC1, "standard"),
            _run_config("bc1_max", src, W, H, BCN_BC1, "max"),
            _run_config("bc1_max_rdo", src, W, H, BCN_BC1, "max", rdo=1.0),
        ]
        if has_alpha or img.shape[2] == 4:
            entries.append(_run_config("bc7", src, W, H, BCN_BC7, "max"))
        out[name] = {"has_alpha": has_alpha, "configs": entries}
    src_rg = normal.tobytes()
    out["normal_map"] = {
        "has_alpha": False,
        "configs": [_run_config("bc5", src_rg, W, H, BCN_BC5, "max")],
    }
    return out


# ---------------- capability matrix (literature, not self-measured) ----------------

CAPABILITIES = {
    "features": [
        "bc1_encode",
        "bc4_encode",
        "bc5_encode",
        "bc7_encode",
        "astc_etc_mobile",
        "rdo_post_codec",
        "simd_encode_speed",
        "seekable_vfs",
        "archive_format",
        "open_source_mit",
        "python_api",
        "c_abi",
        "fuzz_tested_decoder",
    ],
    "vendors": ["vortexpak", "oodle", "basis_ktx2", "nvtt_ispc", "zstd"],
    "matrix": {
        #                 vortexpak,  oodle,     basis,     nvtt,      zstd
        "bc1_encode": ["yes", "yes", "yes", "yes", "no"],
        "bc4_encode": ["yes", "yes", "yes", "yes", "no"],
        "bc5_encode": ["yes", "yes", "yes", "yes", "no"],
        "bc7_encode": ["partial", "yes", "yes", "yes", "no"],
        "astc_etc_mobile": ["no", "yes", "yes", "partial", "no"],
        "rdo_post_codec": ["partial", "yes", "no", "partial", "no"],
        "simd_encode_speed": [
            "partial",
            "yes",
            "partial",
            "yes",
            "yes",
        ],  # zero-alloc hot path; AVX2 intrinsics = roadmap
        "seekable_vfs": ["yes", "no", "no", "no", "partial"],
        "archive_format": ["yes", "yes", "no", "no", "yes"],
        "open_source_mit": ["yes", "no", "yes", "yes", "yes"],
        "python_api": ["yes", "no", "partial", "partial", "yes"],
        "c_abi": ["yes", "yes", "yes", "yes", "partial"],
        "fuzz_tested_decoder": ["yes", "unknown", "partial", "partial", "yes"],
    },
    "notes": "Competitor columns are literature/published-capability references, NOT measurements. 'partial'= subset or lower tier.",
}


def derive_gaps(measured: dict, vfs_lat: dict = None, psnr_floor: float = 30.0) -> list:
    gaps = []
    if vfs_lat:
        w99 = vfs_lat["warm_cache_materialized"]["p99_ms"]
        c99 = vfs_lat["cold_random_materialized"]["p99_ms"]
        if w99 > 1.0:
            gaps.append(
                {
                    "area": "vfs:warm_p99",
                    "severity": "high",
                    "status": "frame_budget",
                    "detail": f"Warm cached read p99 = {w99} ms > 1 ms frame-budget slice — stutter risk.",
                }
            )
        if c99 > 4.0:
            gaps.append(
                {
                    "area": "vfs:cold_p99",
                    "severity": "medium",
                    "status": "frame_budget",
                    "detail": f"Cold streamed read p99 = {c99} ms > 4 ms — must be hidden by async prefetch.",
                }
            )
    cap = CAPABILITIES["matrix"]
    for feature in CAPABILITIES["features"]:
        ours = cap[feature][0]
        if ours == "no":
            gaps.append(
                {
                    "area": feature,
                    "severity": (
                        "high"
                        if feature in ("astc_etc_mobile", "simd_encode_speed")
                        else "medium"
                    ),
                    "status": "missing",
                    "detail": "Nemame ziaden pokrytie — konkurencia ano.",
                }
            )
        elif ours == "partial":
            gaps.append(
                {
                    "area": feature,
                    "severity": "medium",
                    "status": "partial",
                    "detail": "Mame subset/lower tier — plne pokrytie chyba.",
                }
            )
    # measured weak spots
    for scenario, data in measured.items():
        best = max(data["configs"], key=lambda e: e["psnr_db"])
        if best["psnr_db"] < psnr_floor:
            gaps.append(
                {
                    "area": f"{scenario}:{best['config']}",
                    "severity": "medium",
                    "status": "weak_quality",
                    "detail": f"Najlepsi format dosahuje len {best['psnr_db']} dB (< {psnr_floor} dB floor) — "
                    "scenario je mimo toku 4-color palety; zvazit dalsi mode/subset.",
                }
            )
        slowest = min(data["configs"], key=lambda e: e["encode_mpx_s"])
        if slowest["encode_mpx_s"] < 5:
            gaps.append(
                {
                    "area": f"{scenario}:{slowest['config']}",
                    "severity": "low",
                    "status": "slow_encode",
                    "detail": f"Encode throughput {slowest['encode_mpx_s']} Mpx/s — SIMD/vektorizacia kandidat.",
                }
            )
    return gaps


def main() -> None:
    if not IS_NATIVE_AVAILABLE:
        raise SystemExit(
            "Native core required for quality measurements (VORTEXPAK_DISABLE_NATIVE not allowed here)."
        )
    measured = measure()
    vfs_lat = vfs_latency()
    kodak = _run_kodak()
    doc = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "native": IS_NATIVE_AVAILABLE,
        "image_size": f"{W}x{H}",
        "measured": measured,
        "vfs_latency": vfs_lat,
        "kodak": kodak,
        "capabilities": CAPABILITIES,
        "gaps": derive_gaps(measured, vfs_lat),
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"quality matrix -> {OUT} ({len(measured)} scenarios, {len(doc['gaps'])} gaps)"
    )


if __name__ == "__main__":
    main()
