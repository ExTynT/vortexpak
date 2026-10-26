"""Authoritative, impartial measurement of the Xonotic 0.8.6 lossless goal: a
byte-exact VortexPak strict archive of the whole release must be <= 0.80x the
shipped download (the sum of the .pk3 ZIP/DEFLATE files). Verified two ways
(full in-memory decode and per-file engine VFS reads) against SHA-256 of every
extracted asset; xz -9e and zstd --ultra -22 solid streams are the reference bars.
Sizes are deterministic; times are one observation on the recording host.

    python benchmarks/xonotic_lossless.py --release ../vortex_corpus/xonotic/Xonotic/data \
        --assets ../vortex_corpus/xonotic/xonotic_assets \
        --output benchmarks/reports/xonotic-lossless
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vortexpak  # noqa: E402
from vortexpak.codec import VortexArchive, VortexPakCodec  # noqa: E402
from vortexpak.vfs import VortexVFS  # noqa: E402

MB = 1e6
TARGET_NUM = 80  # target = floor(0.80 * download_bytes), computed in integers
TARGET_DEN = 100
COMPETITOR_TIMEOUT_S = 7200


def pct(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def iter_files(assets: Path) -> Iterator[Tuple[str, bytes]]:
    for p in sorted(assets.rglob("*")):
        if p.is_file():
            yield p.relative_to(assets).as_posix(), p.read_bytes()


def run_git(argv: List[str]) -> str:
    try:
        r = subprocess.run(["git"] + argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def git_dirty() -> Any:
    out = run_git(["status", "--porcelain"])
    return "unknown" if out == "unknown" else bool(out)


def run_competitor(tool: str, argv: List[str], out_path: Path) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"tool": tool, "argv": argv}
    if shutil.which(tool) is None:
        rec["status"] = "not found"
        return rec
    try:
        v = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, timeout=60
        )
        rec["version"] = (
            v.stdout.splitlines()[0].strip() if v.stdout.splitlines() else "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        rec["version"] = "unknown"
    t0 = time.perf_counter()
    try:
        with open(out_path, "wb") as fh:
            r = subprocess.run(
                argv, stdout=fh, stderr=subprocess.DEVNULL, timeout=COMPETITOR_TIMEOUT_S
            )
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        return rec
    if r.returncode != 0:
        rec["status"] = f"error {r.returncode}"
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        return rec
    rec["status"] = "ok"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec["bytes"] = out_path.stat().st_size if out_path.exists() else None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--release",
        type=Path,
        required=True,
        help="directory with the shipped .pk3 files",
    )
    ap.add_argument(
        "--assets", type=Path, required=True, help="the same pk3 contents, extracted"
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for results.json / RESULTS.md",
    )
    ap.add_argument(
        "--work",
        type=Path,
        default=Path(tempfile.gettempdir()) / "vpak_xonotic",
        help="scratch dir holding the .vpak and the solid stream (created)",
    )
    ap.add_argument("--layout", choices=("compact", "stream"), default="compact")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="keep only the first N sorted files (smoke tests)",
    )
    ap.add_argument(
        "--no-competitors",
        action="store_true",
        help="skip the xz/zstd solid-stream bars",
    )
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Shipped download and the 0.80x target.
    pk3s = sorted(args.release.glob("*.pk3"))
    if not pk3s:
        raise SystemExit(f"no .pk3 files in {args.release}")
    pk3_files = {p.name: p.stat().st_size for p in pk3s}
    for name, size in pk3_files.items():
        print(f"pk3 {name} {size}")
    download_bytes = sum(pk3_files.values())
    target_bytes = download_bytes * TARGET_NUM // TARGET_DEN

    # 2. Extracted assets and their SHA-256.
    files: List[Tuple[str, bytes]] = list(iter_files(args.assets))
    if args.limit is not None:
        files = files[: args.limit]
    files.sort(key=lambda f: f[0])
    sha = {name: hashlib.sha256(data).digest() for name, data in files}
    print(f"assets files={len(files)} raw_bytes={sum(len(d) for _, d in files)}")

    # 3. Pack.
    codec = VortexPakCodec(mode="strict", layout=args.layout, max_workers=args.workers)
    t0 = time.perf_counter()
    blob = codec.encode_files(files).to_bytes()
    pack_s = time.perf_counter() - t0
    vpak_path = args.work / "xonotic.vpak"
    vpak_path.write_bytes(blob)
    metrics = codec.get_rdo_metrics()
    backend = codec.entropy_backend
    vpak_bytes = len(blob)
    print(f"vpak bytes={vpak_bytes} backend={backend} pack_s={pack_s:.2f}")

    # 4. Verify A: full in-memory decode, byte-exact against SHA-256.
    arch = VortexArchive.from_bytes(blob)
    t0 = time.perf_counter()
    out = codec.decode_archive(arch, materialize_full_rgba=True)
    decode_s = time.perf_counter() - t0
    mismatches_a = 0
    for name in sha:
        if name not in out or hashlib.sha256(bytes(out[name])).digest() != sha[name]:
            mismatches_a += 1
            print(f"MISMATCH A {name}")
    verify_ok, verify_n = codec.verify_archive(arch)
    del out
    gc.collect()
    codec._pool.shutdown()

    # 5. Verify B: the engine read path, one VFS read per file.
    read_ms: List[float] = []
    mismatches_b = 0
    t0 = time.perf_counter()
    with VortexVFS.open(vpak_path) as vfs:
        for name in sorted(sha):
            tr = time.perf_counter()
            data = vfs.read_bytes(name)
            read_ms.append((time.perf_counter() - tr) * 1e3)
            if hashlib.sha256(data).digest() != sha[name]:
                mismatches_b += 1
                print(f"MISMATCH B {name}")
    vfs_total_s = time.perf_counter() - t0
    read_ms.sort()
    p50_ms, p99_ms, max_ms = (
        pct(read_ms, 0.50),
        pct(read_ms, 0.99),
        read_ms[-1] if read_ms else 0.0,
    )

    # 6. Breakdown from the archive's own blocks and entries.
    by_strategy: Dict[str, Dict[str, int]] = {}
    by_lane: Dict[str, Dict[str, int]] = {}
    for b in arch.blocks:
        s = by_strategy.setdefault(
            str(b.strategy_id), {"blocks": 0, "raw": 0, "comp": 0}
        )
        s["blocks"] += 1
        s["raw"] += b.raw_size
        s["comp"] += len(b.compressed_payload)
        lane = re.sub(r"_\d+$", "", b.lane_name)
        v = by_lane.setdefault(lane, {"blocks": 0, "raw": 0, "comp": 0})
        v["blocks"] += 1
        v["raw"] += b.raw_size
        v["comp"] += len(b.compressed_payload)
    by_ext: Dict[str, Dict[str, int]] = {}
    for e in arch.entries:
        t = by_ext.setdefault(e.ext.lower() or "(none)", {"files": 0, "raw": 0})
        t["files"] += 1
        t["raw"] += e.raw_size

    # 7. Competitors: one solid stream of the same bytes in the same order.
    competitors: Dict[str, Dict[str, Any]] = {}
    solid = args.work / "solid.bin"
    if not args.no_competitors:
        with open(solid, "wb") as fh:
            for _, data in files:
                fh.write(data)
        specs = {
            "xz": (
                ["xz", "-9e", "-T1", "-k", "-c", str(solid)],
                args.work / "solid.xz",
            ),
            "zstd": (
                ["zstd", "--ultra", "-22", "--long=31", "-T1", "-c", str(solid)],
                args.work / "solid.zst",
            ),
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                (tool, pool.submit(run_competitor, tool, argv, comp_out))
                for tool, (argv, comp_out) in specs.items()
            ]
            for tool, fut in futures:
                rec = fut.result()
                competitors[tool] = rec
                print(
                    f"competitor {tool} {rec.get('status')} "
                    f"bytes={rec.get('bytes', '-')} s={rec.get('seconds', '-')}"
                )
        for p in (solid, args.work / "solid.xz", args.work / "solid.zst"):
            p.unlink(missing_ok=True)

    # 8. Provenance.
    prov: Dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "vortexpak_version": vortexpak.__version__,
        "git_head": run_git(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "argv": list(sys.argv),
    }

    # 9. Results.
    ratio = vpak_bytes / download_bytes
    passed = vpak_bytes <= target_bytes
    result: Dict[str, Any] = {
        "provenance": prov,
        "release": {
            "pk3_files": pk3_files,
            "download_bytes": download_bytes,
            "target_bytes": target_bytes,
            "target_factor": TARGET_NUM / TARGET_DEN,
        },
        "assets": {
            "files": len(files),
            "raw_bytes": sum(len(d) for _, d in files),
            "limit": args.limit,
        },
        "vortexpak": {
            "layout": args.layout,
            "backend": backend,
            "bytes": vpak_bytes,
            "pack_s": round(pack_s, 2),
            "decode_s": round(decode_s, 2),
            "x_download": round(ratio, 4),
        },
        "verify": {
            "files": len(files),
            "mismatches_a": mismatches_a,
            "verify_archive_ok": verify_ok,
            "verify_archive_files": verify_n,
            "mismatches_b": mismatches_b,
        },
        "vfs": {
            "total_s": round(vfs_total_s, 2),
            "reads": len(read_ms),
            "p50_ms": round(p50_ms, 3),
            "p99_ms": round(p99_ms, 3),
            "max_ms": round(max_ms, 3),
        },
        "dedup": {
            "files": metrics.get("dedup_files"),
            "bytes": metrics.get("dedup_bytes"),
        },
        "breakdown": {
            "by_strategy": by_strategy,
            "by_lane": by_lane,
            "by_extension": by_ext,
        },
        "competitors": competitors,
        "pass": passed,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )

    def row(label: str, b: Optional[int]) -> str:
        if b is None:
            return f"| {label} | - | - | - |"
        return f"| {label} | {b:,} | {b / MB:,.1f} | {b / download_bytes:.4f}x |"

    xz_rec, zst_rec = competitors.get("xz"), competitors.get("zstd")
    verdict = "PASS" if passed else "FAIL"
    best_name: Optional[str] = None
    best_bytes: Optional[int] = None
    for cname, crec in competitors.items():
        cbytes = crec.get("bytes")
        if cbytes is not None and (best_bytes is None or cbytes < best_bytes):
            best_name, best_bytes = cname, cbytes
    lines = [
        "# Xonotic 0.8.6 lossless repack",
        "",
        f"Goal: byte-exact VortexPak strict archive of the whole release <= {TARGET_NUM / TARGET_DEN:.2f}x "
        f"the shipped download.",
        "",
        "| Archive | Bytes | MB | x download |",
        "|---|---:|---:|---:|",
        row("shipped download (pk3, ZIP/DEFLATE)", download_bytes),
        row("target 0.80x", target_bytes),
        row(f"VortexPak strict {args.layout} ({backend})", vpak_bytes),
        row("xz -9e -T1 solid", xz_rec.get("bytes") if xz_rec else None),
        row(
            "zstd --ultra -22 --long=31 -T1 solid",
            zst_rec.get("bytes") if zst_rec else None,
        ),
    ]
    if best_bytes is not None:
        delta_pct = (vpak_bytes - best_bytes) / best_bytes * 100.0
        lines.append("")
        lines.append(
            f"Smallest competitor: {best_name} {best_bytes / MB:,.1f} MB; "
            f"VortexPak is {delta_pct:+.1f} % versus it."
        )
    lines += [
        "",
        f"**{verdict}**: VortexPak {vpak_bytes / MB:,.1f} MB = {ratio:.4f}x download "
        f"{'<=' if passed else '>'} target {target_bytes / MB:,.1f} MB.",
        "",
        f"Verification: {len(files):,} files; A mismatches {mismatches_a}; verify_archive ok {verify_ok} "
        f"({verify_n:,} files); B mismatches {mismatches_b}.",
        "",
        f"Timings: pack {pack_s:.2f} s; full decode {decode_s:.2f} s; VFS {vfs_total_s:.2f} s total "
        f"({len(read_ms):,} reads, p50 {p50_ms:.3f} ms, p99 {p99_ms:.3f} ms, max {max_ms:.3f} ms).",
        "",
        f"Dedup: {metrics.get('dedup_files', 0):,} files, {metrics.get('dedup_bytes', 0) / MB:,.1f} MB.",
    ]
    lines += [
        "",
        "## Strategy breakdown",
        "",
        "| Strategy | Blocks | Raw MB | Comp MB |",
        "|---|---:|---:|---:|",
    ]
    for sid, v in sorted(by_strategy.items(), key=lambda kv: -kv[1]["comp"]):
        lines.append(
            f"| {sid} | {v['blocks']:,} | {v['raw'] / MB:,.1f} | {v['comp'] / MB:,.1f} |"
        )
    lines += [
        "",
        "## Lane breakdown",
        "",
        "| Lane | Blocks | Raw MB | Comp MB |",
        "|---|---:|---:|---:|",
    ]
    for lane, v in sorted(by_lane.items(), key=lambda kv: -kv[1]["comp"]):
        lines.append(
            f"| {lane} | {v['blocks']:,} | {v['raw'] / MB:,.1f} | {v['comp'] / MB:,.1f} |"
        )
    lines += [
        "",
        "## Extension breakdown",
        "",
        "| Extension | Files | Raw MB |",
        "|---|---:|---:|",
    ]
    for ext, v in sorted(by_ext.items(), key=lambda kv: -kv[1]["raw"]):
        lines.append(f"| {ext} | {v['files']:,} | {v['raw'] / MB:,.1f} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- One run; sizes are deterministic, times are a single observation on a shared host and indicative only.",
        "- Competitors are solid streams without random access, while VortexPak reads any file individually.",
        "- The solid concatenation carries no per-file headers, which slightly favours the competitors.",
        "- Competitors run single-threaded at their strongest settings "
        "(xz -9e, zstd --ultra -22 --long=31); multi-threaded modes lose ratio.",
        "- Compact layout decodes a whole 16 MiB block per cold single-file read; "
        "the VFS p99/max above is that cost (use --layout stream for streaming).",
        "",
        "Reproduce with the command in the script docstring.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    # 10. Machine-readable verdict line; mismatch is the only error condition.
    def cb(tool: str) -> str:
        rec = competitors.get(tool)
        b = rec.get("bytes") if rec else None
        return str(b) if b is not None else "-"

    print(
        f"XONOTIC files={len(files)} vpak={vpak_bytes} ratio={ratio:.4f} target={target_bytes} "
        f"pass={passed} xz={cb('xz')} zstd={cb('zstd')} "
        f"mismatches_a={mismatches_a} mismatches_b={mismatches_b}"
    )
    sys.exit(2 if (mismatches_a or mismatches_b) else 0)


if __name__ == "__main__":
    main()
