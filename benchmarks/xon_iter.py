"""Private Xonotic iteration tiers; only xonotic_lossless.py publishes claims.

    python -B benchmarks/xon_iter.py exact --expect 924623172
    python -B benchmarks/xon_iter.py quick
    python -B benchmarks/xon_iter.py calibrate

T1 uses a fixed file sample; T2 rebuilds the whole archive from cached blocks.
T3 remains benchmarks/xonotic_lossless.py, with its full public verification.
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vortexpak.codec import (  # noqa: E402
    MicroBlock,
    VortexArchive,
    VortexPakCodec,
    _decompress_block,
)
from vortexpak.native import NATIVE_LIB_PATH  # noqa: E402
from vortexpak.vfs import VortexVFS  # noqa: E402

CALIBRATION = ROOT / "benchmarks/reports/v2/xon-iter/calibration.json"
CACHE_HEADER = struct.Struct("<IQI")


def fingerprint(codec: VortexPakCodec) -> str:
    """Invalidate cached blocks whenever the loaded implementation changes."""
    digest = hashlib.sha256()
    native = NATIVE_LIB_PATH.read_bytes() if NATIVE_LIB_PATH is not None else b"none"
    for content in (
        native,
        (ROOT / "vortexpak/codec.py").read_bytes(),
        (ROOT / "vortexpak/native.py").read_bytes(),
    ):
        digest.update(hashlib.sha256(content).digest())
    for value in (codec.mode, codec.layout, codec.entropy_backend):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cached_blocks(
    codec: VortexPakCodec,
    raw_blocks: List[Tuple[int, str, bytes]],
    cache_dir: Path,
    fp: str,
) -> Tuple[List[MicroBlock], int, int]:
    """Reuse independent blocks; verify every fresh payload before atomic storage."""
    blocks: List[MicroBlock] = []
    missing: List[Tuple[int, str, bytes]] = []
    destinations: Dict[int, Path] = {}
    for bid, lane, raw in raw_blocks:
        key = hashlib.sha256(
            fp.encode("utf-8") + b"\0" + lane.encode("utf-8") + b"\0" + raw
        ).hexdigest()
        path = cache_dir / key[:2] / (key + ".blk")
        destinations[bid] = path
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        if len(data) >= CACHE_HEADER.size:
            strategy, raw_size, payload_len = CACHE_HEADER.unpack_from(data)
            if strategy in (9, 10):
                raise SystemExit("dictionary strategies cannot be cached")
            if (
                strategy in (0, 1, 2, 3, 4, 5, 8, 11)
                and raw_size == len(raw)
                and payload_len == len(data) - CACHE_HEADER.size
                and (strategy != 0 or payload_len == raw_size)
                and (strategy == 0 or payload_len > 0)
            ):
                blocks.append(
                    MicroBlock(bid, lane, strategy, raw_size, data[CACHE_HEADER.size :])
                )
                continue
        missing.append((bid, lane, raw))
    hits = len(blocks)
    if missing:
        raw_by_id = {bid: raw for bid, _, raw in missing}
        fresh = codec.compress_blocks(missing)
        if sorted(b.block_id for b in fresh) != sorted(raw_by_id):
            raise SystemExit("compression returned inconsistent block IDs")
        for block in fresh:
            if block.strategy_id in (9, 10):
                raise SystemExit("dictionary strategies cannot be cached")
            try:
                valid = _decompress_block(block) == raw_by_id[block.block_id]
            except Exception:
                valid = False
            if not valid:
                raise SystemExit(f"ROUNDTRIP {block.block_id}")
            path = destinations[block.block_id]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fh:
                    tmp = Path(fh.name)
                    fh.write(
                        CACHE_HEADER.pack(
                            block.strategy_id,
                            block.raw_size,
                            len(block.compressed_payload),
                        )
                    )
                    fh.write(block.compressed_payload)
                os.replace(tmp, path)
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            blocks.append(block)
    blocks.sort(key=lambda b: b.block_id)
    return blocks, hits, len(missing)


def quick_sample(files: List[Tuple[str, bytes]]) -> List[Tuple[str, bytes]]:
    """Select a deterministic, extension-stratified six-percent file sample."""
    groups: Dict[str, List[Tuple[str, bytes]]] = {}
    for name, data in files:
        groups.setdefault(Path(name).suffix.lower(), []).append((name, data))
    sample: List[Tuple[str, bytes]] = []
    for group in groups.values():
        total = sum(len(data) for _, data in group)
        if total < 1_000_000:
            sample.extend(group)
            continue
        ordered = sorted(
            group,
            key=lambda item: (
                hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
                item[0],
            ),
        )
        taken = 0
        for item in ordered:
            sample.append(item)
            taken += len(item[1])
            if taken * 100 >= total * 6:
                break
    return sorted(sample, key=lambda item: item[0])


def _git(argv: List[str]) -> str:
    try:
        result = subprocess.run(
            ["git"] + argv, cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _calibrations() -> List[Dict[str, Any]]:
    if not CALIBRATION.exists():
        return []
    records = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit(f"invalid calibration list: {CALIBRATION}")
    return records


def _estimate(sample_bytes: int, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    estimate = None
    delta = 0
    delta_pct = 0.0
    if records:
        last = records[-1]
        estimate = sample_bytes * last["full_bytes"] / last["sample_bytes"]
        delta = sample_bytes - last["sample_bytes"]
        delta_pct = delta * 100.0 / last["sample_bytes"]
    errors = [
        abs(
            prev["full_bytes"] * cur["sample_bytes"] / prev["sample_bytes"]
            - cur["full_bytes"]
        )
        / cur["full_bytes"]
        for prev, cur in zip(records, records[1:])
        if prev["fingerprint"] != cur["fingerprint"]
    ]
    return {
        "est_full": round(estimate) if estimate is not None else None,
        "delta": delta,
        "delta_pct": delta_pct,
        "band": max(errors) if errors else "uncalibrated",
    }


def _encode(
    codec: VortexPakCodec,
    files: List[Tuple[str, bytes]],
    args: argparse.Namespace,
    fp: str,
    download: int,
    provenance: Dict[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    start = time.perf_counter()
    plan = codec.plan_blocks(files)
    planned = time.perf_counter()
    blocks, hits, recompressed = cached_blocks(codec, plan.raw_blocks, args.cache, fp)
    compressed = time.perf_counter()
    archive = codec.finish_archive(plan, blocks)
    del plan
    blob = archive.to_bytes()
    assembled = time.perf_counter()
    verified = 0
    if not args.no_verify:
        parsed = VortexArchive.from_bytes(blob)
        decoded = codec.decode_archive(parsed, materialize_full_rgba=True)
        for name, raw in files:
            if (
                name not in decoded
                or hashlib.sha256(bytes(decoded[name])).digest()
                != hashlib.sha256(raw).digest()
            ):
                print(f"MISMATCH {name}", flush=True)
                raise SystemExit(2)
            verified += 1
        del decoded, parsed
    checked = time.perf_counter()
    by_lane: Dict[str, Dict[str, int]] = {}
    by_strategy: Dict[str, Dict[str, int]] = {}
    for block in archive.blocks:
        for table, key in (
            (by_lane, re.sub(r"_\d+$", "", block.lane_name)),
            (by_strategy, str(block.strategy_id)),
        ):
            row = table.setdefault(key, {"blocks": 0, "raw": 0, "comp": 0})
            row["blocks"] += 1
            row["raw"] += block.raw_size
            row["comp"] += len(block.compressed_payload)
    result: Dict[str, Any] = {
        "tier": "T1" if quick else "T2",
        "public_claim": False,
        "bytes": len(blob),
        "download": download,
        "ratio": len(blob) / download,
        "target": download * 80 // 100,
        "pass": len(blob) <= download * 80 // 100,
        "files": len(files),
        "raw": sum(len(raw) for _, raw in files),
        "blocks": len(archive.blocks),
        "hits": hits,
        "recompressed": recompressed,
        "by_lane": by_lane,
        "by_strategy": by_strategy,
        "timings": {
            "plan": planned - start,
            "blocks": compressed - planned,
            "assemble": assembled - compressed,
            "verify": checked - assembled,
        },
        "verified": verified,
        "fingerprint": fp,
        **provenance,
    }
    if quick:
        reads: List[float] = []
        vfs_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="xon_iter_") as work:
            path = Path(work) / "sample.vpak"
            path.write_bytes(blob)
            with VortexVFS.open(path) as vfs:
                for name, raw in sorted(files):
                    before = time.perf_counter()
                    data = vfs.read_bytes(name)
                    reads.append((time.perf_counter() - before) * 1000.0)
                    if (
                        not args.no_verify
                        and hashlib.sha256(data).digest()
                        != hashlib.sha256(raw).digest()
                    ):
                        print(f"MISMATCH {name}", flush=True)
                        raise SystemExit(2)
        reads.sort()
        result["vfs"] = {
            "p50_ms": reads[max(0, math.ceil(len(reads) * 0.50) - 1)] if reads else 0.0,
            "p99_ms": reads[max(0, math.ceil(len(reads) * 0.99) - 1)] if reads else 0.0,
            "max_ms": reads[-1] if reads else 0.0,
        }
        result["timings"]["vfs"] = time.perf_counter() - vfs_start
        result.update(_estimate(len(blob), _calibrations()))
        result["est_ratio"] = (
            result["est_full"] / download if result["est_full"] is not None else None
        )
    result["secs"] = time.perf_counter() - start
    return result


def _line(result: Dict[str, Any], quick: bool) -> str:
    tail = f"verified={result['verified']}/{result['files']} secs={result['secs']:.2f}"
    if not quick:
        return (
            f"EXACT bytes={result['bytes']} ratio={result['ratio']:.4f} "
            f"pass={result['pass']} blocks={result['blocks']} hits={result['hits']} "
            f"recompressed={result['recompressed']} {tail}"
        )
    band = result["band"]
    band_text = f"{band * 100:.3f}%" if isinstance(band, (int, float)) else band
    estimate = result["est_full"] if result["est_full"] is not None else "uncalibrated"
    ratio = (
        f"{result['est_ratio']:.4f}"
        if result["est_ratio"] is not None
        else "uncalibrated"
    )
    return (
        f"QUICK sample_files={result['files']} sample_raw={result['raw']} "
        f"sample_bytes={result['bytes']} delta={result['delta']:+d} "
        f"({result['delta_pct']:+.3f}%) est_full={estimate} est_ratio={ratio} "
        f"band={band_text} vfs_p99_ms={result['vfs']['p99_ms']:.3f} {tail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    for command in ("exact", "quick", "calibrate"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--assets",
            type=Path,
            default=Path(r"..\vortex_corpus\xonotic\xonotic_assets"),
        )
        sub.add_argument(
            "--release",
            type=Path,
            default=Path(r"..\vortex_corpus\xonotic\Xonotic\data"),
        )
        sub.add_argument("--cache", type=Path, default=Path(".empryo/cache/xon_iter"))
        sub.add_argument("--workers", type=int, default=16)
        sub.add_argument("--no-verify", action="store_true")
        sub.add_argument("--out", type=Path)
        if command == "exact":
            sub.add_argument("--expect", type=int)
    args = parser.parse_args()
    if args.cmd == "calibrate":
        args.no_verify = False
    if not args.assets.is_dir():
        raise SystemExit(f"assets directory not found: {args.assets}")
    pk3s = list(args.release.glob("*.pk3"))
    if not pk3s:
        raise SystemExit(f"no .pk3 files in {args.release}")
    download = sum(path.stat().st_size for path in pk3s)
    if download <= 0:
        raise SystemExit("empty release download")
    files = [
        (path.relative_to(args.assets).as_posix(), path.read_bytes())
        for path in sorted(args.assets.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise SystemExit("no asset files")
    codec = VortexPakCodec(mode="strict", layout="compact", max_workers=args.workers)
    try:
        if codec.entropy_backend != "vxcm":
            raise SystemExit("Xonotic iteration requires the native VXCM backend")
        fp = fingerprint(codec)
        dirty = _git(["status", "--porcelain"])
        provenance = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_dirty": "unknown" if dirty == "unknown" else bool(dirty),
        }
        if args.cmd == "calibrate":
            quick = _encode(
                codec, quick_sample(files), args, fp, download, provenance, True
            )
            print(_line(quick, True), flush=True)
            exact = _encode(codec, files, args, fp, download, provenance, False)
            records = _calibrations()
            records.append(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "git_head": provenance["git_head"],
                    "fingerprint": fp,
                    "sample_files": quick["files"],
                    "sample_bytes": quick["bytes"],
                    "full_bytes": exact["bytes"],
                }
            )
            CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION.write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            result = {"quick": quick, "exact": exact}
            line = _line(exact, False)
        else:
            is_quick = args.cmd == "quick"
            result = _encode(
                codec,
                quick_sample(files) if is_quick else files,
                args,
                fp,
                download,
                provenance,
                is_quick,
            )
            line = _line(result, is_quick)
        out = args.out or Path(f".empryo/probes/out/xon_{args.cmd}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(line, flush=True)
        if (
            args.cmd == "exact"
            and args.expect is not None
            and result["bytes"] != args.expect
        ):
            raise SystemExit(3)
    finally:
        codec._pool.shutdown()


if __name__ == "__main__":
    main()
