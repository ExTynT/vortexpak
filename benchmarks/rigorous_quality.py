"""Independent-decoder texture quality, separate from byte-exact archives."""

import argparse
import io
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.rigorous import (
    digest,
    environment,
    manifest,
    median_ci,
)  # noqa: E402
from benchmarks import rigorous, protocol as bp, measurement as bm  # noqa: E402


def dds(blocks, w, h, fmt):
    dxgi, bb = (71, 8) if fmt == 1 else (98, 16)
    return (
        struct.pack("<4sI", b"DDS ", 124)
        + struct.pack("<IIIIII", 0x81007, h, w, (w // 4) * (h // 4) * bb, 0, 0)
        + bytes(44)
        + struct.pack("<II4sIIIII", 32, 4, b"DX10", 0, 0, 0, 0, 0)
        + struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
        + struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
        + blocks
    )


def reference_decode(blocks, w, h, fmt):
    import numpy as np
    from PIL import Image
    from vortexpak.bcn import BCN_BCX

    if fmt not in (1, 7, BCN_BCX) or min(w, h) <= 0 or w % 4 or h % 4:
        raise ValueError("unsupported reference format or dimensions")
    if fmt != BCN_BCX:
        if len(blocks) != (w // 4) * (h // 4) * (8 if fmt == 1 else 16):
            raise ValueError("invalid reference block length")
        with Image.open(io.BytesIO(dds(blocks, w, h, fmt))) as im:
            return np.array(im.convert("RGBA"))
    count = (w // 4) * (h // 4)
    bitmap_bytes = int.from_bytes(blocks[:4], "little")
    if bitmap_bytes != (count + 7) // 8 or len(blocks) < 4 + bitmap_bytes:
        raise ValueError("invalid BCX bitmap")
    bitmap, pos = blocks[4 : 4 + bitmap_bytes], 4 + bitmap_bytes
    planes = [bytearray(), bytearray()]
    flags = []
    for i in range(count):
        is7 = (bitmap[i // 8] >> (i % 8)) & 1
        flags.append(is7)
        length = 16 if is7 else 8
        planes[0].extend(bytes(8) if is7 else blocks[pos : pos + length])
        planes[1].extend(blocks[pos : pos + length] if is7 else b"\x40" + bytes(15))
        pos += length
    if pos != len(blocks):
        raise ValueError("invalid BCX payload length")
    a = reference_decode(bytes(planes[0]), w, h, 1)
    b = reference_decode(bytes(planes[1]), w, h, 7)
    mask = np.array(flags, dtype=bool).reshape(h // 4, w // 4).repeat(4, 0).repeat(4, 1)
    return np.where(mask[..., None], b, a)


def linear_rgb(a):
    import numpy as np

    a = a.astype(np.float64) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def metrics(original, decoded):
    import numpy as np
    import flip_evaluator as fe

    a, b = original[..., :3], decoded[..., :3]
    delta = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(delta * delta))
    alpha = original[..., 3].astype(np.float64) - decoded[..., 3].astype(np.float64)
    alpha_mse = float(np.mean(alpha * alpha))
    flip = fe.evaluate(
        a.astype(np.float32) / 255,
        b.astype(np.float32) / 255,
        "LDR",
        inputsRGB=True,
        applyMagma=False,
        computeMeanError=True,
        parameters={"ppd": 67.0},
    )
    return {
        "rgb_mse_srgb8": mse,
        "rgb_psnr_srgb_db": 10 * math.log10(255**2 / mse) if mse else None,
        "rgb_exact": mse == 0,
        "rgb_mse_linear": float(np.mean((linear_rgb(a) - linear_rgb(b)) ** 2)),
        "rgb_max_abs_error": int(np.abs(delta).max()),
        "alpha_mse": alpha_mse,
        "alpha_psnr_db": 10 * math.log10(255**2 / alpha_mse) if alpha_mse else None,
        "alpha_exact": alpha_mse == 0,
        "flip_rgb_ppd67": float(flip[1]),
    }


def metric_ci(values):
    if all(v is None for v in values):
        return {
            "n": len(values),
            "median": None,
            "bootstrap_95ci": [None, None],
            "exact": True,
        }
    if any(v is None for v in values):
        raise ValueError("mixed finite and exact quality results")
    return median_ci(values)


def display(value, digits=3):
    return "exact" if value is None else f"{value:.{digits}f}"


class MetricOracle:
    def __init__(self, validation_path="optimized"):
        if validation_path not in ("reference", "optimized"):
            raise ValueError("unknown validation path")
        self.validation_path = validation_path
        self.cache = {}
        self.reference_calls = self.validated_encodes = 0

    def evaluate(self, key, original, blocks, w, h, fmt):
        import numpy as np
        from vortexpak.bcn import decode_bcn
        from benchmarks import protocol

        encoded_key = protocol.sha(protocol.canonical(key))
        source = original.tobytes()
        entry = self.cache.get(encoded_key)
        if entry is not None and (
            entry["source"] != source or entry["blocks"] != blocks
        ):
            raise AssertionError(
                "nondeterministic input or encoded bytes for identical contract"
            )
        if entry is None or self.validation_path == "reference":
            ref = reference_decode(blocks, w, h, fmt)
            met = metrics(original, ref)
            ref.setflags(write=False)
            current = {
                "source": source,
                "blocks": bytes(blocks),
                "reference": ref,
                "metrics": met,
            }
            self.reference_calls += 1
            if entry is not None and (
                met != entry["metrics"] or not np.array_equal(ref, entry["reference"])
            ):
                raise AssertionError("oracle or metrics changed for identical pixels")
            entry = current
            self.cache[encoded_key] = entry
        ours = np.frombuffer(decode_bcn(blocks, w, h, fmt), np.uint8).reshape(h, w, 4)
        if not np.array_equal(ours, entry["reference"]):
            raise AssertionError("independent reference decoder mismatch")
        self.validated_encodes += 1
        return dict(entry["metrics"])


def check_oracle_vectors():
    import numpy as np

    red = reference_decode(struct.pack("<HHI", 0xF800, 0x07E0, 0), 4, 4, 1)
    zero = reference_decode(b"\x40" + bytes(15), 4, 4, 7)
    if not np.all(red == np.array([255, 0, 0, 255])) or np.any(zero):
        raise AssertionError("independent DDS golden vectors failed")


def main():
    started = time.perf_counter_ns()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument(
        "--profile", choices=list(bp.spec()["profiles"]), default="diagnostic"
    )
    ap.add_argument("--receipt", type=Path)
    ap.add_argument(
        "--validation-path", choices=["reference", "optimized"], default="optimized"
    )
    ap.add_argument("--require-kodak", action="store_true")
    args = ap.parse_args()
    if args.corpus.resolve() in [args.output.resolve(), *args.output.resolve().parents]:
        ap.error("output must be outside corpus")
    if args.output.exists() and any(args.output.iterdir()):
        ap.error("new empty output directory required")
    configs, original = bp.spec()["quality_configs"], manifest(args.corpus)
    cfg = bp.settings(
        args.profile,
        original,
        "quality",
        list(configs),
        args.threads,
        args.repeats,
        args.warmups,
        0,
        args.seed,
    )
    cfg.update(
        validation_path=args.validation_path,
        encoder_options=configs,
        ppd=67.0,
        pixel_policy=bp.spec()["contracts"]["pixels"],
        worker_model="resident images, summed encode calls",
    )
    if args.profile != "diagnostic" and (
        args.receipt is None or not args.require_kodak
    ):
        ap.error("quality reference requires --receipt and --require-kodak")
    os.environ.update(
        RAYON_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = (
        bp.capture_receipt(
            json.loads(args.receipt.read_text(encoding="utf-8")), args.output
        )
        if args.receipt
        else None
    )
    bp.activate(receipt)
    rigorous.ACTIVE_RECEIPT = receipt
    affinity = bm.bind_cpu_budget(args.threads)
    import numpy as np
    from PIL import Image
    from vortexpak.bcn import encode_bcn
    from vortexpak.native import IS_NATIVE_AVAILABLE

    if not IS_NATIVE_AVAILABLE:
        raise RuntimeError("native encoder unavailable")
    bp.verify_loaded(receipt)
    m, env = bp.freeze_inputs(original, args.output), environment()
    report = bp.initialize(m, cfg, "quality")
    report.update(
        environment=env,
        receipt=receipt,
        affinity=affinity,
        excluded=[],
        dependencies=bp.dependencies() if args.profile != "diagnostic" else {},
        calibration=bm.calibrate(),
        image_contracts={},
        image_schedule={},
    )
    bp.capture_harness(args.output, report)
    store, oracle = bp.EventStore(args.output, report), MetricOracle(
        args.validation_path
    )
    try:
        images = []
        for row in m["files"]:
            try:
                with Image.open(Path(m["root"]) / row["path"]) as im:
                    arr = np.array(im.convert("RGBA"))
            except (OSError, ValueError) as exc:
                report["excluded"].append({"path": row["path"], "reason": str(exc)})
                continue
            h, w = arr.shape[:2]
            if w % 4 or h % 4:
                report["excluded"].append(
                    {
                        "path": row["path"],
                        "reason": "dimensions not divisible by 4; no padding",
                    }
                )
                continue
            arr.setflags(write=False)
            report["image_contracts"][row["path"]] = {
                "width": w,
                "height": h,
                "channels": "RGBA8 straight",
                "pixels_sha256": digest(arr.tobytes()),
                "source_sha256": row["sha256"],
                "color_space": "sRGB interpretation",
            }
            images.append((row["path"], arr))
        if not images:
            raise ValueError("no eligible images")
        if args.require_kodak:
            if {n for n, _ in images} != {
                f"kodim{i:02}.png" for i in range(1, 25)
            } or report["excluded"]:
                raise ValueError("exact complete Kodak set required")
            if any(sorted(a.shape[:2]) != [512, 768] for _, a in images):
                raise ValueError("unexpected Kodak dimensions")
        check_oracle_vectors()
        identity = metrics(images[0][1], images[0][1])
        if identity["flip_rgb_ppd67"] > 1e-7 or not identity["rgb_exact"]:
            raise AssertionError("metric identity control failed")
        image_rng = random.Random(args.seed ^ 0xFA17)
        for round_ in report["schedule"]:
            rnd = round_["round"]
            image_order = image_rng.sample(images, len(images))
            report["image_schedule"][str(rnd)] = [n for n, _ in image_order]
            for name in round_["order"]:
                store.append("trial_started", {"round": rnd, "config": name})
                opt, rows, elapsed, stored, pixels, sse = (
                    configs[name],
                    [],
                    0,
                    0,
                    0,
                    0.0,
                )
                for path, arr in image_order:
                    h, w = arr.shape[:2]
                    start = time.perf_counter_ns()
                    blocks = encode_bcn(
                        arr.tobytes(),
                        w,
                        h,
                        opt["fmt"],
                        quality=opt["quality"],
                        rdo=opt["rdo"],
                    )
                    enc_ns = time.perf_counter_ns() - start
                    key = {
                        "config": name,
                        "options": opt,
                        "pixels": report["image_contracts"][path],
                        "ppd": 67,
                        "oracle_contract": report["protocol_id"],
                    }
                    met = oracle.evaluate(key, arr, blocks, w, h, opt["fmt"])
                    rows.append(
                        {
                            "path": path,
                            "pixels": w * h,
                            "bytes": len(blocks),
                            "encode_ns": enc_ns,
                            "block_sha256": digest(blocks),
                            **met,
                        }
                    )
                    elapsed, stored, pixels = (
                        elapsed + enc_ns,
                        stored + len(blocks),
                        pixels + w * h,
                    )
                    sse += met["rgb_mse_srgb8"] * w * h * 3
                mse = sse / (pixels * 3)
                trial = {
                    "config": name,
                    "round": rnd,
                    "images": rows,
                    "encode_ms": elapsed / 1e6,
                    "encode_ns": elapsed,
                    "affinity": affinity,
                    "bytes": stored,
                    "bpp": stored * 8 / pixels,
                    "encode_mpx_s": pixels / (elapsed / 1e9) / 1e6,
                    "rgb_mse_srgb8": mse,
                    "rgb_psnr_srgb_db": 10 * math.log10(255**2 / mse) if mse else None,
                    "worst_image_rgb_psnr_db": min(
                        (
                            r["rgb_psnr_srgb_db"]
                            for r in rows
                            if r["rgb_psnr_srgb_db"] is not None
                        ),
                        default=None,
                    ),
                    "flip_image_mean": float(
                        np.mean([r["flip_rgb_ppd67"] for r in rows])
                    ),
                    "flip_worst_image": max(r["flip_rgb_ppd67"] for r in rows),
                }
                store.trial(trial)
                report["trials" if rnd >= 0 else "warmup_trials"].append(trial)
                print(
                    f"round {rnd}: {name}: {display(trial['rgb_psnr_srgb_db'], 2)} dB RGB",
                    flush=True,
                )
                if rigorous.source_fingerprint() != env["source_fingerprint"]:
                    raise ValueError("implementation or harness changed")
        report["summary"] = {}
        for name in configs:
            rows = [t for t in report["trials"] if t["config"] == name]
            report["summary"][name] = {
                k: metric_ci([t[k] for t in rows])
                for k in ("encode_ms", "encode_mpx_s")
            }
            for k in (
                "bytes",
                "bpp",
                "rgb_mse_srgb8",
                "rgb_psnr_srgb_db",
                "worst_image_rgb_psnr_db",
                "flip_image_mean",
                "flip_worst_image",
            ):
                values = [t[k] for t in rows]
                report["summary"][name][k] = (
                    bp.exact_metric(values)
                    if all(v is not None for v in values)
                    else metric_ci(values)
                )
        report["oracle"] = {
            "validated_encodes": oracle.validated_encodes,
            "reference_calls": oracle.reference_calls,
            "unique_contracts": len(oracle.cache),
            "memoization": "within-run; exact input/output bytes",
        }
        report["host_after"] = bm.host_info()
        if env["host"]["power_scheme"] != report["host_after"]["power_scheme"]:
            raise RuntimeError("power scheme changed")
        report["adequacy"] = {
            "at_least_7_rounds": args.repeats >= 7,
            "vendor_encoders_measured": False,
            "real_gpu_execution": False,
            "all_24_kodak": args.require_kodak,
            "qualified_reference": args.profile != "diagnostic",
        }
        report["calibration_after"] = bm.calibrate()
        report["elapsed_wall_ns"] = time.perf_counter_ns() - started
        report["elapsed_scope"] = (
            "entry through measurement, excludes final serialization; suite measures child exit"
        )
        bp.finalize(args.output, report, store)
        lines = [
            "# Independent texture quality v2",
            "",
            "Run: " + report["run_id"],
            "Protocol: " + report["protocol_id"],
            "Evidence: " + report["evidence_head"],
            "",
            "| Config | bpp | RGB PSNR dB | Worst image dB | FLIP mean | Encode ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, s in sorted(report["summary"].items()):
            lines.append(
                "| "
                + name
                + " | "
                + " | ".join(
                    display(s[k]["median"])
                    for k in (
                        "bpp",
                        "rgb_psnr_srgb_db",
                        "worst_image_rgb_psnr_db",
                        "flip_image_mean",
                        "encode_ms",
                    )
                )
                + " |"
            )
        lines += [
            "",
            "BCX storage only. No real-GPU/vendor claim. Exact RGB quality; timing uncertainty over rounds.",
            "Diagnostic runs are not references; full valid evidence: report.json.gz.",
        ]
        with (args.output / "report.md").open("x", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except BaseException as exc:
        report["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        report["errors"].append({"type": type(exc).__name__, "error": str(exc)})
        store.append("run_" + report["status"], report["errors"][-1])
        bp.atomic_json(args.output / "failure.json", report)
        raise


if __name__ == "__main__":
    main()
