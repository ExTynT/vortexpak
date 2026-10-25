"""
Generate the VXTX-GPU quality dashboard from benchmarks/quality_matrix.json.

Panels:
  1. PSNR per scenario x codec configuration (grouped bars, 30 dB floor line)
  2. Encode throughput per configuration (log scale)
  3. Quality vs bpp scatter (every scenario x config point)
  4. Capability coverage heatmap vs published competitors

Run after quality_matrix.py — always renders the CURRENT measured state.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import lz4.frame as lz4_frame
import zstandard as zstd_mod
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

plt.rcParams["font.sans-serif"] = "Segoe UI"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8

BG = "#0f172a"
PANEL = "#1e293b"
FG = "#e2e8f0"
ACCENTS = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#fb7185"]

BENCH = Path(__file__).resolve().parent
MATRIX = BENCH / "quality_matrix.json"
CHARTS = BENCH / "charts"

PSNR_FLOOR = 30.0


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def main() -> None:
    doc = json.loads(MATRIX.read_text(encoding="utf-8"))
    CHARTS.mkdir(exist_ok=True)
    measured = doc["measured"]
    scenarios = [s for s in measured if s != "normal_map"]

    # ---- collect long-form rows: (scenario, config, psnr, bpp, mpx_s)
    rows = []
    for s in scenarios:
        for e in measured[s]["configs"]:
            rows.append((s, e["config"], e["psnr_db"], e["bpp"], e["encode_mpx_s"]))

    configs = []
    for _, cfg, *_ in rows:
        if cfg not in configs:
            configs.append(cfg)
    config_colors = {c: ACCENTS[i % len(ACCENTS)] for i, c in enumerate(configs)}

    fig, axes = plt.subplots(2, 2, figsize=(19, 11), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"VortexPak VXTX-GPU quality dashboard  —  {doc['updated_at'][:19]}  "
        f"({doc['image_size']} corpus, {len(rows)} measured points)",
        color=FG,
        fontsize=15,
        fontweight="bold",
    )

    # ---- Panel 1: PSNR grouped bars
    ax = axes[0][0]
    _style(ax)
    n_s = len(scenarios)
    n_c = len(configs)
    width = 0.8 / n_c
    xs = np.arange(n_s)
    for ci, cfg in enumerate(configs):
        vals = []
        for s in scenarios:
            e = next((e for e in measured[s]["configs"] if e["config"] == cfg), None)
            vals.append(e["psnr_db"] if e else 0.0)
        bars = ax.bar(
            xs + ci * width - 0.4 + width / 2,
            vals,
            width,
            label=cfg,
            color=config_colors[cfg],
            edgecolor=BG,
            linewidth=0.4,
        )
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.6,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=FG,
                )
    ax.axhline(PSNR_FLOOR, color="#fb7185", linestyle="--", linewidth=1.2)
    ax.text(
        n_s - 0.4,
        PSNR_FLOOR + 0.8,
        f"{PSNR_FLOOR:.0f} dB floor",
        color="#fb7185",
        fontsize=8,
    )
    ax.set_xticks(xs + width * (n_c - 1) / 2 - 0.2)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=7.5)
    ax.set_ylabel("PSNR [dB]")
    ax.set_title(
        "1 · Quality per scenario — where do we drop below the floor?", fontsize=11
    )
    ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=8, loc="upper right")

    # ---- Panel 2: encode throughput (log)
    ax = axes[0][1]
    _style(ax)
    agg = {}
    for _, cfg, _, _, mpx in rows:
        agg.setdefault(cfg, []).append(mpx)
    names = sorted(agg, key=lambda c: -np.mean(agg[c]))
    means = [np.mean(agg[c]) for c in names]
    bars = ax.bar(
        range(len(names)),
        means,
        color=[config_colors[c] for c in names],
        edgecolor=BG,
        linewidth=0.4,
    )
    ax.set_yscale("log")
    for b, c in zip(bars, names):
        ax.text(
            b.get_x() + b.get_width() / 2,
            np.mean(agg[c]) * 1.15,
            f"{np.mean(agg[c]):.0f}\nMpx/s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=FG,
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("encode throughput [Mpx/s, log]")
    ax.set_title("2 · Encode speed — SIMD gap is the tallest missing bar", fontsize=11)

    # ---- Panel 3: quality vs bpp scatter
    ax = axes[1][0]
    _style(ax)
    for s, cfg, psnr, bpp, _ in rows:
        if psnr < 5:
            continue
        ax.scatter(
            bpp,
            psnr,
            s=46,
            color=config_colors[cfg],
            edgecolor=BG,
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            f"{s.split('_')[0][:7]}",
            (bpp, psnr),
            fontsize=5.5,
            color=FG,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(PSNR_FLOOR, color="#fb7185", linestyle="--", linewidth=1.2)
    ax.set_xlabel("bitrate [bpp]")
    ax.set_ylabel("PSNR [dB]")
    ax.set_title(
        "3 · Quality vs bitrate — trade-off frontier (all measured points)", fontsize=11
    )
    handles = [mpatches.Patch(color=config_colors[c], label=c) for c in configs]
    ax.legend(handles=handles, facecolor=PANEL, labelcolor=FG, fontsize=8)

    # ---- Panel 4: capability heatmap
    ax = axes[1][1]
    _style(ax)
    cap = doc["capabilities"]
    val = {"yes": 1.0, "partial": 0.5, "no": 0.0, "unknown": 0.25}
    mat = np.array([[val[v] for v in cap["matrix"][f]] for f in cap["features"]])
    ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cap["vendors"])))
    ax.set_xticklabels(cap["vendors"], fontsize=8)
    ax.set_yticks(range(len(cap["features"])))
    ax.set_yticklabels(cap["features"], fontsize=7.5)
    for fi, f in enumerate(cap["features"]):
        for vi, v in enumerate(cap["matrix"][f]):
            ax.text(
                vi,
                fi,
                v,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#0f172a",
                fontweight="bold",
            )
    ax.set_title(
        "4 · Capability coverage vs competition (literature refs, not measurements)",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    out = CHARTS / "quality_dashboard.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"dashboard -> {out}")

    # ---- gap report PNG (text panel) for quick reading
    gaps = doc["gaps"]
    fig2, ax = plt.subplots(figsize=(13, max(3, 0.42 * len(gaps) + 1.5)), dpi=150)
    fig2.patch.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        f"Gap report — {len(gaps)} open items", color=FG, fontsize=13, fontweight="bold"
    )
    colors = {"high": "#fb7185", "medium": "#fbbf24", "low": "#38bdf8"}
    y = 0.97
    for g in gaps:
        ax.text(
            0.01,
            y,
            f"[{g['severity'].upper():6s}]",
            color=colors[g["severity"]],
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.115,
            y,
            f"{g['area']}",
            color=FG,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.33, y, g["detail"], color="#94a3b8", fontsize=8.5, transform=ax.transAxes
        )
        y -= 1.0 / max(len(gaps), 1) * 0.96
    out2 = CHARTS / "quality_gaps.png"
    fig2.savefig(out2, facecolor=BG)
    plt.close(fig2)
    print(f"gap report -> {out2}")

    competitor_chart(doc)
    vfs_latency_chart(doc)
    kodak_chart(doc)


def competitor_chart(doc: dict) -> None:
    """Measured-on-this-machine comparison vs general-purpose codecs
    (zstd-19 / lz4 on the same corpus) + PUBLISHED reference bars for
    specialist BCn encoders (different hardware — indicative only)."""
    scenarios = [
        "smooth_gradient",
        "gradient_noise",
        "structured_shapes",
        "bilinear_patch",
        "noise50",
    ]
    from quality_matrix import _smooth, _grad_noise, _structured, _bilinear, _noise50

    gen = {
        "smooth_gradient": _smooth,
        "gradient_noise": _grad_noise,
        "structured_shapes": _structured,
        "bilinear_patch": _bilinear,
        "noise50": _noise50,
    }
    agg = {}
    for s in scenarios:
        src = gen[s]().tobytes()
        n = len(src)
        t0 = time.perf_counter()
        c19 = zstd_mod.ZstdCompressor(level=19).compress(src)
        t1 = time.perf_counter()
        agg.setdefault("zstd-19 (measured)", []).append(
            (n / len(c19), n / 1e6 / (t1 - t0))
        )
        t0 = time.perf_counter()
        lz = lz4_frame.compress(src, compression_level=0)
        t2 = time.perf_counter()
        agg.setdefault("lz4 (measured)", []).append((n / len(lz), n / 1e6 / (t2 - t1)))
        for e in doc["measured"][s]["configs"]:
            if e["config"] in ("bc1_fast", "bc1_standard", "bc1_max"):
                agg.setdefault(e["config"] + " (measured)", []).append(
                    (n / e["size_kb"] / 1024, e["encode_mpx_s"])
                )
    # published reference points (indicative, other hardware)
    pub = {
        "ispc_texcomp BC1 (published)": 3000.0,
        "NVTT BC1 (published)": 400.0,
        "Oodle Texture (published)": 250.0,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "VortexPak vs competition — same corpus, this machine (blue) vs published references (hatched)",
        color=FG,
        fontsize=13,
        fontweight="bold",
    )

    ax = ax1
    _style(ax)
    names = list(agg)
    means_t = [np.mean([v[0] for v in agg[c]]) for c in names]
    bars = ax.bar(
        range(len(names)),
        means_t,
        color=[ACCENTS[i % len(ACCENTS)] for i in range(len(names))],
        edgecolor=BG,
    )
    for b, c, v in zip(bars, names, means_t):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 1.03,
            f"{v:.1f}x",
            ha="center",
            fontsize=9,
            color=FG,
        )
    ax.axhline(4.0, color="#fb7185", linestyle="--", linewidth=1.1)
    ax.text(
        len(names) - 0.45,
        4.12,
        "BC1/BC7 fixed 4x/2x bound",
        color="#fb7185",
        fontsize=7.5,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (measured)", "\n") for n in names], fontsize=8)
    ax.set_ylabel("compression ratio [x]")
    ax.set_title(
        "Compression ratio on identical corpus (higher = smaller)", fontsize=11
    )

    ax = ax2
    _style(ax)
    spd = {c: np.mean([v[1] for v in agg[c]]) for c in agg if "bc1" in c}
    bars = ax.bar(
        range(len(spd)),
        list(spd.values()),
        color=[ACCENTS[i % len(ACCENTS)] for i in range(len(spd))],
        edgecolor=BG,
        label="measured here",
    )
    for i, (name, ref) in enumerate(pub.items()):
        ax.bar(len(spd) + i, ref, color="#94a3b8", alpha=0.45, hatch="//", edgecolor=BG)
        ax.text(
            len(spd) + i, ref * 1.05, f"{ref:.0f}", ha="center", fontsize=8, color=FG
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(spd) + len(pub)))
    ax.set_xticklabels(
        [k.replace(" (measured)", "") for k in spd] + list(pub),
        fontsize=7.5,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("encode throughput [Mpx/s, log]")
    ax.set_title(
        "Encode throughput — measured here vs published (different HW)", fontsize=11
    )
    for b, c in zip(bars, spd):
        ax.text(
            b.get_x() + b.get_width() / 2,
            spd[c] * 1.1,
            f"{spd[c]:.0f}",
            ha="center",
            fontsize=8,
            color=FG,
        )

    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    out = CHARTS / "competitor_comparison.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"competitor comparison -> {out}")


def vfs_latency_chart(doc: dict) -> None:
    lat = doc.get("vfs_latency")
    if not lat:
        return
    series = [
        ("warm raw", lat["warm_cache_raw"], "#34d399"),
        ("warm materialized", lat["warm_cache_materialized"], "#a78bfa"),
        ("cold raw", lat["cold_random_raw"], "#38bdf8"),
        ("cold materialized", lat["cold_random_materialized"], "#f472b6"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"VFS read latency vs frame budget  —  {lat['entries']} assets, {lat['archive_mb']} MB archive",
        color=FG,
        fontsize=13,
        fontweight="bold",
    )

    ax = ax1
    _style(ax)
    width = 0.2
    xs = np.arange(4)  # p50, p95, p99, p999
    for si, (label, data, color) in enumerate(series):
        vals = [data["p50_ms"], data["p95_ms"], data["p99_ms"], data["p999_ms"]]
        ax.bar(
            xs + si * width - 0.3,
            vals,
            width,
            label=label,
            color=color,
            edgecolor=BG,
            linewidth=0.4,
        )
    for budget, color in [(1.0, "#fbbf24"), (4.0, "#fb7185"), (16.6, "#e2e8f0")]:
        ax.axhline(budget, linestyle="--", linewidth=1.1, color=color)
        ax.text(3.35, budget * 1.08, f"{budget} ms", color=color, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["p50", "p95", "p99", "p99.9"], fontsize=9)
    ax.set_ylabel("latency [ms, log]")
    ax.set_title("Percentiles vs frame-budget slices", fontsize=11)
    ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=8)

    ax = ax2
    _style(ax)
    labels = ["<= 1 ms", "<= 4 ms", "<= 16.6 ms"]
    xs = np.arange(3)
    keys = ["le_1ms_pct", "le_4ms_pct", "le_16ms_pct"]
    width = 0.2
    for si, (label, data, color) in enumerate(series):
        vals = [data[k] for k in keys]
        bars = ax.bar(
            xs + si * width - 0.3, vals, width, color=color, edgecolor=BG, linewidth=0.4
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 1,
                f"{v:.0f}",
                ha="center",
                fontsize=7,
                color=FG,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% reads within budget")
    ax.set_title("Budget compliance (100% = no stutter from this path)", fontsize=11)

    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    out = CHARTS / "vfs_latency.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"vfs latency -> {out}")


def kodak_chart(doc: dict) -> None:
    """Kodak PhotoCD (industry-standard texture codec corpus) with PSNR +
    NVIDIA FLIP perceptual metric. Directly comparable with published
    ISPC/Basis/NVTT numbers."""
    k = doc.get("kodak", {})
    if k.get("status") != "ok":
        return
    configs = ["bc1_max", "bc1_max_rdo", "bc7"]
    labels = ["BC1 MAX", "BC1 MAX+RDO", "BC7"]
    colors = ["#38bdf8", "#f472b6", "#a78bfa"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Kodak PhotoCD ({k.get('images', 24)} images, {k.get('image_size', '768x512')}) — industry-standard corpus",
        color=FG,
        fontsize=13,
        fontweight="bold",
    )

    psnrs = [k[c]["psnr_avg_db"] for c in configs]
    psnr_mins = [k[c]["psnr_min_db"] for c in configs]
    flips = [k[c]["flip_avg"] for c in configs]
    flip_worsts = [k[c]["flip_worst"] for c in configs]

    ax = ax1
    _style(ax)
    xs = np.arange(len(configs))
    bars = ax.bar(xs, psnrs, 0.55, color=colors, edgecolor=BG, linewidth=0.5)
    for b, v, mn in zip(bars, psnrs, psnr_mins):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.15,
            f"{v:.1f}",
            ha="center",
            fontsize=11,
            color=FG,
            fontweight="bold",
        )
        ax.plot(
            [b.get_x() + 0.1, b.get_x() + b.get_width() - 0.1],
            [mn, mn],
            color="#fb7185",
            linewidth=1.5,
            linestyle="--",
        )
        ax.text(
            b.get_x() + b.get_width() + 0.05,
            mn,
            f"min {mn:.0f}",
            fontsize=7,
            color="#fb7185",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PSNR [dB]")
    ax.set_title("PSNR (dashed = worst image)", fontsize=11)
    ax.set_ylim(15, 45)

    ax = ax2
    _style(ax)
    bars = ax.bar(xs, flips, 0.55, color=colors, edgecolor=BG, linewidth=0.5)
    for b, v, w in zip(bars, flips, flip_worsts):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.001,
            f"{v:.4f}",
            ha="center",
            fontsize=10,
            color=FG,
            fontweight="bold",
        )
        ax.plot(
            [b.get_x() + 0.1, b.get_x() + b.get_width() - 0.1],
            [w, w],
            color="#fbbf24",
            linewidth=1.5,
            linestyle="--",
        )
        ax.text(
            b.get_x() + b.get_width() + 0.03,
            w,
            f"worst {w:.3f}",
            fontsize=7,
            color="#fbbf24",
        )
    ax.axhline(0.10, color="#fb7185", linestyle="--", linewidth=1.2)
    ax.text(2.2, 0.102, "visible artifact threshold", color="#fb7185", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("FLIP (lower = better)")
    ax.set_title("NVIDIA FLIP perceptual metric (dashed = worst image)", fontsize=11)

    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    out = CHARTS / "kodak_quality.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"kodak quality -> {out}")


if __name__ == "__main__":
    main()
