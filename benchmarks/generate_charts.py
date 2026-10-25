"""
Generate High-Resolution Benchmark Visualizations for VortexPak vs Competitors
(Includes Ablation Baselines, Modern WebP-Lossless, and Outlier-Free Breakdown)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = "Segoe UI"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8


def main() -> None:
    bench_dir = Path(__file__).resolve().parent
    results = json.loads((bench_dir / "results.json").read_text(encoding="utf-8"))
    charts_dir = bench_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    comb = results["combined_all_3_games"]

    # 1. Chart 1: Pareto Frontier (Archive Size MB vs Ready-to-GPU Load Time & Archive Unpack Time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=180)
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
        ax.tick_params(colors="#e2e8f0", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    colors = {
        "Per-File (.pak)": "#64748b",
        "Solid Competitor (1T)": "#38bdf8",
        "Parallel-8T Competitor": "#22d3ee",
        "Ablation Baseline (8T)": "#f43f5e",
        "Modern Image Baseline (8T)": "#fb923c",
        "VortexPak (Ours)": "#10b981",
    }
    markers = {
        "Per-File (.pak)": "s",
        "Solid Competitor (1T)": "o",
        "Parallel-8T Competitor": "D",
        "Ablation Baseline (8T)": "^",
        "Modern Image Baseline (8T)": "P",
        "VortexPak (Ours)": "*",
    }

    offsets_a = {
        "Per-File LZ4": (0, 8),
        "Per-File ZIP/Zlib-6": (38, 6),
        "Per-File Zstd-3": (-36, 6),
        "Per-File Zstd-9": (-36, -10),
        "Solid LZ4": (0, 8),
        "Solid Zlib-6": (34, -8),
        "Solid Zstd-3": (-32, 6),
        "Solid Zstd-9": (34, 4),
        "Solid Zstd-19 (Ultra)": (0, -14),
        "Solid Brotli-6": (0, 8),
        "Solid LZMA (7z/XZ)": (0, 8),
        "Parallel-8T LZ4": (14, 8),
        "Parallel-8T Zstd-3": (36, 7),
        "Parallel-8T Zstd-9": (42, -6),
        "Parallel-8T VXLZ-ANS (Proprietary)": (38, -14),
        "WebP-Lossless + Parallel-8T Zstd-9": (-36, 9),
        "Parallel-8T Raw-Pixels + Zstd-9": (0, 9),
        "RDO-Quantized + Raw-Zstd-9 (8T)": (56, -4),
        "VortexPak-Strict (Pure VXLZ-ANS)": (-18, -20),
        "VortexPak-Engine (Pure VXLZ-ANS)": (36, 10),
        "VortexPak-RDO (Pure VXLZ-ANS)": (52, 8),
        "VortexPak-Engine (VXPK-Zstd)": (-46, -14),
        "VortexPak-RDO (VXPK-Zstd)": (-42, -16),
    }

    # Left subplot: Compressed Size (MB) vs Raw Archive Decompress Time (ms) — Log X
    for row in comb:
        cat = row["codec"]
        grp = row["category"]
        sz_mb = row["compressed_bytes"] / (1024 * 1024)
        dec_ms = row["decode_ms"]
        is_ours = grp == "VortexPak (Ours)"
        if "VortexPak-RDO" in cat:
            c = "#a855f7"
        elif "Strict" in cat:
            c = "#f59e0b"
        elif is_ours:
            c = "#10b981"
        else:
            c = colors.get(grp, "#38bdf8")

        sz_pt = 260 if is_ours else 95
        ax1.scatter(
            dec_ms,
            sz_mb,
            c=c,
            marker=markers.get(grp, "o"),
            s=sz_pt,
            zorder=5,
            edgecolors="white",
            linewidth=1.0,
        )
        dx, dy = offsets_a.get(cat, (0, 8))
        weight = "bold" if is_ours else "normal"
        txt_color = (
            "#c084fc"
            if "VortexPak-RDO" in cat
            else (
                "#fde047" if "Strict" in cat else ("#6ee7b7" if is_ours else "#e2e8f0")
            )
        )
        short_lbl = (
            cat.replace(" (Ultra)", "")
            .replace(" (7z/XZ)", "")
            .replace(" (Perceptual RDO)", " (56 dB / 0.998 SSIM)")
            .replace("WebP-Lossless + Parallel-8T Zstd-9", "WebP-Lossless+Zstd9")
            .replace("Parallel-8T Raw-Pixels + Zstd-9", "Ablation: RawPix+Zstd9")
            .replace("RDO-Quantized + Raw-Zstd-9 (8T)", "Ablation: RDO+RawZstd9")
        )
        ax1.annotate(
            short_lbl,
            (dec_ms, sz_mb),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=8.0 if not is_ours else 9.2,
            fontweight=weight,
            color=txt_color,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel(
        "Čas dekompresie archívu (ms, log škála — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_ylabel(
        "Veľkosť skomprimovanej hry (MB — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_title(
        "A) Pareto Hranica: Veľkosť Archívu vs. Čas Dekompresie",
        color="#f8fafc",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylim(3.7, 12.6)
    ax1.set_xlim(1.8, 500.0)

    offsets_b = {
        "Per-File LZ4": (0, 8),
        "Solid LZ4": (24, 6),
        "Solid Zstd-9": (38, -6),
        "Parallel-8T Zstd-9": (-45, 14),
        "Parallel-8T VXLZ-ANS (Proprietary)": (42, -16),
        "WebP-Lossless + Parallel-8T Zstd-9": (0, 9),
        "Parallel-8T Raw-Pixels + Zstd-9": (0, 9),
        "RDO-Quantized + Raw-Zstd-9 (8T)": (-36, 9),
        "Solid LZMA (7z/XZ)": (0, 8),
        "VortexPak-Strict (Pure VXLZ-ANS)": (-54, -12),
        "VortexPak-Engine (Pure VXLZ-ANS)": (36, 9),
        "VortexPak-RDO (Pure VXLZ-ANS)": (54, 6),
        "VortexPak-Engine (VXPK-Zstd)": (-46, -12),
        "VortexPak-RDO (VXPK-Zstd)": (-42, -16),
    }

    # Right subplot: Compressed Size (MB) vs Total Ready-to-GPU Load Time (ms)
    for row in comb:
        cat = row["codec"]
        grp = row["category"]
        sz_mb = row["compressed_bytes"] / (1024 * 1024)
        gpu_ms = row["gpu_ready_ms"]
        is_ours = grp == "VortexPak (Ours)"
        if "VortexPak-RDO" in cat:
            c = "#a855f7"
        elif "Strict" in cat:
            c = "#f59e0b"
        elif is_ours:
            c = "#10b981"
        else:
            c = colors.get(grp, "#38bdf8")

        sz_pt = 260 if is_ours else 95
        ax2.scatter(
            gpu_ms,
            sz_mb,
            c=c,
            marker=markers.get(grp, "o"),
            s=sz_pt,
            zorder=5,
            edgecolors="white",
            linewidth=1.0,
        )
        if cat in offsets_b:
            dx, dy = offsets_b[cat]
            txt_color = (
                "#c084fc"
                if "VortexPak-RDO" in cat
                else (
                    "#fde047"
                    if "Strict" in cat
                    else ("#6ee7b7" if is_ours else "#e2e8f0")
                )
            )
            short_lbl = (
                cat.replace(" (Ultra)", "")
                .replace(" (7z/XZ)", "")
                .replace(" (Perceptual RDO)", " (56 dB)")
                .replace("WebP-Lossless + Parallel-8T Zstd-9", "WebP-Lossless+Zstd9")
                .replace("Parallel-8T Raw-Pixels + Zstd-9", "Ablation: RawPix+Zstd9")
                .replace("RDO-Quantized + Raw-Zstd-9 (8T)", "Ablation: RDO+RawZstd9")
            )
            ax2.annotate(
                short_lbl,
                (gpu_ms, sz_mb),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                fontsize=8.0 if not is_ours else 9.2,
                fontweight="bold" if is_ours else "normal",
                color=txt_color,
            )

    ax2.set_xscale("log")
    ax2.set_xlabel(
        "Celkový čas načítania do RAM/GPU vrátane 695 textúr (ms, log — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Veľkosť skomprimovanej hry (MB — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.set_title(
        "B) Ready-to-GPU Load: Rozbalenie + Plná Materializácia RGBA",
        color="#f8fafc",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylim(3.7, 12.6)
    ax2.set_xlim(8.0, 700.0)

    plt.tight_layout()
    p1 = charts_dir / "pareto_frontier.png"
    plt.savefig(p1, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Chart 2: Bar comparison of Compressed Size (MB) & Materialized GPU-Ready Throughput (GB/s)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=180)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, axis="x", linestyle="--", alpha=0.25, color="#94a3b8")
        ax.tick_params(colors="#e2e8f0", labelsize=9.2)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    selected = [
        r
        for r in comb
        if r["codec"]
        in (
            "Per-File LZ4",
            "Per-File Zstd-9",
            "Solid LZ4",
            "Solid Zstd-9",
            "Solid Zstd-19 (Ultra)",
            "Solid LZMA (7z/XZ)",
            "Parallel-8T LZ4",
            "Parallel-8T Zstd-9",
            "Parallel-8T VXLZ-ANS (Proprietary)",
            "WebP-Lossless + Parallel-8T Zstd-9",
            "Parallel-8T Raw-Pixels + Zstd-9",
            "RDO-Quantized + Raw-Zstd-9 (8T)",
            "VortexPak-Strict (Pure VXLZ-ANS)",
            "VortexPak-Engine (Pure VXLZ-ANS)",
            "VortexPak-RDO (Pure VXLZ-ANS)",
            "VortexPak-Engine (VXPK-Zstd)",
            "VortexPak-RDO (VXPK-Zstd)",
        )
    ]

    labels = [r["codec"] for r in selected]
    sizes_mb = [r["compressed_bytes"] / (1024 * 1024) for r in selected]
    gpu_gbps = [r["gpu_materialized_gbps"] for r in selected]
    bar_colors = [
        (
            "#a855f7"
            if "VortexPak-RDO" in l
            else (
                "#10b981"
                if "VortexPak-Engine" in l
                else (
                    "#f59e0b"
                    if "Strict" in l
                    else (
                        "#fb923c"
                        if "WebP" in l
                        else (
                            "#f43f5e"
                            if ("Raw-Pixels" in l or "RDO-Quantized" in l)
                            else (
                                "#22d3ee"
                                if "Parallel-8T" in l
                                else ("#64748b" if "Per-File" in l else "#38bdf8")
                            )
                        )
                    )
                )
            )
        )
        for l in labels
    ]

    y_pos = np.arange(len(labels))
    bars1 = ax1.barh(
        y_pos, sizes_mb, color=bar_colors, height=0.68, edgecolor="#0f172a"
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel(
        "Skomprimovaná veľkosť všetkých 3 hier (MB — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.2,
        fontweight="bold",
    )
    ax1.set_title(
        "Veľkosť Archívu (Pôvodne 82.93 MB súbory / 323.25 MB GPU dáta)",
        color="#f8fafc",
        fontsize=11.0,
        fontweight="bold",
    )
    ax1.set_xlim(0, 13.5)
    for bar, val in zip(bars1, sizes_mb):
        ax1.text(
            val + 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} MB",
            va="center",
            color="#f8fafc",
            fontsize=8.8,
            fontweight="bold",
        )

    bars2 = ax2.barh(
        y_pos, gpu_gbps, color=bar_colors, height=0.68, edgecolor="#0f172a"
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in labels])
    ax2.invert_yaxis()
    ax2.set_xlabel(
        "Materializovaná GPU priepustnosť (GB/s z 323.25 MB RGBA + dát — viac je lepšie)",
        color="#f8fafc",
        fontsize=10.2,
        fontweight="bold",
    )
    ax2.set_title(
        "End-to-End GPU-Ready Priepustnosť (GB/s vrátane textúr)",
        color="#f8fafc",
        fontsize=11.0,
        fontweight="bold",
    )
    ax2.set_xlim(0, 26.5)
    for bar, val in zip(bars2, gpu_gbps):
        ax2.text(
            val + 0.22,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} GB/s",
            va="center",
            color="#f8fafc",
            fontsize=8.8,
            fontweight="bold",
        )

    plt.tight_layout()
    p2 = charts_dir / "bars_size_speed.png"
    plt.savefig(p2, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 3. Chart 3: Per-Game Size & Ready-to-GPU Speedup Breakdown (including Outlier-Free Corpus)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=180)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color="#94a3b8")
        ax.tick_params(colors="#e2e8f0", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    g_keys = [
        "minetest_game",
        "kenney_3d_platformer",
        "browserquest",
        "combined_no_tmx_outlier",
    ]
    g_titles = [
        "Minetest Game\n(1 434 súborov)",
        "Kenney 3D\n(100 súborov)",
        "BrowserQuest\n(452 súborov)",
        "Bez 67MB Mapy\n(1 985 súborov)",
    ]
    compare_codecs = [
        "Parallel-8T Zstd-9",
        "WebP-Lossless + Parallel-8T Zstd-9",
        "Parallel-8T Raw-Pixels + Zstd-9",
        "VortexPak-Engine (Pure VXLZ-ANS)",
        "VortexPak-RDO (Pure VXLZ-ANS)",
    ]
    c_map = {
        "Parallel-8T Zstd-9": "#22d3ee",
        "WebP-Lossless + Parallel-8T Zstd-9": "#fb923c",
        "Parallel-8T Raw-Pixels + Zstd-9": "#f43f5e",
        "VortexPak-Engine (Pure VXLZ-ANS)": "#10b981",
        "VortexPak-RDO (Pure VXLZ-ANS)": "#a855f7",
    }

    x = np.arange(len(g_keys))
    width = 0.16

    for idx, cname in enumerate(compare_codecs):
        vals_mb = []
        vals_gpu_ms = []
        for gk in g_keys:
            row = next(r for r in results[gk] if r["codec"] == cname)
            vals_mb.append(row["compressed_bytes"] / (1024 * 1024))
            vals_gpu_ms.append(row["gpu_ready_ms"])
        offset = (idx - 2.0) * width
        b1 = ax1.bar(
            x + offset,
            vals_mb,
            width,
            label=cname,
            color=c_map[cname],
            edgecolor="#0f172a",
        )
        b2 = ax2.bar(
            x + offset,
            vals_gpu_ms,
            width,
            label=cname,
            color=c_map[cname],
            edgecolor="#0f172a",
        )
        for bar, v in zip(b1, vals_mb):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.12,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=7.0,
                fontweight="bold",
            )
        for bar, v in zip(b2, vals_gpu_ms):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 2.5,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=7.0,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(g_titles, fontweight="bold", color="#f8fafc")
    ax1.set_ylabel(
        "Veľkosť po kompresii (MB — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.2,
        fontweight="bold",
    )
    ax1.set_title(
        "Porovnanie Veľkosti: Hry + Korpus bez 67MB Mapy (MB)",
        color="#f8fafc",
        fontsize=11.0,
        fontweight="bold",
    )
    ax1.set_ylim(0, 10.6)
    ax1.legend(
        facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=7.8
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(g_titles, fontweight="bold", color="#f8fafc")
    ax2.set_ylabel(
        "Ready-to-GPU Load Time (ms — menej je lepšie)",
        color="#f8fafc",
        fontsize=10.2,
        fontweight="bold",
    )
    ax2.set_title(
        "Čas Načítania do RAM/VRAM (ms): Hry + Korpus bez Mapy",
        color="#f8fafc",
        fontsize=11.0,
        fontweight="bold",
    )
    ax2.set_ylim(0, 235)
    ax2.legend(
        facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=7.8
    )

    plt.tight_layout()
    p3 = charts_dir / "per_game_breakdown.png"
    plt.savefig(p3, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Charts generated and copied to artifact directory successfully.")


if __name__ == "__main__":
    main()
