/**
 * VortexPak Engine Loader Demo
 * ============================
 * Demonstrates the complete texture-loading flow a game engine
 * (Godot / Bevy / Unity native plugin / custom engine) would use
 * via the VortexPak C-ABI:
 *
 *   1. Open a .vpak archive (seekable VFS)
 *   2. List assets / query metadata
 *   3. Stream a single texture's GPU block data on demand
 *   4. Read the stored GPU block payload (upload as-is for BC1/BC4/BC5/BC7)
 *   5. Read decoded RGBA texels (required for storage-only BCX)
 *   6. Measure load latency
 *
 * Build:  see CMakeLists.txt or compile directly against vortexpak.h
 * Run:    engine_loader <archive.vpak> [texture_name]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

#include "vortexpak.h"

/* ── VFS handle wrapper for clean RAII in C ─────────────────────── */
typedef struct {
    void *vfs;
} VpakArchive;

static VpakArchive vpak_open(const char *path) {
    VpakArchive a = {NULL};
    a.vfs = vfs_open_c(path);
    return a;
}

static void vpak_close(VpakArchive *a) {
    if (a->vfs) {
        vfs_close_c(a->vfs);
        a->vfs = NULL;
    }
}

/* ── Demo flow ──────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <archive.vpak> [asset_path]\n", argv[0]);
        return 1;
    }
    const char *vpak_path = argv[1];
    const char *asset = (argc > 2) ? argv[2] : NULL;

    printf("── VortexPak Engine Loader Demo ──\n\n");

    /* 1. Open archive (reads only the header — not the whole file) */
    double t0 = now_ms();
    VpakArchive ark = vpak_open(vpak_path);
    double t_open = now_ms() - t0;

    if (!ark.vfs) {
        fprintf(stderr, "Failed to open %s\n", vpak_path);
        return 1;
    }
    printf("Opened %s in %.3f ms (header-only seek)\n", vpak_path, t_open);

    /* 2. Query archive metadata */
    size_t n_assets = vfs_get_assets_count_c(ark.vfs);
    printf("Assets in archive: %zu\n", n_assets);

    /* 3. Find and stream a texture */
    const char *target = asset ? asset : "textures/t0.png";
    printf("\nStreaming '%s'...\n", target);

    /* Query size first (metadata-only, no decompression) */
    int64_t raw_size = vfs_get_asset_size_c(ark.vfs, target);
    if (raw_size < 0) {
        /* Try without extension */
        fprintf(stderr, "  Asset not found (tried '%s')\n", target);
        printf("  Hint: run `vortexpak ls %s` to list available assets\n",
               vpak_path);
        vpak_close(&ark);
        return 0;
    }
    printf("  Raw size: %lld B\n", (long long)raw_size);

    /* Allocate generously: raw_size is the original file size (e.g. PNG) and
     * the stored payload can be larger. A production loader would take the
     * texture dimensions from the entry metadata instead. */
    size_t buf_cap = (size_t)raw_size * 100 + 65536;
    uint8_t *buf = malloc(buf_cap);
    if (!buf) {
        vpak_close(&ark);
        return 1;
    }

    /* a) Stored payload (materialize=0). For BC1/BC4/BC5/BC7 textures this is
     *    spec-conformant GPU block data: hand it to glCompressedTexImage2D /
     *    vkCmdCopyBufferToImage as-is. BCX is a storage-only format and must
     *    go through path b) before upload. */
    t0 = now_ms();
    int64_t raw_got = vfs_read_asset_c(ark.vfs, target, buf, buf_cap, 0);
    double t_raw = now_ms() - t0;

    /* b) Fully decoded RGBA texels (materialize=1); the block is now cached. */
    t0 = now_ms();
    int64_t rgba_got = vfs_read_asset_c(ark.vfs, target, buf, buf_cap, 1);
    double t_rgba = now_ms() - t0;

    free(buf);
    vpak_close(&ark);
    if (raw_got < 0 || rgba_got < 0) {
        fprintf(stderr, "  Read failed (raw %lld, rgba %lld)\n",
                (long long)raw_got, (long long)rgba_got);
        return 1;
    }
    printf("  Stored payload: %lld B in %.3f ms (cold block)\n",
           (long long)raw_got, t_raw);
    printf("  Decoded RGBA:   %lld B in %.3f ms (warm block)\n",
           (long long)rgba_got, t_rgba);

    printf("\n── Key metrics for engine integration ──\n");
    printf("  Archive open (header-only): %.3f ms\n", t_open);
    printf("  Single-asset cold read:     %.3f ms\n", t_raw);
    printf("  GPU upload: BC1/BC4/BC5/BC7 payloads directly; BCX via RGBA.\n");

    return 0;
}
