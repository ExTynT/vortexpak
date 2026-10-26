# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.1] - 2026-10-26

### Fixed
- CI: macOS jobs never started (the retired `macos-13` runner); they now run
  on `macos-15-intel`.
- CI: benchmark-harness tests failed without `psutil`; CI installs the
  `benchmark` extra and the tests skip cleanly when it is missing.
- Build-receipt test compared an unresolved temp path (Windows 8.3 short
  names, macOS `/private`).

## [3.0.0] - 2026-10-26

**Breaking:** `mode="strict"` rejects `rdo` and `gpu`; the `gpu_avg_psnr_db`
metric is replaced by `gpu_color_psnr_db` / `gpu_min_color_psnr_db`; the
default block layout is `stream`; archives with mip chains (`TX_MIPS`) need a
3.x reader; BC4/BC5/BC7 archives written by 2.1.0 must be repacked (they were
not spec-conformant).

### Added
- North Star "Xonotic −40 %" in the README roadmap: six measurable gates (size,
  exactness, texture quality, real game, load speed, iteration) with measured
  current values; `benchmarks/xonotic_baseline.py` reproduces the baseline.
- README roadmap with milestones, progress and a Mermaid diagram, generated
  from `docs/roadmap.json` by `tools/roadmap.py` (`done M9` / `set M13 active`);
  CI fails when the README view is stale.
- Engine-ready GPU textures: BC1/BC4/BC5/BC7 payloads carry a full mip chain
  (`TX_MIPS`, default on, `--no-mips` / `mips=False` to disable), box filtered
  in linear light for sRGB color; color textures are tagged `TX_SRGB`. The
  `--gpu auto` floor applies to level 0; the worst mip is reported
  (`gpu_min_mip_color_psnr_db`).
- DDS export: `VortexVFS.read_dds`, `vortexpak extract --dds` (DX10 header,
  `*_UNORM_SRGB` formats, mip count). `VortexVFS.texture_info` and C-ABI
  `vfs_get_texture_info_c` describe format, size, mips and color space; the
  engine loader example uses it and runs in CI.
- `layout="stream"` (new default, `--layout`): assets >= 64 KiB get their own
  block(s), smaller ones share ~256 KiB blocks, bounding bytes decoded per
  random read. `layout="compact"` keeps the previous larger shared blocks.
- `gpu_min_psnr_db` / `--gpu-min-psnr` (default 35 dB) quality floor for
  `--gpu auto`; per-archive metrics report formats chosen, lossless fallbacks,
  pooled color PSNR and the worst texture.
- Strict impartial benchmark protocol: solid, indexed and per-file archive baselines; SHA-256 verification of every original file, matched CPU affinity, randomized isolated trials, raw latency/CPU/RSS data and bootstrap intervals.
- Separate reference-decoded texture benchmark with RGB-only pooled-MSE PSNR, explicit FLIP viewing parameters, complete Kodak validation and per-image observations.
- Benchmark adapter, worker, statistics and quality regression tests; optional `benchmark` dependency extra.
- Reference-decoder conformance tests: every GPU format is wrapped in a DDS
  container and decoded by Pillow; native and fallback paths must match it
  bit-for-bit.
- Kodak PhotoCD benchmark with NVIDIA FLIP perceptual metric.
- BCX opt-in storage format (per-block BC1/BC7 mix, ~4.8 bpp); decoded at
  load time, not GPU-native.
- Native VFS (C-ABI) decodes BCn textures when RGBA is requested.
- C engine-loader example (`examples/engine_loader`).

### Changed
- GPU texture encoding runs per texture across the codec thread pool
  (deterministic, identical archive bytes): Xonotic textures with mips pack in
  33.6 s instead of 75.0 s sequentially, without mips 23.5 s instead of 44.5 s.
- `--gpu auto` now tries BC1 then BC7 and accepts the first that meets the
  floor (previously: BC1 for every opaque texture regardless of quality).
- GPU and RDO texture PSNR exclude alpha; alpha is checked separately. Metric
  `gpu_avg_psnr_db` (mean of dB, alpha included) is replaced by pooled
  `gpu_color_psnr_db` and `gpu_min_color_psnr_db`.
- BC1 MAX encoder ~13x faster (greedy 565 polish, quantile split, integer
  metric, AVX2 evaluation); costs ~1.9 dB avg on Kodak vs the previous
  exhaustive search.

### Fixed
- Without the native core, BCX textures were decoded as BC1 and every BCX
  archive failed to read (`BCN payload length ... != expected`).
- BC7 mode 6 shares one P-bit across RGBA; the encoder chose P=0 freely and
  turned opaque alpha 255 into 254 (1861 of 4096 texels on an opaque noise
  test image). Opaque blocks now use P=1 and fully transparent blocks P=0,
  in both the native and NumPy encoders.
- `mode="strict"` accepted `rdo=True`, which rewrote PNG bytes and broke the
  byte-exact contract. Strict now rejects `rdo` and `gpu`.
- **BC4/BC5/BC7 were not spec-conformant**: GPUs would have decoded them
  incorrectly. BC4 palette order, BC5 block interleaving and BC7 now follow
  the D3D11/Vulkan spec (BC7 = real mode 6 with 4-bit indices). Archives
  written by 2.1.0 containing BC4/BC5/BC7 textures must be re-packed;
  `vortexpak info` reports them as failing XXH3 verification.
- `--gpu auto` no longer selects BCX (not GPU-native); it picks BC7 for
  alpha content and BC1 for opaque.
- BCX decoder rejects truncated or oversized payloads instead of returning
  partial data, and computes block offsets in O(n).

### Removed
- Proprietary BC7 2-subset "APR" layout (not decodable by GPUs).

## [2.1.0] - 2026-10-24

First public release.

### Added
- VXPK container format: semantic lane packing, parallel micro-blocks,
  XXH3-64 per-asset integrity, fuzz-hardened parser (`VXPKError` contract).
- VXLZ-ANS proprietary entropy codec (Rust, Rayon-parallel) with
  high-entropy bypass and token folding for repetitive XML/JSON.
- VXTX-GPU block compression: BC1/BC4/BC5 encode+decode, BC7 mode-6
  (8 bpp RGBA, smooth alpha), endpoint fitting via exact 1D segmentation
  DP + least-squares refinement, closed-loop post-codec RDO, and
  `--gpu auto` per-texture format selection by measured PSNR.
- Lane dictionaries: per-lane trained zstd dictionaries stored as blocks
  (strategies 9/10); selection-guarded — never enlarges the archive.
- Seekable VFS: 16-block LRU cache, microsecond warm reads, frame-budget
  latency metrics (p50/p95/p99/p99.9 vs 1/4/16.6 ms budgets).
- C-ABI for engines (`vortexpak_native/vortexpak.h`, cbindgen-generated,
  CI-verified) with a pure-C consumer example.
- Python package with CLI (`vortexpak pack/info/ls/extract/vfs-bench`),
  NumPy fallback for every native path, installable wheel (setuptools +
  cargo build).
- CI: 3-OS matrix (Windows/Linux/macOS), pure-fallback job, header
  freshness check; benchmark workflow with artifacts.
- Benchmark suite: 10-scenario quality matrix (PSNR/bpp/throughput +
  auto-derived gap list), full-game peak benchmark, chart dashboards.
- Format specification (`docs/FORMAT.md`).

### Measured
- Xonotic 0.8.6 (2.47 GB, 14 597 files): 1 078 MB (2.29×) vs zstd-19
  1 117 MB (2.21×); full unpack 0.9 s; warm VFS read p99 6 µs.
- GPU texture lane: 183 MB → 87.5 MB, post-codec PSNR avg 37 dB.

### Security
- Corrupt/truncated archives raise `VXPKError` — no panics across the FFI
  boundary; decoder bounds-checked and adversarially fuzz-tested
  (300 mutation rounds Python, block-level Rust fuzz).

[Unreleased]: https://github.com/ExTynT/vortexpak/compare/v3.0.1...HEAD
[3.0.1]: https://github.com/ExTynT/vortexpak/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/ExTynT/vortexpak/compare/v2.1.0...v3.0.0
[2.1.0]: https://github.com/ExTynT/vortexpak/releases/tag/v2.1.0
