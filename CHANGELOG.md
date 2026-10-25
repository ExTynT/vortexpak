# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Reference-decoder conformance tests: every GPU format is wrapped in a DDS
  container and decoded by Pillow; native and fallback paths must match it
  bit-for-bit.
- Kodak PhotoCD benchmark with NVIDIA FLIP perceptual metric.
- BCX opt-in storage format (per-block BC1/BC7 mix, ~4.8 bpp); decoded at
  load time, not GPU-native.
- Native VFS (C-ABI) decodes BCn textures when RGBA is requested.
- C engine-loader example (`examples/engine_loader`).

### Changed
- BC1 MAX encoder ~13x faster (greedy 565 polish, quantile split, integer
  metric, AVX2 evaluation); costs ~1.9 dB avg on Kodak vs the previous
  exhaustive search.

### Fixed
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

[Unreleased]: https://github.com/ExTynT/vortexpak/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/ExTynT/vortexpak/releases/tag/v2.1.0
