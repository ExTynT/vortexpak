# Xonotic 0.8.6 lossless repack

Goal: byte-exact VortexPak strict archive of the whole release <= 0.80x the shipped download.

| Archive | Bytes | MB | x download |
|---|---:|---:|---:|
| shipped download (pk3, ZIP/DEFLATE) | 1,185,158,790 | 1,185.2 | 1.0000x |
| target 0.80x | 948,127,032 | 948.1 | 0.8000x |
| VortexPak strict compact (vxcm) | 924,623,172 | 924.6 | 0.7802x |
| xz -9e -T1 solid | 951,234,812 | 951.2 | 0.8026x |
| zstd --ultra -22 --long=31 -T1 solid | 1,014,118,420 | 1,014.1 | 0.8557x |

Smallest competitor: xz 951.2 MB; VortexPak is -2.8 % versus it.

**PASS**: VortexPak 924.6 MB = 0.7802x download <= target 948.1 MB.

Verification: 14,597 files; A mismatches 0; verify_archive ok True (14,597 files); B mismatches 0.

Timings: pack 292.41 s; full decode 4.33 s; VFS 229.11 s total (14,597 reads, p50 0.005 ms, p99 367.456 ms, max 1182.895 ms).

Dedup: 1,511 files, 22.3 MB.

## Strategy breakdown

| Strategy | Blocks | Raw MB | Comp MB |
|---|---:|---:|---:|
| 11 | 147 | 2,409.3 | 884.2 |
| 0 | 2 | 40.1 | 40.1 |

## Lane breakdown

| Lane | Blocks | Raw MB | Comp MB |
|---|---:|---:|---:|
| LANE_SCRIPTS | 114 | 1,966.9 | 579.5 |
| LANE_MEDIA | 19 | 331.6 | 316.1 |
| LANE_MAP_STRIPE | 14 | 149.9 | 28.5 |
| LANE_3D_GEOM | 1 | 1.0 | 0.2 |
| LANE_STRUCTURED | 1 | 0.0 | 0.0 |

## Extension breakdown

| Extension | Files | Raw MB |
|---|---:|---:|
| .dds | 8,186 | 1,630.0 |
| .bsp | 31 | 201.4 |
| .jpg | 3,585 | 182.5 |
| .ogg | 846 | 147.9 |
| .map | 150 | 68.2 |
| .md3 | 294 | 52.7 |
| .dpm | 40 | 48.8 |
| .ase | 183 | 32.4 |
| .iqm | 137 | 31.2 |
| .dem | 2 | 15.3 |
| .po | 53 | 14.2 |
| .dat | 3 | 12.4 |
| .ttf | 1 | 12.3 |
| .lno | 3 | 8.3 |
| .mp3 | 2 | 3.4 |
| .sp2 | 31 | 2.2 |
| .cache | 32 | 1.5 |
| .xcf | 1 | 1.1 |
| .obj | 38 | 1.0 |
| .zym | 2 | 0.9 |
| .cfg | 117 | 0.9 |
| .shader | 186 | 0.7 |
| .otf | 2 | 0.4 |
| .mdl | 19 | 0.4 |
| .txt | 66 | 0.3 |
| .pot | 1 | 0.2 |
| .ent | 1 | 0.2 |
| .waypoints | 32 | 0.2 |
| .sh | 31 | 0.1 |
| .wav | 3 | 0.1 |
| .framegroups | 88 | 0.1 |
| .license | 1 | 0.0 |
| .sounds | 22 | 0.0 |
| .pl | 7 | 0.0 |
| .bgs | 1 | 0.0 |
| .spr32 | 11 | 0.0 |
| .skin | 268 | 0.0 |
| .hardwired | 32 | 0.0 |
| .mapinfo | 30 | 0.0 |
| .zh_cn | 1 | 0.0 |
| .zh_hk | 1 | 0.0 |
| .zh_tw | 1 | 0.0 |
| (none) | 5 | 0.0 |
| .width | 2 | 0.0 |
| .svg | 1 | 0.0 |
| .rtlights | 6 | 0.0 |
| .xpm | 1 | 0.0 |
| .options | 30 | 0.0 |
| .rb | 1 | 0.0 |
| .subr | 1 | 0.0 |
| .spr | 2 | 0.0 |
| .cmake | 3 | 0.0 |
| .mid | 1 | 0.0 |
| .rc | 1 | 0.0 |
| .in | 1 | 0.0 |
| .lmp | 1 | 0.0 |

## Caveats

- One run; sizes are deterministic, times are a single observation on a shared host and indicative only.
- Competitors are solid streams without random access, while VortexPak reads any file individually.
- The solid concatenation carries no per-file headers, which slightly favours the competitors.
- Competitors run single-threaded at their strongest settings (xz -9e, zstd --ultra -22 --long=31); multi-threaded modes lose ratio.
- Compact layout decodes a whole 16 MiB block per cold single-file read; the VFS p99/max above is that cost (use --layout stream for streaming).

Reproduce with the command in the script docstring.
