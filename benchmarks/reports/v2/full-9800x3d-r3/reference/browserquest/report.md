# Byte-exact archive benchmark v2

Run: 381a849beea3445694e78167a5e0e2a6
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 30d3aa5bee83239c5176a71fcb76410d6004afac0814d5b6fd52f4c999ddd359
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.799 | 54.633 | 30.326 | 258.5 | 1.7380 |
| perfile-zstd3 | 2.990 | 61.230 | 20.650 | 260.4 | 0.3192 |
| tar-lz4 | 3.699 | 104.580 | 41.418 | 239.4 | N/A (solid) |
| tar-zstd19 | 2.543 | 1186.552 | 42.294 | 264.4 | N/A (solid) |
| tar-zstd3 | 2.749 | 116.160 | 34.693 | 239.8 | N/A (solid) |
| tar-zstd9 | 2.687 | 201.550 | 48.653 | 240.1 | N/A (solid) |
| vpak | 2.774 | 697.966 | 31.676 | 192.3 | 1.4537 |
| zip-deflate | 3.137 | 154.495 | 32.644 | 118.0 | 0.4574 |
| zip-store | 69.083 | 23.862 | 13.366 | 249.1 | 0.0620 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
