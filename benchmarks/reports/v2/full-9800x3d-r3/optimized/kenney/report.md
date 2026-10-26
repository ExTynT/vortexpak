# Byte-exact archive benchmark v2

Run: 75caec0cd635445c9a5611f5231a1e16
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: c7ebd2db9c6c3f9f0223a4d0c6b77bcd0b0761095ba53f4cb4c956d8e6be929c
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 4.297 | 10.071 | 3.086 | 70.0 | 2.0712 |
| perfile-zstd3 | 4.340 | 15.989 | 2.109 | 70.7 | 0.1126 |
| tar-lz4 | 4.412 | 11.090 | 4.324 | 60.7 | N/A (solid) |
| tar-zstd19 | 4.261 | 505.461 | 4.828 | 137.2 | N/A (solid) |
| tar-zstd3 | 4.296 | 18.395 | 4.455 | 57.3 | N/A (solid) |
| tar-zstd9 | 4.288 | 23.853 | 4.060 | 66.7 | N/A (solid) |
| vpak | 4.257 | 396.813 | 8.224 | 82.3 | 6.0646 |
| zip-deflate | 4.340 | 98.720 | 9.173 | 56.4 | 0.1625 |
| zip-store | 4.979 | 2.833 | 2.530 | 56.8 | 0.0731 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
