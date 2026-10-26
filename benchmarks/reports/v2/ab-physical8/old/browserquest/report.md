# Byte-exact archive benchmark v2

Run: 116a9f4eac7f49e29a391f64ace25afc
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: bbdacd52cda552f3a1b9742c8b8815bd9f3b58c8c6f3e897638f02205d5aaf3a
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.799 | 31.181 | 29.566 | 258.7 | 1.6331 |
| perfile-zstd3 | 2.990 | 43.421 | 20.286 | 262.5 | 0.3186 |
| tar-lz4 | 3.699 | 103.133 | 41.189 | 239.5 | N/A (solid) |
| tar-zstd19 | 2.544 | 948.626 | 41.875 | 498.0 | N/A (solid) |
| tar-zstd3 | 2.743 | 110.303 | 34.221 | 259.4 | N/A (solid) |
| tar-zstd9 | 2.671 | 156.722 | 45.962 | 306.5 | N/A (solid) |
| vpak | 2.774 | 167.442 | 17.415 | 222.5 | 1.4463 |
| zip-deflate | 3.137 | 154.078 | 32.728 | 118.1 | 0.4542 |
| zip-store | 69.083 | 24.140 | 13.065 | 249.1 | 0.0607 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
