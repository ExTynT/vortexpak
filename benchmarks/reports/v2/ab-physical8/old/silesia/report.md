# Byte-exact archive benchmark v2

Run: 554c5de5f01c4a7098b8c843ffeab37e
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: f529a5b611fdf4fa422e3a5d73146c634afd206e9309ab9b1f015793e8c5e669
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 63.717 | 174.381 | 252.336 | 843.8 | 66.6324 (exploratory) |
| perfile-zstd3 | 63.074 | 261.648 | 223.074 | 839.0 | 57.4997 (exploratory) |
| tar-lz4 | 96.236 | 541.261 | 129.310 | 638.8 | N/A (solid) |
| tar-zstd19 | 50.400 | 13200.898 | 255.920 | 1316.6 | N/A (solid) |
| tar-zstd3 | 63.150 | 375.814 | 249.764 | 630.6 | N/A (solid) |
| tar-zstd9 | 56.442 | 875.540 | 240.166 | 780.4 | N/A (solid) |
| vpak | 57.919 | 1923.229 | 122.377 | 567.9 | 133.6395 (exploratory) |
| zip-deflate | 65.655 | 2555.280 | 256.771 | 362.9 | 74.3052 (exploratory) |
| zip-store | 202.121 | 132.464 | 24.927 | 622.3 | 10.5279 (exploratory) |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
