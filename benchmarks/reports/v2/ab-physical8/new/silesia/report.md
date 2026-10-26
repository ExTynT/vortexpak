# Byte-exact archive benchmark v2

Run: 0876324b8b394e59812fdfdfef50649b
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 5fcb169edfad1943b3b3a66b019d3116416164c7a622cb98f16b53044edef36b
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 63.717 | 176.479 | 252.362 | 844.2 | 66.2659 (exploratory) |
| perfile-zstd3 | 63.074 | 263.786 | 223.620 | 838.7 | 57.8593 (exploratory) |
| tar-lz4 | 96.236 | 543.370 | 129.512 | 638.6 | N/A (solid) |
| tar-zstd19 | 50.400 | 13655.727 | 255.794 | 1316.8 | N/A (solid) |
| tar-zstd3 | 63.150 | 377.369 | 249.840 | 630.5 | N/A (solid) |
| tar-zstd9 | 56.442 | 889.483 | 238.222 | 780.3 | N/A (solid) |
| vpak | 57.944 | 1587.803 | 102.364 | 568.5 | 133.2874 (exploratory) |
| zip-deflate | 65.655 | 2555.958 | 256.394 | 363.1 | 74.3732 (exploratory) |
| zip-store | 202.121 | 131.232 | 24.669 | 622.3 | 10.6645 (exploratory) |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
