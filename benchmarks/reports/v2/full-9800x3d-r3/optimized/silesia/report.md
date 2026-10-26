# Byte-exact archive benchmark v2

Run: 6bbed797d75a41f28e82a18069d19b21
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: c01ff5d183f4f2bd6e30c40092d225466ed27657aa3dcf4684a0377cddf3dabf
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 63.717 | 777.252 | 247.518 | 843.9 | 67.0213 (exploratory) |
| perfile-zstd3 | 63.074 | 763.801 | 222.353 | 838.6 | 57.4708 (exploratory) |
| tar-lz4 | 96.236 | 547.004 | 129.601 | 638.6 | N/A (solid) |
| tar-zstd19 | 50.395 | 49132.617 | 256.653 | 630.6 | N/A (solid) |
| tar-zstd3 | 63.069 | 934.050 | 250.599 | 630.4 | N/A (solid) |
| tar-zstd9 | 56.337 | 3173.972 | 238.980 | 630.4 | N/A (solid) |
| vpak | 57.919 | 10369.756 | 494.494 | 566.7 | 137.4509 (exploratory) |
| zip-deflate | 65.655 | 2573.008 | 257.807 | 362.9 | 75.4681 (exploratory) |
| zip-store | 202.121 | 130.170 | 24.730 | 622.3 | 10.4413 (exploratory) |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
