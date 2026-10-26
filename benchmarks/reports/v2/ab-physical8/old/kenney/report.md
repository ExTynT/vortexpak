# Byte-exact archive benchmark v2

Run: 90e0248b94224115a86461f25ba32487
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 530d8738045cc96b62d83086824e4cde0ff09aa6b18bd4d0a313ec3424603e74
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 4.297 | 8.109 | 3.013 | 70.0 | 1.3691 |
| perfile-zstd3 | 4.340 | 13.293 | 2.076 | 71.2 | 0.1018 |
| tar-lz4 | 4.412 | 10.926 | 4.086 | 60.7 | N/A (solid) |
| tar-zstd19 | 4.261 | 340.796 | 4.759 | 147.2 | N/A (solid) |
| tar-zstd3 | 4.296 | 18.915 | 4.081 | 67.2 | N/A (solid) |
| tar-zstd9 | 4.285 | 24.277 | 4.057 | 76.5 | N/A (solid) |
| vpak | 4.257 | 345.414 | 7.106 | 83.8 | 5.9568 |
| zip-deflate | 4.340 | 96.782 | 9.224 | 56.3 | 0.1710 |
| zip-store | 4.979 | 2.767 | 2.559 | 56.8 | 0.0614 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
