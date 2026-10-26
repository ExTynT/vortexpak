# Byte-exact archive benchmark v2

Run: d40b3e818b8a4affa3c44588fda141b4
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 05e8ce7cd269baba0fa8e90e9239b4cf0ab9dd33f0cf7a686f88d86b18d18900
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 4.297 | 8.097 | 2.997 | 70.0 | 1.3005 |
| perfile-zstd3 | 4.340 | 13.424 | 2.106 | 70.8 | 0.0917 |
| tar-lz4 | 4.412 | 10.870 | 4.098 | 60.8 | N/A (solid) |
| tar-zstd19 | 4.261 | 342.359 | 4.819 | 147.1 | N/A (solid) |
| tar-zstd3 | 4.296 | 19.096 | 4.152 | 67.2 | N/A (solid) |
| tar-zstd9 | 4.285 | 24.850 | 4.224 | 76.8 | N/A (solid) |
| vpak | 4.274 | 182.953 | 4.468 | 87.5 | 0.3565 |
| zip-deflate | 4.340 | 96.908 | 9.149 | 56.1 | 0.1595 |
| zip-store | 4.979 | 2.873 | 2.522 | 57.1 | 0.0573 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
