# Byte-exact archive benchmark v2

Run: fbb95baf5cca4e24b7001e8deed76b6b
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 4840094e94d44dde36e360ebf00270a982712f69114ab13707b933f0e3e3d9a2
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 4.297 | 10.096 | 3.041 | 69.9 | 2.0645 |
| perfile-zstd3 | 4.340 | 15.818 | 2.121 | 70.7 | 0.1100 |
| tar-lz4 | 4.412 | 11.124 | 4.207 | 60.7 | N/A (solid) |
| tar-zstd19 | 4.261 | 385.684 | 4.938 | 137.2 | N/A (solid) |
| tar-zstd3 | 4.296 | 17.516 | 4.177 | 57.3 | N/A (solid) |
| tar-zstd9 | 4.288 | 22.149 | 3.961 | 66.6 | N/A (solid) |
| vpak | 4.257 | 400.788 | 8.245 | 82.4 | 6.0323 |
| zip-deflate | 4.340 | 97.451 | 9.264 | 56.1 | 0.1904 |
| zip-store | 4.979 | 2.765 | 2.518 | 57.8 | 0.1191 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
