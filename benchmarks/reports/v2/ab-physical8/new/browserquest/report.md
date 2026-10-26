# Byte-exact archive benchmark v2

Run: ea981090af8a4f25a966018eb278f1c1
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 72e0f9313069b8aa3ef0b7d693396010e93e2ea889f1b13a77394cad593a7ddc
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.799 | 31.143 | 29.592 | 258.9 | 1.6447 |
| perfile-zstd3 | 2.990 | 41.972 | 20.516 | 262.4 | 0.3196 |
| tar-lz4 | 3.699 | 103.218 | 41.113 | 239.4 | N/A (solid) |
| tar-zstd19 | 2.544 | 955.823 | 42.011 | 497.9 | N/A (solid) |
| tar-zstd3 | 2.743 | 111.026 | 34.591 | 259.5 | N/A (solid) |
| tar-zstd9 | 2.671 | 157.453 | 45.981 | 306.5 | N/A (solid) |
| vpak | 2.800 | 181.606 | 17.096 | 220.0 | 0.6879 |
| zip-deflate | 3.137 | 153.891 | 32.652 | 117.9 | 0.4552 |
| zip-store | 69.083 | 23.922 | 13.078 | 249.3 | 0.0616 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
