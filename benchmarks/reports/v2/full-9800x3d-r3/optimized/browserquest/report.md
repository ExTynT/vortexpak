# Byte-exact archive benchmark v2

Run: 15b7ccdfc01b4ff2b83ff170e1e8935d
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 4c1d809de9f71a41f34eff67d06b2db1498a6322135c145bc6aecb34aceb76dd
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.799 | 54.596 | 30.417 | 258.6 | 1.7114 |
| perfile-zstd3 | 2.990 | 61.715 | 20.616 | 260.3 | 0.3176 |
| tar-lz4 | 3.699 | 103.755 | 41.627 | 239.3 | N/A (solid) |
| tar-zstd19 | 2.543 | 1229.283 | 41.727 | 264.3 | N/A (solid) |
| tar-zstd3 | 2.749 | 116.680 | 34.471 | 239.4 | N/A (solid) |
| tar-zstd9 | 2.687 | 201.098 | 48.648 | 239.6 | N/A (solid) |
| vpak | 2.774 | 699.661 | 31.693 | 192.8 | 1.4513 |
| zip-deflate | 3.137 | 153.775 | 32.703 | 118.0 | 0.4571 |
| zip-store | 69.083 | 23.692 | 13.012 | 249.0 | 0.0630 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
