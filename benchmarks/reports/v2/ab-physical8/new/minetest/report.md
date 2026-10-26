# Byte-exact archive benchmark v2

Run: b7041a4ab6984768b8c23330a52d7217
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 148593b9117985c20fd07169bedab18dcf1bfbf65528da5883063a973dd726a8
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.384 | 9.558 | 7.548 | 65.9 | 1.3782 |
| perfile-zstd3 | 2.866 | 26.497 | 9.720 | 70.6 | 0.0210 |
| tar-lz4 | 2.844 | 40.477 | 25.581 | 59.6 | N/A (solid) |
| tar-zstd19 | 2.153 | 737.640 | 26.780 | 145.9 | N/A (solid) |
| tar-zstd3 | 2.345 | 49.647 | 26.860 | 66.9 | N/A (solid) |
| tar-zstd9 | 2.211 | 76.995 | 25.979 | 75.7 | N/A (solid) |
| vpak | 2.375 | 129.627 | 11.915 | 95.6 | 0.5694 |
| zip-deflate | 3.003 | 91.075 | 25.063 | 54.6 | 0.0825 |
| zip-store | 5.297 | 14.718 | 15.088 | 58.6 | 0.0401 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
