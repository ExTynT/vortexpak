# Byte-exact archive benchmark v2

Run: a943b2d2714e430394588acf03f8c757
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 11a52a1cb2780c92d399236eefb5aa3057dc171184139da7a741cc555046cde0
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.384 | 17.221 | 7.392 | 65.6 | 1.4021 |
| perfile-zstd3 | 2.866 | 49.633 | 9.844 | 70.9 | 0.0212 |
| tar-lz4 | 2.844 | 40.525 | 25.852 | 59.5 | N/A (solid) |
| tar-zstd19 | 2.153 | 768.638 | 26.795 | 137.1 | N/A (solid) |
| tar-zstd3 | 2.343 | 48.267 | 27.542 | 58.8 | N/A (solid) |
| tar-zstd9 | 2.210 | 79.079 | 26.438 | 66.3 | N/A (solid) |
| vpak | 2.293 | 335.025 | 17.386 | 64.9 | 0.8289 |
| zip-deflate | 3.003 | 90.885 | 25.125 | 54.4 | 0.0953 |
| zip-store | 5.297 | 14.827 | 15.187 | 58.1 | 0.0402 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
