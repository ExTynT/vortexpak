# Byte-exact archive benchmark v2

Run: dc1fc94fa0394af5a700571dddbaa4b2
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: a9a0dfabd151b1f8c1691193834eb614a11ae1be99f34427be6cefa559089856
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.384 | 17.053 | 7.720 | 65.7 | 1.5709 |
| perfile-zstd3 | 2.866 | 50.093 | 10.062 | 70.8 | 0.0211 |
| tar-lz4 | 2.844 | 40.987 | 26.324 | 59.4 | N/A (solid) |
| tar-zstd19 | 2.153 | 738.257 | 27.528 | 137.0 | N/A (solid) |
| tar-zstd3 | 2.343 | 48.329 | 27.649 | 58.8 | N/A (solid) |
| tar-zstd9 | 2.210 | 75.568 | 26.885 | 66.4 | N/A (solid) |
| vpak | 2.293 | 342.085 | 17.730 | 64.6 | 0.8324 |
| zip-deflate | 3.003 | 91.395 | 25.400 | 54.5 | 0.0873 |
| zip-store | 5.297 | 14.739 | 15.309 | 58.0 | 0.0377 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
