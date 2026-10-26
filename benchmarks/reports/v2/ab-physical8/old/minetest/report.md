# Byte-exact archive benchmark v2

Run: 09eaabb8446846afbbcee64659bf5960
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 688dd123f576adb27abfecd81945feaccfb4e2367dfbd6e38ce758b5e2e01a60
Profile: physical8

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 2.384 | 9.575 | 7.382 | 65.7 | 1.3805 |
| perfile-zstd3 | 2.866 | 26.638 | 9.778 | 70.5 | 0.0194 |
| tar-lz4 | 2.844 | 40.573 | 25.450 | 59.5 | N/A (solid) |
| tar-zstd19 | 2.153 | 731.978 | 26.812 | 146.1 | N/A (solid) |
| tar-zstd3 | 2.345 | 49.640 | 26.873 | 66.3 | N/A (solid) |
| tar-zstd9 | 2.211 | 77.034 | 26.181 | 75.5 | N/A (solid) |
| vpak | 2.293 | 112.837 | 11.948 | 96.7 | 0.8309 |
| zip-deflate | 3.003 | 91.319 | 25.122 | 54.4 | 0.0832 |
| zip-store | 5.297 | 14.590 | 15.201 | 58.0 | 0.0408 |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
