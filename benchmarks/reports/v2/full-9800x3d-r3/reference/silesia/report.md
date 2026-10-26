# Byte-exact archive benchmark v2

Run: 8ed927ea7eeb43098e6df9f0e3278f1a
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: c7c957f29d35baba0da4f6cd050b0f0c4685a264c036fb506a5b6510ff4a7c48
Profile: reference

| Configuration | Archive MiB | Pack ms | Restore ms | Process lifetime peak to pack MiB | Median round reset p99 ms |
|---|---:|---:|---:|---:|---:|
| indexed-zstd3 | 63.717 | 779.807 | 247.417 | 843.8 | 66.9005 (exploratory) |
| perfile-zstd3 | 63.074 | 771.322 | 220.673 | 838.5 | 60.1571 (exploratory) |
| tar-lz4 | 96.236 | 546.736 | 129.874 | 638.7 | N/A (solid) |
| tar-zstd19 | 50.395 | 54269.933 | 258.450 | 630.5 | N/A (solid) |
| tar-zstd3 | 63.069 | 939.504 | 251.606 | 630.3 | N/A (solid) |
| tar-zstd9 | 56.337 | 3396.684 | 241.186 | 630.5 | N/A (solid) |
| vpak | 57.919 | 10386.573 | 493.861 | 566.9 | 134.1151 (exploratory) |
| zip-deflate | 65.655 | 2572.977 | 257.329 | 362.8 | 74.6716 (exploratory) |
| zip-store | 202.121 | 130.191 | 24.715 | 622.2 | 10.5113 (exploratory) |

Full raw evidence and intervals: report.json.gz. CPU times below resolution are unresolved, not zero.
Memory is OS process-lifetime high water, NOT exact per-phase heap allocation.
OS page cache uncontrolled. No cold physical IO, real GPU or production SLO claim.
Every restore SHA-256 checked; every read and prime verified against original bytes or SHA control.
Timing CIs are descriptive over rounds. Diagnostic runs are not references. No universal winner.
