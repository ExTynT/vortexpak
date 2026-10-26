# Byte-exact archive benchmark

Corpus SHA-256: e1cdb3af9b6381a39f2978ac6c2aadfa8a89e89aeaee4897c4ba56041934d31b

1434 files; 5342715 input bytes; 7 measured rounds; 1 CPU(s).

| Configuration | Archive MiB | Pack ms | Restore ms | Pack peak RSS MiB | Reset p99 ms |
|---|---:|---:|---:|---:|---:|
| tar-zstd9 | 2.210 | 80.921 | 26.702 | 62.9 | N/A (solid) |
| perfile-zstd3 | 2.866 | 50.713 | 9.717 | 58.6 | 0.0252 |
| vpak | 2.293 | 393.937 | 17.654 | 61.5 | 0.9257 |
| indexed-zstd3 | 2.384 | 16.787 | 7.609 | 48.0 | 1.6037 |
| tar-zstd19 | 2.153 | 902.886 | 27.254 | 135.5 | N/A (solid) |
| tar-lz4 | 2.844 | 41.372 | 26.534 | 49.0 | N/A (solid) |
| zip-store | 5.297 | 14.869 | 15.659 | 51.4 | 0.0529 |
| zip-deflate | 3.003 | 96.779 | 27.848 | 49.2 | 0.1046 |
| tar-zstd3 | 2.343 | 50.479 | 27.994 | 48.5 | N/A (solid) |

Values: medians across rounds. Full JSON includes raw trials, latency samples, CPU time,
sampled RSS, deadline misses, size buckets, bootstrap intervals and paired differences.

Limits: OS cache uncontrolled; reset means codec cache only, not cold disk.
Memory-to-memory pack/restore; archive writes, fsync and filesystem extraction are NOT timed.
RSS sampled every 5 ms; short transient allocations may be missed. No universal winner.
Confidence intervals are conditional on this corpus/host; no hardware or content generalization.
p99.9 has weak support unless each trial contains at least 100000 requests.
No lossy codecs or published competitor numbers participate in this table.
