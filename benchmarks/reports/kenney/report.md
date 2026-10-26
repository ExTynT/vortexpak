# Byte-exact archive benchmark

Corpus SHA-256: 51fb7d8968929cab11f6fad797dc18df9fa9ed056e1a9bc3aa3dd543effcf08d

100 files; 5208758 input bytes; 7 measured rounds; 1 CPU(s).

| Configuration | Archive MiB | Pack ms | Restore ms | Pack peak RSS MiB | Reset p99 ms |
|---|---:|---:|---:|---:|---:|
| tar-zstd9 | 4.288 | 21.770 | 3.836 | 48.7 | N/A (solid) |
| perfile-zstd3 | 4.340 | 15.396 | 2.287 | 49.2 | 0.1868 |
| vpak | 4.257 | 462.398 | 8.172 | 75.5 | 6.0172 |
| indexed-zstd3 | 4.297 | 9.474 | 3.031 | 48.6 | 2.1268 |
| tar-zstd19 | 4.261 | 444.457 | 4.833 | 135.1 | N/A (solid) |
| tar-lz4 | 4.412 | 11.112 | 3.989 | 48.6 | N/A (solid) |
| zip-store | 4.979 | 2.624 | 2.511 | 49.2 | 0.1873 |
| zip-deflate | 4.340 | 106.438 | 9.370 | 49.3 | 0.2379 |
| tar-zstd3 | 4.296 | 17.688 | 4.192 | 48.7 | N/A (solid) |

Values: medians across rounds. Full JSON includes raw trials, latency samples, CPU time,
sampled RSS, deadline misses, size buckets, bootstrap intervals and paired differences.

Limits: OS cache uncontrolled; reset means codec cache only, not cold disk.
Memory-to-memory pack/restore; archive writes, fsync and filesystem extraction are NOT timed.
RSS sampled every 5 ms; short transient allocations may be missed. No universal winner.
Confidence intervals are conditional on this corpus/host; no hardware or content generalization.
p99.9 has weak support unless each trial contains at least 100000 requests.
No lossy codecs or published competitor numbers participate in this table.
