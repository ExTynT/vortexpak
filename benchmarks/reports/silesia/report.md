# Byte-exact archive benchmark

Corpus SHA-256: 319dd01d676ca8828d5ae5a15b236bca5d7ec4d55b8b9e3c7f6eec2aa3fe55ca

12 files; 211938580 input bytes; 7 measured rounds; 1 CPU(s).

| Configuration | Archive MiB | Pack ms | Restore ms | Pack peak RSS MiB | Reset p99 ms |
|---|---:|---:|---:|---:|---:|
| tar-zstd9 | 56.337 | 3700.190 | 305.598 | 510.9 | N/A (solid) |
| perfile-zstd3 | 63.074 | 900.297 | 254.691 | 709.5 | 62.7843 |
| vpak | 57.919 | 12450.220 | 584.905 | 526.5 | 140.5896 |
| indexed-zstd3 | 63.717 | 918.349 | 295.524 | 715.1 | 71.1088 |
| tar-zstd19 | 50.395 | 67691.491 | 318.946 | 576.0 | N/A (solid) |
| tar-lz4 | 96.236 | 620.727 | 143.953 | 532.6 | N/A (solid) |
| zip-store | 202.121 | 141.442 | 28.544 | 444.3 | 15.3852 |
| zip-deflate | 65.655 | 2882.821 | 314.973 | 307.5 | 78.3361 |
| tar-zstd3 | 63.069 | 1065.808 | 282.208 | 505.4 | N/A (solid) |

Values: medians across rounds. Full JSON includes raw trials, latency samples, CPU time,
sampled RSS, deadline misses, size buckets, bootstrap intervals and paired differences.

Limits: OS cache uncontrolled; reset means codec cache only, not cold disk.
Memory-to-memory pack/restore; archive writes, fsync and filesystem extraction are NOT timed.
RSS sampled every 5 ms; short transient allocations may be missed. No universal winner.
Confidence intervals are conditional on this corpus/host; no hardware or content generalization.
p99.9 has weak support unless each trial contains at least 100000 requests.
No lossy codecs or published competitor numbers participate in this table.
