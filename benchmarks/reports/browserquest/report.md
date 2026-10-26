# Byte-exact archive benchmark

Corpus SHA-256: 0702a565fe85fa16f374c531dd14386099a2e669099d74357f7c112a72e60641

452 files; 72381221 input bytes; 7 measured rounds; 1 CPU(s).

| Configuration | Archive MiB | Pack ms | Restore ms | Pack peak RSS MiB | Reset p99 ms |
|---|---:|---:|---:|---:|---:|
| tar-zstd9 | 2.687 | 219.192 | 49.835 | 192.0 | N/A (solid) |
| perfile-zstd3 | 2.990 | 71.834 | 20.991 | 253.1 | 0.3275 |
| vpak | 2.774 | 785.222 | 34.352 | 190.2 | 1.5961 |
| indexed-zstd3 | 2.799 | 68.515 | 30.383 | 250.8 | 2.6261 |
| tar-zstd19 | 2.543 | 1336.563 | 43.410 | 262.8 | N/A (solid) |
| tar-lz4 | 3.699 | 116.353 | 42.995 | 165.5 | N/A (solid) |
| zip-store | 69.083 | 25.379 | 13.519 | 177.7 | 0.0688 |
| zip-deflate | 3.137 | 171.436 | 35.903 | 113.0 | 0.5690 |
| tar-zstd3 | 2.749 | 134.511 | 36.128 | 181.9 | N/A (solid) |

Values: medians across rounds. Full JSON includes raw trials, latency samples, CPU time,
sampled RSS, deadline misses, size buckets, bootstrap intervals and paired differences.

Limits: OS cache uncontrolled; reset means codec cache only, not cold disk.
Memory-to-memory pack/restore; archive writes, fsync and filesystem extraction are NOT timed.
RSS sampled every 5 ms; short transient allocations may be missed. No universal winner.
Confidence intervals are conditional on this corpus/host; no hardware or content generalization.
p99.9 has weak support unless each trial contains at least 100000 requests.
No lossy codecs or published competitor numbers participate in this table.
