# Benchmark protocol

## Authoritative v2 experiment

The normative, machine-readable contract is benchmarks/protocol.json.
Historical v1 results below are preserved, not re-labelled or overwritten.
The threat model is accidental stale binaries, changed inputs, incomplete work,
incorrect units, biased comparisons and invalid aggregation—not a malicious
administrator able to rewrite an entire local evidence bundle and all hashes.
SHA-256 is integrity and provenance binding here, not remote authentication.
No absolute absence of bugs or host noise is claimed.

### Build and execute

~~~powershell
pip install -e ".[benchmark,quality]"
python benchmarks/protocol.py build --output benchmarks/reports/v2/build-new
python benchmarks/run_reference.py --corpus-root ../vortex_corpus --output benchmarks/reports/v2/run-new --receipt benchmarks/reports/v2/build-new/receipt.json
~~~

Build uses cargo build --locked --release and verifies unchanged source and
build context. Its receipt binds source snapshot, Cargo.lock, compiler/cargo,
flags/config fingerprints and the loaded DLL. Every run captures an immutable
implementation copy and the actual measurement harness. Original corpus bytes
are frozen and SHA-256 checked outside the repository under the corpus cache.
Each fresh archive worker loads only its declared, receipted implementation.
Dirty Git state is recorded and exact source bytes are preserved.

The complete one-CPU experiment retains all nine archive configurations across
four complete pinned corpora, seven measured rounds plus one warm-up, and
10,000 per-state queries on game assets. Silesia uses 100 queries; its tail is
exploratory, never an SLO. Kodak uses all 24 original images and five declared
settings. Profile diagnostic is explicitly not a qualified reference. Profile
physical8 selects one logical processor per physical core and has a distinct
identity; it is not a faster substitute for one-CPU evidence. No global Windows
power plan, cache state or background application is modified.

### Precision and safe acceleration

Use integer monotonic nanoseconds and retain empty-timer calibration before and
after. Nothing is mechanically subtracted. Windows process CPU times were
observed in 15.625-ms increments on this host: zero/subresolution values are
unresolved, not exact zero. If any replicate is unresolved, do not publish a
biased median conditional on just the nonzero replicates. Keep raw counters.
Memory high water is OS process-lifetime working set, not per-phase allocated
heap. RSS before/after are snapshots; no sampling thread runs in timed regions.

Every full restore retains original SHA-256 verification. Every measured read
and untimed prime is validated; the optimized path uses full byte equality
against a verified immutable original instead of repeatedly hashing it.
The control path uses SHA-256. Their clear/prime/read sequences and sample
counts are identical. Image encoding still executes every repetition; every
block byte and our decoded pixels are checked. Independent reference decoding
and expensive metrics may be reused only for the exact same input/output,
format, dimensions, policy and oracle contract within that run. No persistent
cross-run quality cache or historical timing reuse is permitted.

### Evidence, statistics and time

Write each trial atomically, index attempts in a hash-chained append-only ledger,
and serialize full JSON/GZIP once at finalization. Progress is the committed
ledger tail, not a repeatedly replaced state.json cache: on Windows a live
reader can deny replacement even when ordinary writes are allowed. No
authoritative event depends on publication of a redundant progress snapshot.
Record warm-ups, failed
attempts, timeouts, interruptions and session boundaries. A resumed invocation
seals a stopped session and starts a new full plan; it never silently fills
missing favorable observations. Complete renderers verify the bundle, ledger,
configuration/query coverage, implementation and summaries. Missing/incompatible
or altered evidence fails closed. Legacy rendering requires explicit --legacy.

The replication unit is a round/session. A displayed p99 is the median of
within-round nearest-rank p99 estimates, not a population p99. Coverage flags
are not precision guarantees. All outliers remain. Timing bootstrap intervals
are descriptive; exact sizes and deterministic quality do not get fabricated
sampling CIs. Paired same-protocol operational regression gates require exact
input/resource matching, zero unannounced size/per-image quality regression,
and predeclared timing tolerance; uncertainty is inconclusive, not success.
No family-wise population-significance or production-SLO claim is made.

run_reference executes control then optimized full suites sequentially and
checks every warm-up/measured output, quality result and query-validation count.
It measures child start-to-exit wall time, including setup, oracles and durable
evidence work. Per-run elapsed time stops before final evidence serialization;
the outer suite duration includes it. Only the first predeclared qualifying
complete suite can produce REFERENCE.json; no best-of selection is allowed.
Images and HTML include identities and are derived from the same verified data.

Measured v1 full runtime on Ryzen 7 9800X3D was approximately 48m21s; measured
operations alone occupied about 25m21s before warming, priming and checks.
The v2 optimization target of 38–45 minutes is a planning estimate, not an
observed speedup. Reserve about 90–110 minutes for initial full control and
optimized validation, plus compilation if needed. --estimate explicitly reports
uncalibrated state without matching whole-run history; it never fabricates ETA.
Use the ordinary correctness suite (~6–10 seconds in prior runs) for immediate
feedback; a shorter diagnostic experiment cannot replace this reference.

## Historical v1 protocol (not upgraded)

The canonical runners are [rigorous.py](../benchmarks/rigorous.py) and
[rigorous_quality.py](../benchmarks/rigorous_quality.py). The older runners,
JSON files and charts are exploratory historical evidence, not a competitive
scoreboard. Results produced by different protocols must not be spliced together.

## Reproduce

Install a release Rust core and the optional measurement dependencies:

~~~powershell
pip install -e ".[benchmark,quality]"
python benchmarks/rigorous.py --corpus ../vortex_corpus/legacy_games/minetest_game --output benchmarks/reports/minetest --threads 1 --repeats 7 --requests 10000
python benchmarks/rigorous.py --corpus ../vortex_corpus/legacy_games/kenney_3d_platformer --output benchmarks/reports/kenney --threads 1 --repeats 7 --requests 10000
python benchmarks/rigorous.py --corpus ../vortex_corpus/legacy_games/browserquest --output benchmarks/reports/browserquest --threads 1 --repeats 7 --requests 10000
python benchmarks/rigorous_quality.py --corpus ../vortex_corpus/kodak --output benchmarks/reports/kodak --threads 1 --repeats 7 --require-kodak
~~~

Use a new output directory for every experiment; existing reports are not
silently overwritten. Full Xonotic can use the same archive command, but its
results cannot be inferred from the smaller corpora. The preloaded protocol
requires available RAM of at least seven times the input bytes. It fails rather
than silently sampling, swapping knowingly, skipping files or changing protocol.

An additional general-purpose compression control is the complete official
[Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia):
12 files, 211,938,580 bytes. Fetch outside the repo with
`python benchmarks/fetch_silesia.py --output ../vortex_corpus/silesia`.
The downloader checks the exact member list, published sizes and legacy MD5
identities over HTTPS; the runner additionally records SHA-256. MD5 is an
identity compatibility check here, not a modern security guarantee.
For its much larger individual members, use a separately labelled run:
`python benchmarks/rigorous.py --corpus ../vortex_corpus/silesia --output benchmarks/reports/silesia --threads 1 --repeats 7 --requests 100`.
Archive repetitions remain seven, but tail adequacy is false.
These observations are not p99/p99.9 evidence and must not be pooled with the
10,000-query asset trials.

Close other compute-heavy applications, use AC power, record power/thermal
settings, and run sequentially. Affinity caps CPU availability, not exclusivity;
other applications, SMT siblings, turbo and thermal drift remain confounders.
For stronger evidence, repeat on independently controlled hardware. This runner
makes those limitations visible; it cannot remove them from a desktop session.

## Track A: byte-exact archives

All configurations receive identical preloaded original files, names and order.
The corpus is the **entire specified directory**, excluding only .git metadata.
Symlinks are rejected. A sorted SHA-256 manifest records every input, size and
excluded Git file. Sources are revalidated before each trial and after the run.
No compressibility-based filtering, file truncation or extension sampling occurs.

| Configuration | Representation | Access topology |
|---|---|---|
| vpak | strict mode, VXLZ backend, native core | semantic lanes, indexed blocks |
| tar-zstd3 / 9 / 19 | deterministic PAX tar + zstd, checksum enabled | solid stream |
| tar-lz4 | identical tar + LZ4 level 0, content checksum | solid stream |
| zip-store | standard ZIP, stored members | independent members |
| zip-deflate | standard ZIP, DEFLATE level 6 | independent members |
| indexed-zstd3 | indexed contiguous stream, 2 MiB zstd frames | fixed blocks |
| perfile-zstd3 | same index format, one frame per nonempty member | independent frames |

The two indexed zstd controls are benchmark containers, **not a claim about a
shipping competing product**. Their compressed JSON index, names, offsets,
lengths, frame headers and checksums all count toward archive size. VortexPak's
own header, index, dictionary and checksums also count. Tar includes its real
padding and PAX metadata. ZIP includes its real directory and member overhead.
Container differences are part of the archive-product comparison; this is not
an isolated entropy-codec comparison. Indexed controls expose the topology
trade-off instead of hiding the solid baseline.

File contents and relative names are the preservation contract. Ownership,
permissions, timestamps, symlinks and empty directories are not in this contract;
no configuration gets credit for preserving them.

### Resource and timing boundaries

- Default budget: one logical CPU. Every trial is a fresh subprocess with the
  same affinity, Rayon budget and numerical-library thread limits. No concurrent
  benchmark trials. Multicore experiments use a separate output and explicit
  --threads N; they are not mixed with single-core numbers.
- Solid zstd uses zero worker threads at budget 1 and N-1 workers at budget N.
  Indexed zstd uses up to N independent frame workers. VortexPak uses its normal
  thread pools, restricted by the same affinity. LZ4 and ZIP remain single-threaded.
  A CPU allowance is not a promise that every algorithm exploits every CPU.
- Pack includes container construction, compression and serialization from the
  preloaded file list. Restore includes parsing, decompression and materializing
  every original file into owned bytes. Codec thread-pool construction/teardown
  belongs to these timings. Module import and process startup are excluded.
- Corpus read/hash, output persistence, filesystem extraction and fsync are
  excluded for **all** configurations. This is a memory-to-memory archive test,
  not install-time or durable-write throughput. Do not label it end-to-end disk IO.
- Every restore checks exact path set, length and SHA-256 of **every file** against
  the original manifest, outside timing. Every random read is similarly checked.
  A failure stops the run, persists the error, and prevents a complete scoreboard.
  Internal format checksum work remains included where the real API performs
  it (e.g. ZIP CRC); the common independent SHA-256 oracle is outside timing.
- Report bytes, ratio, pack/restore wall time, process CPU time, MiB/s, sampled
  process RSS and RSS delta, sampled thread count, archive hashes and determinism.
  RSS sampling interval is 5 ms: transient peaks can be missed. RSS includes the
  runtime/input baseline; it is not allocated heap bytes. No tracemalloc overhead.
- Environment includes dependency versions, CPU topology/frequency snapshot,
  RAM, OS, Python, Git revision/dirty state and hashes of runner, package sources
  and loaded native library. Source changes invalidate a running experiment.

### Random access and cache states

All seekable artifacts are file-backed, on the same temporary filesystem.
Index/open time is reported separately from steady-state single-file reads.
Solid archives have no random-access score: decompressing them once up front and
calling it a microsecond random read would conceal the actual work.

The seeded query stream is identical for every configuration within a round:
uniform sampling **with replacement over file names**, not bytes. Each measured
round gets a new fixed seed. Small files are therefore intentionally represented
by their frequency; per-size buckets expose what the aggregate hides.

1. codec_reset: clear decompressed-block cache before **every** query. A shuffled
   pass alone is not called cold, since neighboring assets may share a block.
2. repeat_after_one_prime: clear, read that same asset once, then time one repeat.
   This is not a guaranteed cache hit: an asset can exceed cache capacity.

VortexPak and indexed controls use 16-block LRUs; actual memory capacity differs
with block sizes. ZIP has no decompressed member cache. This compares the real
read APIs, not equal-byte-cache entropy kernels. OS page cache is uncontrolled;
there is **no cold-disk claim** and no system-wide cache-flush operation. Writing
artifacts itself tends to warm OS pages. A subsequent physical-IO study needs
storage identification, residency verification and a separately specified
cache-eviction protocol; it cannot reuse these labels as cold IO.

Raw latency samples are retained in query order for both states. Report p50,
p90, p95, p99, p99.9, max, mean, standard deviation, size buckets and misses at
1 / 4 / 8.33 / 16.67 ms. Quantiles use nearest-rank/inverted-CDF, not interpolated
fictional observations. Ten thousand requests give about 100 observations above
p99 but only 10 above p99.9. The report flags this distinction explicitly; maxima
and percentiles are observations, not production worst-case guarantees.

### Repetitions and inference

One randomized warm-up round per configuration is discarded. Seven or more
measured complete rounds use independently randomized configuration order.
Warm-ups condition the host/OS, not persistent state in the fresh worker process.
No outlier trimming, best-of-N, adaptive stopping or reruns until a favorite wins.

For each scalar, report median, min/max, median absolute deviation and a seeded
5000-resample percentile bootstrap 95% interval over **rounds**. For latency,
bootstrap per-round quantiles, not thousands of correlated reads treated as
independent experiments. Paired pack/restore percentage differences use the
same round's tar-zstd3 baseline. Intervals are descriptive, conditional on this
host and corpus, and uncorrected for multiple comparisons. Seven rounds do not
establish population-level significance. No p-values, absolute superiority
claims, arbitrary weighted score, or cross-corpus average of compression ratios.

Interpret size, pack time, restore time, access latency and memory together.
A size win against one topology is not a universal win. Publish all registered
configurations, complete failures, raw observations and limitations.

## Track B: lossy textures

Quality results **never share a lossless archive leaderboard**. The strict Kodak
mode requires exactly kodim01.png through kodim24.png at their original 768x512
or 512x768 dimensions. Generic image corpora record every exclusion and reason;
there is no silent padding or resizing. The full archive track still includes
all non-image files and images regardless of texture eligibility.

Compare our BC1 fast/MAX/MAX+RDO, BC7 mode 6 and storage-only BCX separately.
No NVTT/Basis/ISPC numbers are borrowed from papers or other machines. A vendor
comparison requires executing that encoder at specified settings, identical
formats/bpp/content, hardware and resource budgets, with its exact version.

Every encoded image is independently decoded by Pillow DDS. BCX is demultiplexed
into BC1 and BC7 streams and independently decoded before block recombination.
A mismatch with our decoder fails the run. This covers tested outputs, not
proof of the entire format or a substitute for execution on actual GPUs.

Metrics retain every image's block hash, pixel count, encoded bytes, RGB MSE and
PSNR in sRGB 8-bit space, linear-RGB MSE, maximum RGB error, separate alpha MSE
and PSNR, and RGB LDR FLIP at explicit 67 pixels/degree. Identical-image controls
must yield zero MSE and FLIP. Opaque alpha cannot inflate RGB PSNR. Corpus PSNR
comes from pixel-weighted pooled RGB squared error, **not mean per-image dB**.
Exact quality is JSON null plus an explicit exact flag, never nonstandard Infinity.
FLIP's per-image mean and worst image are reported without a universal invisible
threshold. Alpha compositing, cutout coverage and normal-map angular error remain
separate unmeasured workloads; RGB scores are not evidence for them.

Encoding time sums per-image calls, including source byte conversion, but
excludes reference decoding, quality computation, corpus IO and hashing. One
warm-up and seven randomized measured rounds use the same CPU affinity. There
is no vendor-speed claim and no direct-GPU-upload claim for BCX.

## Artifacts and tests

Each experiment writes manifest.json (archive track), report.json (full raw
observations/environment/adequacy) and report.md (readable descriptive table).
Incomplete reports are not valid results. JSON preserves all trials and latency
samples, including slow observations; do not publish just a chart.

Run python -m unittest discover -s tests -p "test_rigorous*.py" -v.
Tests cover all archive adapters, file-backed readers, empty and Unicode files,
multi-block reads, corruption of the original input, exact integrity checks,
quantile/CI definitions, isolated worker execution, source hashing, output
protection, RGB-only quality and independent BCX decoding. Optional dependencies
are explicit extras; standard installs may skip the optional metric/protocol tests.
CI smoke timings validate execution only and must not be used as performance data.

Build the standalone graphical report after all runs complete:
`python benchmarks/report_benchmarks.py benchmarks/reports/minetest benchmarks/reports/kenney benchmarks/reports/browserquest benchmarks/reports/silesia benchmarks/reports/kodak --output benchmarks/reports/index.html`.
The HTML links to full raw observations; a chart alone is not the evidence.
