# Measured evidence

## Qualified v2 reference

The completed `v2/full-9800x3d-r3` campaign is the first qualified reference.
[Identity and evidence](v2/full-9800x3d-r3/REFERENCE.json) ·
[Measured runtime and validation](v2/full-9800x3d-r3/RESULTS.md) ·
[PNG report](v2/full-9800x3d-r3/optimized/benchmark-report.png).

## Latest A/B (stream layout + texture quality floor)

[`v2/ab-physical8/RESULTS.md`](v2/ab-physical8/RESULTS.md): previous vs current implementation on all
8 physical cores. Not a new qualified 1-CPU reference.

[`v2/textures-mips/RESULTS.md`](v2/textures-mips/RESULTS.md): size and pack-time cost of default GPU mip chains.

## Authority and versions

The folders linked below are preserved historical v1 measurements. They are
not retrospectively upgraded to v2. A qualified v2 reference requires a complete
`run_reference.py` suite, an exact-output equivalence proof and `REFERENCE.json`.
Incomplete/failed/diagnostic runs are never substitutes. Keep the complete
local bundle, including raw trial files, source/DLL snapshots and manifests;
Git-tracked summaries alone are not a reproducible reference.

The v2 source of truth is `bundle.json` + hash-chained `events.jsonl`, registered
protocol and exact implementation receipt. HTML/PNG are derived views.

The initial v2 suite `v2/full-9800x3d` stopped on Windows file-sharing denial
while replacing an auxiliary progress snapshot. Its failed evidence is retained;
it did not produce REFERENCE.json and is not a baseline. The corrected run uses
a new directory and protocol fingerprint; no trials are silently reused.

### What is in git

Tracked: every RESULTS/BASELINE summary, suite and reference identities, per-case
`report.md`, `bundle.json` and receipts, and the compressed raw reports of the
qualified reference (`v2/full-9800x3d-r3/optimized/*/report.json.gz`). Full
bundles (implementation and DLL snapshots, harness copies, per-trial files,
event ledgers, logs) and the failed, stopped and canary runs stay on the
recording machine; `protocol.load_report` needs the full bundle.

## Historical v1 results


Open [index.html](index.html) locally for all configurations, charts, timings,
confidence intervals and limitations. GitHub does not render arbitrary HTML;
the individual Markdown tables are readable there:

- [Minetest Game](minetest/report.md) — entire non-Git corpus, 1,434 files.
- [Kenney 3D Platformer](kenney/report.md) — entire non-Git corpus, 100 files.
- [BrowserQuest](browserquest/report.md) — entire non-Git corpus, 452 files.
- [Kodak PhotoCD](kodak/report.md) — exactly 24 images, separate lossy track.
- [Silesia](silesia/report.md) — complete official 12-file corpus, 211,938,580 bytes.
  Its 100-query latency measurements have insufficient tail support.

Full raw results, manifests, all latency samples, source/native-library hashes,
environment and adequacy flags are retained in each report.json.gz. Decompress
with any gzip reader; these are evidence, not just summary charts. Uncompressed
report.json and standalone manifest.json remain local outputs and are ignored
by Git to avoid duplicating large artifacts.

All five experiments are complete: 252 measured archive trials across four
corpora, plus 35 measured Kodak configuration rounds. The full Python suite
passes 73 tests; benchmark source lint passes.

Archive trials use seven randomized measured rounds and one discarded warm-up,
one CPU, and nine solid/indexed/per-file configurations. Asset latency uses
10,000 identical seeded queries per cache state and round. Codec reset is not
cold OS/disk IO. Quality uses RGB-only pooled-MSE PSNR and independent Pillow
DDS decoding, including demultiplexed BCX; no actual GPU was executed.

No corpus sampling, adaptive stopping, best-of selection, hidden failed decode,
borrowed vendor measurements or universal-winner score. These are descriptive
single-host observations, not industry certification or a production SLO.
Full definitions: [BENCHMARKING.md](../../docs/BENCHMARKING.md).
