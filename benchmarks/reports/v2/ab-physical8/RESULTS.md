# A/B: stream block layout + texture quality guarantees

Baseline: previous implementation, receipt `../build-9800x3d-v2`. Candidate: receipt `../build-9800x3d-v3`.
Same protocol fingerprint, profile `physical8` (8 physical cores, one logical CPU each), 7 measured rounds + 1 warm-up,
10,000 queries per cache state and round (Silesia: 100, tail exploratory), all nine archive configurations per run,
old/new interleaved per corpus. Every restore and every read was byte-verified against the originals.

## Archive (strict mode, VortexPak row)

| Corpus | Size | Codec-reset p99, median of rounds | Paired p99 change [95% CI] | p50 | Pack | Restore |
|---|---:|---:|---:|---:|---:|---:|
| minetest | +3.57% | 0.831 → 0.569 ms | -31.6% [-32.3, -30.6] | 0.567 → 0.377 ms | +13.1% | -1.1% |
| kenney | +0.4% | 5.957 → 0.356 ms | -94% [-94.1, -40.8] | 0.126 → 0.065 ms | -46.7% | -37.1% |
| browserquest | +0.92% | 1.446 → 0.688 ms | -52.4% [-53, -51.5] | 0.436 → 0.152 ms | +5% | -2.7% |
| silesia | +0.04% | 133.639 → 133.287 ms | n/a (tail coverage insufficient) | 31.125 → 27.094 ms | -17.4% | -16.5% |

Gate (`protocol.compare_runs`): **potential_regression** on every corpus, because the protocol tolerates 0 extra bytes;
the stream layout trades ratio for bounded reads by design. Minetest pack time is also flagged (+13%, CI 5–15%).
No other configuration changed. Kenney: `sprites/skybox.png` (3.8 MB, 1 of 100 uniformly queried files) sits at the
p99 boundary, so its own ~3.5–6 ms decode appears in some rounds (new p99 CI 0.35–3.54 ms). Silesia reads are bound by
decoder throughput, not layout.

Context, candidate run (same rounds):

| Corpus | VortexPak MiB / p99 ms | indexed-zstd3 | per-file zstd3 | solid tar-zstd9 |
|---|---:|---:|---:|---:|
| minetest | 2.375 / 0.569 | 2.384 / 1.378 | 2.866 / 0.021 | 2.211 / solid |
| kenney | 4.274 / 0.356 | 4.297 / 1.3 | 4.34 / 0.092 | 4.285 / solid |
| browserquest | 2.800 / 0.688 | 2.799 / 1.645 | 2.99 / 0.32 | 2.671 / solid |
| silesia | 57.944 / 133.287 | 63.717 / 66.266 | 63.074 / 57.859 | 56.442 / solid |

## Kodak (independent decoder)

BC1 settings unchanged. BC7 mode 6 RGB PSNR 41.961 → 41.837 dB, worst image 38.467 → 38.39 dB:
opaque blocks now keep P=1, so alpha stays exactly 255 (max per-image alpha MSE 0.189 → 0). BCX 38.557 → 38.522 dB.


## Texture quality floor (`--gpu auto`, engine mode)

Exploratory, single deterministic run per setting (`benchmarks/gpu_floor_sweep.py`, all CPUs). PSNR is RGB-only, measured on
the decoded archive independently of the codec. Every lossless fallback was checked bit-exact.

| Corpus (textures) | Previous auto: archive, GPU textures < 35 / < 30 dB, worst | Floor 30 | Floor 32 | Floor 35 (default) |
|---|---|---:|---:|---:|
| xonotic (3118) | 87.64 MB, 1563 / 335, 9.7 dB | 119.36 MB (+36.2%) | 132.87 MB (+51.6%) | 152.59 MB (+74.1%) |
| minetest (451) | 182.4 KB, 244 / 144, 14.52 dB | 183.6 KB (+0.6%) | 183.8 KB (+0.8%) | 194.7 KB (+6.7%) |
| browserquest (238) | 2.23 MB, 140 / 76, 18.22 dB | 2.11 MB (-5.4%) | 2.05 MB (-8.1%) | 1.69 MB (-24.1%) |
| kenney (8) | 876.6 KB, 0 / 0, 36.99 dB | 850.6 KB (-3.0%) | 853.3 KB (-2.7%) | 853.3 KB (-2.7%) |

Previously `auto` coded every opaque texture as BC1 regardless of result. With the floor no GPU texture is below it.
Xonotic formats, previous → floor 35: BC1 3029 → 1500, BC7 35 → 1304, lossless 54 → 314.
The size cost is driven by the BC7 encoder supporting only mode 6: textures BC7 mode 6 cannot fit fall back to lossless.
Pixel-art corpora shrink because lossless VXTX beats BC7 on flat-color art.

## Reproduce

```bat
python -B benchmarks\gpu_floor_sweep.py --corpus-root ..\vortex_corpus --impl benchmarks\reports\v2\build-9800x3d-v3\snapshot --floors 30 32 35 --output benchmarks\reports\v2\ab-physical8\gpu_auto_new.json
python -B benchmarks\protocol.py build --output benchmarks\reports\v2\build-9800x3d-v3
rem per corpus and receipt (old = build-9800x3d-v2), interleaved:
python -B benchmarks\rigorous.py --corpus ..\vortex_corpus\legacy_games\kenney_3d_platformer --output benchmarks\reports\v2\ab-physical8\new\kenney --receipt benchmarks\reports\v2\build-9800x3d-v3\receipt.json --profile physical8 --threads 8 --repeats 7 --warmups 1 --requests 10000
python -B benchmarks\rigorous_quality.py --corpus ..\vortex_corpus\kodak --output benchmarks\reports\v2\ab-physical8\new\kodak --receipt benchmarks\reports\v2\build-9800x3d-v3\receipt.json --profile physical8 --threads 8 --repeats 7 --warmups 1 --require-kodak
```

`ab_summary.json` is derived from the validated bundles with `protocol.load_report` and `protocol.compare_runs`.

Limits: one host, `physical8` profile only (the 1-CPU reference was not re-run), OS page cache uncontrolled, no cold-disk
or real-GPU measurement.
