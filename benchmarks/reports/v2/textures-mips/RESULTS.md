# GPU textures: mip chains (exploratory)

`benchmarks/gpu_floor_sweep.py`, engine mode, `--gpu auto`, floor 35 dB, 16 workers, one deterministic run per setting,
Ryzen 7 9800X3D. Archives contain only the corpus images. Every lossless fallback was checked bit-exact; GPU PSNR is RGB on level 0.

| Corpus (textures) | Without mips | With mips (default) | Size | Pack time |
|---|---:|---:|---:|---:|
| xonotic (3118) | 152.59 MB, 23.5 s | 186.60 MB, 33.6 s | +22.3% | +43% |
| minetest (451) | 194.7 KB, 0.2 s | 207.6 KB, 0.3 s | +6.6% | +50% |
| browserquest (238) | 1.69 MB, 2.1 s | 1.76 MB, 2.7 s | +4.2% | +29% |
| kenney (8) | 853.3 KB, 0.7 s | 1.04 MB, 1.1 s | +21.8% | +57% |

Same format choices with and without mips (level-0 floor). Size rises less than the theoretical +33 % because lossless
fallbacks carry no mips. Pack times include parallel per-texture GPU encoding (new); before it, the same Xonotic pack
took 44.5 s without and 75.0 s with mips, and the identical archive bytes confirm the parallel path is deterministic.

Mips below ~16x16 of strongly 2-D color gradients cannot be represented by one-subset BC modes (a single 4x4 block);
they are not floor-gated and their worst PSNR is reported as `gpu_min_mip_color_psnr_db`.

Reproduce:

```bat
python -B benchmarks\gpu_floor_sweep.py --corpus-root ..\vortex_corpus --floors 35 --output mips_on.json
python -B benchmarks\gpu_floor_sweep.py --corpus-root ..\vortex_corpus --floors 35 --no-mips --output mips_off.json
```
