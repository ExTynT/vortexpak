# North-Star baseline: Xonotic 0.8.6

Shipped download: **1,185.2 MB** in 7 pk3 (ZIP/DEFLATE) files, 14,597 entries. Target ≤ 0.60× = **711.1 MB**.

| Type | Files | Raw MB | In pk3 MB | Share of download |
|---|---:|---:|---:|---:|
| .dds | 8,186 | 1,630.0 | 730.3 | 61.6 % |
| .jpg | 3,585 | 182.5 | 172.6 | 14.6 % |
| .ogg | 846 | 147.9 | 146.0 | 12.3 % |
| .bsp | 31 | 201.4 | 53.6 | 4.5 % |
| .dpm | 40 | 48.8 | 20.8 | 1.8 % |
| .iqm | 137 | 31.2 | 16.2 | 1.4 % |
| .md3 | 294 | 52.7 | 13.7 | 1.2 % |
| .map | 150 | 68.2 | 6.9 | 0.6 % |
| .dem | 2 | 15.3 | 5.4 | 0.5 % |
| .mp3 | 2 | 3.4 | 3.4 | 0.3 % |

DXT textures (.dds): 1,630.0 MB raw; solid zstd-19 (lossless) 639.6 MB. Formats: DXT1 982.4 MB, DXT5 621.0 MB, DXT3 21.1 MB, DXT2 5.6 MB, other 0.0 MB.

Byte-exact VortexPak strict repack of the extracted files (current code, 16 workers):

- stream: **1,114.5 MB** (0.940× download), pack 18.0 s, 14,597 files verified byte-exact
- compact: **1,077.6 MB** (0.909× download), pack 250.2 s, 14,597 files verified byte-exact

Already-compressed .jpg and .ogg cannot shrink much losslessly, so the target
requires lossy rate-distortion re-encoding of the DXT textures (quality-gated against the shipped
texture) plus lossless JPEG recompression. Reproduce with the command in the script docstring.
