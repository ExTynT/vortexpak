# Texture quality (not lossless archive comparison)

RGB PSNR from pooled MSE; Pillow reference decode; FLIP at 67 pixels/degree.

| Setting | bpp | RGB PSNR dB | Worst image dB | FLIP mean | Encode ms |
|---|---:|---:|---:|---:|---:|
| bc1-fast | 4.000 | 34.619 | 31.009 | 0.05255 | 228.21 |
| bc1-max | 4.000 | 35.231 | 31.373 | 0.04297 | 3908.18 |
| bc1-max-rdo | 4.000 | 35.249 | 31.381 | 0.04287 | 17595.49 |
| bc7-mode6 | 8.000 | 41.961 | 38.467 | 0.02393 | 5885.58 |
| bcx-storage | 4.668 | 38.557 | 36.190 | 0.03623 | 9840.68 |

BCX is storage-only, not GPU-native. No vendor performance claims.
Per-image observations, hashes and bootstrap timing intervals: report.json.
No FLIP number guarantees invisibility; real-GPU tests and alpha compositing are not covered.
