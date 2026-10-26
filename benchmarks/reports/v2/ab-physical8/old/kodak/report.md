# Independent texture quality v2

Run: 23c03d50539f474a9ddfac2849c62286
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 1530a99d26f084d5ee5861ef63c3be237766d2285fb03ea7e08344d28a0a8075

| Config | bpp | RGB PSNR dB | Worst image dB | FLIP mean | Encode ms |
|---|---:|---:|---:|---:|---:|
| bc1-fast | 4.000 | 34.619 | 31.009 | 0.053 | 36.485 |
| bc1-max | 4.000 | 35.231 | 31.373 | 0.043 | 520.818 |
| bc1-max-rdo | 4.000 | 35.249 | 31.381 | 0.043 | 2285.701 |
| bc7-mode6 | 8.000 | 41.961 | 38.467 | 0.024 | 763.219 |
| bcx-storage | 4.668 | 38.557 | 36.190 | 0.036 | 1291.680 |

BCX storage only. No real-GPU/vendor claim. Exact RGB quality; timing uncertainty over rounds.
Diagnostic runs are not references; full valid evidence: report.json.gz.
