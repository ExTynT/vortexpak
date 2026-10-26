# Independent texture quality v2

Run: a488fdf0bfa243c4b9147772bc7a1a52
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 2eba4686d1e66439893abeb9e20b320a0a6b6566588cf61946fca25f57f8d707

| Config | bpp | RGB PSNR dB | Worst image dB | FLIP mean | Encode ms |
|---|---:|---:|---:|---:|---:|
| bc1-fast | 4.000 | 34.619 | 31.009 | 0.053 | 36.140 |
| bc1-max | 4.000 | 35.231 | 31.373 | 0.043 | 515.799 |
| bc1-max-rdo | 4.000 | 35.249 | 31.381 | 0.043 | 2298.592 |
| bc7-mode6 | 8.000 | 41.837 | 38.390 | 0.025 | 251.419 |
| bcx-storage | 4.665 | 38.522 | 36.157 | 0.036 | 767.924 |

BCX storage only. No real-GPU/vendor claim. Exact RGB quality; timing uncertainty over rounds.
Diagnostic runs are not references; full valid evidence: report.json.gz.
