# Independent texture quality v2

Run: db358f87627f41ffaa9ddb6f92a64c37
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: 431d5724ab6491f68e7a88852591abcb60c46729f759aeddaa51b8a237f4f37a

| Config | bpp | RGB PSNR dB | Worst image dB | FLIP mean | Encode ms |
|---|---:|---:|---:|---:|---:|
| bc1-fast | 4.000 | 34.619 | 31.009 | 0.053 | 218.162 |
| bc1-max | 4.000 | 35.231 | 31.373 | 0.043 | 3779.212 |
| bc1-max-rdo | 4.000 | 35.249 | 31.381 | 0.043 | 16997.695 |
| bc7-mode6 | 8.000 | 41.961 | 38.467 | 0.024 | 5716.407 |
| bcx-storage | 4.668 | 38.557 | 36.190 | 0.036 | 9512.644 |

BCX storage only. No real-GPU/vendor claim. Exact RGB quality; timing uncertainty over rounds.
Diagnostic runs are not references; full valid evidence: report.json.gz.
