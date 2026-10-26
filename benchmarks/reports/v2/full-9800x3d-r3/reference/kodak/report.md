# Independent texture quality v2

Run: 49cb8243a335484781d3d956164bcb04
Protocol: 8bb498a52152a07332b29eff1703b53e2248b5f6895fbc503849277dae62259c
Evidence: cb85476277c1b9d6600797149a509f2f56fe145bfb1ad2bf711c3354fa3ccfe3

| Config | bpp | RGB PSNR dB | Worst image dB | FLIP mean | Encode ms |
|---|---:|---:|---:|---:|---:|
| bc1-fast | 4.000 | 34.619 | 31.009 | 0.053 | 218.507 |
| bc1-max | 4.000 | 35.231 | 31.373 | 0.043 | 3749.420 |
| bc1-max-rdo | 4.000 | 35.249 | 31.381 | 0.043 | 17007.268 |
| bc7-mode6 | 8.000 | 41.961 | 38.467 | 0.024 | 5715.690 |
| bcx-storage | 4.668 | 38.557 | 36.190 | 0.036 | 9511.206 |

BCX storage only. No real-GPU/vendor claim. Exact RGB quality; timing uncertainty over rounds.
Diagnostic runs are not references; full valid evidence: report.json.gz.
