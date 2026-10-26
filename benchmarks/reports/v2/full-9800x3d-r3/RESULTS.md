# Verified reference v2 — Ryzen 7 9800X3D

Status: complete. All ten registered experiments exited successfully.

Control validation path: 47m 37s.
Optimized validation path: 39m 48s.
Observed harness wall-time reduction: 16.42% (7m 49s saved).
Whole campaign including finalization/rendering: 87m 41s.

This is one predeclared sequential same-host harness comparison, not a codec speedup, population significance result, or hardware-independent guarantee.
Source of truth: REFERENCE.json -> suite.json -> validated per-case raw bundles and ledger chains.

## Equivalence

287 paired measured trials and 41 paired warmup trials verified. Archive bytes/hashes, encoded image blocks, every per-image quality metric and query coverage match between control and optimized validation paths. No trial, corpus, configuration or integrity check was removed.

| Case | Control | Optimized | Paired measured trials | Paired warmups |
|---|---:|---:|---:|---:|
| browserquest | 8m 56s | 7m 37s | 63 | 9 |
| kenney | 7m 27s | 6m 08s | 63 | 9 |
| kodak | 8m 39s | 5m 25s | 35 | 5 |
| minetest | 6m 01s | 5m 55s | 63 | 9 |
| silesia | 16m 34s | 14m 43s | 63 | 9 |

## Verification and limits

All ten raw bundles, trial artifacts, implementation receipts and ledger chains independently revalidated after completion. Local current-workspace checks: 88 Python tests, flake8 benchmarks/tests and mypy vortexpak passed. Both generated PNGs decoded and verified (3026 x 9788).

Measurements refer to the archived receipted implementation, not automatically to later working-tree changes. Single host, one CPU allowance, OS-cache uncontrolled. No real GPU execution or cold physical IO measurement. Silesia tail coverage insufficient for a qualified p99 claim. Prior failed/stopped runs remain preserved and are not mixed into this reference.

[Full PNG](optimized/benchmark-report.png) · [Raw reference identity](REFERENCE.json)
