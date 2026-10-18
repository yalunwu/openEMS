AI disclosure: Measurements and report prepared with Codex.

# HIP setup measurements

These historical measurements use the original `2a0e069` base and modified
CSXCAD/fparser, before the port onto upstream `ff68e3d` and the 2026-10-18 scope
reduction to stock dependencies. They have not been repeated for the current
solver. The STL, PLY, weighted and voxel construction cases now require CPU
construction; these timings do not describe the current supported GPU scope.

Measured on Windows with an NVIDIA RTX 3070 (8 GiB), driver 617.42, MSVC 19.44.35229, CUDA 12.6.77 and HIP 6.2.4 headers.
Both construction modes use identical HIP stepping for 100 timesteps. CPU construction uses its automatic thread count.
Each case/mode has 4 fresh worker processes and 3 runs per process. CPU/HIP ordering alternates by process.
Process-cold means the first run in a worker; OS file/driver disk caches are not flushed. Warm means subsequent runs in that worker.
Per-run wall time includes model creation, setup, stepping and output finalization; Python process startup/imports are outside that timer.
Setup includes operator/waveform construction, backend initialization (including fusion buffers) and processing initialization. Times below are medians in milliseconds.

## Warm runs

| Case | Setup CPU / HIP | Total CPU / HIP |
|---|---:|---:|
| analytic | 25.396 / 12.668 | 52.534 / 38.553 |
| many | 121.896 / 19.272 | 152.254 / 47.385 |
| stl | 165.090 / 46.231 | 362.285 / 239.601 |
| ply | 75.081 / 34.097 | 153.004 / 110.905 |
| weighted | 18.704 / 8.985 | 55.226 / 45.341 |
| voxel | 67.714 / 14.967 | 99.948 / 43.669 |
| cylinder | 40.662 / 27.445 | 70.310 / 54.928 |
| multigrid | 51.542 / 19.320 | 94.430 / 61.374 |

## Process-cold runs

| Case | Setup CPU / HIP | Total CPU / HIP |
|---|---:|---:|
| analytic | 214.789 / 221.450 | 268.891 / 274.468 |
| many | 299.409 / 226.326 | 367.790 / 293.473 |
| stl | 344.609 / 254.542 | 579.610 / 489.811 |
| ply | 258.644 / 244.512 | 378.847 / 364.086 |
| weighted | 181.171 / 191.838 | 245.055 / 255.600 |
| voxel | 255.017 / 221.521 | 331.626 / 290.210 |
| cylinder | 266.290 / 279.793 | 332.902 / 345.603 |
| multigrid | 244.681 / 227.260 | 319.007 / 296.728 |

## Memory and transfers

These transfer/device-memory numbers come from a separate instrumented run per case/mode, with kernel timing enabled. They are not throughput timings.
The shared-context counters cover setup through the first voltage update, including requested time-zero output and waveform files. Device peaks count application buffers; opaque driver allocations are excluded.
Host peaks are process-lifetime peak resident memory from warm throughput runs and include library/driver initialization.

| Case | Host peak MiB CPU / HIP | Device peak MiB CPU / HIP | H2D KiB CPU / HIP | D2H KiB CPU / HIP |
|---|---:|---:|---:|---:|
| analytic | 169.5 / 196.8 | 0.44 / 1.44 | 19.8 / 17.4 | 217.1 / 6.8 |
| many | 173.6 / 199.1 | 0.99 / 3.08 | 281.7 / 352.8 | 366.2 / 15.1 |
| stl | 190.5 / 209.5 | 0.96 / 5.55 | 252.5 / 1029.5 | 366.2 / 9.1 |
| ply | 180.5 / 209.4 | 0.97 / 5.56 | 257.1 / 1038.6 | 366.2 / 9.1 |
| weighted | 166.2 / 195.2 | 0.08 / 0.49 | 30.9 / 37.9 | 0.0 / 1.2 |
| voxel | 172.3 / 198.8 | 0.97 / 2.70 | 260.1 / 21.9 | 366.2 / 9.1 |
| cylinder | 177.1 / 199.7 | 0.22 / 3.47 | 21.6 / 31.5 | 204.8 / 198.2 |
| multigrid | 169.6 / 198.8 | 0.57 / 2.63 | 54.5 / 40.8 | 393.8 / 14.2 |

## Interpretation

No case has a warm setup median more than 10% slower on HIP. These are measurements for this machine and these models; no fixed speedup is promised.

Every paired output comparison uses finite-value checks, matching inventories/meshes/times, `1e-4` peak-relative and `1e-3` relative-L2 limits.
The release validation record contains the full per-run phase data, source states, compiler/device diagnostics and binary hashes. Earlier benchmark series are preserved separately.

Solver DLL SHA-256: `584482ef9b2dbcff30abb69855ba24b0bb8db41ccb0bd79be5fbed9e6eb11247`.
Throughput JSON SHA-256: `4c9fbfffb8520c2bcdbb3add20927a737c765d439cb2e33ab4290695bac407bd`.
Diagnostic JSON SHA-256: `c9de6e8ac75d3ed242b2eeebf8ff49dcb3303675bc417d537aa23bea5f52177c`.

AMD/ROCm, Metal and older NVIDIA/P5000 hardware were unavailable and remain untested.
