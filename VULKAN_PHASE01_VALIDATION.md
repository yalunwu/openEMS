# Vulkan Phase 0 / Phase 1 validation

Measured 2026-10-03 on Windows x64 with an NVIDIA GeForce RTX 3070.
Vulkan reported vendor 4318, device 9348, driver identifier 2480046080 and API
version 4211013. The benchmark used GCC 16.1.0, Release, revision
`Vulkan-Test-1-gfdf08cb` plus the uncommitted Phase 0/1 changes, and loaded
`C:\Users\Allen\openEMS-Project\openEMS\build\libopenEMS.dll`.

## Correctness and configuration coverage

- MinGW Vulkan build: native CTest passed, **52 tests**, 38.36 seconds.
- MSVC Vulkan build: native CTest passed, **52 tests**, 49.89 seconds.
- MinGW Vulkan-disabled Release build: built and CTest passed; GPU checks are
  skipped in that configuration.
- Rebuilt local Cython bindings: all **21 Python GPU tests** passed, 40.28 seconds.
- Octave 11.3.0: `RunOpenEMS` accepted batch sizes 1/32 and profiling, activated
  Vulkan, completed 97 steps, and produced matching probe times and values.
- Command-line checks reject limits 0/65 cleanly in MinGW, MSVC and the
  Vulkan-disabled build.
- `git diff --check` passed.

Focused tests cover batch limits 1/8/16/32/64, caller intervals 0/1/7/33/65,
continuation, a delayed source, mixed Mur/PML boundaries, overflow rejection,
field edits/uploads, profiling toggles, reset with pending work and solver error
propagation. Submission counts equal the ceiling of interval length divided by
the configured batch limit; direct stepping downloads no fields.

The mixed-boundary CPU/GPU comparison uses a 1e-4 absolute and 0.1% peak-relative
field tolerance. Observed maximum error was below 4e-10 across all batch limits.
The longer test exposed an existing PML pre-update ordering error at Mur/PML
intersections. Matching the CPU extension priority order fixed it.

Multigrid equivalence now includes caller intervals 1/7/19/33/65, spanning several
internal batches and a final partial batch, across the existing material,
excitation, boundary and hierarchy fixtures. The five-level fixture's observed
maximum field error was below 3e-9. Python additionally compares probe and HDF5
dump timestamps across batch sizes 1/32/64, with 1e-4 / 0.1% field tolerances.

## Profiling-disabled performance

The matrix contains 14 cases, five batch limits and five measured repetitions
per point: **350 measurements** after warm-up. Initialization is excluded from
elapsed times; final GPU synchronization is included. Case shapes and materials
are defined in `test_vulkan_performance.cpp`.

The reference is batch size 1 in the same rebuilt binary. These comparisons
isolate submission batching rather than comparing different compiler builds or
older binaries. Median elapsed milliseconds are followed by the minimum–maximum
range. The reduction column is `(time1 - time32) / time1`.

| Case, 256 steps | Batch 1, ms (range) | Batch 32, ms (range) | Time reduction |
| --- | ---: | ---: | ---: |
| small, 25³ | 18.662 (18.098–20.732) | 6.284 (6.018–6.851) | 66.3% |
| large, 129³ | 184.835 (182.199–187.182) | 164.599 (164.540–166.161) | 10.9% |
| asymmetric, 97×31×65 | 33.359 (32.866–35.106) | 18.413 (18.094–18.876) | 44.8% |
| thin, 129×129×3 | 21.843 (21.760–24.474) | 15.507 (15.278–17.060) | 29.0% |
| lossy/nonuniform, 65×49×33 | 30.902 (30.514–33.393) | 16.389 (16.057–16.588) | 47.0% |
| PML, 65³ | 74.973 (72.671–96.257) | 59.262 (58.393–59.774) | 21.0% |
| two-pole Debye, 49³ | 33.185 (33.163–35.395) | 28.856 (27.064–29.635) | 13.0% |
| cylinder, 41×129×33 | 46.040 (45.656–48.287) | 31.617 (30.821–32.139) | 31.3% |
| multigrid, one level | 32.399 (31.985–35.183) | 17.028 (16.758–17.982) | 47.4% |
| multigrid, two levels | 46.249 (45.404–48.106) | 31.177 (30.597–31.613) | 32.6% |
| multigrid, five levels | 60.726 (60.426–63.240) | 47.073 (46.506–47.907) | 22.5% |
| probes / energy checks | 568.528 (472.761–604.601) | 470.840 (465.840–585.602) | 17.2% |
| HDF5 dumps / energy checks | 5453.826 (5357.516–5478.234) | 5406.186 (5368.665–5691.294) | 0.9% |
| steady-state | 122.293 (114.033–125.976) | 113.783 (106.860–116.526) | 7.0% |

The final three cases include actual solver processing, output and final field
downloads. Their sampling requires one timestep per solver call in these
fixtures, so increasing the internal cap cannot reduce stepping submissions.
Treat their differences as run variation, not evidence of batching gains.
The dump case shows that output costs dominate this workload.

Short small-grid results varied enough to make the 32-step result slower than
16 steps in the first sweep. A separate 4096-step sweep resolves that uncertainty:

| Batch limit | Median ms | Minimum–maximum ms |
| ---: | ---: | ---: |
| 1 | 336.097 | 334.219–341.191 |
| 8 | 90.347 | 80.272–96.274 |
| 16 | 64.277 | 63.467–65.851 |
| 32 | 53.507 | 53.008–54.430 |
| 64 | 51.079 | 50.135–51.642 |

That longer run gives **6.28×** stepping throughput at 32 versus 1, with an 84.1%
elapsed-time reduction. Keep the bounded default of 32; 64 offers a modest further
gain here. Workload-specific limits remain configurable.

## Profiling cost and diagnostic findings

Five measured profiling runs at batch size 32, compared with the corresponding
profiling-disabled measurements above:

| Stepping case | Disabled median ms | Enabled median ms (range) | Added elapsed time |
| --- | ---: | ---: | ---: |
| small, 4096 steps | 53.507 | 58.860 (56.140–60.527) | 10.0% |
| large, 256 steps | 164.599 | 174.745 (174.022–175.547) | 6.2% |
| PML, 256 steps | 59.262 | 62.423 (61.977–62.961) | 5.3% |
| multigrid, five levels, 256 steps | 47.073 | 53.149 (51.960–53.689) | 12.9% |

GPU timestamps are available on this device and all categories were exercised.
The direct 256-step cases report eight submissions and zero field downloads.
The five-level case reports 9480 dispatches, about 12 ms of CPU command recording
and about 38 ms of GPU batch time; command-buffer overlap is a candidate for a
separate follow-up experiment.

The probe fixture reports 598 total submissions, including 256 stepping and 256
probe gathers, 37,086,984 field-download bytes and 16,384 mapped probe bytes.
The dump fixture reports 768 submissions and 220,796,928 field-download bytes.
The steady-state fixture reports 538 submissions and 256 stepping batches.
These counters identify processing/readback work for Phase 2. Output profiling
times varied substantially; they are diagnostic runs, not throughput baselines.
Sampled GPU category times are approximate and overlap the full batch measure.

## Reproduction and artifacts

See `VULKAN_PERFORMANCE.md` for options and benchmark semantics. From a reserved
output directory, with the matching DLL dependencies on the executable path:

```text
test_backend --vulkan-benchmark --csv=matrix.csv
test_backend --vulkan-benchmark --case=small --steps=4096 --csv=small-longer.csv
test_backend --vulkan-benchmark --case=large --batch-size=32 --profile --csv=profile-large.csv
test_backend --batching-tests
```

Local raw CSV/log pairs are in `openEMS/build/phase01-benchmark/`: `matrix`,
`small-longer`, `profile-small`, `profile-large`, `profile-pml`,
`profile-multigrid-5`, `profile-probes`, `profile-dumps`, `profile-steady-state`
and `profile-small-longer`. The directory also stores the tracked source diff
and copies of the new benchmark sources. Test logs are under `openEMS/build/`
with the `phase01-` prefix. These generated build artifacts are gitignored.

The Python test run loaded the rebuilt local binding from `build/python/openEMS/`
and preloaded its matching DLL from `openEMS/build/libopenEMS.dll`. Installed
headers came from `build/install/`. Rebuild bindings with matching installed
headers when replacing the native library. The older system binding was
excluded from these results.

Coverage is Windows / RTX 3070 with MinGW and MSVC. Linux, additional vendors and
Vulkan validation layers were not tested in this run. Portable performance claims
require additional hardware measurements.
