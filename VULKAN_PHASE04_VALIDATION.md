# Vulkan Phase 4 validation

Windows / NVIDIA RTX 3070, driver 591.74, 2026-10-04. Source: `9e94c73` plus
the Phase 4 working-tree changes. Release builds: MinGW GCC 16.1.0 and MSVC 19.44.
Throughput measurements use the MinGW executable and `openEMS/build/libopenEMS.dll`,
SHA256 `5D2915B1A0B5420E67628C6D565CD5ADD522C6FAFB596B035A5705C606427D26`.

## Representation and selection

The palette stores all twelve final coefficient float bit patterns per node,
ordered by direction and then vv/vi/ii/iv. This follows the existing SSE
compression approach of comparing complete operators, without its four-lane
packing. Material IDs are insufficient: geometry, anisotropy and boundary
corrections are included through the final coefficients. No quantization is used.

Analysis compared this layout (4 bytes per node plus 48 per unique tuple) with
direction-specific component palettes (12 bytes per node plus 16 per unique
component tuple). The following are real tutorial/antenna meshes, including
smoothed nonuniform spacing. MB below means 1,000,000 bytes of coefficient payload.

| Model | Stored nodes | Unique nodes | Unique components | Dense MB | Node palette MB | Component palette MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSL notch filter | 269,780 | 5,054 | 7,935 | 12.949 | 1.322 | 3.364 |
| Simple patch antenna | 103,635 | 6,300 | 16,401 | 4.974 | 0.717 | 1.506 |
| Large dipole array | 5,517,027 | 4,557 | 95 | 264.817 | 22.287 | 66.206 |

Whole-node indexing won on storage in all three cases and needs one index load
per half-step instead of three. The component layout was counted, not implemented
as a second shader. The two Python tutorials were exported with rebuilt bindings;
the dipole array used the existing `build/large-antenna/large-mesh` export.

`--vulkan-coefficients=palette` is opt-in. Each hierarchy level retains dense
storage if the candidate saves no bytes, exceeds 65,536 tuples or the device
buffer limit, or cannot allocate optional storage. The tuple cap limits resource
usage; it is not an inferred GPU cache size. Upload/synchronization errors still
fail initialization. `analyze` counts complete dictionaries and uses dense shaders.

For the 129-cubed benchmark, the six-entry palette uses 8,587,056 allocated device
bytes versus 103,041,088 for dense coefficients. These measured allocation totals
include Vulkan alignment and exclude fields, extensions and staging. Host palettes
are discarded after initialization.

## Correctness

- All 67 native backend tests pass with both compilers. CTest passes 4/4 with
  each compiler and 4/4 with Vulkan disabled, including invalid-option diagnostics.
- Exact reconstruction tests distinguish signed zero, adjacent floats, subnormal
  bits and NaN payloads. They also cover duplicate tuples, empty input and bounded
  construction. These are integer bit comparisons, not floating-point tolerances.
- CPU/dense-GPU/palette-GPU comparisons cover PEC/PMC, anisotropic lossy nonuniform
  meshes, PML and two-pole Debye, with batches 1/31/33/64 and continuation. CPU
  tolerance remains 1e-4 absolute and 0.1% of peak. The two GPU layouts had zero
  measured field difference on these fixtures (allowed bound: 1e-7 of peak).
- An all-unique synthetic decay operator verifies storage-cost fallback. A
  185,115-node nonuniform cylindrical multigrid root exceeds the tuple cap and
  uses dense storage while its children use palettes. Closed/open angular grids,
  PML/Mur, Debye, shared pipelines and reset/reinitialization are covered.
- All 26 Python GPU tests pass against the newly built DLL and rebuilt extension.
  The new test compares nonuniform anisotropic lossy lumped-port V/I records,
  derived impedance and HDF5 dump samples/timestamps in dense and palette modes.
- The Octave CPU/dense/palette probe check stops at step 96 in every mode with
  97 samples. Maximum relative voltage/H-probe differences are 2.44e-7 / 2.02e-7
  against the CPU reference, below the existing 0.1% bound.

An MSVC rebuild exposed Windows min/max macros in the existing batching code;
`NOMINMAX` is now set before including its headers. C++ layout changes require
rebuilding Python extensions with the updated native headers.

## Profiling-disabled performance

Batch limit 32, one untimed warm-up and five measured runs per point. The complete
15-case matrix used 256 fixed timesteps. Times include GPU completion and exclude
initialization; output cases include processing and the final field readback.
No compiler jobs or other test suites ran during measurement. These are local
interactive-desktop measurements, without locked GPU clocks.

| Case | Dense median ms | Palette median ms | Dense / palette |
| --- | ---: | ---: | ---: |
| small | 3.583 | 3.309 | 1.083 |
| large | 164.308 | 118.053 | 1.392 |
| asymmetric | 18.246 | 13.479 | 1.354 |
| thin | 7.407 | 11.427 | 0.648 |
| lossy-nonuniform | 13.895 | 15.893 | 0.874 |
| pml | 60.321 | 54.176 | 1.113 |
| debye | 18.626 | 14.419 | 1.292 |
| cylinder | 31.192 | 40.106 | 0.778 |
| multigrid-1 | 17.157 | 14.488 | 1.184 |
| multigrid-2 | 47.069 | 30.975 | 1.520 |
| multigrid-5 | 49.856 | 60.053 | 0.830 |
| probes | 39.850 | 69.948 | 0.570 |
| dumps | 3169.937 | 3145.286 | 1.008 |
| steady-state | 38.175 | 42.828 | 0.891 |
| energy-large | 627.071 | 579.791 | 1.082 |

Short-run regressions above 5% were checked with 2,048-step runs, reversing the
mode order (palette then dense) and retaining five repeats. Ranges are min–max.

| Case | Dense median ms (range) | Palette median ms (range) | Dense / palette |
| --- | ---: | ---: | ---: |
| large | 1342.37 (1316.54–1346.27) | 940.98 (922.45–941.35) | 1.427 |
| thin | 60.35 (58.86–63.13) | 58.27 (57.82–58.44) | 1.036 |
| lossy-nonuniform | 77.25 (76.95–78.07) | 78.59 (77.01–79.58) | 0.983 |
| cylinder | 252.07 (249.04–254.85) | 221.80 (206.19–225.70) | 1.136 |
| multigrid-5 | 402.96 (399.09–409.20) | 375.43 (373.16–381.29) | 1.073 |
| probes | 269.83 (174.60–352.69) | 322.22 (165.85–370.36) | 0.837 |
| steady-state | 123.87 (93.22–209.56) | 219.52 (127.65–232.13) | 0.564 |

The large grid shows a stable 1.43x stepping throughput gain, exceeding run
spread. The plan's 120-byte/node dense traffic model gives about 393 GB/s for
that baseline. Single-fixture initialization grew from 5.09 to 5.42 seconds;
palette construction and two extra shader compilations add setup cost.

Thin/cylindrical/multigrid short-run slowdowns did not reproduce in the longer
checks. The lossy-nonuniform candidate hits the cap and actually uses dense
shaders; its longer-run difference is small. Probe and steady-state runs have
wide, overlapping timing ranges and worse palette medians. They do not establish
a benefit. Dense remains the default and the reference for workloads where lookup
or setup cost is unfavorable. Automatic device/workload selection is deferred.

## Tutorial throughput

The exported notch and patch tutorials were also run for 1,024 steps, with
field dumps disabled and their ordinary probes retained. Each mode had one
warm-up and five measured solver runs. XML used `endCriteria=1e-30` (the existing
XML reader replaces zero with its default); every run was checked to reach
exactly 1,024 steps. Reported simulation time excludes initialization.

| Tutorial | Dense median ms (range) | Palette median ms (range) | Dense / palette |
| --- | ---: | ---: | ---: |
| MSL_NotchFilter | 171.41 (168.13–172.17) | 147.21 (146.42–152.22) | 1.164 |
| Simple_Patch_Antenna | 81.70 (79.54–93.60) | 63.57 (60.11–88.54) | 1.285 |

The notch filter's gain exceeds these run ranges. Patch timing ranges overlap,
so its lower median should not be treated as a guaranteed gain. The large dipole
array was measured for coefficient reuse/storage, not repeated throughput.

## Reproduction and limits

From a scratch output directory, using the matching dependency DLLs:

```text
ctest --test-dir openEMS/build --output-on-failure
openEMS/build/test_backend --coefficient-tests
openEMS/build/test_backend --vulkan-benchmark --batch-size=32 --coefficients=dense --csv=dense.csv
openEMS/build/test_backend --vulkan-benchmark --batch-size=32 --coefficients=palette --csv=palette.csv
openEMS/build/test_backend --vulkan-benchmark --case=large --steps=2048 --repeats=5 --batch-size=32 --coefficients=palette --csv=large.csv
python -m unittest discover -s openEMS/python/Tests -p test_gpu_engine.py -v
```

Local CSVs, tutorial exports, run logs and the Octave check are under
`openEMS/build/phase04`; build/CTest/Python logs are under the meta-repository
`build/phase04-*.log`. Generated results remain untracked.

Only this NVIDIA device and Windows were tested. Vulkan validation layers,
other GPU vendors and forced allocation exhaustion were not tested. Allocation
failure has a dense fallback, but the tests exercise capacity/storage selection
rather than inducing out-of-memory conditions. No synchronization, extension
ordering, workgroup or precision changes are part of Phase 4.
