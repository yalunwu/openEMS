# Vulkan Phase 2 validation

Windows / NVIDIA RTX 3070, driver 591.74, Vulkan 1.2 backend, 2026-10-03/04.
Release builds use MinGW GCC 16.1.0 and MSVC 19.44.35229. The starting solver
revision is `a0e67e0` (Phases 0/1); Phase 2 was measured as a working-tree change
and later committed as `33807c7`.

The final cached-memory implementation passes all 56 native backend tests with
both compilers, CTest 3/3 with each compiler and with Vulkan disabled, all 22
Python GPU tests, and the Octave probe/stopping check below. These correctness
checks finished before the final performance measurements.

The final benchmark loads `openEMS/build/libopenEMS.dll`, SHA256
`B689FBAB33B3711C3CBA7F4159930B4594955CF7FE72ED542C0A98DDA9BD7E65`.
The executable's embedded version remains `Vulkan-Test-1-gfdf08cb` from the
cached CMake configuration; the actual checkout is
`a0e67e05754afe9b6eb403e7411d3adda615db01` plus the recorded Phase 2 diff.
Both optimized and reference measurements use this same DLL and executable.

## Implementation and numerical coverage

The final timestep submission gathers registered probes after hierarchy
projection. Cached gathers survive repeat synchronization but are invalidated by
stepping, setters, re-registration and reset. Standalone synchronization still
gathers when necessary. Twenty consecutive re-registrations test descriptor reuse
and resource cleanup; registration also waits for pending GPU work.

Basic Cartesian energy uses float products with 256-cell FP64 or FP32 partials,
followed by CPU double accumulation and EPS0/MUE0 weighting. The final cell plane
on each axis is excluded, matching the basic interface. SSE accumulates four
float lanes, includes the final z vector and may use different angular extents;
SSE, cylindrical and multigrid energy therefore keep their CPU implementation.
The SSE fallback test explicitly checks a nonzero final z lane; all multigrid
fixtures assert that the reduction capability is unavailable.

Reduction comparisons use identical field snapshots, independently of the
existing CPU/GPU stepping tolerance of 0.1%. Random fields span roughly twelve
orders of magnitude and include excluded boundary planes. The lossy PML fixture
runs 991 timesteps through excitation and decay, ending at about 7.93e-12 of its
sampled peak energy.

| Reduction test | FP64 maximum relative error | Forced FP32 maximum relative error |
| --- | ---: | ---: |
| Wide-range fields | 4.55e-14 | 5.18e-9 |
| Lossy PML excitation/decay | 3.37e-15 | 2.28e-8 |
| 5,000,211-node snapshot | 2.47e-13 | Not tested |

Acceptance tolerances are 1e-12 for FP64 and 3e-7 for FP32. Zero fields return
zero. Infinity, NaN and overflowing FP32 partials request CPU fallback; the latter
also verifies that the CPU can retain a finite result. These are measured bounds
on these fixtures, not guarantees for every input or vendor.
Tiny estimates use CPU fallback when the bounded loss from subnormal products
could exceed the active precision tolerance. A 1e-20 field test preserves a
positive CPU result with x86 flushing explicitly disabled. A zero GPU result
requires a complete, all-zero host mirror before it is cached directly.

Near-threshold exact-endcriteria tests choose a recorded decay ratio and perturb
it by plus/minus 1e-6 relative. CPU energy, FP64 reduction and forced FP32 stop at
84 steps below that ratio and 81 steps above it, with identical check schedules.
Sinusoidal steady-state detection stops at step 64 in all three modes. Energy-only
optimized runs download full fields only for the final compatibility sync.

Cached probe tests verify on-demand gathers, repeat syncs, edits, pending
re-registration, empty registries and reference mode. A 33-step request with
batch size 32 takes two submissions including its gather; the subsequent full
field sync takes one additional submission. A deliberately smaller staging
buffer exercises the two-copy fallback. The optimized basic CPU mirror copies
contiguous arrays directly; other layouts retain setters.
Mapped buffers prefer cached coherent memory, with the original coherent type
retained when a cached type is unavailable. On this device the optimized staging
uses host-cached memory type 4; the benchmark reference uses uncached type 3.

Python checks cover voltage/current integrals, E/H point probes at different
sampling rates, their timestamps and a derived Fourier voltage/current ratio
within 0.1%. Existing tests cover field dumps, supported materials/boundaries,
cylindrical and child-level multigrid output. An Octave CPU/Vulkan probe test
with exact-endcriteria stops at the same step 144 and agrees within 0.1%.

## Profiling-disabled performance

The final 15-case matrix records 150 measured runs: five repeats in each mode,
256 timesteps and batch limit 32. The reference disables Phase 2 readbacks in
the same executable and retains Phase 1 batching. Both modes use identical
processing schedules. Initialization and warm-up are excluded; completion of
pending GPU work and the final full-field synchronization are included.

| Solver workload | Reference median seconds (min–max) | Phase 2 median seconds (min–max) | Speedup |
| --- | ---: | ---: | ---: |
| Probes / energy checks, 33³ | 0.5669 (0.4818–0.6059) | 0.05660 (0.03982–0.07062) | 10.0× |
| HDF5 dumps / energy checks, 33³ | 5.257 (5.202–5.402) | 3.640 (3.609–3.810) | 1.44× |
| Steady-state, 17³ | 0.10098 (0.09992–0.10575) | 0.05496 (0.05354–0.07705) | 1.84× |
| Energy checks, 171³ / 5,000,211 nodes | 11.669 (11.624–11.792) | 0.6743 (0.5544–0.7015) | 17.3× |

The large case enables frequent deterministic energy checks and writes no field
dump. Its roughly 11-second setup is excluded from the speedup above; initialization
can dominate such a short simulation. Gains depend on how often the model asks
for energy or output. Full dumps still transfer the whole domain and retain
their filesystem/processing costs.

Separate one-repeat profiles explain the readback changes; their timings are
not used for the speedups above. Counts exclude the untimed energy verification.

| Workload | Submissions, reference → Phase 2 | Full-field bytes, reference → Phase 2 | Phase 2 energy-result bytes |
| --- | ---: | ---: | ---: |
| Probes | 598 → 299 | 37,086,984 → 862,488 | 86,016 |
| HDF5 dumps | 768 → 554 | 220,796,928 → 220,796,928 | 86,016 |
| Steady-state | 536 → 268 | 1,414,944 → 117,912 | 2,816 |
| Five-million-node energy checks | 34 → 25 | 1,080,045,576 → 120,005,064 | 2,456,576 |

The large case performs eight reductions, each returning 307,072 bytes instead
of copying 120,005,064 bytes of fields (391× fewer result bytes per check).
Its final compatibility sync still copies the full domain. In the optimized
profile, CPU readback costs about 74 ms and mirroring 8 ms; GPU timestep batches
take about 516 ms. Readback timings include waits and can overlap other columns.
Probe-result bytes are unchanged by fusion: 16,384 for the probe case and 21,504
for steady-state. Both modes preserve CPU interpolation and output scheduling.

Direct-stepping cases do not consume fields or probes during the measured run.
Their initial 256-step medians are recorded below; these short timings also
capture clock and scheduling variation.

| Stepping workload | Reference median ms | Phase 2 median ms |
| --- | ---: | ---: |
| Small, 25³ | 3.224 | 3.316 |
| Large, 129³ | 165.514 | 163.690 |
| Asymmetric, 97×31×65 | 17.803 | 18.475 |
| Thin, 129×129×3 | 6.968 | 6.959 |
| Lossy/nonuniform, 65×49×33 | 14.171 | 9.983 |
| PML, 65³ | 58.171 | 64.420 |
| Two-pole Debye, 49³ | 18.378 | 19.355 |
| Cylinder, 41×129×33 | 33.043 | 32.567 |
| Multigrid, one level | 16.823 | 22.138 |
| Multigrid, two levels | 30.541 | 32.224 |
| Multigrid, five levels | 48.197 | 47.397 |

Apparent slowdowns above 5% in PML, Debye and the one/two-level multigrid cases
were checked with 4096 timesteps, five repeats per mode and the reference run
first. They did not repeat: median differences range from -1.54% to +1.92%.
No direct-stepping gain is claimed from the short sweep.

| Follow-up, 4096 steps | Reference median ms (min–max) | Phase 2 median ms (min–max) |
| --- | ---: | ---: |
| PML | 948.485 (942.026–950.001) | 941.583 (936.874–948.105) |
| Debye | 305.742 (297.629–314.324) | 301.024 (299.441–303.548) |
| Multigrid, one level | 282.730 (279.490–285.864) | 278.441 (276.759–283.334) |
| Multigrid, two levels | 504.108 (497.380–509.378) | 513.764 (512.203–516.692) |

## Reproduction

```text
ctest --test-dir build --output-on-failure
test_backend --readback-tests
test_backend --vulkan-benchmark --batch-size=32 --csv=final-optimized-matrix.csv
test_backend --vulkan-benchmark --batch-size=32 --reference-readback --csv=final-reference-matrix.csv
test_backend --vulkan-benchmark --case=energy-large --batch-size=32 --verify-energy --csv=energy-large.csv
test_backend --vulkan-benchmark --case=energy-large --batch-size=32 --repeats=1 --profile --verify-energy --csv=profile-energy-large.csv
test_backend --vulkan-benchmark --case=pml --steps=4096 --batch-size=32 --csv=long-pml.csv
```

Local generated results, logs, the Octave script, source diffs and library hashes
are under `openEMS/build/phase02-benchmark/` and remain gitignored. Python loads
`build/python/openEMS` and preloads the matching `openEMS/build/libopenEMS.dll`.
Performance runs are separate from compiler jobs and correctness tests, exclude
initialization and warm-up, include final GPU completion, and use five repeats.
Final measurements are named `final-optimized-matrix`, `final-reference-matrix`,
`final-profile-<mode>-<case>` and `final-long-<mode>-<case>`; earlier measurements
remain available as development history. `source.diff`, copied validation notes
and SHA256 manifests identify the final source and binaries.

Coverage is Windows / RTX 3070. FP32 was forced on this FP64-capable device;
hardware without FP64 and other vendors remain untested. Linux and Vulkan
synchronization validation are also untested: the Khronos validation layer is
not installed on this host. The five-million-node energy check compares the
reduction against CPU accumulation on GPU field snapshots; it is not a separate
five-million-node CPU-versus-GPU physics run. Larger 19M/40M models remain outside
this validation.
