# Vulkan batching, readbacks and profiling

The Vulkan backend records up to 32 timesteps per queue submission by default.
The limit controls submission size. Ordinary V/I/E/H probes can keep their sample
intervals while the GPU records several steps: a mapped buffer retains each
intermediate result, then the CPU replays processing at the original timestep.
Submissions end at the next full-field consumer or energy/stopping check.
Steady-state detection also retains its one-step samples and period checks.
Every-step full-field output or stopping checks still require one-step submissions.

During history recording, probe gathering and multigrid projection run after each
step. Direct `IterateTS()` calls project and gather at the end of the requested
interval. Repeated probe syncs reuse the completed result.
Field edits, stepping, probe registration and reset invalidate cached results;
standalone probe synchronization can still gather on demand. Partial probe updates
do not mark the entire CPU field mirror current. Interpolation and integration
remain in the existing CPU processing code.
History storage grows lazily, up to 64 frames of registered probe values;
allocation failure retains ordinary stepping and the previous probe resources.
Full fields are available only after replay reaches the recorded batch end.

Energy-only checks use a GPU reduction for the basic Cartesian engine. It squares
the stored float fields over the same interior cells as `CalcFastEnergy()`, sums
256-cell partials, and applies EPS0/MUE0 weighting in CPU double precision. FP64
partials are used when supported; otherwise the reduction uses FP32 partials.
The validated relative tolerances are 1e-12 and 3e-7 respectively. Zero fields
produce zero energy; non-finite results, including FP32 partial-sum overflow,
request the synchronized CPU calculation. Very small estimates also use the CPU
when flushing subnormal float products could exceed the tolerance; a zero GPU
result is accepted directly only when the reduction's GPU-side bit check finds no
nonzero fields in the energy domain. SSE energy uses different FP32 lane
accumulation and extents, so SSE, cylindrical and multigrid cases retain that CPU
calculation. This estimates the existing stopping metric, not physical energy.

Full-field consumers still synchronize all fields. Voltage and current copies
share one submission and wait when a two-field staging buffer is available.
This adds 12 bytes of host-visible staging storage per stored node; allocation
failure retains the smaller buffer and separate copies. The final synchronization
still populates all CPU fields before the solver reports success.
Basic-engine mirrors copy the matching contiguous arrays directly; other field
layouts retain their existing setter-based conversion.
Mapped buffers prefer host-cached coherent memory when the device exposes a
matching memory type, retaining the original coherent type otherwise. This
reduces CPU read costs for staging, probe values and energy partials without
requiring explicit cache invalidation. Device metadata reports
`staging_host_cached` for the selected field staging allocation.

## Simulation options

Use `--vulkan-batch-size=N` to select a limit from 1 to 64. A limit of 1 provides
the single-step submission reference. Enable `--vulkan-profile` for diagnostics:

```text
openEMS model.xml --engine=vulkan --vulkan-batch-size=32 --vulkan-profile
```

Python forwards the same options through `Run()`:

```python
fdtd.Run(sim_path, engine='vulkan', vulkan_batch_size=32, vulkan_profile=True)
```

In MATLAB/Octave, pass them in the existing `RunOpenEMS` options string:

```matlab
RunOpenEMS(sim_path, 'model.xml', '--engine=vulkan --vulkan-batch-size=32 --vulkan-profile')
```

Profiling is disabled by default. `VULKAN_PROFILE` reports runtime timesteps,
submissions, dispatches, emitted/coalesced memory barriers, field upload/download
bytes, probe and energy result
bytes and CPU recording/submission/wait/readback/mirror costs. Initialization is
excluded.
Readback costs include their waits, so CPU timing columns can overlap. Probe
bytes represent GPU writes to mapped host-visible memory, not measured PCIe
traffic. `energy_bytes` similarly counts mapped reduction results; they are
separate from `downloaded_bytes`, which counts field copies. Counts are collected
only when profiling is enabled.

GPU timestamps measure each timestep submission and sampled dispatches from the
first step in that submission, plus final multigrid projection, probe gathering
and field/energy readbacks. Dispatch categories are voltage, current, extension and
multigrid. GPU category totals describe their measured samples; do not add them
to the batch total or extrapolate them without using sample counts. A bounded
query pool can omit later dispatch samples in unusually deep/complex hierarchies.
Dispatch timestamps can include scheduling/overlap costs and are approximate.
Per-category dispatch counters cover all recorded work, including child levels;
GPU category times still cover only sampled work. The barrier category brackets
sampled runtime barriers with timestamps. It includes scheduling and timestamp
overhead and is not an isolated hardware barrier latency. Consecutive identical
compute barriers can be coalesced without changing dispatch or extension order,
but this candidate is disabled by default because local throughput measurements
did not show a consistent benefit.
See [Synchronization dependencies](#synchronization-dependencies) for the
recording invariants and their validation.
GPU timing instrumentation itself can alter scheduling; use profiling-disabled
runs for speed comparisons. Unsupported timestamps retain CPU diagnostics.

The backend checks the compute queue's timestamp support, scales ticks with the
device period, masks timestamp wraparound, and collects available results after
existing fence waits. It adds no query-result waits. `Synchronize()` waits for
stepping without downloading fields; `ClearProfile()` synchronizes before
resetting statistics. `Reset()` waits before destroying pending GPU resources.
GPU execution/synchronization errors propagate to the command-line solver or
Python caller instead of continuing with stale fields.

## Frequency-domain fields

`--vulkan-fd=gpu` opts into GPU E/H Fourier accumulation on the basic Cartesian
engine. `--vulkan-fd=cpu` is the default and the explicit reference. Python uses
`fdtd.Run(sim_path, engine='vulkan', vulkan_fd='gpu')`; MATLAB/Octave passes
`--engine=vulkan --vulkan-fd=gpu` in the `RunOpenEMS` options string.
No XML property or output format changes are needed.

Supported consumers are frequency-domain E/H dumps (`DumpType` 10/11), including
planar FD NF2FF recordings, with native (0), node (1) or cell (2) sampling.
They use the resolved output mesh, including subsampling and `OptResolution`.
Nonuniform edge scaling and boundary interpolation match the existing interface.
Cylindrical/multigrid/SSE operators, custom interfaces/processings, other field types and SAR
retain CPU accumulation. TD output retains its existing sampling and readbacks.
The angular NF2FF transform still runs in the standalone CPU postprocessor.

Eligible samples accumulate after each complete timestep within a submission.
They retain the start/stop window, FD oversampling, negative Fourier phase,
staggered H time and `2*dt*FD_interval` weight. Complex FP32 sums have the same
precision as CPU FD storage. The CPU prepares a small batch of phase factors
once per frequency/sample using the reference `std::exp` arithmetic; all output
points reuse those factors. It introduces no wait per sample. Supported FD
samples stop forcing full-field downloads or batch boundaries. CPU consumers,
ordinary probe replay and stopping checks still determine their own boundaries.
Completion, convergence and graceful abort download sums before the existing
HDF5/legacy HDF5 or VTK writers run. The existing final CPU field sync remains.

Each dump needs 24 bytes per output point per frequency for its sums, plus a
compact separable axis mapping and 64 phase slots per frequency. E and H over
10 million points need 480 MB per frequency, or 2.4 GB for five frequencies,
in addition to the solver's existing allocations. `VULKAN_FD` reports each
dump's choice, sample interval, estimated bytes, actual sum allocation bytes,
mapping bytes, chunk count and accumulator read/write traffic (`2B/I` bytes
per timestep). The output point count can differ between E and H.
The existing CPU FD arrays remain allocated for finalization and file writers,
so GPU accumulation adds device storage rather than reducing host memory.

Memory selection uses `VK_EXT_memory_budget` when available, otherwise a
conservative quarter of each device-local heap's capacity. Driver estimates
must fit within 75% of the available allowance. The fallback applies that limit
across all FD dumps using actual allocation sizes on their selected heaps,
including device-local mapping, phase and shared staging allocations.
Allocation failure selects CPU accumulation for that dump before its first
sample. Size, storage-buffer,
allocation-count and dispatch limits are checked; sums are partitioned where
needed. A shared 16 MiB readback buffer bounds final staging. Partitioning does
not reduce the total sum storage. An accumulated prefix is never discarded to
fall back during a run; GPU submission/upload/readback failures remain errors.
This is an opt-in resource policy, not an automatic performance selector.
Changing registered frequencies or interpolation requires reset and
reinitialization; stepping rejects an inconsistent registration. Reset also
clears FD sums and sample counters. Re-registration refuses to discard live sums.
Enabling GPU accumulation also rejects a supported dump with a CPU-accumulated
prefix. All candidates are checked before changing any registration; reset and
reinitialize processing before selecting a different accumulation path.

Profiling adds `fd_samples` (sum of per-dump samples), `fd_dispatches`,
`fd_accumulator_bytes`, `fd_mapping_bytes`, `fd_phase_bytes` (mapped host writes),
`fd_phase_ms` (CPU preparation), `fd_download_bytes` and `fd_download_ms`
(bounded final readback, waits and copying). FD download bytes are separate from
`downloaded_bytes`, which continues to count full fields. Storage byte gauges
describe live registered resources even when timing is disabled. `gpu_fd_ms`
samples dispatches from the first eligible FD timestep in each submission, plus
the initial sample, within the bounded timestamp pool. It is not the total
accumulation time. Final file-writing time is included
in the benchmark's solver-output runtime, outside GPU timestamps.

Explicit benchmark cases are `fd-volume`, `fd-surface`, `fd-line` (65 cubed),
`fd-large` (129 cubed), `fd-10m` (217 cubed, 10,218,313 stored nodes) and `fd-none`
(65 cubed, diagnostic no-dump bound). They are excluded from the default matrix.
Use matching regions/frequencies/steps for CPU and GPU comparisons:

```text
test_backend --vulkan-benchmark --case=fd-volume --fd=cpu --fd-frequencies=5 --fd-over-sampling=32 --steps=257 --batch-size=64 --repeats=3 --csv=cpu.csv
test_backend --vulkan-benchmark --case=fd-volume --fd=gpu --fd-frequencies=5 --fd-over-sampling=32 --steps=257 --batch-size=64 --repeats=3 --csv=gpu.csv
```

`--fd-dump-mode=0|1|2` selects interpolation. Outputs use the
`vulkan_benchmark_fd_cpu_*` and `vulkan_benchmark_fd_gpu_*` prefixes. Run in a
scratch directory and compare complex files before interpreting timings.
`--field-accumulation-tests` covers partitioning, reset/reuse, allocation fallback,
registration rejection and graceful interruption. Its spectrum comparisons reject
NaN and infinity in either complex component before applying the 0.1% tolerance.
The Python and Octave tests exercise scripting and
file/NF2FF compatibility. The independent synchronization check is:

```text
cmake -DBACKEND_EXECUTABLE=/absolute/path/to/test_backend -DTEST_ARGUMENT=--field-accumulation-tests -DSHADER_HEURISTIC=1 -P cmake/tests/check_vulkan_synchronization.cmake
```

`--fd-phase-benchmark` evaluates an experimental batched GLSL sine/cosine
generator against the CPU factors, including large timestep offsets. It does
not select a runtime implementation. See
[Recorded validation and measurements](#recorded-validation-and-measurements)
for accuracy, phase-generation and end-to-end measurements and their limits.

## Coefficient storage

`--vulkan-coefficients=palette` enables an exact whole-node palette for the four
operator coefficients in all three directions. Each stored node has one 32-bit
index into twelve float bit patterns. Geometry, anisotropy, loss and boundary
corrections come from the final operator values; material IDs are not used.
Fields and arithmetic retain their existing precision.

Selection is independent at each multigrid level. Dense storage is retained if
index plus palette bytes do not improve on 48 bytes per node, the palette exceeds
65,536 tuples (3 MiB of payload), device buffer limits are exceeded, or optional
host/device allocation fails. Device upload errors still fail initialization.
The tuple limit bounds dictionary growth and is a resource cap, not a measured
cache-size threshold. Dense remains the default because storage savings alone do
not establish a throughput benefit on every device and workload.

`--vulkan-coefficients=analyze` counts both whole-node and per-component tuples
without the tuple cap and runs the dense reference. It can use substantial host
memory on mostly unique meshes. Run a short simulation to initialize the backend;
`--no-simulation` stops before backend initialization. `VULKAN_COEFFICIENTS` reports
counts, candidate payload bytes, selected storage and coefficient device allocation
bytes for each hierarchy level. `complete=0` marks an abandoned candidate: its
unique count and candidate bytes are lower bounds. Component counts are populated
only in analysis mode. Allocation bytes include Vulkan alignment; they exclude
field, extension and staging buffers.

Python accepts `vulkan_coefficients='palette'` through `Run()`. MATLAB/Octave
accepts the same command-line option in the `RunOpenEMS` options string.
Rebuild Python extensions against the updated C++ headers when updating an
existing native installation.

## Benchmark utility

From a directory reserved for benchmark output, run the newly built
`test_backend` executable. Ensure its matching dependency DLLs are available.
On Windows the commands use `test_backend.exe`.

```text
test_backend --vulkan-benchmark --csv=matrix.csv
test_backend --vulkan-benchmark --case=large --steps=1024 --repeats=5 --csv=large.csv
test_backend --vulkan-benchmark --case=multigrid-5 --batch-size=32 --profile --csv=profile.csv
test_backend --vulkan-benchmark --case=energy-large --batch-size=32 --verify-energy --csv=energy.csv
test_backend --vulkan-benchmark --case=energy-large --batch-size=32 --reference-readback --csv=reference.csv
test_backend --vulkan-benchmark --case=large --batch-size=32 --coefficients=palette --csv=palette.csv
```

The default sweep tests batch sizes 1, 8, 16, 32 and 64, with 256 timesteps and
five measured runs per point. An untimed warm-up precedes recorded results.
Reported totals include final GPU completion; initialization, iteration and
synchronization are recorded separately. Results include median, minimum and
maximum elapsed time, compiler/build/revision, GPU/driver identifiers, loaded
library path when available, stored hierarchy nodes, active voltage-update nodes
and the operator's existing cell metric. The revision alone does not identify an
uncommitted change: save the diff with the measurement record.

Cases: `small`, `large`, `asymmetric`, `thin`, `lossy-nonuniform`, `pml`, `debye`,
`mur`, `tfsf`, `lorentz`, `lorentz-large` (three poles, 25/49 cubed), `sheet`,
`rlc`, `absorber`,
`cylinder`, `multigrid-1`, `multigrid-2`, `multigrid-5`, `probes`, `dumps`,
`steady-state`, `energy-large` (171 cubed, 5,000,211 stored nodes).
Cases without processing measure stepping directly without field downloads;
repeats continue the warmed simulation. The last four run fresh fixed-length
solver instances and include actual processing/output and final field readback.
Output filenames use the `vulkan_benchmark_` prefix and can be overwritten on
repeated benchmark runs. Initialization timing for direct stepping is shared by
that fixture's repeats; output cases initialize a fresh fixture for each repeat.

`--reference-readback` disables Phase 2 energy reduction, fused/cached probe
gathering and combined field copies in the same executable. It retains Phase 1
batching and uses the original setter-based CPU mirror. It retains the same
processing schedule and restores original uncached single-field staging when
that is the first matching coherent memory type. `--verify-energy` compares a
supported GPU estimate against the CPU calculation on synchronized fields after each
measured run, outside the timed interval. Unsupported energy capabilities fail
verification explicitly. Benchmark CSVs record energy bytes and reference mode.

Use fixed timesteps (`EndCriteria=0`) for throughput. The output cases also enable
`--exact-endcriteria` to exercise a deterministic energy-check schedule. Hardware
load, clocks, compiler, driver and output filesystem all affect these numbers.
Run performance measurements separately from other GPU tests and compiler jobs.

`test_backend --batching-tests` runs focused CPU/GPU equivalence checks for
submission limits, zero calls, tails, continuation, delays, field edits, profiling
toggles, pending resets and failure propagation. The full native and Python GPU
suites additionally cover supported physics, geometries, probes and dump timing.
`test_backend --readback-tests` exercises energy precision/overflow/decay,
SSE fallback, cached probes, re-registration, field edits, staging fallback and
exact/steady-state stopping. It also forces FP32 on FP64-capable hardware.

`--coalesce-barriers` enables the duplicate-barrier candidate for Phase 6
comparisons. `--reference-barriers` restores the default, which retains every
runtime barrier. Both retain the probe-history synchronization fix.
`--reference-dispersive-pre` records separate Lorentz/conducting-sheet pre passes
instead of the default combined dispatch. The combined dispatch uses the existing shader
over disjoint state records and falls back to separate passes when the device's
workgroup limit requires it. Apply passes retain their order and barriers.
CSVs report both switches, barriers, coalesced barriers, dispatch categories and
sampled barrier timestamps. See
[Recorded validation and measurements](#recorded-validation-and-measurements)
for results.

`test_backend --synchronization-tests` compares CPU/reference/coalesced/fused fields
with intersecting Mur/PML/local boundaries, multiple TFSF faces, overlapping
electric/magnetic dispersive poles, RLC/excitation priority, probe history and
nested multigrid. CPU tolerance is 1e-4 absolute and 0.1% of peak; all GPU
paths must agree within 1e-7 of peak. It also forces combined-dispatch capacity
fallback. The full native suite includes these cases.

An installed Khronos layer can validate the focused suite without changing the
solver's Vulkan 1.2 requirements:

```text
cmake -DBACKEND_EXECUTABLE=/absolute/path/to/test_backend -P cmake/tests/check_vulkan_synchronization.cmake
```

Set `VK_LAYER_PATH` if the loader cannot locate the layer. Configure with
`OPENEMS_VULKAN_SYNC_TESTS=ON` to add this opt-in CTest. The wrapper verifies
layer loading and fails on validation messages as well as numerical failures;
a missing layer fails explicitly. Standard full synchronization validation is
enabled. `SHADER_HEURISTIC=1` enables supplemental shader-access analysis;
use `TEST_ARGUMENT=--synchronization-shader-tests` for boundaries, Debye, RLC,
probe history and multigrid, or `--batching-tests` for a shorter check. The shared
Lorentz shader produces known false positives under that heuristic because
it cannot distinguish runtime modes or disjoint state ranges (see
[Synchronization dependencies](#synchronization-dependencies)).
Host accesses and individual shader invocations still need manual/numerical
verification. Performance runs must disable validation.

`test_backend --coefficient-tests` checks exact reconstruction (including signed
zeros and NaN payloads), bounded construction, dense fallback, CPU/dense/palette
field agreement, reset, anisotropic lossy nonuniform meshes, PEC/PMC, PML, Debye
and cylindrical multigrid. Benchmark CSV coefficient columns describe the root
level; initialization diagnostics include every level. The earlier standalone
Phase 4 validation note is absent from this checkout; rerun the relevant commands
and retain results before making current numerical or performance claims.

## Synchronization dependencies

The Phase 6 changes retain extension priorities, ordered apply passes and
multigrid transfers. The two recording optimizations have narrower invariants:

- Duplicate-barrier coalescing removes only identical compute-to-compute memory
  barriers with shader-write sources and shader-read/write destinations. Only
  bindings, push constants and profiling timestamps may intervene. Dispatch,
  buffer-copy and command-buffer begin invalidate the coalescing state. Barriers
  involving transfer or host access are retained.
- The combined Lorentz/conducting-sheet pre dispatch reads fields and updates
  one independent state record per invocation. Different poles may read the same
  field position without sharing state. Apply mode writes those positions, so
  apply passes keep their original order and barriers. If the combined dispatch
  exceeds the device's workgroup limit, the original pre passes are used.

| Producer and consumer | Required dependency |
|---|---|
| Voltage/current updates, Mur, PML, RLC, Debye, TFSF and local boundary corrections | Compute shader-write to shader-read/write barriers preserve field and extension-state dependencies in priority order. Overlapping face/pole apply passes remain ordered. |
| Child/root multigrid updates and interpolation/projection | Compute barriers publish each phase and transfer before the next level or timestep consumes it. |
| Field updates to probe gather | Shader/transfer writes are made visible to the gather. The gather's trailing barrier both publishes results to the host and orders its field reads before subsequent compute writes; the latter is an execution dependency. |
| Initial FD sum fill and completed fields to FD accumulation | Transfer/compute writes become visible to accumulation. Its trailing compute barrier publishes sums and orders field reads before the next timestep overwrites them. Separate chunks and dumps write disjoint sums. |
| Device sums/fields to final readback | Compute/transfer writes become visible to transfer reads; transfer writes become visible to host reads, followed by a fence wait before copying mapped results. |
| CPU phase factors to FD dispatch | Host-coherent phase memory is written after the previous fence wait and before submission. A batch never overwrites phase slots still in use. |

`--synchronization-tests` exercises intersecting boundaries, overlapping
electric/magnetic poles, excitation/RLC priority, probe history and nested
multigrid against CPU and separate-pass references. It also forces dispatch-limit
fallback. `--field-accumulation-tests` validates FD mapping, chunking and lifecycle
independently. Standard synchronization validation checks recorded dependencies;
the optional shader heuristic cannot distinguish the shared dispersive shader's
pre/apply branches or disjoint state records. Numerical checks and the access
invariants above remain necessary for those cases and for host memory access.

## Recorded validation and measurements

Local measurements recorded on 2026-10-05 used Windows, a MinGW Release build,
an RTX 3070 with 8 GiB VRAM and NVIDIA driver 591.74. They are historical results,
not measurements of the later registration/test fixes. Profiling and validation
were disabled for throughput comparisons; instrumented runs checked counters
separately. The focused native FD tests passed in both MinGW and MSVC builds.
Recorded Python/Octave runs covered FD file formats, interpolation and NF2FF.
Those binding results require rebuilding the matching CSXCAD and openEMS modules.

Two five-repeat dispersive comparisons used batch size 32 and 2,048 timesteps.
The ranges below are the two rounds' median total stepping times:

| Case | Separate pre passes, ms | Combined pre dispatch, ms |
|---|---:|---:|
| Lorentz, 25 cubed | 54.772-56.318 | 52.805-52.860 |
| Lorentz, 49 cubed | 165.498-170.069 | 163.459-166.086 |
| Conducting sheet, 25 cubed | 48.306-49.934 | 48.305-55.710 |

Separate 256-timestep profile runs reduced total dispatches from 2,304 to 1,792
for the Lorentz fixtures and from 1,792 to 1,536 for the sheet, retaining 2,056
and 1,800 barriers respectively. The sheet timing varied between rounds. Duplicate-barrier
coalescing also gave mixed throughput results, including slower Lorentz/RLC runs;
it remains disabled by default. Dispatch reduction alone is not a speed claim.

The FD sweep compared 22 CPU/GPU output pairs across volumes, planes and lines,
one to five frequencies, and Nyquist/every-step sampling. All complex fields and
meshes passed; the largest peak-relative complex error was 8.78e-8 against a
0.001 tolerance. A separate three-repeat confirmation of a 65-cubed volume,
five frequencies, 257 timesteps, every-step sampling and batch size 64 gave
median solver-output times of 8.881 s for CPU accumulation and 0.546 s for GPU
accumulation. Those times include finalization and file writing but exclude setup.
Sparse outputs can be slower on the GPU, so accumulation remains opt-in.

The 217-cubed five-frequency E/H case needed 2,452,395,120 sum bytes and 20 sum
chunks plus the shared 16 MiB readback buffer. Every-step solver-output medians
were 99.900 s CPU and 30.778 s GPU for 65 timesteps; HDF5 output was a substantial
part of GPU runtime. These results do not establish performance on smaller-VRAM
devices, including the unmeasured RTX 2060 workload.

The experimental GLSL phase generator reached relative errors of 0.0084-0.0175
at timestep offset 100,000 and about 2 at offset 100,000,000. The runtime therefore
retains the CPU `std::exp` phase factors. Reproduce accuracy/lifecycle checks with
`--field-accumulation-tests`, dependencies with `--synchronization-tests` and the
validation wrapper above, and timings with the explicit benchmark cases and
reference switches in this document. Run benchmark modes separately from other
GPU tests and compare their output files before interpreting performance.

## References

- [Khronos timestamp queries](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html)
- [Timestamp command semantics](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html)
- [Query-result availability](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html)

Retained historical measurements and their limits are in
[Recorded validation and measurements](#recorded-validation-and-measurements).
Earlier standalone Phase 1/2 validation notes are absent from this checkout;
the test commands above support reproduction but are not fresh execution results.
