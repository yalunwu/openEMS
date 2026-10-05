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
result is accepted directly only when a complete host mirror confirms that all
fields are zero. SSE energy uses different FP32 lane accumulation and extents, so
SSE, cylindrical and multigrid cases retain that CPU calculation. This estimates
the existing stopping metric, not physical energy.

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
submissions, dispatches, field upload/download bytes, probe and energy result
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
GPU timing instrumentation itself can alter scheduling; use profiling-disabled
runs for speed comparisons. Unsupported timestamps retain CPU diagnostics.

The backend checks the compute queue's timestamp support, scales ticks with the
device period, masks timestamp wraparound, and collects available results after
existing fence waits. It adds no query-result waits. `Synchronize()` waits for
stepping without downloading fields; `ClearProfile()` synchronizes before
resetting statistics. `Reset()` waits before destroying pending GPU resources.
GPU execution/synchronization errors propagate to the command-line solver or
Python caller instead of continuing with stale fields.

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
`cylinder`, `multigrid-1`, `multigrid-2`, `multigrid-5`, `probes`, `dumps`,
`steady-state`, `energy-large` (171 cubed, 5,000,211 stored nodes).
The first eleven time stepping directly without field downloads;
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

`test_backend --coefficient-tests` checks exact reconstruction (including signed
zeros and NaN payloads), bounded construction, dense fallback, CPU/dense/palette
field agreement, reset, anisotropic lossy nonuniform meshes, PEC/PMC, PML, Debye
and cylindrical multigrid. Benchmark CSV coefficient columns describe the root
level; initialization diagnostics include every level. Phase 4 measurements are
recorded in `VULKAN_PHASE04_VALIDATION.md`.

## References

- [Khronos timestamp queries](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html)
- [Timestamp command semantics](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html)
- [Query-result availability](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html)

Measured results and configuration coverage are recorded in
`VULKAN_PHASE01_VALIDATION.md` and `VULKAN_PHASE02_VALIDATION.md`.
