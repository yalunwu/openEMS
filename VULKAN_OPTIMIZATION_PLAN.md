# Vulkan optimization implementation plan

Status: Phases 0 and 1 implemented and validated on Windows / RTX 3070,
2026-10-03. Phases 2 through 6 remain proposed. See
`VULKAN_PERFORMANCE.md` for usage and `VULKAN_PHASE01_VALIDATION.md` for results.

## Objective and scope

Improve Vulkan simulation throughput by reducing submission and readback costs,
then optimize shader execution and memory traffic where measurements justify it.
Deliver each phase as a separately reviewable change with numerical validation
and before/after measurements.

Target the existing C++11 / Vulkan 1.2 backend and current dependencies. Keep
existing XML models, Python and MATLAB engine selection, physics, output timing,
and CPU fallback working. Use the RTX 3070 as the initial measurement target;
cross-vendor validation is required before claiming portable performance gains.

## Implementation before Phase 0

- `FDTD/vulkan/engine_vulkan.cpp::IterateTS()` records and submits one command
  buffer per timestep and waits before reusing it. Fields remain device-local.
- `shaders_glsl.h` uses fixed 32 x 4 x 2 groups for the core updates, with the
  contiguous physical z coordinate mapped to invocation x. It already uses
  separate component arrays and computes all three components per invocation.
- `SyncProbesToHost()` submits a separate gather and waits. Full-field reads
  download voltage and current separately and populate CPU field mirrors.
- `openems.cpp::RunFDTD()` schedules processing boundaries and energy checks.
  Steady-state detection forces one timestep per call and consumes CPU probes.
- Four dense coefficient buffers cost 48 bytes per stored grid node; voltage
  and current buffers add 24 bytes, excluding staging and extension state.
- Multigrid shares a device and pipelines, but extensions and interface transfers
  add dispatches and compute memory barriers at every level.

The timings in `VULKAN_MULTIGRID_VALIDATION.md` are historical baselines, not
predictions. Collect fresh measurements from the actual DLL under test.

## Phase 0: measurement and reference results

Implement opt-in profiling and expand the native benchmark utility before
changing execution. Instrument CPU recording, submission, fence waiting,
readback and mirror updates; count submissions, dispatches and transferred bytes.
Use GPU timestamp queries for batches and sampled update/extension phases.
Check timestamp support, period and valid bits, handle wraparound, and collect
available results without adding waits to normal execution. Profiling stays off
by default; compare instrumented and uninstrumented runs to quantify overhead.

Benchmark fixed timestep counts after warm-up, with at least five timed runs and
reported median and spread. Separate initialization, stepping, output and final
synchronization. Every stepping measurement must wait for its final GPU work.
Record revision, compiler, build type, loaded DLL, GPU, driver, grid shape,
physical cells, active hierarchy cells, extensions and processing intervals.

Include small and large Cartesian domains, asymmetric and thin grids, lossy and
nonuniform meshes, PML, dispersive materials, cylindrical grids, and one/two/five
multigrid levels. Exercise no-output, probes, field dumps and steady-state runs.
Use fixed timesteps for throughput and deterministic energy schedules for
end-criteria comparisons. Confirm that Vulkan execution did not fall back.

Completion: reproducible performance records and numerical reference results;
profiling-disabled behavior passes the existing backend and Python GPU suites.

## Phase 1: batch timesteps into fewer submissions

Refactor phase-recording helpers to take an explicit timestep. Excitation,
TFSF and Mur start conditions must use the recorded timestep rather than a host
counter advanced during command construction.

Record several complete voltage/current hierarchy steps in one command buffer.
Start with a bounded batch size of 32 and benchmark 1, 8, 16, 32 and 64. Retain
all required inter-phase and inter-timestep GPU dependencies. Apply initialization
visibility barriers appropriately; preserve projection at the end of the caller's
`IterateTS()` interval, including a partial final submission.

Initially reuse one command buffer and fence per submission. Update published
hierarchy counters only after successful submission and track pending work for
subsequent host synchronization. Handle zero-step calls, overflow, recording and
submission failures, field edits, reset, and resource destruction. Make
`RunFDTD()` honor backend iteration/synchronization failures before processing
data or reporting success.

Do not advance past the interval selected by processing or stopping checks.
Keep steady-state calls at one step until its sampling is redesigned. Internal
submission chunks alone do not improve cancellation latency; any later change to
caller intervals must preserve processing and multigrid projection semantics.

Validation: compare batch sizes and continuation across calls, including 0/1
steps, non-multiple tails, delayed sources, Mur activation, explicit field edits,
PML, multi-pole Debye and all existing multigrid cases. Check output timestamps,
reset after pending work and failure propagation.

Completion: submission count scales with submission chunks rather than steps;
numerical tolerances remain unchanged and timed workload results are recorded.
If recording/waiting remains significant, separately prototype two command
buffers with independent fences so CPU recording can overlap GPU execution.

## Phase 2: integrate probe gathering and GPU energy estimation

### 2a. Probe gathering

Append the existing gather dispatch to the final submission when registered
probes are needed, after hierarchy projection and a correct producer/consumer
dependency. Make `SyncProbesToHost()` consume that submission's result instead of
submitting another gather. Track result generation/timestep and pending work;
invalidate results on field edits, stepping, re-registration and reset. Preserve
the standalone synchronization API with an on-demand gather for uncached data.
Partial probe updates must not mark the whole CPU field mirror valid.

Keep existing probe decoding, integrals and interpolation first. The registry
already deduplicates points. Add scheduling-aware subsets only if gathering
inactive points is measurable, and preserve the fallback for processing that
requires full fields or another hierarchy level.

Validation: compare voltage/current integrals, field probes, steady-state probes,
repeat syncs, field edits, mixed sampling intervals and multigrid output. Verify
that the ordinary sample boundary no longer incurs a separate gather submission.

### 2b. GPU energy estimate

Add a backend capability for obtaining the fast energy estimate, with the CPU
path retained for unsupported cases. Implement a staged GPU reduction and read
back partial sums or a scalar instead of downloading the entire domain.

Reproduce the effective `CalcFastEnergy()` behavior: cell extents, component
selection, multiplication/accumulation precision, and EPS0/MUE0 weighting. Audit
the active interface's SSE behavior and cylindrical/multigrid projection rules.
Do not substitute a different physical energy formula. Evaluate optional FP64
support; otherwise use a validated reduction scheme with CPU accumulation of
partial sums, retaining the full-field path where accuracy is insufficient.

Separate energy-only checks from processing that actually needs fields. Route
steady-state energy requests through the backend capability while keeping its
per-step probe sampling. Retain full downloads for field dumps and other field
consumers, and batch voltage/current copies if transfer setup is significant.

Validation: compare estimates through excitation and decay, zero/non-finite
cases, lossy materials and all supported geometries. Test stopping decisions
near thresholds under `--exact-endcriteria`; measure reduction precision before
setting a documented tolerance. Preserve the same sampling phase and schedule.

Completion: supported energy-only checks transfer scalar/partial results and
produce validated stopping behavior. Full dumps remain numerically equivalent.

## Phase 3: tune workgroup shapes

Parameterize core shader group dimensions through compile-time definitions
compatible with Vulkan 1.2. Test a small candidate set such as 32 x 4 x 2,
32 x 4 x 1, 16 x 4 x 2 and 64 x 2 x 1. Update dispatch rounding and device-limit
checks together; preserve contiguous physical-z accesses.

Use benchmark results to select conservative variants by device limits and grid
shape. Initially provide overrides in the benchmark utility, avoiding startup
autotuning in production. Keep the current shape as the fallback. Evaluate
extension and multigrid transfer group sizes separately when they are significant.

Validation: dimensions around every candidate group boundary, sub-group-sized
grids, thin slabs, nonuniform shapes, all physics tests and supported device
limits. Default selection requires repeatable wins without material regressions.

## Phase 4: reduce coefficient traffic

Measure how often complete coefficient tuples repeat. Build an exact float-bit
palette and compact index buffer during initialization, with matching shader
variants. Include mesh geometry, direction and all coefficients in the tuple;
material IDs alone cannot represent these operators correctly. Compare whole-node
tuples with per-component tuples before choosing the representation.

Retain dense buffers when palette/index storage or lookup cost is unfavorable.
Report palette size, compression ratio, VRAM and throughput. Do not quantize
coefficients or field storage in this phase. Consider uniform-region constants
only after the palette measurements justify a separate specialization.

Validation: uniform/repeated regions, nonuniform spacing, anisotropy, lossy
materials, PEC/PMC, PML, dispersive and cylindrical operators; verify exact
coefficient reconstruction and unchanged field tolerances.

## Phase 5: reduce stencil loads

Prototype shared-memory tiling for one half-step on the simple Cartesian path.
Load the necessary field components and halo, synchronize the whole workgroup,
then evaluate curls from shared data. Out-of-domain lanes must participate in
barriers safely. Keep the direct-load shader available and benchmark against
hardware cache reuse, shared-memory cost and occupancy.

Evaluate subgroup exchange only as a separate feature-gated variant. Do not
assume a subgroup size or that subgroup operations cross workgroup boundaries.
Combine tiling with coefficient compression only after measuring each alone.

Validation: halo faces/edges/corners, arbitrary dimensions and boundaries, long
decay runs and equivalence with the direct-load implementation. Extend to other
physics/geometry only after the base variant provides a repeatable benefit.

## Phase 6: streamline extensions and synchronization

Build a dependency map for each supported extension phase and multigrid transfer,
including overlapping points, face ordering and shared state. Measure dispatch
and barrier costs using Phase 0 instrumentation.

Combine independent work or fuse selected corrections with compatible kernels
where ordering permits. Remove redundant barriers only after proving the data
dependencies remain satisfied. Global memory barriers are not automatically
slower than buffer barriers; compare valid alternatives on the actual workloads.
Do not require synchronization2 or a newer Vulkan version for the base path.

Validation: intersecting Mur/PML/absorbing boundaries, multiple TFSF faces,
overlapping dispersive passes, lumped elements, excitation priority and nested
multigrid transfers. Run synchronization validation when the development layer
is available, alongside numerical tests.

## Later experiments

Temporal blocking across multiple timesteps requires halo expansion, global
dependencies and extension scheduling. Prototype it separately after Phases 0-6;
a simple voltage/current fusion cannot synchronize independent workgroups.

GPU steady-state histories and period comparisons could remove the remaining
per-step host sampling constraint. Treat this as a distinct implementation with
explicit sampling-phase and stopping-equivalence tests.

Pipeline/shader caches and batched initialization uploads address setup time.
Prioritize them only if setup dominates real workloads. Keep device/driver cache
compatibility checks and a safe cache-miss path.

## Release and acceptance

Recommended first milestone: Phases 0, 1 and 2a. Next deliver GPU energy estimation
and workgroup selection; use those profiles to choose the order of Phases 4-6.
Keep each optimization independently selectable in the benchmark harness until
its default-selection criteria are established.

For each solver change, build Release, run the complete native backend suite and
CTest, and run Python GPU equivalence tests against the exact newly built DLL.
Use fixed-step and exact-endcriteria integration cases, including probes, field
dumps and derived port results. Run the relevant Octave/MATLAB cases when solver
scheduling or scripting behavior is affected. Check Vulkan-disabled compilation,
Windows MSVC and the existing MinGW configuration; test Linux and an additional
GPU vendor before making broad compatibility/performance claims.

Keep existing numerical tolerances; scheduling-only changes must preserve sample
timestamps and deterministic stopping decisions. Reduction changes need explicit
precision and near-threshold evidence. Record any untested configurations.

Performance acceptance uses warm, repeated runs with profiling disabled. As an
initial review rule, investigate repeatable regressions above 5% on the workload
matrix and retain a fallback for configurations that lose performance. A claimed
gain must exceed measurement spread. Performance timing is informational rather
than a flaky CI threshold. Report improvements by workload and separate setup,
stepping and output costs.

Update `CHANGELOG.md` for user-visible changes and the validation notes with
reproduction commands and results. Follow `AGENTS.md` / `AI_POLICY.md` for C++11,
dependencies, clean diffs and disclosure/sign-off when changes are submitted.

## Technical references

- [Khronos timestamp queries](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html): GPU timing support and deferred query collection.
- [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html): compute and host dependencies; adapt examples to the existing Vulkan 1.2 API.
- [NVIDIA Vulkan guidance](https://developer.nvidia.com/blog/vulkan-dos-donts/): submission overhead, synchronization and profiling considerations.
