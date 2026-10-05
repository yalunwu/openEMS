# Vulkan optimization implementation plan

Status: Phases 0, 1 and 2 implemented on Windows / RTX 3070, 2026-10-03/04
(commits `a0e67e0` and `33807c7`).
Phase 4 remains explicitly opt-in with dense storage as the default; further
tuning and automatic selection are deferred. See `VULKAN_PHASE04_VALIDATION.md`.
Phases 3 and 5 are cancelled. Phase 6a and the first measured Phase 6b delivery
are implemented locally on 2026-10-05: the dependency audit, probe-history
synchronization fix, profiling and independent dispersive pre-pass fusion.
Duplicate-barrier coalescing remains opt-in because throughput results were
inconsistent. See `VULKAN_PHASE06_AUDIT.md` and `VULKAN_PHASE06_VALIDATION.md`.
Further Phase 6b candidates and Phases 7-10 remain proposed. The
2026-10-05 review keeps field accumulation (7) first for the reported workload,
including NF2FF surface validation formerly listed as 10b. GPU energy (8a)
remains planned; steady-state (8b), probe spectra (9) and mode matching (10a)
require measured comparisons before choosing where their arithmetic runs.
See `VULKAN_PERFORMANCE.md` for usage,
`VULKAN_PHASE01_VALIDATION.md` and `VULKAN_PHASE02_VALIDATION.md` for results.

## Objective and scope

Improve Vulkan simulation throughput by reducing submission and readback costs,
then optimize shader execution and memory traffic where measurements justify it.
Deliver each phase as a separately reviewable change with numerical validation
and before/after measurements. Minimize end-to-end runtime and unnecessary
transfers; moving arithmetic onto the GPU is not itself an acceptance criterion.

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

## Current baseline, including probe-history follow-ups

The section above describes the code before this plan. Remaining phases build on
the following state:

- `IterateTS()` records up to the batch limit (default 32, range 1-64) of
  complete hierarchy steps per command buffer, reusing one command buffer and
  fence. Commands are re-recorded for every submission because excitation, TFSF
  and Mur activation embed the recorded timestep.
- When probes are registered, the final submission of an interval gathers them
  after hierarchy projection; cached results are invalidated by stepping, edits,
  re-registration and reset.
- Ordinary V/I/E/H probes now retain intermediate results in a bounded history
  buffer and replay CPU processing at the original sample times. Full-field
  consumers and energy/stopping checks still bound submissions.
- Steady-state detector probes are already registered for GPU gathering.
  `RunFDTD()` uses `BeginProbeHistory(horizon)` to record multiple GPU steps,
  bounded by detector decision times, then replays samples through the CPU
  detector. Its `step = 1` replay loop does not imply one GPU submission per
  timestep. Histories and period comparisons still incur CPU work to measure.
- Energy-only checks use a GPU reduction for the basic Cartesian engine only.
  Cylindrical (SSE-based operator), SSE and multigrid energy keep the CPU path
  and a full-field download. Field dumps still transfer the whole domain.
- Frequency-domain E/H dumps (`DumpType` 10/11) already write field results in
  `ProcessFieldsFD::PostProcess()` at the end. During stepping, `RunFDTD()`
  synchronizes the full grid and `ProcessFieldsFD::Process()` accumulates the
  Fourier sums on the CPU. Small dump regions still trigger full-grid readbacks.
- Core update shaders, workgroup shapes and dense coefficient buffers are
  unchanged.

Measured headroom on the RTX 3070 (`VULKAN_PHASE01_VALIDATION.md`):

- The core updates move at least 60 bytes per node per half-step: 24 of
  coefficients, 12 field read, 12 field write and 12 of neighbor field. The
  129^3 case (256 steps, 164.6 ms) therefore sustains about 400 GB/s, roughly
  89% of the 448 GB/s peak. Large grids are bandwidth bound with near-ideal
  neighbor reuse; coefficients are 40% of the minimum traffic.
- Small grids are overhead bound: 25^3 runs at about 13 us per step at batch 32.
- The five-level multigrid case spends about 12 ms recording commands against
  about 38 ms of GPU batch time.
- The Khronos validation layer is not installed on the measurement host, so
  synchronization validation has not yet been run.

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
Preserve steady-state sample times; later probe-history support separates
one-step CPU replay from multi-step GPU submissions. Internal submission chunks
alone do not improve cancellation latency; any later change to
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

Implemented scope: the reduction covers the basic Cartesian engine. SSE energy
accumulates padded z lanes in FP32 and cylindrical/multigrid energy uses
different extents or projected root fields, so those configurations keep the
CPU estimate and full-field download. Phase 8a now explicitly targets these
remaining Vulkan configurations; Phase 2's implemented scope is unchanged.

## Phase 3: tune workgroup shapes (cancelled)

Cancelled to avoid device-specific tuning and additional shader variants without
evidence of a portable benefit. Retain the existing workgroup shapes and device
limit checks. This work is outside the remaining implementation roadmap.

## Phase 4: reduce coefficient traffic (implemented, opt-in)

Whole-node and per-component counts on smoothed tutorial meshes favored the
whole-node layout. The exact palette, shader variant, per-level dense fallback,
benchmark selection and regression tests are implemented. Keep dense storage as
the default and palette selection explicitly opt-in. Further palette tuning,
automatic selection and promotion to a default are outside the active roadmap.
Reconsider them only in a separate review. Results and limitations are in
`VULKAN_PHASE04_VALIDATION.md`.

The implemented representation stores exact float-bit coefficient tuples and a
compact index per node, including geometry, direction and boundary corrections.
It does not quantize coefficients or fields. Its tradeoff is reduced coefficient
storage/traffic against indexed lookup, cache behavior and extra initialization.

Recorded RTX 3070 results show a repeatable 1.43x large-grid stepping gain, but
probe and steady-state cases have worse palette medians with wide, overlapping
ranges. Those cases establish no benefit. Other GPU vendors and forced allocation
exhaustion remain untested. Keep the existing capability available without
depending on it for the output/convergence work in Phases 7-10.

Use dense storage for the primary before/after benchmarks of subsequent phases.
Keep palette compatibility and regression coverage, and label any separately
measured palette results so coefficient-mode changes cannot obscure the gain.

## Phase 5: reduce stencil loads (cancelled)

Cancelled because shared-memory tiling and subgroup variants introduce tile-size,
barrier and resource tradeoffs whose performance depends on the GPU. Retain the
direct-load shaders. This work is outside the remaining implementation roadmap.

## Phase 6a: audit extension and synchronization dependencies

Implemented locally, 2026-10-05. The separate dependency map is in
`VULKAN_PHASE06_AUDIT.md`; baseline findings, numerical/validation coverage and
dispatch/barrier measurements are in `VULKAN_PHASE06_VALIDATION.md`.

Build a dependency map for each supported extension phase and multigrid transfer,
recording buffer reads/writes, overlapping points, face ordering, extension
priorities and shared state. Document the execution and memory dependencies each
barrier satisfies, including dependencies across submissions and host readbacks.
Measure dispatch and barrier costs using Phase 0 instrumentation.

Validate the existing path with synchronization validation and numerical tests.
Cover intersecting Mur/PML/absorbing boundaries, multiple TFSF faces, overlapping
dispersive passes, lumped elements, excitation priority and nested multigrid
transfers. Record any validation findings and untested configurations.

Completion: a reviewed dependency map, baseline measurements and a short list of
individually justified optimization candidates. Deliver the audit separately
before changing execution in Phase 6b.

## Phase 6b: optimize measured extension and synchronization costs

First delivery implemented locally, 2026-10-05. Independent Lorentz/conducting-
sheet pre passes share one dispatch, retaining ordered apply passes and every
required dependency. Device-limit fallback and the original recording path are
tested. Duplicate-barrier coalescing is available for benchmark comparisons but
disabled by default. Command-buffer reuse and hierarchy projection fusion remain
separate future candidates; see the Phase 6 audit and measurement record.

Use the Phase 6a audit and profiles to select one candidate at a time. State its
expected benefit and required dependencies before implementing it. Prioritize
removing unnecessary work while preserving the extension architecture.

Combine independent work or fuse selected corrections with compatible kernels
where ordering permits. Remove redundant barriers only after proving the data
dependencies remain satisfied. Global memory barriers are not automatically
slower than buffer barriers; compare valid alternatives on the actual workloads.
Keep shader combinations limited and retain a reference path for comparisons.
Do not require synchronization2 or a newer Vulkan version for the base path.

Validation: require synchronization validation and numerical equivalence on the
Phase 6a cases before accepting each execution change. Preserve output timestamps,
stopping schedules and existing numerical tolerances. Measure performance with
validation and profiling disabled, including representative extension and
multigrid workloads.

Completion: separately reviewable changes with demonstrated performance benefits
and documented dependency proofs. Retain the reference path where an optimization
does not meet the performance acceptance criteria.

## Phase 7: accumulate frequency-domain E/H fields on the GPU (proposed)

Keep running complex Fourier sums on the GPU at the existing sample times, then
download the completed results for the existing CPU file writers. No timestep
history is required. Sampling only the last timestep cannot recover these
frequency-domain results.

### Scope and integration

Deliver native Yee sampling (`DumpMode 0`) first, followed by node (`1`) and
cell (`2`) interpolation as separately validated increments. Precompute mesh
indices/weights where useful; retain CPU handling for modes not yet supported.
Planar frequency-domain NF2FF recording uses this same E/H dump machinery and
belongs in this phase's integration tests, rather than a second implementation.

- Start with `DumpType` 10/11 on the basic Cartesian Vulkan engine. Register
  enabled dump regions, frequencies, sample schedules and interpolation data
  after processing initialization has resolved their output meshes. Add an
  opt-in runtime mode through the command line, Python and MATLAB/Octave, with
  an explicit CPU reference mode and per-dump selection/fallback diagnostics.
  Keep existing XML and file formats; this execution choice needs no new
  CSXCAD property.
- Gather and convert the requested output points on-device using the active
  E/H interface semantics. Preserve native/node/cell interpolation, nonuniform
  mesh scaling, component ordering, boundary behavior, spatial subsampling and
  `OptResolution`. A mapping that is not validated must use the CPU path.
- Match `ProcessFieldsFD::Process()` exactly in sample eligibility, start/stop
  windows, FD oversampling and E/H staggered time. Preserve its negative Fourier
  phase and `2 * dt * FD_interval` weighting. Accumulate complex FP32 results
  initially, matching the current storage; validate phase-factor generation,
  rounding and long-run cancellation rather than assuming bitwise equivalence.
- Record accumulation dispatches at each eligible timestep within a batch,
  after all required field updates. Preserve sample times across batch tails
  and probe-history replay, without duplicate CPU accumulation. Supply phase
  factors once per frequency/sample and reuse them across output points.
  Compare batched CPU preparation/upload with GPU generation; account for
  trigonometric cost and long-run phase accuracy. Avoid a host wait per sample.
  Audit producer/consumer barriers and validate synchronization independently
  of Phase 6b.
- Classify supported GPU FD consumers separately in `RunFDTD()` and
  `ProcessingArray::GetNextFullFieldInterval()`: their samples must no longer
  force full-field downloads or end a batch. Retain CPU boundaries for other
  consumers, probes and stopping checks as required. Mixed GPU/CPU dumps must
  still sample the same fields at the same timesteps.
- At normal completion, early convergence or graceful interruption, finish
  pending accumulation and populate the existing `m_FD_Fields` before
  `DumpFDData()`. Preserve HDF5/legacy HDF5 and VTK output metadata and values.
  Reset and repeated runs must clear sums and counters after pending GPU work
  completes. GPU failures must propagate rather than write incomplete results
  as successful output.

### Memory and fallback

Budget complex FP32 sums at 24 bytes per output point per frequency per E or H
dump. Both fields over 10 million output points need approximately 480 MB
(458 MiB) per frequency, or 2.4 GB for five frequencies. These are additional
allocations beyond fields, coefficients, extensions, mapping and staging; output
point counts can differ from solver cell counts. Sum the costs across all dump
boxes and allocate only their requested regions, not the whole grid by default.

Check size arithmetic, device buffer/allocation limits and available memory
before enabling a dump. Partition buffers where required by Vulkan limits and
bound final readback staging. Report estimated and allocated bytes. Choose CPU
accumulation per dump before the first sample if resources or mappings are
unsupported; the FDTD solver and eligible dumps remain GPU accelerated. For
small unsupported regions, evaluate compact gathers plus CPU accumulation as an
alternative to full-grid copies. Preserve a working full-field reference path;
never discard an accumulated prefix or silently reduce frequencies or precision.
Mid-run GPU errors remain errors, not a restart of accumulation on the CPU.

Buffer partitioning satisfies allocation limits but does not reduce the total
sum storage. Every requested point/frequency needs contributions throughout the
run. Frequency chunking or spatial tiling reduces residency only with an explicit
spill/restore, retained-sample or rerun strategy; measure its memory and transfer
costs before adding it. Do not treat chunking as a free out-of-memory remedy.

Report actual FD sample intervals and an accumulator traffic estimate: for B
bytes of sums sampled every I timesteps, read/write traffic is about 2B/I bytes
per timestep, plus field/mapping reads. The five-frequency example adds about
4.8 GB per sample, not necessarily per timestep. Diagnose expensive full-volume
sampling without changing requested temporal or spatial resolution automatically.

Initially retain CPU handling for cylindrical/multigrid dumps, other FD types
including SAR, time-domain output and unvalidated interpolation modes. Small
probe/energy transfers, unsupported energy full-field reads and the existing
final CPU field synchronization can remain. The target is to eliminate
recurring full-grid transfers caused by supported FD dumps, not all CPU work.
Phase 10 considers mode matching and extends accumulation to SAR. Phase 8
addresses convergence; Phase 9 compares ordinary probe integration and spectra
on GPU versus compact readback and CPU processing.

### Validation and performance acceptance

Compare the new and CPU FD paths using identical Vulkan timesteps and fields,
then compare supported solver cases against CPU-engine references. Cover E and
H separately/together; one/multiple frequencies and boxes; full volumes, planes
and lines; all supported interpolation modes; nonuniform/lossy meshes and
material boundaries; sampling windows, oversampling, disabled dumps and batch
sizes 1/32/64 with tails. Include mixed TD/FD output, ordinary probes, early
convergence, graceful interruption, reset/reuse and forced allocation fallback.

Include frequency-domain NF2FF surfaces with enabled/disabled faces, supported
interpolation, orientation and symmetry/mirror handling. Compare surface spectra,
file metadata and derived far-field patterns, polarization and radiated power.
Keep time-domain NF2FF recording unchanged; it does not become final-only FD
recording automatically. The angular far-field transform remains separate CPU
post-processing, with any later GPU port measured independently.

Check complex fields and output metadata. Use existing field tolerances and
explicit absolute tolerances near zeros; phase comparisons need an amplitude
floor. Long decaying signals and
cancellation-heavy cases must pass without loosening tolerances to conceal
accumulation error. Preserve sample counts and deterministic stopping decisions.

Extend profiling with FD sample/dispatch counts, accumulator/mapping bytes,
GPU accumulation time and final FD download bytes, separate from full-field
downloads. Benchmark CPU versus GPU FD accumulation with matching outputs and
fixed timesteps; a no-dump run is a diagnostic upper bound, not the reference.
Include the 10M-cell workload on the RTX 2060 when available and the existing
RTX 3070 matrix, sweeping regions, sample rates and frequency counts. Record
actual VRAM capacity/usage, total runtime, MC/s and final transfer/file costs;
measure repeated runs with profiling disabled. GPU accumulator traffic can
become bandwidth-bound as the frequency count grows, so measure the crossover.

Completion: supported FD samples cause no recurring full-field readback;
completed sums are downloaded at finalization, existing output comparisons pass,
and measured end-to-end gains exceed run-to-run spread. Keep the path opt-in
until memory and performance selection criteria are supported by evidence.

## Phase 8: convergence checks (proposed)

Keep GPU energy reduction as the target for eliminating full-field energy
readbacks. Decide how to handle the remaining steady-state histories/comparisons
from measurements against existing batched replay. Deliver these separately,
retain CPU references, and preserve the stopping metric and decision boundaries.

### 8a. Cylindrical and multigrid energy

Extend Phase 2's energy capability to the actual interfaces used by cylindrical
and multigrid models. Audit the active `CalcFastEnergy()` path, including SSE
lane order, padded z values, grid extents, FP32 accumulation and EPS0/MUE0
weighting. Reproduce the existing projected-root metric for multigrid; summing
every hierarchy level would double-count overlapping regions and change the
criterion. Keep projection and reduction on-device and download only compact
partial sums or the completed estimate. Do not add a new hierarchy masking or
physical-energy definition to the existing convergence criterion.

Measure download, CPU mirroring and energy arithmetic under default wall-clock
checks, exact-endcriteria and steady-state period checks. Four-second polling
alone establishes no overhead percentage; record actual check costs and their
share of total runtime. GPU energy remains in scope even without exact-endcriteria.

Validate against the current CPU estimate on identical field snapshots, including
non-multiple-of-four z sizes, closed/open angular domains, one/two/five nested
levels, lossy decay, zero/subnormal/non-finite fields and near-threshold cases.
Reduction order can affect stopping: require documented error bounds and the
same deterministic stopping decisions, with a reference fallback for ambiguous
cases. Profile any fallbacks rather than counting those runs as GPU-only checks.

Completion: validated cylindrical/multigrid energy checks no longer routinely
download full fields. Preserve basic Cartesian behavior and benchmark check
costs separately from stepping, with and without frequent exact-endcriteria.

### 8b. Measure remaining steady-state overhead, then select execution

Use the current GPU probe-history plus CPU replay path as the baseline. Measure
submission counts, probe transfers, per-sample replay, period comparisons and
energy separately across probe counts and period lengths. Do not assume a fixed
1-4 probes or infer lost GPU batching from the one-step CPU loop.

First evaluate processing compact history batches on the CPU with less replay
overhead. Prototype GPU histories/reductions only if the remaining costs justify
them. Compare both against the existing path with identical sample/decision
times; retaining CPU comparisons is an acceptable outcome when faster or when
GPU gains do not exceed measurement spread.

For either design, preserve `Engine_Ext_SteadyState::ApplyVoltages()` sample
phase and timestep mapping, two-period warm-up, signal-power eligibility,
squared-difference ratios, energy comparison and maximum/clamping rules. Reuse
the validated energy path, including Phase 8a where needed. End batches at the
original decision boundaries and check convergence before advancing further;
do not overshoot a converged period. Preserve progress and graceful interruption,
and budget two periods of detector history with checked reset/reuse behavior.

Validate period lengths both smaller and larger than a batch, warm-up, absent
or weak signals, multiple probe powers, long runs, exact stop timesteps,
cylindrical/multigrid projection and mixed output consumers. Completion: a
documented choice backed by equivalent stopping decisions and end-to-end
measurements. Accept execution changes only with a demonstrated benefit.

## Phase 9: compare GPU and CPU probe processing (proposed)

Benchmark existing batched probe gathering and CPU integration/spectra first.
Compare improved compact batch readback plus CPU processing with on-device
integration and accumulation for few/many probes, short/long integrals and
frequency counts. Keep GPU spectra conditional on an end-to-end benefit; avoid
mandatory offload of tiny workloads. Use the actual direct-DFT/port workflow as
the reference, not an assumed FFT timing or an unrelated occupancy estimate.

For the GPU candidate, extend gathering with voltage line integrals, current
contour integrals, E/H point conversion and running Fourier sums for frequencies
specified before the run. Reuse Phase 7's scheduling/phase-factor infrastructure
and batch small probes together; avoid a kernel launch or host wait per probe.
Preserve integration signs, mesh/interpolation rules, weighting, E/H time bases
and independent TD/FD sample intervals. `ProcessIntegral` accumulates complex
double values, unlike FP32 field dumps: validate precision explicitly and retain
a reference path where the device cannot meet its numerical requirements. Do
not silently lower accumulation precision. Floating-point atomics are not
required by the design; use independent outputs or staged reductions where
appropriate and measure the actual device's precision/throughput tradeoff.

Support both existing time histories and an explicit final-spectra-only mode.
When histories are requested, buffer reduced samples for bounded batch readback
and preserve their files/timestamps. GPU spectra alone cannot support arbitrary
frequencies chosen later. Python/MATLAB port post-processing must either consume
compatible precomputed spectra or continue reading histories; preserve pulse
versus periodic normalization, reference-plane shifts and derived port results.
Expose any new mode consistently through the existing scripting interfaces.

Compare integrals, complex spectra, impedance and S-parameters with current
probe and port paths, including many ports, multiple frequencies, mixed sampling
and cancellation-heavy signals. Measure few-probe and many-probe cases with
histories enabled/disabled. GPU math capacity alone does not establish a speedup:
small reductions may be dominated by dispatch cost and compete with FDTD memory
traffic. Choose defaults from measured crossover and compatibility evidence.

Completion: preserve ordinary time-history workflows and document the useful
workload range of each selected path. Deliver GPU final-only spectra only if
numerical tests and profiles support them; retaining CPU spectra is valid.

## Phase 10: application-specific processing (proposed)

Keep mode matching and SAR as separate deliveries with capability checks,
memory accounting, reference paths and numerical/performance acceptance. Reuse
Phases 7/9 where applicable, without making a GPU probe-spectrum implementation
a prerequisite for compact mode readback. Select order by actual model usage.
The former Phase 10b is folded into Phase 7's NF2FF integration and validation.

- **10a. Waveguide mode matching:** first gather only the required surface
  values/interpolation footprint into bounded history buffers, and use CPU
  overlaps, purity and spectra. Compare against GPU weighted-overlap and power
  reductions with compact result readback or GPU spectra where justified.
  Mode weights are already prepared and normalized on the CPU at initialization;
  a GPU candidate uploads them once and needs no GLSL parser or mode-file reader.
  Measure actual surface/component counts and transfer sizes. Preserve template
  normalization, surface area weights, interpolation and both instantaneous
  mode-purity and voltage/current results. The nonlinear purity history cannot
  be reconstructed from final field phasors alone. Validate analytic and
  file-defined modes, origins, port normalization and resulting S-parameters.
  Keep batches bounded by actual CPU consumers and stopping checks; avoid
  replacing full-grid copies with a new synchronous transfer per sample where
  compact history replay suffices. Completion: no whole-domain readbacks for
  supported mode probes, with execution chosen by measured end-to-end benefit.
- **10c. SAR frequency accumulation:** accumulate the E and optional J spectra
  consumed by `ProcessFieldsSAR`, preserving cell interpolation, conductivity
  handling, material assumptions and normalization. Validate raw spectra and
  resulting local/1g/10g SAR against existing tests. Keep mass averaging as
  separate post-processing initially; quantify any later GPU port independently.

## Later experiments

Temporal blocking across multiple timesteps requires halo expansion, global
dependencies and extension scheduling. Prototype it separately after Phases 4 and 6;
a simple voltage/current fusion cannot synchronize independent workgroups.

Pipeline/shader caches and batched initialization uploads address setup time.
Prioritize them only if setup dominates real workloads. Keep device/driver cache
compatibility checks and a safe cache-miss path.

## Release and acceptance

Phases 0-2 and the opt-in Phase 4 path are implemented; Phases 3 and 5 are
cancelled. Remaining work is:

1. Phase 7: profile the E/H FD workload and implement opt-in GPU accumulation,
   native then interpolated, with per-dump memory budgeting and CPU fallback.
   Include NF2FF surface/far-field validation. This remains first for the
   reported output-heavy large-grid workload.
2. Phase 8a: retain GPU cylindrical/multigrid energy as planned work; measure
   default, exact-endcriteria and steady-state check costs and preserve the
   existing projected-root metric and stopping decisions.
3. Phase 8b: measure the existing batched steady-state path and compare reduced
   CPU replay overhead with GPU history/comparison only where justified.
4. Phase 9 and Phase 10a: compare compact batch readback/CPU processing against
   GPU integration and spectra. Prioritize by measured costs; mode gathering
   need not wait for GPU probe spectra. Keep time histories and precision.
5. Phase 10c: extend validated field accumulation to SAR as a separate delivery.
6. Phase 6a audit and first Phase 6b delivery completed locally with SDK
   synchronization validation; independent dispersive pre passes are combined.
7. Further Phase 6b candidate: remove per-submission command recording by reusing
   pre-recorded batch command buffers, reading the timestep from device memory
   instead of recording it. Prototype two command buffers with independent
   fences only if reuse is not feasible.

Phase 7 leads for the current workload. Phase 8a and the profiling/comparisons
in 8b/9/10a can proceed independently; any steady-state implementation needs the
appropriate validated energy path. SAR follows validated field accumulation.
Phases 7-10 do not depend on Phase 6b, but each execution change requires its own
synchronization audit and validation. Phase 6a/6b remain separate work selected
by profiles, especially for dispatch, barrier and recording costs in small or
multigrid models. Phase 4 stays opt-in and is not a prerequisite for this work;
its automatic selection remains deferred.
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
