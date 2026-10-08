# Vulkan optimization implementation plan

Status as of 2026-10-07:

| Phase | Delivery status |
|---|---|
| 0 | Core profiling and benchmarks implemented; timing breakdown and convergence fixtures need extension. |
| 1 | Timestep batching implemented. |
| 2 | Probe gathering/history and basic Cartesian GPU energy implemented. |
| 3 | Workgroup tuning cancelled. |
| 4 | Exact coefficient palettes implemented, opt-in; dense remains the default. |
| 5 | Stencil-load tuning cancelled. |
| 6a | Extension and synchronization audit implemented. |
| 6b | First delivery implemented; command reuse, recording overlap and projection fusion remain candidates. |
| 7 | Cartesian FD E/H accumulation implemented, opt-in, including FD NF2FF recording. |
| 7 follow-ups | Proposed; measure final-readback pipelining and host-result memory lifetime separately. |
| 8 prerequisite | Next delivery: define convergence snapshots, validate detector traces and measure check costs. |
| 8a | Proposed; select cylindrical/multigrid GPU energy by measured check cost or capability demand. |
| 8b | Proposed; measure steady-state overhead before selecting execution changes. |
| 9 | Proposed; separate compact integration, GPU spectra and final-only output deliveries. |
| 10a | Proposed; compact waveguide mode processing. |
| 10b | Folded into Phase 7's FD NF2FF integration and validation. |
| 10c | Proposed; begin with supported Cartesian cell-conductivity SAR accumulation. |

See [Vulkan performance](VULKAN_PERFORMANCE.md) for options, test commands,
[synchronization dependencies](VULKAN_PERFORMANCE.md#synchronization-dependencies)
and [recorded measurements](VULKAN_PERFORMANCE.md#recorded-validation-and-measurements).
The measurements are historical RTX 3070 results, not portable speedup claims.
"Implemented" describes code delivery, not a fresh validation of the current
revision. Track correctness results, measured benefit and hardware coverage
separately under [Release and acceptance](#release-and-acceptance). The convergence
phase audit below is unresolved; this plan does not establish CPU/Vulkan detector
equivalence. Earlier standalone Phase 1/2/4 validation notes are absent from this
checkout; use retained evidence and mark missing measurements as unverified.

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

## Current implementation

- `IterateTS()` records up to the batch limit (default 32, range 1-64) of
  complete hierarchy steps per command buffer, reusing one command buffer and
  fence. Commands are re-recorded for every submission because excitation, TFSF
  and Mur activation embed the recorded timestep.
- Direct stepping projects multigrid fields and gathers registered probes at the
  end of the requested interval. History recording does both after each step.
  Stepping, field edits, re-registration and reset invalidate cached results.
- Ordinary V/I/E/H probes and steady-state detector probes use bounded GPU
  histories and CPU replay at the original sample times. The CPU detector's
  one-step replay loop does not imply one GPU submission per timestep.
- Energy-only checks use a GPU reduction for the basic Cartesian engine only.
  SSE, cylindrical and multigrid cases retain CPU energy and full-field reads.
- With `--vulkan-fd=gpu`, supported Cartesian FD E/H dumps accumulate on-device
  at their original sample times and download completed sums at finalization.
  Native/node/cell sampling and FD NF2FF surfaces are supported. Unsupported or
  resource-limited dumps retain CPU accumulation per dump; CPU mode is the default.
- Time-domain fields, CPU FD dumps and other full-field consumers still require
  synchronized fields and bound batches. Energy/stopping decision times also
  bound batches. Final CPU field synchronization remains.
- Dense coefficients and direct-load core shaders remain the defaults. Exact
  palettes are opt-in. Multigrid shares device resources but records updates,
  extensions and transfers at each level.

## Next priorities

1. **Convergence contract:** define the reference voltage/current phases,
   projection state and timestep mapping; compare CPU/Vulkan detector traces.
   Resolve discrepancies in a separate correctness change before optimization.
2. **Measurement extension:** separate convergence, stepping, finalization and
   file-writing costs; add cylindrical/multigrid solver cases with energy checks.
3. **Select one candidate:** publish measured cost fractions and choose among
   Phase 6b recording changes, Phase 7 finalization/memory work, Phase 8a energy,
   Phase 8b CPU replay, Phase 9 compact probes, Phase 10a modes and Phase 10c SAR.
   Include capability demand, memory limits, implementation effort and validation
   cost; Phase 8a has no automatic priority after the prerequisite.
4. **Prototype, decide, then remeasure:** keep accepted changes independently
   reviewable and select the next candidate from the resulting workload profile.
   GPU probe spectra, final-only output and SAR J accumulation remain separately
   gated follow-ups to their initial deliveries.

Phase 8a closes a capability gap for cylindrical/multigrid models; it does not
accelerate basic Cartesian energy, which already runs on the GPU. For large
Cartesian FD workloads, separate stepping, finalization and file-writing costs
before selecting more work: output was substantial in the recorded 217-cubed run.

Phase 6b command-buffer reuse is a separate prototype for workloads with material
recording overhead, particularly small grids and multigrid. Projection fusion is
lower priority until projection costs justify it. Phases 8-10 do not depend on
these candidates. Phase 8b needs a validated energy path for its target geometry;
the CPU reference remains valid, so replay work need not wait for Phase 8a's GPU
implementation. Mode gathering need not wait for Phase 9. SAR can reuse Phase 7
infrastructure but requires its own processing integration and resource policy.

Before each prototype, record the target workloads, measured baseline, removable
cost or memory requirement, expected benefit, numerical/resource limits, bounded
effort and a review milestone. Stop or defer it if correctness requires relaxed
semantics, resource use exceeds the agreed budget, or repeated measurements fail
to show the intended benefit beyond variation. A memory/capability delivery may
instead meet a stated capacity requirement without unacceptable runtime regression.
Document that outcome separately. Recording share alone is not a universal cutoff;
do not assign priorities from assumed model prevalence or unmeasured transfer costs.

## Phase 0: measurement and reference results (core implemented; extension next)

Opt-in profiling records CPU recording, submission, fence waiting, readback and
mirror costs, plus submission/dispatch counts and transferred bytes. GPU queries
time batches and sampled phases. Profiling is off by default; instrumentation
can change scheduling and is not the throughput acceptance configuration.
Retain timestamp support/period/valid-bit checks and wraparound handling; collect
completed query results without adding waits to normal execution.

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

### Required measurement extension

The harness times solver setup separately. Output-case `iteration_s` wraps all
of `RunFDTD()`, including processing initialization/FD registration, final field
synchronization, FD finalization and file writing; its subsequent
`synchronization_s` does not isolate these costs. Add separate timings for solver
and processing setup, stepping, field download/mirroring, energy checks, detector
replay/comparison, FD finalization and file writing. State which timers overlap;
do not add nested CPU timings or sampled GPU categories to derive total runtime.
Retain an
uninstrumented end-to-end measurement as the performance acceptance result.

Extend the existing transfer-byte counters with check counts and fallback reasons.
For FD finalization, distinguish device copies, fence waits, host copies and file
serialization/writes. A benchmark-only null sink may help isolate output costs,
but must preserve sampling, accumulation and final readback, and state which
serialization work it omits. It is a diagnostic; acceptance still compares real
output files and end-to-end runs with matching processing schedules.

Add cylindrical and one/two/five-level multigrid solver fixtures that actually
execute convergence checks; existing direct-stepping cases are insufficient.
Include default wall-clock checks, exact-endcriteria and steady-state periods,
with and without other full-field consumers. Count checks, readback bytes and
fallbacks by reason. Vary probe counts, period lengths and output intervals.

Keep fixed-step throughput, deterministic stopping and wall-clock stopping as
separate experiments. Faster execution can move a wall-clock check to another
timestep, so identical stop timesteps are not its acceptance criterion. For each
candidate, report the measured fraction of runtime it can remove and the expected
end-to-end benefit. A capability improvement may still be useful with little
speedup; label that outcome explicitly and reconsider performance priority.

## Phase 1: batch timesteps into fewer submissions (implemented)

Phase-recording helpers use explicit timesteps for excitation, TFSF and Mur
activation. Several complete hierarchy steps share one command buffer, retaining
inter-phase/inter-timestep dependencies and projection at the caller's interval
end, including partial submissions.

One command buffer/fence is reused after pending work completes. Hierarchy
counters advance after successful submission. Recording, submission and
synchronization failures propagate through `RunFDTD()` rather than producing
successful output from incomplete fields.

Keep processing/stopping boundaries, steady-state sample times and multigrid
projection semantics. Internal chunks alone do not improve cancellation latency.

Regression requirements: batch sizes 1/8/16/32/64 and continuation, including 0/1
steps, non-multiple tails, delayed sources, Mur activation, explicit field edits,
PML, multi-pole Debye and all existing multigrid cases. Check output timestamps,
reset after pending work and failure propagation.

## Phase 2: probe gathering and GPU energy estimation (implemented)

### 2a. Probe gathering

Gathering is appended after projection when probes are needed. Bounded histories
retain intermediate samples for CPU replay. `SyncProbesToHost()` consumes cached
results or gathers on demand; partial probe updates do not mark the full CPU
field mirror valid.

Probe decoding, integrals and interpolation remain on the CPU. The registry
deduplicates points. Processing that needs full fields or another hierarchy level
retains its reference path; scheduling-aware subsets require measured benefit.

Retain checked sizes, device buffer limits and history-growth allocation fallback.
Measure total history bytes by registered point count and batch capacity, including
point metadata; account for two-period CPU detector records separately. Apply a
workload-aware byte budget if measurements justify one; 64 frames alone is not a
total-memory budget. Preserve sampling through the reference path on fallback.

Regression requirements: voltage/current integrals, field/steady-state probes,
repeat syncs, field edits, mixed sampling intervals and multigrid output. Verify
that the ordinary sample boundary no longer incurs a separate gather submission.

### 2b. GPU energy estimate

Basic Cartesian energy uses a staged GPU reduction with FP64 partials when
supported, otherwise FP32 partials, followed by CPU double accumulation and
EPS0/MUE0 weighting. It preserves `CalcFastEnergy()` extents and field products.
Unsupported precision/resources and exceptional values retain synchronized CPU
calculation. SSE, cylindrical and multigrid remain in Phase 8a's scope.

Energy-only requests avoid full-field reads. Full-field consumers retain their
downloads; voltage/current copies share a submission when staging resources allow.
Keep the existing energy metric, sample phase and stopping schedule.

Regression requirements: excitation/decay, lossy fields, zero/subnormal/non-finite
values, FP32 overflow and near-threshold deterministic stopping. Precision bounds
and fallback behavior are documented in [Vulkan performance](VULKAN_PERFORMANCE.md).

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
Reconsider them only in a separate review. See
[coefficient storage](VULKAN_PERFORMANCE.md#coefficient-storage) for current usage.

The implemented representation stores exact float-bit coefficient tuples and a
compact index per node, including geometry, direction and boundary corrections.
It does not quantize coefficients or fields. Its tradeoff is reduced coefficient
storage/traffic against indexed lookup, cache behavior and extra initialization.

Use dense storage for the primary before/after benchmarks of subsequent phases.
Keep palette compatibility and regression coverage, and label any separately
measured palette results so coefficient-mode changes cannot obscure the gain.

## Phase 5: reduce stencil loads (cancelled)

Cancelled because shared-memory tiling and subgroup variants introduce tile-size,
barrier and resource tradeoffs whose performance depends on the GPU. Retain the
direct-load shaders. This work is outside the remaining implementation roadmap.

## Phase 6a: extension and synchronization audit (implemented)

Implemented locally, 2026-10-05. The dependency invariants, numerical/validation
coverage and dispatch/barrier measurements are recorded in
[Vulkan performance](VULKAN_PERFORMANCE.md#synchronization-dependencies).

The audit covers buffer reads/writes, overlapping points, face ordering,
extension priorities and shared state, including dependencies across submissions
and host readbacks. Regression coverage includes intersecting Mur/PML/absorbing
boundaries, multiple TFSF faces, overlapping dispersive passes, lumped elements,
excitation priority and nested multigrid transfers. Future execution changes
require their own dependency proof, synchronization validation and numerical
comparisons; retain a record of untested configurations.

## Phase 6b: extension and recording costs (first delivery implemented)

First delivery implemented locally, 2026-10-05. Independent Lorentz/conducting-
sheet pre passes share one dispatch, retaining ordered apply passes and every
required dependency. Device-limit fallback and the original recording path are
tested. Duplicate-barrier coalescing is available for benchmark comparisons but
disabled by default. Command-buffer reuse and hierarchy projection fusion remain
separate future candidates; see the Phase 6 audit and measurement record.

Remaining candidates:

- **Command-buffer reuse:** first prototype fixed full batches with a stable
  command sequence and no scheduled FD/history consumers. Supply changing
  timesteps through device memory and retain rerecording for unsupported cases.
  Then evaluate tails, history/projection placement and FD schedules. Moving the
  timestep alone is insufficient: FD eligibility changes recorded dispatches,
  and sample counters currently advance during recording. Define per-submission
  bookkeeping, phase-buffer ownership and bounded recording variants before
  extending reuse. Invalidate recordings on relevant batch, registration,
  schedule, profiling, pipeline, buffer or descriptor changes and reset. Use
  reusable command-buffer flags, preserve resource lifetimes and wait before
  modifying in-flight resources. Prioritize material recording overhead.
- **Recording/execution overlap:** independently prototype two command buffers
  with separate fences and safe per-submission data to overlap CPU recording
  with GPU execution. This addresses a different cost from reuse and is not
  conditional on reuse failing. Preserve field dependencies and stopping bounds;
  measure each candidate separately before considering a combination. Compare
  recording and execution durations: ideal overlap changes their sum toward the
  larger duration, so recording longer than execution remains a bottleneck.
  Include phase/history buffer ownership, replay and decision-boundary waits in
  the design and measurements; two command buffers alone do not remove those costs.
- **Hierarchy projection fusion:** measure projection separately from hierarchy
  updates/transfers. Ordinary stepping projects at the caller's interval end;
  history recording projects every step. Select fusion only where it saves enough
  work and preserves child/root dependencies.

Select one candidate at a time from profiles. State the expected benefit and
required dependencies, keep shader combinations limited and retain the reference
path. Compare valid barrier alternatives on actual workloads; do not assume
global memory barriers are slower than buffer barriers. Keep the Vulkan 1.2 base
path without requiring synchronization2.

Validation: require synchronization validation and numerical equivalence on the
Phase 6a cases before accepting each execution change. Preserve output timestamps,
stopping schedules and existing numerical tolerances. Measure performance with
validation and profiling disabled, including representative extension and
multigrid workloads.

Completion: separately reviewable changes with demonstrated performance benefits
and documented dependency proofs. Retain the reference path where an optimization
does not meet the performance acceptance criteria.

## Phase 7: accumulate frequency-domain E/H fields on the GPU (implemented, opt-in)

`--vulkan-fd=gpu` keeps running complex Fourier sums on-device at the existing
sample times, then downloads completed results for the existing CPU file writers.
CPU accumulation remains the default reference. The option is available through
the command line, Python and MATLAB/Octave without new XML properties or formats.

### Scope and integration

Implemented scope is basic Cartesian `DumpType` 10/11 with native Yee
(`DumpMode 0`), node (`1`) and cell (`2`) sampling, including FD NF2FF surfaces.
Cylindrical/multigrid/SSE operators, custom interfaces/processings, SAR and other
unsupported field types retain CPU handling. The angular NF2FF transform remains
standalone CPU post-processing; time-domain NF2FF recording is unchanged.

Preserve these contracts in subsequent changes:

- Register enabled regions, frequencies, schedules and interpolation data after
  processing initialization resolves the output meshes. Preserve nonuniform mesh
  scaling, component/boundary rules, spatial subsampling and `OptResolution`.
- Match `ProcessFieldsFD::Process()` eligibility, start/stop windows,
  oversampling, staggered E/H time, negative Fourier phase and
  `2 * dt * FD_interval` weighting. Complex FP32 sums match existing storage;
  CPU-prepared phase factors are reused across points. Experimental GPU phase
  generation did not meet long-run accuracy requirements.
- Accumulate after the required field updates at each eligible timestep across
  batch tails and history replay, without duplicate CPU accumulation or a host
  wait per sample. Supported GPU FD samples do not force full-field reads or end
  batches; mixed GPU/CPU consumers retain their original schedules.
- At normal completion, convergence or graceful interruption, finish pending
  work and populate `m_FD_Fields` before `DumpFDData()`. Preserve HDF5/legacy HDF5
  and VTK metadata/values, reset/reuse and error propagation. Registration must
  not discard an accumulated CPU prefix. A mid-run GPU error is not a restart
  of accumulation on the CPU.

### Memory and fallback

Budget complex FP32 sums at 24 bytes per output point per frequency per E or H
dump. Both fields over 10 million output points need approximately 480 MB
(458 MiB) per frequency, or 2.4 GB for five frequencies, in addition to fields,
coefficients, extensions, mapping, phases and staging. Output point counts can
differ from solver cell counts; allocate only the requested regions.

Budget host memory separately: `ProcessFieldsFD::InitProcess()` allocates complete
CPU result arrays before GPU registration, and those arrays remain alongside
device sums. Include host field mirrors, processing arrays and writer temporaries
in peak RAM measurements, as well as device allocations in peak VRAM. Bounded
readback staging does not bound either total. Account for simultaneous FD/SAR
outputs when extending the accumulator; reducing host residency is separate work.

Check size arithmetic, device buffer/allocation limits and available memory
before enabling a dump. Use driver heap budgets when available; otherwise cap
cumulative actual FD allocations by heap, including mappings, phases, all sum
chunks and shared staging once. Retain the policies described in
[FD memory handling](VULKAN_PERFORMANCE.md#frequency-domain-fields). Partition
buffers where required and bound final readback staging.

Choose CPU accumulation per dump before its first sample when mappings or
resources are unsupported. Keep eligible dumps and FDTD stepping on the GPU;
never silently reduce requested frequencies, precision or resolution. Compact
gathers plus CPU accumulation for unsupported regions remain a separate
candidate, not an implemented replacement for the full-field reference.

Buffer partitioning satisfies allocation limits but does not reduce the total
sum storage. Every requested point/frequency needs contributions throughout the
run. Frequency chunking or spatial tiling reduces residency only with an explicit
spill/restore, retained-sample or rerun strategy; measure its memory and transfer
costs before adding it. Do not treat chunking as a free out-of-memory remedy.

Report actual FD sample intervals and accumulator traffic: for B
bytes of sums sampled every I timesteps, read/write traffic is about 2B/I bytes
per timestep, plus field/mapping reads. The five-frequency example adds about
4.8 GB per sample; the sample interval determines the per-timestep cost.

### Conditional finalization and host-memory experiments

Select these independently from Phase 0's finalization timings and peak-memory
measurements; neither is a required precursor to unrelated optimization work.

- **Pipelined final readback:** the current path copies a bounded device chunk,
  waits, then copies staging data into the CPU result array. Compare bounded
  staging sizes and a two-buffer pipeline that overlaps a subsequent device copy
  with the preceding host copy. Give each in-flight slot safe command/fence
  ownership; never read before completion or reuse staging before its host copy
  finishes. Count both buffers in allocation policy, retain single-buffer fallback
  on optional allocation failure, and preserve errors and output equivalence.
  Accept only a measured finalization and end-to-end benefit. This is separate
  from Phase 6b's overlap of timestep recording with GPU execution.
- **Deferred host allocation:** measure memory lifetimes before moving result
  allocation to finalization. Delaying all allocations can lower stepping-time
  residency without lowering peak RAM, and can move allocation failure to the end
  of a long run. Define result ownership and late-failure behavior while preserving
  CPU accumulation fallback, registration checks and reset/reuse. Claim peak-memory
  improvement only when measurements include finalization and writing.
- **Bounded-memory output:** separately evaluate streaming completed results to
  writers or materializing/releasing smaller result units. Specify the supported
  HDF5, legacy HDF5 and VTK paths, writer temporaries, format compatibility and
  processing-object lifetime. Preserve the current materialized reference where
  needed. Budget host RAM and device heaps separately; reducing CPU result storage
  does not make oversized device accumulators fit. Accept a demonstrated capacity
  improvement under the prototype's runtime and compatibility limits.

### Regression requirements and remaining measurements

Compare GPU and CPU FD paths using identical Vulkan timesteps and fields,
then compare supported solver cases against CPU-engine references. Cover E and
H separately/together; one/multiple frequencies and boxes; full volumes, planes
and lines; all supported interpolation modes; nonuniform/lossy meshes and
material boundaries; sampling windows, oversampling, disabled dumps and batch
sizes 1/32/64 with tails. Include mixed TD/FD output, ordinary probes, early
convergence, graceful interruption, reset/reuse, forced chunking, cumulative
allocation fallback, mixed GPU/CPU residency and cylindrical/multigrid CPU fallback.

Include frequency-domain NF2FF surfaces with enabled/disabled faces, supported
interpolation, orientation and symmetry/mirror handling. Compare surface spectra,
file metadata and derived far-field patterns, polarization and radiated power.

Check complex fields and output metadata. Use existing field tolerances and
explicit absolute tolerances near zeros; phase comparisons need an amplitude
floor. Reject non-finite comparisons. Long decaying and cancellation-heavy
signals must pass without loosening tolerances. Preserve sample counts and
deterministic stopping decisions.

Profiling distinguishes FD samples/dispatches, accumulator/mapping storage,
phase preparation, GPU accumulation and final FD downloads from full-field reads.
Continue CPU/GPU comparisons with matching outputs, fixed timesteps and profiling
disabled; a no-dump run is only a diagnostic bound. Sweep regions, sample rates
and frequency counts, recording VRAM use, total runtime, MC/s and final transfer/
file costs. The RTX 2060 10M-cell workload and cross-vendor measurements remain
unverified. Keep GPU accumulation opt-in until memory/performance selection is
supported by evidence.

## Phase 8: convergence checks (proposed)

Keep GPU energy reduction as the target for eliminating full-field energy
readbacks. Decide how to handle the remaining steady-state histories/comparisons
from measurements against existing batched replay. Deliver these separately,
retain CPU references, and preserve the stopping metric and decision boundaries.

### Prerequisite: define and validate the convergence contract

Before changing reduction or replay, specify the field snapshot for every check:
voltage phase, current phase, extension ordering, hierarchy projection state and
logical timestep. The CPU detector runs in `Apply2Voltages()` before the current
update. Vulkan currently calls it after a complete timestep and adjusts the CPU
timestep counter; this does not restore the earlier current field. Establish the
intended snapshot and resolve any discrepancy separately from performance work.

Ordinary processing probes run after stepping; distinguish them from the detector's
voltage-update hook. Preserve the existing weighted voltage/current stopping metric
and separate decay (`currE/maxE`) from steady-state period comparisons. Do not adopt
a full-step snapshot on an assumed negligible phase error. Evaluate an intermediate
device-side gather/reduction within the existing command buffer, with dependencies
before field mutation and compact readback at the decision boundary, if required
to preserve the contract. Audit hierarchy projection and extension ordering; this
approach does not inherently require a host wait or separate submission per step.

Add CPU/Vulkan detector traces for sampled voltages, electric/magnetic energy,
projection state, period index, comparison result and stop timestep. Cover basic
Cartesian, cylindrical and nested multigrid paths. Existing comparisons of GPU
energy against CPU reduction under Vulkan stepping do not prove CPU-engine sample
phase equivalence. Record both same-snapshot arithmetic comparisons and independent
solver traces, using existing field tolerances for the latter.

Define two acceptance levels. Scheduling changes on the same stepping path must
preserve samples and deterministic stopping decisions. Reduction changes must
also bound arithmetic disagreement and resolve ambiguous decisions using the
reference state. Independent CPU/GPU simulations retain their field tolerances;
do not infer universal stop-step equality at arbitrary thresholds from those
tolerances. Add explicit near-threshold cases and document their expected decisions.

Completion: the snapshot contract, trace fixtures and numerical acceptance are
reviewable, any phase correction is validated, and Phase 0's check-cost measurements
are available. No new GPU convergence capability is claimed by this prerequisite.

### 8a. Cylindrical and multigrid energy

Extend Phase 2's energy capability to the actual interfaces used by cylindrical
and multigrid models. Audit the active `CalcFastEnergy()` path, including SSE
lane order, padded z values, grid extents, FP32 accumulation and EPS0/MUE0
weighting. Reproduce the existing projected-root metric for multigrid; summing
every hierarchy level would double-count overlapping regions and change the
criterion. Keep projection and reduction on-device and download only compact
partial sums or the completed estimate. Do not add a new hierarchy masking or
physical-energy definition to the existing convergence criterion.

First prototype reference-order arithmetic on identical snapshots, including the
four SSE FP32 lanes for each of electric and magnetic energy and their final
combination; measure whether it offers useful parallelism. FP64 or a tree reduction
is not automatically equivalent to that reference. If a faster reduction is needed,
specify conservative error bounds and reference recovery before implementation.
Any required projection must follow the snapshot contract for that check type.
Judge total check cost including avoided readback and mirroring; limited reduction
parallelism alone does not establish whether the prototype is useful.

Fallback must cover historical state as well as the current estimate: `currE/maxE`
depends on previous maxima, and steady-state energy uses the previous period.
Recomputing only current energy cannot repair an uncertain historical value.
Specify how reference values are retained or recovered, or how bounds propagate
through the complete decision. Extend the estimate/decision interface if needed
to expose validity, uncertainty and fallback reason; a boolean plus scalar alone
does not carry this contract. Keep CPU reduction for configurations where the
contract cannot be met efficiently; do not silently loosen stopping semantics.

Measure download, CPU mirroring and energy arithmetic under default wall-clock
checks, exact-endcriteria and steady-state period checks. Four-second polling
alone establishes no overhead percentage; record actual check costs and their
share of total runtime. GPU energy remains in scope even without exact-endcriteria.

Validate against the current CPU estimate on identical field snapshots, including
non-multiple-of-four z sizes, closed/open angular domains, one/two/five nested
levels, lossy decay, zero/subnormal/non-finite fields and near-threshold cases.
Test thresholds on both sides of, and within, the candidate error bound; include
uncertain historical maxima and previous-period estimates. Require deterministic
decisions against the same-snapshot reference under the contract above. Report
fallback reasons, frequency and cost rather than counting those runs as GPU-only.

Completion: validated cylindrical/multigrid energy checks no longer routinely
download full fields. Preserve basic Cartesian behavior and benchmark check
costs separately from stepping, with and without frequent exact-endcriteria.

### 8b. Measure remaining steady-state overhead, then select execution

Use the GPU probe-history plus CPU replay path validated by the prerequisite as
the baseline. Measure submission counts, probe transfers, per-sample replay,
period comparisons and energy separately across probe counts and period lengths.
Do not assume a fixed 1-4 probes or infer lost GPU batching from the one-step
CPU loop.

First evaluate processing compact history batches on the CPU with less replay
overhead. Prototype GPU histories/reductions only if the remaining costs justify
them. Compare both against the existing path with identical sample/decision
times; retaining CPU comparisons is an acceptable outcome when faster or when
GPU gains do not exceed measurement spread.

For either design, preserve the validated detector snapshot and timestep mapping,
two-period warm-up, signal-power eligibility, squared-difference ratios, energy
comparison and maximum/clamping rules. Reuse the validated CPU energy path or
Phase 8a's accepted GPU path. End batches at the original decision boundaries
and check convergence before advancing further;
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

Deliver separately: (1) compact integration and bounded sample readback with
existing histories and CPU spectra, (2) an optional GPU spectrum prototype while
histories remain available, and (3) a final-spectra-only mode with its scripting
and file contract. Each delivery needs its own numerical/performance evidence;
success of integration does not require shipping either later mode.

For the GPU candidate, extend gathering with voltage line integrals, current
contour integrals, E/H point conversion and running Fourier sums for frequencies
specified before the run. Reuse Phase 7's scheduling/phase-factor infrastructure
and batch small probes together; avoid a kernel launch or host wait per probe.
Preserve integration signs, mesh/interpolation rules, weighting, E/H time bases
and independent TD/FD sample intervals. `ProcessIntegral` accumulates complex
double values, unlike FP32 field dumps: validate precision explicitly and retain
a reference path where the device cannot meet its numerical requirements. Do
not silently lower accumulation precision or reuse Phase 7's FP32 phase factors
unchanged. Floating-point atomics are not required by the design; use independent
outputs or staged reductions where appropriate and measure the actual device's
precision/throughput tradeoff.

Preserve existing time histories in the first two deliveries. For a final-only mode,
define frequency matching, spectral metadata, normalization, sample counts,
missing-frequency errors and graceful-interruption/finalization behavior before
implementation. Retain histories for later frequency selection and history-based
processing such as MATLAB autoregressive estimation; do not silently substitute
stored spectra for these workflows.
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

### 10a. Waveguide mode matching

First gather only the required surface values/interpolation footprint into
bounded history buffers, and use CPU overlaps, purity and spectra. Compare
against GPU weighted-overlap and power reductions with compact result readback
or GPU spectra where justified. Mode weights are already prepared and normalized
on the CPU at initialization; a GPU candidate uploads them once and needs no GLSL
parser or mode-file reader.

Define a processing capability for satisfying a sample from compact data while
CPU arithmetic remains active. Update both `GetNextFullFieldInterval()` and
`RunFDTD()`'s full-field synchronization decision; registering a gather alone
leaves mode probes classified as full-field consumers. Keep unsupported probes
on the reference path and never mark the full host field mirror valid after a
partial update. Specify component/interpolation footprints, field phases,
hierarchy levels, sample mapping, checked history byte limits and allocation
fallback. Add focused scheduling tests with supported and unsupported consumers.

Measure actual surface/component counts and transfer sizes. Preserve template
normalization, surface area weights, interpolation and both instantaneous
mode-purity and voltage/current results. The nonlinear purity history cannot be
reconstructed from final field phasors alone. Validate analytic and file-defined
modes, origins, port normalization and resulting S-parameters.

Keep batches bounded by actual CPU consumers and stopping checks; avoid replacing
full-grid copies with a new synchronous transfer per sample where compact history
replay suffices. Completion: no whole-domain readbacks for supported mode probes,
apart from final synchronization or another consumer's needs, with equivalent
overlap/purity histories and execution chosen by measured end-to-end benefit.

### 10c. SAR frequency accumulation

First support basic Cartesian, cell-interpolated E accumulation for the existing
cell-conductivity path. Preserve conductivity, density, material assumptions and
normalization. Keep unsupported geometry/material configurations and the optional
J path on CPU initially. Add J accumulation as a separate delivery with its own
field conversion, conductivity and timing validation.

Integrate with `ProcessFieldsSAR` explicitly: Phase 7's capability check accepts
the exact `ProcessFieldsFD` class, while SAR owns separate processing and E/J
result arrays. Define registration, result ownership, sample counts, scheduling,
reset/reuse and finalization into those arrays; merely relaxing the type check is
insufficient. Preserve normal completion, convergence, graceful interruption and
error propagation. Select CPU fallback before the first sample without losing a
prefix, and keep stepping plus other eligible dumps on the GPU.

Apply checked cumulative memory accounting across simultaneous FD/SAR outputs,
including host result arrays, device sums, mappings, phases, staging and CPU
post-processing temporaries. Measure peak RAM/VRAM and allocation fallback.
Validate raw complex E and, when supported, J spectra before local/1g/10g SAR,
including material boundaries, zero conductivity, multiple frequencies, mixed
outputs and lifecycle cases. Keep mass averaging on the CPU initially and report
accumulation, finalization and averaging costs separately. Completion requires
equivalent SAR output and measured end-to-end benefit for the supported range.

## Later experiments

Temporal blocking across multiple timesteps requires halo expansion, global
dependencies and extension scheduling. Prototype it separately after Phases 4 and 6;
a simple voltage/current fusion cannot synchronize independent workgroups.

Pipeline/shader caches and batched initialization uploads address setup time.
Prioritize them only if setup dominates real workloads. Keep device/driver cache
compatibility checks and a safe cache-miss path.

## Release and acceptance

Keep each optimization independently selectable in the benchmark harness until
its default-selection criteria are established. Retain dense coefficients as the
primary reference and report palette comparisons separately. Each execution
change requires its own synchronization audit and validation.

For each solver change, build Release, run the complete native backend suite and
CTest, and run Python GPU equivalence tests against the exact newly built DLL.
Use fixed-step and exact-endcriteria integration cases, including probes, field
dumps and derived port results. Run the relevant Octave/MATLAB cases when solver
scheduling or scripting behavior is affected. Check Vulkan-disabled compilation,
Windows MSVC and the existing MinGW configuration; test Linux and an additional
GPU vendor before making broad compatibility/performance claims.

For each selected delivery, name the implementation and validation owners and
record the available build/GPU matrix before work starts. Mark unavailable Linux,
vendor or device checks as unassigned/unverified with the evidence needed to close
them. Cover the default path, the candidate and its reference, then the relevant
interactions with batch size, coefficient mode, FD mode and other execution
switches. Retain reference switches until those comparisons are recorded; avoid
requiring every unrelated combination for each delivery.

Keep existing numerical tolerances; scheduling-only changes must preserve sample
timestamps and deterministic stopping decisions on the same stepping path.
Reduction changes follow Phase 8's snapshot and historical-state contract with
explicit precision and near-threshold evidence. Record independent CPU/GPU solver
comparisons separately from same-snapshot arithmetic tests, and list untested
configurations. Wall-clock polling retains its policy, not an identical stop step.

Performance acceptance uses warm, repeated runs with profiling disabled. As an
initial review rule, investigate repeatable regressions above 5% on the workload
matrix and retain a fallback for configurations that lose performance. Use paired
or interleaved baseline/candidate runs, with at least five measured runs per path;
retain raw timings and report medians, minimum/maximum and interquartile ranges.
Describe spread using those statistics and report paired differences when used.
For small or overlapping gains, repeat an independent round and require consistent
improvement beyond observed run-to-run variation before claiming a benefit.
Performance timing is informational rather than a flaky CI threshold. Report
improvements by workload and separate setup, stepping and output costs.

For each delivery, retain an evidence record with four distinct statuses: code
implemented, correctness validated, performance measured and hardware/build
coverage. Record the revision and dirty diff if applicable, exact commands,
configuration, numerical tolerances/results, raw timing location and remaining
gaps. Link only retained files or accessible artifacts; missing historical notes
are not current validation. Documentation-only plan updates do not change these
statuses or establish new solver correctness/performance claims.

Update `CHANGELOG.md` for user-visible changes and the validation notes with
reproduction commands and results. Follow `AGENTS.md` / `AI_POLICY.md` for C++11,
dependencies, clean diffs and disclosure/sign-off when changes are submitted.

## Technical references

- [Khronos timestamp queries](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html): GPU timing support and deferred query collection.
- [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html): compute and host dependencies; adapt examples to the existing Vulkan 1.2 API.
- [Khronos command-buffer lifecycle](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html#commandbuffers-lifecycle): reuse flags, pending work and resource lifetimes.
- [NVIDIA Vulkan guidance](https://developer.nvidia.com/blog/vulkan-dos-donts/): submission overhead, synchronization and profiling considerations.
