# Vulkan memory-access implementation plan

Selected on 2026-10-10: direct PML addressing (discussion item 2), PML/Yee
same-phase fusion (item 3), and compiled core-load tuning (item 4).
Status as of 2026-10-11: M0 instrumentation and M1/M3 candidates are implemented
and locally validated. M2 inspection retains the existing core shader; explicit
center-load reuse is included in the new fused current shader. The
[M1–M3 evidence record](VULKAN_MEMORY_M123_RESULTS.md) distinguishes numerical
acceptance, measured performance and hardware coverage. Release defaults remain
indexed; candidates are selectable in the native experiment harness.

## Objective and evidence

Reduce time per cell update by removing dependent index loads, unnecessary
field/flux transfers, and demonstrated redundant loads or spills. Acceptance
uses unprofiled stepping and total runtime with validated numerical outputs;
neither cache-hit rate nor the percentage of memory-wait samples is the target.

The retained [full antenna capture](../build/nsight-full-20261010/REPORT.md)
used installed revision `c849f07`, driver 617.42 and an RTX 3070. Both captures
converged at 17,400 steps / -40.06 dB. Early/decay memory-dependency samples
were 93.89%/93.88%; device activity stayed about 98.7% across the full run.
The four dominant shader hashes remain unmapped to source passes. Earlier
extension samples were 55.6% of the sampled Yee/extension total; this does not
attribute all that cost to PML. These historical results motivate investigation,
not a speedup estimate or validation of the current checkout.

Record a fresh baseline from the implementation revision and its dirty diff.
The checkout contains newer operator-setup work than the captured binary; fix
setup mode, thread count, coefficients, output schedules and energy checks in
each comparison. Keep dense coefficients as the primary reference and preserve
palette compatibility. Palette experiments and CPU setup changes are separate
from this delivery.

## Delivery order

| Milestone | Deliverable | Required evidence before proceeding |
|---|---|---|
| M0 | Attribute shaders, split pass timing, inspect optimized loads | Named pass costs and producer-load evidence; fresh reference outputs/timings |
| M1: item 2 | Direct addressing for regular Cartesian PML regions | Reference-equivalent addressing/state; measured addressing and dispatch tradeoff |
| M2: item 4 | Bounded core-load reuse or spill reduction | A demonstrated compiled-code issue and an independently measured improvement |
| M3: item 3 | Fuse PML pre/Yee/post within one field phase | Component ownership and dependency proof; numerical, synchronization and performance checks |

M2's inspection begins in M0 and can finish without a shader change if optimized
code already eliminates redundant expressions. M3 uses M1's region mapping;
it need not wait for an unsuccessful M2 experiment. Keep each candidate
independently selectable and separately reviewable. Compare each against the
original path, then measure accepted combinations without attributing combined
gains to an individual change.

## M0: attribution and reference results

Extend `FDTD/vulkan/engine_vulkan.{h,cpp}` profiling to distinguish Yee voltage,
Yee current, UPML pre/post voltage, UPML pre/post current, and excitation from
other extensions. Preserve existing aggregate output for consumers and avoid
double-counting nested samples. Collect deferred timestamp results without new
waits or queries when profiling is disabled.

Use optional `VK_EXT_debug_utils` instrumentation: name `VkPipeline` and
`VkShaderModule` objects with `vkSetDebugUtilsObjectNameEXT`, and mark pass
boundaries with `vkCmdBeginDebugUtilsLabelEXT` / `vkCmdEndDebugUtilsLabelEXT`.
Check instance-extension availability before enabling it and guard loaded
function pointers; missing debug utilities must not prevent solver operation.
See the [Khronos debug-utils guide](https://docs.vulkan.org/samples/latest/samples/extensions/debug_utils/README.html#setup).

Enable optional shader source/debug information with shaderc's
`options.SetGenerateDebugInfo()` in profiling builds. Keep performance
optimization enabled and verify the profiler reports successful correlation;
object names alone do not establish source-line or shader-hash attribution.
Record shader source, SPIR-V identity, pipeline identity, variant and captured
Nsight hash; do not assume different tools use the same hash algorithm. Use
available instruction-range correlation if full assembly is unavailable, and
record the resulting attribution limits.

Inspect the load producing each dominant dependency, not only the instruction
where the waiting sample appears. Record actual global load/store instructions,
sectors/transactions where available, register use, local-memory spills and
occupancy/resource limits. Repeated GLSL expressions alone are insufficient
evidence of repeated hardware loads. NVIDIA documents this distinction in its
[shader profiler guide](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html#instruction-mix).

Deliver a pass-cost table, annotated access/dependency inventory, reference
outputs, raw timing locations and a short choice of the first bounded candidate.
First try the retained native traces offline; make a new short capture only
where naming/correlation or timing data is missing. Instrumented timings remain
diagnostic and are separate from performance acceptance.

## M1: direct PML addressing — item 2

Today `AllocateUpmlBuffers()` creates per-component field indices and coefficient/
flux arrays. `kShaderUpmlPre` and `kShaderUpmlPost` read an index before fetching
the field. Introduce immutable descriptors for the existing regular Cartesian
regions: origin, extents, field strides, component and coefficient/flux offset.
Derive field addresses from those descriptors without a per-component index load.

Preserve the original region/component ordering, coefficients and flux ownership.
Prove exact address equivalence, checked size arithmetic and complete coverage
before replacing the lookup. Exercise all six faces, corners/edges, thin regions,
nonuniform meshes and the distinct voltage/current boundary guards. At setup,
validate bounds and sizes, then check pairwise AABB intersections within each
grid level and field component. Account for inclusive grid endpoints: opposing
slabs can share a plane on a thin mesh. Overlap fails candidate eligibility;
the indexed path also needs proven ownership because duplicate field addresses
can race there. Reject an invalid layout before stepping or use a separately
validated path that preserves the required ordering.

Compare the current single indexed dispatch, bounded per-region dispatches,
and a unified multi-box dispatch using a small read-only descriptor array and
prefix offsets. Respect device push-constant/buffer limits and preserve
contiguous z traversal within regions. Account for extra dispatches, descriptor
reads/selection, integer division/address arithmetic, register use and memory
transactions. With all six faces dispatched separately per component, M1's
pre/post voltage/current passes would require `6 * 3 * 2 * 2 = 72` PML dispatches
per timestep, or 2,304 per 32-step batch, versus four per timestep today.
These counts exclude Yee/other extensions and change with the chosen design;
measure command-recording and GPU phase costs to select a design.
Keep the first experiment's coefficient layout unchanged so its effect is
measurable.
Any later coefficient packing needs separate evidence of the loads/transactions
actually removed; a GLSL `vec4` does not prove every member is fetched.

Initial eligibility is basic Cartesian PML. Retain the indexed implementation
for unsupported region layouts, cylindrical/multigrid cases and optional
resource failures before stepping, subject to the ownership checks above.
Runtime execution failures still propagate; they must not silently restart a
simulation. Record selected path and fallback reason in diagnostic/benchmark
metadata. Cover initialization, reset and reuse.

Primary files: `FDTD/vulkan/engine_vulkan.{h,cpp}`, `shaders_glsl.h`,
`test_backend.cpp` and `test_vulkan_performance.cpp`.

## M2: compiled core-load tuning — item 4

Retain the existing component planes, contiguous z traversal and workgroup
shape. After M0, select one actual issue in `kShaderVoltageUpdate` or
`kShaderCurrentUpdate`: repeated field/coefficient loads, a long load-use chain,
or local-memory spilling. Make the smallest change that addresses that issue,
such as explicit reuse of a loaded component or reduced temporary live ranges.
Only assert non-aliasing if descriptor/buffer ownership proves it.

Compare optimized code before/after and measure the register/occupancy tradeoff.
Moving loads earlier or holding more values can increase register pressure;
source simplification without changed generated code is not an optimization.
Use measured device/compiler resource limits and spill instructions, not fixed
32/64-register thresholds: occupancy and spilling depend on the architecture,
allocation granularity and live values.
Preserve arithmetic order, precision, index guards and dense/palette behavior.

Broad workgroup sweeps, shared-memory tiling, subgroup rewrites and multi-timestep
blocking remain deferred. A larger experiment needs measured reuse opportunity
and its own accounting for halo work, barriers, resources and portability.
If no worthwhile compiled-code defect is found, record that result and proceed
to M3; do not manufacture a rewrite to complete the milestone.

## M3: same-phase PML/Yee fusion — item 3

Combine PML pre-update, the corresponding Yee update and PML post-correction
for eligible components, retaining intermediates in registers. Voltage and
current remain separate globally ordered phases. Preserve lower-priority pre
extensions before the fused work and post/apply/excitation operations after it,
matching `RecordVoltagePhase()` / `RecordCurrentPhase()`.

For one component, the reference sequence is:

```text
p = a * old_field - b * old_flux      # PML pre intermediate
u = Yee(old_flux, opposite_field)     # unchanged Yee coefficients and curl
new_flux = u
new_field = p + c * u                 # PML post correction
```

This is a dependency sketch, not authorization to reassociate the arithmetic.
Keep the reference FP32 intermediate rounding and contraction behavior. Where
the original current shader skips a boundary component, retain that behavior:
PML pre/post may still execute, but the omitted Yee update must not be invented.
In that case the sketch uses `u = old_flux`; an early return must not omit the
PML correction.

Define one writer for every field component and flux entry. Partition interior
and PML work so the original full-domain kernel does not also update fused
components. Prove that opposite-phase fields stay immutable during a fused
phase, that no workgroup needs another's intermediate result, and that all
outputs are visible before the next phase or consumer. Keep required global
barriers; a workgroup barrier cannot replace them.

Compare an interior coordinate-box dispatch plus fused PML-box dispatches with
a unified domain dispatch that branches between interior and PML work. Apply
the same single-writer proof to both. Measure branch coherence, register use,
partially occupied workgroups, dispatch overhead and complete phase time;
branching alone does not establish that a design is slower. Keep fusion free
of subgroup operations and assumptions about hardware warp/subgroup width.

Start with a documented basic Cartesian subset and explicit eligibility checks.
Prove each participating extension's ordering and overlap behavior, including
the antenna's lumped ports and sources. Use the separate-pass path for unproven
combinations. Exercise faces/corners, PEC/PMC, Mur/local absorbers, RLC/TFSF,
lossy/dispersive materials and palette selection through either the validated
candidate or its declared fallback. Expand eligibility only with evidence.

Compare against both the indexed reference and accepted direct-address path.
Measure total affected phase cost, saved transactions, dispatch count, extra
mapping/branching, registers, spills and end-to-end runtime. Fewer dispatches
or stores alone do not establish a gain.

## Correctness and performance acceptance

Extend existing native fixtures in `test_backend.cpp` for address equivalence,
per-phase field/flux checkpoints, single-writer coverage, fallback and reset.
Include deliberately overlapping regions, shared inclusive endpoint planes,
and valid thin/opposing slabs. Verify profiling instrumentation remains optional
when disabled or when `VK_EXT_debug_utils` is unavailable.
Reuse the synchronization fixtures for intersecting boundaries and extension
priority; include single steps, partial batches and limits 1/32/64. Compare
candidate and reference on identical independently maintained state. Preserve
the existing GPU-reference threshold of 1e-7 of peak where that fixture uses it,
and existing CPU/GPU tolerances; never relax tolerances to accept a candidate.

For the 285 x 138 x 106 antenna, retain eight port traces and twelve complex
77 GHz FD surfaces with identical axes, timestamps and sample schedules. Use
the current numerical/derived-port checks with stated tolerances. Require
deterministic energy checks at the same 120-step schedule and -40 dB criterion
to preserve the reference stop decision. A changed decision is investigated,
not dismissed as a speed consequence. Wall-clock polling is a separate case.
The broader CPU/Vulkan detector-phase audit remains open; same-path memory
optimizations must not claim to close it or depend on changed stopping semantics.

For every solver delivery, follow the roadmap's [release requirements](VULKAN_OPTIMIZATION_PLAN.md#release-and-acceptance):
Release build, complete native backend suite/CTest, exact newly built DLL in
Python GPU tests, relevant Octave/MATLAB cases and synchronization validation.
Include Vulkan-disabled compilation, MSVC and the existing MinGW configuration.
Record Linux/additional-vendor coverage as unverified until actually run.
Portable Vulkan source does not establish portable performance. Record correctness
and performance separately for each tested GPU/driver/toolchain; ordinary CI
correctness checks do not replace hardware timing. Require cross-vendor evidence
before claiming gains across vendors or broadening defaults on that basis.

Measure 4,096 and 17,400 fixed steps with criterion 1e-30, dense coefficients,
GPU FD accumulation, batch limit 32 and matching energy-check schedules. Use
one untimed warm-up per path, then at least five interleaved timed runs per path.
Disable profiling/validation and competing profiler sessions for acceptance.
Record revision/dirty diff, loaded DLL, XML hash, setup mode/threads, driver,
clocks, selected candidate/fallback, memory use, setup, stepping, output and total
runtime. Keep the separate converged -40 dB numerical comparison.

Report medians, ranges, interquartile ranges and paired differences. Accept an
execution candidate only for repeatable stepping and useful total-runtime gains
beyond observed variation, with no unexplained resource/correctness regression.
Investigate repeatable regressions above the roadmap's 5% review threshold;
retain a fallback for losing configurations. Repeat an independent round for
small/overlapping gains. Do not predict a percentage speedup from stall samples.

## Delivery record

M0 implementation and local validation owner: Codex. Independent review,
Linux and additional GPU/vendor coverage remain unassigned/unverified.
The [M0 evidence record](VULKAN_MEMORY_M0_RESULTS.md) identifies the exact
source diff/binary, named pass costs, optimized SPIR-V dependency inventory,
numerical comparisons and available build matrix. Instrumentation and
pass attribution are available. Optimized SPIR-V load inspection and native
executable register statistics are recorded; native producer instructions,
reliable spill/occupancy metrics and verified source-line correlation remain
open. M1 and M3 candidates are implemented; M2 retains the existing core after
compiled-code inspection and removes repeated center loads in the new fused
current candidate.
The [M1–M3 evidence record](VULKAN_MEMORY_M123_RESULTS.md) tracks correctness,
timing, build coverage and remaining acceptance gaps independently. Release
defaults remain unchanged.

For M0-M3, record implementation owner, validation owner, exact scope, baseline
revision, code/correctness/performance/hardware statuses, evidence paths and
remaining gaps before claiming completion. Assign owners when each milestone
starts; documentation alone does not establish execution or validation.
Prototype/reference switches belong in the native harness initially. Any public
selection added later must include CLI, Python and MATLAB/Octave integration,
documentation and the required changelog entry. Keep release defaults unchanged
until the candidate's eligibility and acceptance evidence are established.
