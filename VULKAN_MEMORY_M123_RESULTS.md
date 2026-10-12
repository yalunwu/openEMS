# Vulkan memory candidates: M1–M3

Implementation and local validation owner: Codex. Baseline: `0e8de576b6a93ce4575d72cfbebfb0290fa03d96`, with the retained M0 working changes. Experiments use workspace builds; the installed solver and external antenna project are unchanged.

Full-simulation comparison passes for all five paths. On the tested RTX 3070, unified fusion reduces median full-length stepping time by 28.26% and total solver process time by 25.67% against the indexed reference. Direct addressing alone saves 19.15% and 17.45% in its separate paired round.

## Implementation

M1 implements `indexed`, `direct` (one prefix-selected region dispatch), and `regions` (one dispatch per region/component). Validated 48-byte descriptors retain the original component and x/y/z flux/coefficient order. Bounds, checked products, offsets and same-component ownership are checked before stepping, including GPU operator construction. Shared inclusive endpoint planes are rejected. Optional resource exhaustion selects the indexed reference before stepping; execution/device failures propagate. Selection after the first timestep or during probe replay is rejected.

Reference index storage remains allocated for pre-step reselection.

M2 leaves the core shader unchanged. The optimized reference SPIR-V already reduces twelve source opposite-field reads to nine loads. There is no demonstrated repeated-load defect to fix. Native executable statistics report 21 registers for each dense Yee shader, 16 for indexed PML and 20 for direct PML on the tested driver. These counts do not establish occupancy or absence of spills. The driver reports `68719476736` bytes of per-thread local memory for every inspected shader, including trivial ones, and returns no internal representation. That metric is unresolved and is not used as spill evidence. No speculative load scheduling or register threshold was substituted for measurements.

Inspection of the first fused current candidate did expose a new compiled-code issue: independent boundary guards prevented reuse of three center voltage loads. Explicit center reuse reduces its optimized opposite-field load count from twelve to nine. The loaded center addresses are valid on outer planes, where unused center reads do not invent a Yee update. Dense current register use rises from 25 to 29; the timing rounds retain this tradeoff explicitly. `before-load-reuse/` retains the earlier modules and native statistics, and `source-short/` identifies the first timing binary.

M3 implements `fused` (unified domain) and `fused-boxes` (interior plus disjoint PML coordinate boxes). Both preserve the workgroup shape and z traversal. One invocation owns the three components of its node and corresponding unique flux entries. Box subtraction partitions the full grid without duplicate writers or holes. Within a phase only its own field and flux are written; the opposite field is immutable. Existing global phase/consumer barriers remain. Conservative barriers between box dispatches also satisfy synchronization validation's buffer-level alias tracking.

Fused operations retain the reference expression order: pre correction, Yee on old flux, then post correction. On outer current planes the Yee update is skipped while PML pre/post still executes, without out-of-range curl loads. Dense and palette coefficients use the original layouts. Final fused shaders use 29 registers in dense mode and 31 in palette mode for both phases; these counts alone do not predict performance. Floating-point equivalence is established on the tested compiler/device; native instruction/contraction inspection and additional-vendor validation remain open.

Fusion eligibility is basic Cartesian PML with active UPML, excitation, steady-state and lumped RLC extensions only. Material losses, PEC/PMC and nonuniform meshes are core coefficient behavior. Mur, local absorbing sheets, TFSF, dispersive materials/conducting sheets, cylindrical/multigrid and unknown active extensions retain the indexed path. Lower-priority pre operations, including RLC pre-update, remain before the replacement block. RLC apply, other post/apply operations and excitation remain after it. The antenna's lumped ports and sources retain their original priority, including shared edges.

Release defaults remain indexed. Selection is confined to the native experiment harness; no new Python/MATLAB public option is introduced.

## Reproduction

```text
test_backend --pml-memory-tests
test_backend --vulkan-memory-inspect
test_backend --vulkan-benchmark --case=pml --pml=fused-boxes --steps=4096 --repeats=5 --batch-size=32 --csv=pml.csv
test_backend --vulkan-memory-model model.xml --pml=fused-boxes --vulkan-setup=cpu --numThreads=1 --vulkan-fd=gpu --vulkan-coefficients=dense --vulkan-batch-size=32 --exact-endcriteria --dump-statistics
```

The harness accepts `indexed|direct|regions|fused|fused-boxes`. CSV and console metadata identify requested/selected paths and fallback reasons. With fusion, voltage/current profiling categories include complete fused PML/Yee work; separate PML categories have zero dispatches. Existing extension aggregate semantics are retained for separate passes.

## Evidence and status

Raw local evidence is retained under `../build/memory-m123-20261011/`: scripts, shader sources/modules/disassemblies, logs, per-run XML and hashes. The M0 binary is preserved in `baseline-bin/`. `source-final/manifest.json` identifies the frozen tested solver and source; `source-delivery/manifest.json` additionally contains the completed documentation. Native process module inspection and Python loading independently confirm final DLL SHA256 `8f533a56d5268915667ddb9062900f19a1eabbe5d3cc1a5c3e6893f247b0a4e5`.

Code is implemented for M1 and M3. M2 retains the compiled core and removes three repeated center loads in the new fused current shader; no isolated speedup is attributed to this change. Independent review, Linux and additional GPU/vendor runs are unassigned/unverified. No portable performance or spill-free claim is made.

## Numerical acceptance

Final acceptance uses the fully converged 4,168,980-cell antenna, with the original maximum of 50,000 steps, deterministic energy checks every 120 steps and the original -40 dB criterion. Setup uses one CPU thread for direct comparison with the saved M0 reference; stepping and FD accumulation use Vulkan, dense coefficients and batch limit 32. All five paths (`indexed`, `direct`, `regions`, `fused`, `fused-boxes`) stop at **17,400 steps / -40.06 dB**.

All eight port traces have identical timestamps and values. All twelve complex 77 GHz E/H surfaces have identical axes, dimensions, frequency and values. Relative L2 error is **zero** for every output, against an unchanged acceptance tolerance of 1e-7. Derived spectral voltage/current, incident/reflected waves, reflection, impedance and source-normalized port ratios also have zero difference. The new indexed path independently matches the saved M0 full reference exactly. These comparisons are retained in `converged-*/comparison.json` and `full-checks.log`.

Every timed 17,400-step run in both acceptance rounds also passes all twenty raw-output and derived-port comparisons with zero error and exact saved-value equality. FD norms accumulate in complex double precision, and explicit array equality independently checks the zero-error report. [Full-duration port overlays](../build/memory-m123-20261011/full-converged-ports.png), [full-length timing figure](../build/memory-m123-20261011/full-timing.png) and the consolidated [machine-readable checks](../build/memory-m123-20261011/evidence-checks.json) retain the final comparison.

Native fixtures compare fields and both PML flux buffers after voltage/current checkpoints on independently maintained state. Dense/palette coefficients, all-six-face PML, mixed PEC/PMC with four PML faces, nonuniform/lossy materials and overlapping RLC/source edges pass all 80 candidate/checkpoint comparisons per validation run with zero measured error (tolerance 1e-7 of peak). Mapping checks cover thin opposing slabs, shared endpoint rejection, invalid bounds/overflow, component grouping and exact single-writer coverage. Lifecycle checks cover selection restrictions, reset/reuse, optional allocation fallback and execution-failure propagation.

## Full-length performance

Acceptance timing uses fixed **17,400 steps**, criterion 1e-30, dense coefficients, GPU FD accumulation, batch limit 32 and exact energy checks every 120 steps. GPU setup is required with one setup thread. Each path has one excluded warm-up and five interleaved measurements. Profiling, synchronization validation and competing profiler sessions are disabled. Setup, XML, DLL hashes and selected paths are retained per run. This timing configuration is separate from the CPU-setup converged numerical gate above.

On the RTX 3070 / driver 617.42, the final unified fused candidate reduces median stepping time by **28.26%** and whole solver process time by **25.67%**:

| Measure | Indexed median / range / IQR (s) | Unified fused median / range / IQR (s) |
|---|---:|---:|
| Setup | 2.770 / 2.730–2.810 / 0.020 | 3.350 / 3.340–3.360 / 0.010 |
| Stepping | 57.440 / 57.260–57.680 / 0.250 | 41.210 / 31.580–41.400 / 0.090 |
| Run including output | 58.140 / 57.940–58.410 / 0.210 | 41.900 / 32.250–42.080 / 0.140 |
| Whole solver process | 61.021 / 60.877–61.307 / 0.244 | 45.359 / 35.830–45.553 / 0.143 |
| Run minus stepping | 0.700 / 0.680–0.760 / 0.050 | 0.690 / 0.670–0.750 / 0.020 |

Paired stepping savings are 16.54, 16.21, 25.68, 16.23 and 16.17 seconds. Paired whole-process savings are 15.992, 15.563, 25.047, 15.688 and 15.638 seconds. Every pair improves; the faster third fused run is retained in the range and raw data, and does not set the median. The cause of this variation is unresolved. Run minus stepping is a remainder rather than a pure output timer. Whole-process time includes native solver setup and output, and excludes Python geometry construction and subsequent post-processing. See `timing-17400.json` and each `17400-*/result.json`.

An independent interleaved full-length round measures M1 alone. Direct addressing reduces median stepping time by **19.15%** and whole-process time by **17.45%** against that round's indexed reference:

| Measure | Indexed median / range / IQR (s) | Direct median / range / IQR (s) |
|---|---:|---:|
| Setup | 2.710 / 2.700–2.740 / 0.020 | 2.990 / 2.950–3.120 / 0.070 |
| Stepping | 56.070 / 55.900–56.290 / 0.040 | 45.330 / 45.250–51.000 / 0.230 |
| Run including output | 56.750 / 56.600–56.960 / 0.010 | 46.020 / 45.930–51.700 / 0.220 |
| Whole solver process | 59.601 / 59.428–59.791 / 0.032 | 49.198 / 49.003–54.943 / 0.218 |
| Run minus stepping | 0.680 / 0.650–0.700 / 0.010 | 0.680 / 0.650–0.700 / 0.030 |

Paired stepping savings are 4.90, 10.60, 11.04, 10.73 and 10.80 seconds; whole-process savings are 4.485, 10.313, 10.750, 10.403 and 10.601 seconds. All pairs improve, including the slower first direct run. Raw evidence is in `m1-full/`. M1 and M3 percentages use their own paired references and are not additive; fusion includes direct mapping and the new fused shader's load reuse.

### Short diagnostic selection

These 4,096-step measurements select variants and retain compiled-load tradeoffs. They do not replace the full-length performance or converged numerical gates.

| Path | First round stepping / process median (s) | Final shader round stepping / process median (s) |
|---|---:|---:|
| Indexed | 13.410 / 16.847 | 13.730 / 17.319 |
| Direct | 10.880 / 14.632 | — |
| Per-region | 11.700 / 15.480 | — |
| Unified fused | 7.920 / 11.988 | 8.320 / 12.507 |
| Boxed fused | 8.720 / 12.764 | 8.800 / 12.980 |

The first round uses DLL `b2dfe65fd32471fad29e36958329dadd09fdbffc871a07a4a2b1b1b6519b9df8`, retained by `source-short/manifest.json`. The final round and every full-run result use DLL `8f533a56d5268915667ddb9062900f19a1eabbe5d3cc1a5c3e6893f247b0a4e5`. Each round excludes one warm-up and includes five interleaved runs per tested path. Prefix-selected direct addressing beats per-region dispatches; unified fusion beats coordinate-box dispatches in both short rounds. Full-length timing is measured for the selected direct and unified fused designs. Per-region and boxed alternatives retain full converged correctness coverage, with performance status limited to the short diagnostic cases.

Before/after load-reuse rounds were not interleaved with each other and show different reference times. They do not establish an isolated speedup from reducing twelve fused current loads to nine. The final combined fused implementation meets the full-length acceptance gate; native transactions, occupancy and spill behavior remain unverified.

## Full-run profiling diagnostics

Separate profiled runs span all 17,400 steps. Dispatch timestamps sample the first step of each batch (581 sampled steps); the table reports sampled phase cost rather than extrapolated full-run phase totals. Voltage/current categories include PML when fused. Leaf PML categories and the extension aggregate must not be summed together.

| Path | Yee + PML microseconds / sampled step | Yee + PML dispatches over full run | All dispatches over full run | CPU recording (s) |
|---|---:|---:|---:|---:|
| Indexed | 3214.4 | 104,400 | 142,984 | 0.350 |
| Direct | 2590.9 | 104,400 | 142,984 | 0.368 |
| Unified fused | 1753.7 | 34,800 | 73,384 | 0.199 |

Fusion removes four dispatches per step (69,600 over the full run) while preserving probe, FD and energy schedules. Direct addressing improves the sampled phase cost with the same dispatch count. Fusion lowers sampled GPU phase cost as well as CPU recording work. These diagnostics support the mechanism; their instrumented runtime and larger apparent GPU-time reduction are not substituted for the unprofiled median gain. No new L1/L2 hit-rate, transaction or memory-wait percentage claim is made.

Raw `VULKAN_PROFILE`, shader identities and selected-path metadata are in `17400-*-profile/console.log`; summarized metrics are in [the full report](../build/memory-m123-20261011/full-report.md). Shader/native statistics are retained in `pipeline-inspection.log`, `spirv/inspection.json` and `before-load-reuse/`. Telemetry CSV files record driver, clocks, power, temperature and device memory; they include setup/idle transitions and desktop allocation, so memory use is not solver-only storage.

## Build and interface coverage

| Check | Result / evidence |
|---|---|
| MSVC Release; complete native backend suite | Pass: 78 tests, `ctest.log` / native test log |
| Complete CTest | Pass: 9/9, `ctest.log` |
| Python GPU tests against the exact final DLL | Pass: 27 tests, `python-gpu.log` |
| Octave GPU setup cases | Pass: all five modes, `octave.log` |
| Khronos synchronization validation | Pass: PML, mixed-boundary, synchronization and batching fixtures; no validation errors |
| Validation-layer loading | Confirmed in process: `validation-layer-loaded.json` |
| Optimized SPIR-V | Ten modules validated; hashes match runtime, `spirv/inspection.json` |
| Vulkan-disabled MinGW build | Pass: `vulkan-off-build.log` |
| MinGW Vulkan solver DLL/CLI | Pass: `mingw-solver-build.log` |
| Complete MinGW backend executable | Blocked by the existing `Excitation::buildExcitationSignal` export/link failure, `mingw-build.log` |
| Linux / additional GPU vendors | Unverified |

The private Python test environment warns that its HDF5 2.0.0 headers differ from the solver's HDF5 2.2.0 runtime. The exact rebuilt solver DLL was preloaded and the GPU tests passed; this does not close general HDF5 ABI compatibility. The installed solver and external antenna files were not replaced.

## Delivery status

| Milestone | Code / numerical status | Performance status on the tested GPU |
|---|---|---|
| M1 | Direct and per-region paths implemented; mapping, state, fallback, reset and full convergence pass | Prefix-selected direct path passes full-length acceptance; per-region timing remains diagnostic |
| M2 | Reference core retained after inspection; new fused current shader reuses three center loads | No isolated load-reuse gain established; its combined cost is included in M3 |
| M3 | Unified and boxed fusion implemented; ownership, ordering, synchronization, dense/palette and full convergence pass | Unified fusion passes full-length acceptance; boxed timing remains diagnostic |

All local checks available for this delivery are complete. Release defaults remain indexed and selection remains in the native experiment harness. Native SASS/contraction inspection, reliable spill/occupancy metrics, verified Nsight source correlation, independent review, Linux and additional-vendor validation remain open. The existing MinGW backend test-link failure is recorded above. These limits do not change the successful full-simulation numerical comparison or the measured result on this Windows/MSVC/RTX 3070 configuration.

The subsequent [independent physics audit](VULKAN_PHYSICS_VALIDATION.md) passes analytical waveguide, dielectric propagation, radiation/power and full CPU comparison checks. Indexed and fused paths also produce identical saved values across the entire 33,016,225-cell, 43,259-step refined simulation. Spatial refinement changes incident-normalized radiated power by 17.9% at 77 GHz, so the original antenna mesh does not meet the audit's convergence targets. Numerical equivalence of the optimization and physical mesh convergence are separate acceptance conditions.
