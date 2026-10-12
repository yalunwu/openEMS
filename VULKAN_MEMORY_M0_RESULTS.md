# M0 memory profiling implementation — 2026-10-11

Implementation and local validation owner: Codex. Independent review,
Linux and additional GPU/vendor validation: unassigned/unverified.

M0's instrumentation is implemented and locally validated. Named-pass
attribution and optimized SPIR-V inspection are available. NVIDIA native
producer instructions, source-line correlation, registers, spills and resource
limits remain unverified. M1-M3 execution candidates are not implemented or
accepted by this delivery; there is no memory-optimization speedup claim.

## Scope and reproducibility

Profiling now splits the legacy extension aggregate into UPML pre/post voltage,
UPML pre/post current, excitation and other extensions. The aggregate reuses
leaf samples without additional timestamp spans. Optional debug-utils names,
dispatch labels and optimized shader debug information are enabled when
profiling is selected before initialization. Missing debug functions remain
nonfatal. Profiling-disabled dispatches add no labels, queries or counters.

The tested source was `0e8de576b6a93ce4575d72cfbebfb0290fa03d96` plus the
instrumentation/test diff retained with each antenna result. The MSVC DLL's
SHA-256 is `93f401acfc86698139a0a3910f0956a671e3ef706a843b22ab02aa47a10a72f2`.
The cached version banner still says `Vulkan-Test-19-g125b384`; identify this
build by the recorded source diff and binary hash, not that banner.

Hardware: RTX 3070 / GA104, driver 617.42, Windows 10. Nsight Graphics
2026.3.1 used unaltered clocks; the idle clock snapshot is not a sustained-run
clock measurement. Additional GPUs and Linux were not tested.

All fresh antenna runs used workspace copies of the original XML, CPU operator
setup with one thread, dense coefficients, GPU FD accumulation, batch limit 32
and exact end-criteria checks every 120 steps. The original XML SHA-256 is
`b8c01e6a7f0f36aeb488a0188192e9e97857c239c2e3692b93981511a92c6ab5`.
The original antenna project and its solver installation were not modified.
Commands, XML/DLL hashes, source diffs and raw outputs are in
[`build/memory-m0-20261011`](../build/memory-m0-20261011/).

## Pass attribution

The converged profiling run recorded 581 sampled dispatches per listed pass.
These are diagnostic samples, not full-run pass totals or additive wall time.

| Pass | Sampled total (ms) | Mean sampled dispatch (ms) |
|---|---:|---:|
| Yee voltage | 354.22 | 0.610 |
| Yee current | 347.75 | 0.599 |
| UPML pre voltage | 362.13 | 0.623 |
| UPML post voltage | 224.38 | 0.386 |
| UPML pre current | 367.59 | 0.633 |
| UPML post current | 224.63 | 0.387 |
| Excitation | 1.42 | 0.002 |

UPML accounts for 62.6% of this sampled Yee-plus-extension total. The short
Nsight capture independently exports the named dispatch regions and associates
the four dominant shader hashes with them through per-region residency:

| Named region | Dominant Nsight hash | Runtime SPIR-V FNV-1a |
|---|---|---|
| UPML pre, voltage and current | `671c00cae96d436c` | `1ae1d1015a435d60` |
| UPML post, voltage and current | `b9924a2cd08a7eab` | `31954668df0ef887` |
| Yee voltage | `364f29a9cf8eb4bd` | `26600281a3678775` |
| Yee current | `b6d50dc447173510` | `3b390e5a4b81a701` |

The hash association is temporal evidence from named regions, with small
cross-boundary sampling contamination; it is not proof that Nsight opened the
correct GLSL line or uses the SPIR-V identity algorithm. The capture reports
93.91% long-scoreboard/L1TEX dependency states among reported sampled warp
states. That percentage does not measure removable time or predict a speedup.

Raw capture, exported events/regimes, identity logs and analysis:
[`nsight-4096`](../build/memory-m0-20261011/nsight-4096/).
The trace contains 296 occurrences of each main phase and is a short diagnostic
window; it is not a new full-run Nsight performance baseline.

## Access and dependency inventory

Offline `glslc -O -g --target-env=vulkan1.2` compilation produced byte-identical
SPIR-V to all four runtime identities above; `spirv-val` passed. Retained GLSL,
disassembly, commands and checks are in
[`offline-spirv`](../build/memory-m0-20261011/offline-spirv/).

| Shader | Observed optimized SPIR-V access/dependency |
|---|---|
| UPML pre | One index load produces the field address: `%39 = OpLoad ... %38`, then `%60 = OpAccessChain ... %39`, then `%61 = OpLoad ... %60`. Separate coefficient-vector and flux loads feed the arithmetic; two stores update field and flux. |
| UPML post | The same index-to-field address chain remains. Coefficient-vector and flux loads plus the indexed field load feed two stores. |
| Yee voltage | Nine opposite-field loads, three own-field loads and six coefficient loads remain, followed by three field stores. Repeated center-field expressions are already reused in SPIR-V. |
| Yee current | The same static load/store counts as voltage, retaining the distinct current boundary guard. |

These are intermediate-code accesses, not native instruction counts or physical
memory transactions. The exported capture does not establish individual load
sectors, native register use, spills, occupancy limits or successful source-line
correlation. Inspect those before accepting an M2 rewrite. In particular, the
SPIR-V evidence does not support adding manual core-load reuse solely because
the GLSL repeats expressions.

The first bounded execution candidate remains M1 direct PML addressing:
remove the demonstrated index dependency for validated regular regions, then
compare indexed, per-region and unified multi-box dispatches. Require exact
address/ownership equivalence, overlap handling and measured complete-phase
cost before choosing a design. Fusion and additional core rewrites remain
separate experiments.

## Validation and timing

| Check | Result / retained evidence |
|---|---|
| MSVC Release build | Passed; `build-msvc.log` |
| Complete native backend / CTest | 76 backend tests and 9/9 CTest entries passed; `ctest-msvc.log` |
| Python GPU equivalence | 27/27 passed using the hash-verified new DLL and rebuilt CPython 3.14 bindings; `python-gpu.log`. Known HDF5 version warning remains. |
| Octave profiling/setup integration | Passed all five setup/thread selections, profiling keys, probe schedules and values; `octave.log` |
| Vulkan synchronization validation | Six synchronization and six batching/profiling fixtures passed; no VUID or synchronization hazards reported. Loaded layer DLL recorded in `validation-layer-modules.json`; `vk_layer_settings.txt` enables synchronization checks. |
| Vulkan-disabled MinGW build | Passed; `build-vulkan-off.log` |
| Vulkan-enabled MinGW | Shared-library target built. Full executable/test linking remains incomplete because of an unchanged excitation-symbol export/link failure; an export-all experiment also hit Boost duplicate symbols. Original linker flags were restored. |
| Profiling CSV contract | 75 columns, matching header/data widths, correct five detailed dispatch counts and legacy aggregate; `evidence-checks.json` |
| Antenna, 4,096 fixed steps | Eight port traces and twelve complex 77 GHz FD surfaces identical with profiling off/on; `validation-4096.json` |
| Antenna, converged criterion | Both stopped at 17,400 steps / -40.06 dB; the same twenty outputs and axes/sample times were identical; `validation-full.json` |

Output comparisons used relative L2 tolerance `1e-7`; measured maximum error was
zero. Existing native and scripting tolerances were not relaxed. The reference
here is the freshly built M0 solver with profiling disabled, not an independent
CPU antenna run or the historical installed binary.

| Run | Stepping (s) | Process elapsed (s) |
|---|---:|---:|
| 4,096 steps, profiling off | 13.657 | 58.786 |
| 4,096 steps, profiling on | 13.698 | 58.428 |
| 17,400 converged steps, profiling off | 59.403 | 104.786 |
| 17,400 converged steps, profiling on | 57.383 | 102.960 |

These are single diagnostic runs. They do not satisfy the warm, repeated,
interleaved performance-acceptance protocol and establish neither overhead nor
a gain. No execution/default optimization was selected from them.

Python validation uses a private staging prefix, hash-verified MSVC DLL and
bindings rebuilt against current headers. The old installed bindings crashed
when paired with the new class layout and were excluded. Native HDF5 2.2.0 is
preloaded consistently; h5py was built against 2.0.0 and still emits the known
version warning. This environment limitation remains open independently of M0.
All 27 Python GPU tests passed in 201.512 seconds; the runtime warning does
not establish ABI compatibility for other HDF5 workloads.
