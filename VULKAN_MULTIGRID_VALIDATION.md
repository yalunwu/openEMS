# Vulkan cylindrical multigrid validation

Validated on 2026-10-01 with a Release build, MinGW GCC 16.1, and an NVIDIA
GeForce RTX 3070 (driver 591.74) on Windows. All levels use one Vulkan device.

Existing cylindrical `MultiGrid` settings now select Vulkan automatically when
the Vulkan engine is requested. The existing 20-split hierarchy limit remains;
storage-buffer, dispatch, index, and allocation limits can still require CPU
fallback. No simulation CLI, XML, MATLAB, or Python options were added.

## Correctness and lifecycle

- Complete native backend suite: **49 passed, 0 failed**.
- CTest: **1/1 passed**, running the complete native backend suite.
- Complete Python GPU suite: **20 passed**, loading the local build DLL.
- Vulkan-disabled compilation: `engine_vulkan.cpp` passed a syntax-only check.
- `git diff --check`: passed.

Native multigrid cases compare every voltage/current component at every level,
including the projected outer fields, over batches of 1, 7, and 19 timesteps.
Coverage includes closed alpha and the innermost r=0 grid, two nested splits,
nonuniform open-alpha interpolation, outer radial UPML, child axial UPML/Mur,
cloned Debye material, reset/reinitialization, and five-level execution.
The maximum observed multigrid field error was **1.08e-8**. All comparisons
passed both the absolute 1e-4 and peak-relative 0.1% limits.

The Python cases run 160 timesteps and compare nonzero voltage probes in inner,
middle, and outer radial regions. The nested case also compares HDF5 field dumps
from all three levels, including output timesteps and finite, nonzero fields.
Probe errors passed 0.1%; dump errors passed 1e-4 absolute and 0.1% relative.

Rejection coverage includes an unknown innermost extension, overflowing/zero
dimensions, out-of-range interpolation indices, and non-finite interpolation
coefficients. Successful execution after failed initialization is tested, as are
child timestep propagation and clearing previously nonzero fields on reset.

## Non-gating throughput

Run `test_backend --multigrid-benchmark` from the build directory. This is a
native test utility option, not a new simulation option.

All three cases use the same outer input grid: 41 radial, 129 angular, and 17
axial lines, with closed alpha. Each timing measures 1,024 GPU timesteps and
one final full-field synchronization. Initialization and CPU reference checks
are excluded. Cell throughput uses the operator's existing cell-count metric.

| Levels | Split radii | Operator cells | Seconds | Timesteps/s | MCells/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | none | 90,610 | 0.163622 | 6,258 | 567.1 |
| 2 | 24 | 63,410 | 0.213981 | 4,785 | 303.4 |
| 5 | 8, 16, 24, 32 | 35,258 | 0.286569 | 3,573 | 126.0 |

These are short, non-gating measurements, not large-model runtime predictions.
GPU clock state, grid shape, extensions, batch size, and output frequency affect
throughput. In particular, the active cell counts differ with refinement.

## Host-transfer audit

Source inspection covered the `IterateTS` per-step loop and its voltage,
current, interpolation, projection, and timestep-recording helpers. None contain
field copies, host memory mappings, staging-buffer accesses, or field-sync calls.
Excitation waveforms and point metadata are uploaded during initialization and
evaluated on the GPU. Parent/child interface transfers and final projections are
compute dispatches operating on device buffers.

Explicit field edits are uploaded before the batch, not inside its timestep
loop. Requested probes and full-field dumps are synchronized outside that loop.
Queue submissions, fences, and push constants remain per-timestep CPU work.
This was a source audit, not a PCIe profiler capture. The Khronos validation
layer was not installed on this machine, so a validation-layer run was not done.

## Reproduction

With the existing Vulkan toolchain and Python dependencies available:

```text
cmake --build build
build/test_backend
ctest --test-dir build --output-on-failure
python python/Tests/test_gpu_engine.py
build/test_backend --multigrid-benchmark
```

On Windows use the `.exe` executables and put the native dependency DLL
directories on `PATH`; the Python GPU tests prefer `build/libopenEMS.dll` over
an older installed backend. Raw validation logs are in `build/multigrid-*.log`.

Multi-GPU, MPI+Vulkan, and manual GPU selection remain out of scope. CPU
cylindrical-multigrid extension compatibility rules remain unchanged.
