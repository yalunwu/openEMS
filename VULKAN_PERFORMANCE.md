# Vulkan batching and profiling

The Vulkan backend records up to 32 timesteps per queue submission by default.
The limit controls submission size; the processing scheduler still determines
when probes, dumps and end-criteria checks must run. Steady-state detection keeps
its existing one-step sampling interval. Multigrid projects fields at the end of
the requested iteration interval, rather than at every internal submission.

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
submissions, dispatches, field upload/download bytes, probe result bytes and CPU
recording/submission/wait/readback/mirror costs. Initialization is excluded.
Readback costs include their waits, so CPU timing columns can overlap. Probe
bytes represent GPU writes to mapped host-visible memory, not measured PCIe
traffic. Counts are collected only when profiling is enabled.

GPU timestamps measure each timestep submission and sampled dispatches from the
first step in that submission, plus final multigrid projection, probe gathering
and field readbacks. Dispatch categories are voltage, current, extension and
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

## Benchmark utility

From a directory reserved for benchmark output, run the newly built
`test_backend` executable. Ensure its matching dependency DLLs are available.
On Windows the commands use `test_backend.exe`.

```text
test_backend --vulkan-benchmark --csv=matrix.csv
test_backend --vulkan-benchmark --case=large --steps=1024 --repeats=5 --csv=large.csv
test_backend --vulkan-benchmark --case=multigrid-5 --batch-size=32 --profile --csv=profile.csv
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
`steady-state`. The first eleven time stepping directly without field downloads;
repeats continue the warmed simulation. The last three run fresh fixed-length
solver instances and include actual processing/output and final field readback.
Output filenames use the `vulkan_benchmark_` prefix and can be overwritten on
repeated benchmark runs. Initialization timing for direct stepping is shared by
that fixture's repeats; output cases initialize a fresh fixture for each repeat.

Use fixed timesteps (`EndCriteria=0`) for throughput. The output cases also enable
`--exact-endcriteria` to exercise a deterministic energy-check schedule. Hardware
load, clocks, compiler, driver and output filesystem all affect these numbers.
Run performance measurements separately from other GPU tests and compiler jobs.

`test_backend --batching-tests` runs focused CPU/GPU equivalence checks for
submission limits, zero calls, tails, continuation, delays, field edits, profiling
toggles, pending resets and failure propagation. The full native and Python GPU
suites additionally cover supported physics, geometries, probes and dump timing.

## References

- [Khronos timestamp queries](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html)
- [Timestamp command semantics](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html)
- [Query-result availability](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html)

Measured results and configuration coverage are recorded in
`VULKAN_PHASE01_VALIDATION.md`.
