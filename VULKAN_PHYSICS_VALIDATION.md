# Independent Vulkan physics validation — 2026-10-11

The optimized solver passes independent analytical and CPU checks. The antenna does not meet every stated mesh-convergence target.

This extends the memory-path equivalence checks with independent analytical references, a full CPU comparison, antenna convergence studies and a frequency-domain power budget. No solver code or installed files were changed for this audit. All simulation copies and logs are under `build/physics-validation-20261011` in the parent workspace.

Tested device: NVIDIA GeForce RTX 3070, 8 GiB, driver 617.42. Native Release DLL SHA-256: `8f533a56d5268915667ddb9062900f19a1eabbe5d3cc1a5c3e6893f247b0a4e5`. Every successful native run retains executable/DLL/XML hashes, command, backend selection and complete timestep count in `result.json`. `support-check.log` verifies these hashes.

| Independent check | Observed result | Acceptance |
|---|---|---|
| WR-42 TE10, x/y/z, CPU and all five GPU paths (18 runs) | Impedance error 1.322%; phase error 0.007284 rad; reflection −43.634 dB; transmission error 0.0703%; lossless power defect 0.1445% | <3%; <0.05 rad; <−35 dB; <1%; <2% |
| Air-filled coax, fused GPU | Maximum reflection −50.9 dB; transmission approximately 0 dB; analytical impedance assertion passes | <−50 dB; transmission −0.1 to +0.05 dB; impedance <5% |
| Uniform RO3003 guide, CPU and fused GPU | Complex impedance error 0.574%; attenuation error 0.279%; phase error 0.01634 rad | <3%; <2%; <0.05 rad |
| Uniform FR4 guide, CPU and fused GPU | Complex impedance error 0.661%; attenuation error 0.819%; phase error 0.02566 rad | <3%; <2%; <0.05 rad |
| Half-wave dipole, CPU and fused GPU | Directivity 1.658 versus 1.64; radiated/accepted power 0.999; normalized pattern error 0.007; surface/far-field power ratio 0.9956 | Directivity and efficiency within 5%; pattern <0.05; flux agreement within 2% |
| Quarter-wave monopole, CPU and fused GPU | Directivity 3.332 versus 3.28; radiated/accepted power 0.968; normalized pattern error 0.013 | Directivity and efficiency within 5%; pattern <0.05 |
| Full antenna CPU versus fused GPU | Largest relative L2 error 1.2573075e-05 (0.00125731%) across 20 outputs | ≤0.1% |

The waveguide compares measured voltage/current impedance and transmission phase with continuum TE10 formulas; it does not compare the analytic port reference impedance with itself. Every candidate reports its requested mode as selected. The new benchmark also passes direct Python execution and the simulation-script contract tests.

The dielectric guide checks the same constant permittivity and conductivity values used by the antenna, against analytical complex propagation and impedance over 20–26 GHz. It tests the implemented material model, not the real materials' frequency dependence.

The CPU antenna run covers all 17,400 timesteps of the original converged GPU run, using the same mesh and timestep. A fixed length makes every port timestamp and all twelve complex 77 GHz E/H surface arrays directly comparable. The eight complete port traces and twelve surface outputs all pass; this is CPU stepping, not CPU operator setup feeding GPU stepping. These audit runs overlapped and are not performance benchmarks.

The entire 33,016,225-cell refined simulation was independently repeated with the original indexed GPU path, using the same eight-thread CPU construction and −60 dB stopping criterion. All eight port traces, twelve near-field surfaces and two dielectric-loss field volumes pass with maximum relative L2 error 0; all 22 outputs have identical saved values. This isolates the large coarse/fine difference from the memory optimization.

Antenna sensitivity is evaluated at 77 GHz. Thresholds were set in `tools/analyze_vulkan_physics.py` before inspecting the results: maximum absolute complex S-parameter change 0.005 (0.5% of incident amplitude), incident-normalized radiated-power change 1%, and maximum normalized far-field amplitude change 0.01. Each quantity has its own gate.

| Full-run variation | Cells / steps | S change (% incident amplitude) | Radiated-power change (%) | Pattern change (%) | Result |
|---|---|---|---|---|---|
| Original −40 dB versus −60 dB | 4,168,980 / 17,400 | 0.299207 | 0.0958593 | 0.387182 | Pass |
| PML 8→16 cells; inner PML boundary preserved | 5,655,188 / 22,560 | 0.00455907 | 0.00802691 | 0.0873458 | Pass |
| 16 extra air cells on each outer face | 7,436,820 / 22,560 | 0.00543071 | 0.0113406 | 0.125361 | Pass |
| Timestep multiplied by 0.5 | 4,168,980 / 45,120 | 0.0576274 | 0.0572389 | 0.0711406 | Pass |
| Every cell split in two on all three axes; PML 8→16 | 33,016,225 / 43,259 | 21.3188 | 17.9025 | 39.753 | Outside target |

The −60 dB baseline stops at 22,560 steps. The boundary, half-timestep and refined runs all reach their −60 dB criterion; none is a partial 4,096-step comparison. Thicker PML adds eight outer cells before increasing PML to sixteen, preserving the original inner absorber boundary. The clearance test moves the absorber outward while retaining eight PML cells. Refinement preserves geometry and physical absorber thickness. Its smaller timestep comes from the solver stability calculation.

The [refinement control check](../build/physics-validation-20261011/refinement-check.json) confirms byte-identical physical-property/source XML, identical outer bounds and inner PML boundaries, and exact preservation of every original mesh line. Only midpoint lines are inserted. This rules out an accidental geometry or material edit during refinement.

![Full antenna convergence](../build/physics-validation-20261011/physics-convergence.png)

![Far-field mesh sensitivity](../build/physics-validation-20261011/physics-pattern-cuts.png)

| Antenna power budget at 77 GHz | Radiated / accepted (%) | Dielectric loss / accepted (%) | Residual (%) |
|---|---|---|---|
| antenna-minus60 | 92.361 | 5.56275 | 2.07629 |
| antenna-halfdt | 92.3497 | 5.56157 | 2.08869 |
| antenna-refined-cpu-setup | 90.7362 | 5.09099 | 4.17282 |

Accepted power is the sum of net voltage/current power entering through all four ports; this accounts for energy leaving through passive ports. Radiated power comes from the closed NF2FF surface. Loss uses ½∫κ|E|² dV over RO3003 and FR4, with cell-center FD E samples and cell volumes clipped to each material box. This is an approximate continuum quadrature, not an exact discrete-energy identity. Its diagnostic target is a residual within 5% of accepted power. Fourier amplitudes retain the common pulse spectral scaling; ratios cancel that scaling, and the raw spectral powers should not be interpreted as calibrated watts.

Two execution issues are retained, not counted as passing tests:

- The HDF5-mode waveguide XML rounds its zero-thickness source plane from the nearest mesh coordinate 4163.78413888889 to 4163.784 drawing units. The exported source is unused and both CPU and GPU produce zero excitation. That attempt was stopped. The analytical-mode fixture uses exactly representable integer drawing coordinates, and the runner now rejects unaligned exported source planes before starting. `export-alignment.json` retains the mismatch.
- Strict GPU construction of the 33M-cell refined mesh exceeds the construction memory budget. Retrying with eight-thread CPU construction succeeds and explicitly selects fused GPU stepping. The failed attempt and successful retry have separate directories.

The stock suite launcher also encountered restricted temporary-directory permissions in this sandbox. The new test passes when executed directly through Python and through the native validation runner; the stock-launcher log is retained.

Reproduction uses `tools/validate_vulkan_physics.py` with the staged native executable, matching bindings and an isolated output directory. `benchmark SCRIPT.py` preserves each existing assertion and exports its setup to the selected native engine. `antenna MODEL.xml` supports the criterion, refinement, timestep, PML and clearance variations above. `tools/analyze_vulkan_physics.py` computes the comparisons and returns a failing exit status when a preset numerical target is exceeded. The exact commands are retained in the result JSON files and `gpu-checks.json` / `waveguide-candidates.json`.

Evidence: [machine-readable analysis](../build/physics-validation-20261011/analysis.json), [hash and candidate checks](../build/physics-validation-20261011/support-check.log), [direct Python test](../build/physics-validation-20261011/python-direct.log).

These results apply to the tested models, device and build. Two meshes measure sensitivity; they do not establish an asymptotic convergence order. This audit does not constitute a proof for every extension combination, GPU vendor or antenna material model.

The original antenna mesh should not be signed off at the stated convergence targets. The implementation passes the independent physics and CPU checks, while the spatial discretization still changes the physical observables beyond the chosen limits. A further mesh study should focus on the feed, metal edges and thin dielectric layers before treating the antenna response as accurate to these targets.
