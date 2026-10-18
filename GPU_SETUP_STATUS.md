AI disclosure: Implementation and validation notes prepared with Codex.

# HIP setup with stock dependencies

The user-approved scope on 2026-10-18 keeps CSXCAD and fparser unchanged.
The solver is on `codex/gpu-setup-upstream`, based on upstream
`ff68e3d8d355df5ac5f0a4e09ea3be6965232119`, after the GPU engine merge.
CPU construction remains the default. GPU stepping retains upstream behavior.

## Supported GPU construction

- Cartesian and cylindrical meshes, axis/seam rules and nested cylindrical
  multigrid, with both material averaging methods and shared timestep selection.
- Box, Sphere, Cylinder, Polygon, LinPoly, MultiBox, RotPoly, SphericalShell,
  CylindricalShell, Wire and Point, plus ordered metal/excitation Curve paths.
  Transforms follow stock CSXCAD semantics. Candidate order follows the stock
  sorted primitive query used by the CPU operator.
- Constant materials, density, dispersion and source weights. Coordinate-free
  expressions are evaluated once by the stock host parser and uploaded as
  constants; no parser bytecode or CSXCAD input-export API is required.
- Gaussian, sinusoidal, Dirac and step waveforms. Default UPML grading is
  implemented directly in openEMS and checked against the stock parser.
- UPML, sources, multiple TFSF sources, Mur/local absorbers, RC/RLC,
  Lorentz/Drude, multipole Debye, conducting sheets and steady-state tables.
- Device-owned coefficients, palettes, extension tables, initialization,
  histories and fusion metadata, with explicit materialization for host callers.
- Probe/dump snapping and mesh selection, E/H sampling and DFT maps,
  constant-function mode matching, material dumps, constant-material SAR
  (including 1 g/10 g) and NF2FF inputs. File encoding and standalone SAR/NF2FF
  post-processing stay on the host.

## Inputs requiring CPU construction

Spatial material/source/dispersion/density weights, UserDefined geometry,
native/imported polyhedra, discrete/voxel materials, file-weighted sources,
spatial/file-based mode matching, custom waveforms and custom PML grading are
rejected before device construction. `gpu_setup='auto'` reports the input,
rebuilds a fresh CPU operator and retains HIP stepping. Strict `gpu` reports an
error. Custom C++ types still require explicit adapters and paired extensions.

Recoverable resource failures also rebuild from scratch in `auto`. Runtime
device faults and invalid numerical constants remain errors. Cancellation
restores primitive-used flags and permits reuse. Construction indices retain
their checked 32-bit limit; handoff widens indices for upstream 64-bit stepping.
CPU construction retains upstream large-grid support.

## Validation on stock CSXCAD/fparser

Fresh native libraries and both binding sets are installed under
`build/gpu-standalone/msvc/`, separately from earlier builds. Windows uses an
RTX 3070 (SM 86), MSVC 19.44, CUDA 12.6 and HIP 6.2.4 headers. Both local Python
environments retain DLL search-directory handles for the new prefix and shared
dependency directory; this test bootstrap does not change either dependency.
A separate GCC 13.3 Linux CPU build uses a fresh prefix under `/tmp`.

- Windows native/device solver contracts: 12/12; stock CSXCAD contracts: 2/2.
- Linux CPU solver contracts: 7/7; stock CSXCAD contracts: 2/2.
- CPython 3.13 and 3.14: each passes 150 CSXCAD and 313 openEMS unit tests,
  plus the official Python notch-filter smoke. Skips cover optional SciPy,
  filesystem symlinks and the Linux `/proc` thread test.
- All 30 CPU/reference/HIP engine comparisons pass, including the
  100,000-step coaxial case, full/indexed coefficients, 32/64-bit stepping,
  snapshots, host DFT and SAR.
- The full strict construction matrix passes, including constant-material
  SAR/NF2FF, nested multigrid, all extensions and sampling maps. All 21
  removed-input fixtures pass fresh CPU fallback and strict rejection checks.
  Memory limits, partial-hierarchy failure, invalid constants, cancellation
  and reuse pass. Native tests also cover UserDefined/custom-PML rejection.
- Five strict construction cases pass with 64-bit stepping. NF2FF passes
  with fused and separate updates/UPML.
- The official Octave notch-filter smoke and explicit constant-expression
  Cartesian/cylindrical HIP construction comparisons pass. The isolated
  Octave PATH includes a trailing separator for stock executable lookup.

Logs and the final validation record are kept under `build/gpu-standalone/` in
the meta-repository. AMD/ROCm, Metal and older NVIDIA hardware remain untested.
This SM-86 validation build has not been packaged or benchmarked across the
other architectures.

## Preserved earlier work

The complete implementation before the scope reduction is backed up, with
verified hashes, in `build/gpu-standalone/pre-scope-20261018/source.zip` and
`state.json`. CSXCAD and fparser changes are preserved in their own Git stashes;
the exact stash IDs are recorded there. Neither dependency has local changes.

The earlier upstream-port build and validation remain under `build/gpu-rebase/`.
The original implementation on `2a0e069` remains on
`codex/gpu-setup-original-base`, with its build, source archives and release
preview under `build/gpu-setup/` and `build/release/`. Those builds used modified
dependencies and cover a broader scope. Their test counts and
[historical measurements](GPU_SETUP_BENCHMARKS.md) are not validation or timing
claims for the current build. Existing release files have been preserved.
