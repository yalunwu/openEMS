# openEMS

**openEMS** is a free, open-source FDTD (Finite-Difference Time-Domain) electromagnetic field solver.
It is the core simulation engine of the [openEMS-Project](https://github.com/thliebig/openEMS-Project)
and is used together with [CSXCAD](https://github.com/thliebig/CSXCAD) for geometry and material description.

- **License:** GPLv3
- **Website:** https://openEMS.de
- **Docs:** https://docs.openems.de
- **Discussions:** https://github.com/thliebig/openEMS-Project/discussions

---

## Features

- 3-D FDTD solver for electromagnetic wave propagation
- Cartesian and cylindrical (including multi-grid) coordinate systems
- SIMD-accelerated engines (SSE2, multi-threaded)
- Uniaxial PML and Mur ABC absorbing boundary conditions
- Total-field / scattered-field (TFSF) excitation
- Lumped RLC elements
- Dispersive materials: Drude, Lorentz, Debye (full multi-polar)
- Near-field to far-field (NF2FF) transformation (standalone tool + library)
- SAR calculations (1g / 10g averaging)
- HDF5 and VTK field dump output
- Scripting via **Octave / Matlab** and **Python** (Cython bindings)

---

## Repository Layout

```
openEMS/
├── openems.h / openems.cpp   # Top-level openEMS class (orchestrator)
├── main.cpp                  # openEMS binary entry point
├── FDTD/                     # Operator + Engine (Yee-mesh EC coefficients, time-stepping)
│   └── extensions/           # Physics add-ons (PML, excitation, lumped elements, …)
├── Common/                   # Processing classes (field dumps, ports, NF2FF, SAR)
├── tools/                    # Utility helpers (array types, constants, …)
├── nf2ff/                    # Near-field to far-field tool (libnf2ff + nf2ff binary)
├── matlab/                   # Octave/Matlab scripting interface
└── python/openEMS/           # Cython bindings + pure-Python helpers (ports, automesh, …)
```

---

## Building

openEMS is normally built as part of the **openEMS-Project** meta-repo using the
`update_openEMS.sh` script, which handles all dependencies automatically.
See the [openEMS-Project README](https://github.com/thliebig/openEMS-Project) for the
recommended full-stack build.

### Stand-alone CMake build

Required dependencies: CSXCAD, fparser, TinyXML, HDF5, VTK, Boost (≥ 1.46,
components: thread, date\_time, serialization, chrono, program\_options).

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=~/opt/openEMS \
  -DFPARSER_ROOT_DIR=~/opt/openEMS \
  -DCSXCAD_ROOT_DIR=~/opt/openEMS
make -j$(nproc)
make install
```

Developer-local path overrides go in `localConfig.cmake` (gitignored) at the repo
root — this avoids having to pass `-D` flags every time:

```cmake
# localConfig.cmake
SET(CMAKE_INSTALL_PREFIX  "$ENV{HOME}/opt/openEMS")
SET(FPARSER_ROOT_DIR      "$ENV{HOME}/opt/openEMS")
SET(CSXCAD_ROOT_DIR       "$ENV{HOME}/opt/openEMS")
```

The build produces `libopenEMS.so` and the `openEMS` binary, both installed under
`<prefix>/lib` and `<prefix>/bin`.

### Octave post-install

After `make install`, run the Octave setup once to compile the HDF5 helper:
```bash
octave --no-gui --eval "setup()"
```

### Python bindings

The CSXCAD Python module must be installed first (from the CSXCAD repo):
```bash
python3 -m venv ~/.venv/openEMS
source ~/.venv/openEMS/bin/activate

# 1. Install CSXCAD Python module
export CSXCAD_INSTALL_PATH=~/opt/openEMS
cd /path/to/CSXCAD/python && pip install .

# 2. Install openEMS Python module
export OPENEMS_INSTALL_PATH=~/opt/openEMS
cd /path/to/openEMS/python && pip install .
```

Verify from outside the source tree:
```bash
python3 -c "import CSXCAD; print(CSXCAD.__version__)"
python3 -c "import openEMS; print(openEMS.__version__)"
```

---

## Getting Started

After installation, follow the official tutorials to run your first simulation:

- **Octave/Matlab tutorials:** https://docs.openems.de/en/latest/octave/Tutorials/index.html — also available locally in `matlab/Tutorials/`
- **Python tutorials:** https://docs.openems.de/en/latest/python/openEMS/Tutorials/index.html — also available locally in `python/Tutorials/`

---

## Testing

**Octave smoke-test** (fastest sanity check):
```bash
cd .github/smoketests/octave
octave MSL_NotchFilter.m
```

**Python smoke-test:**
```bash
cd .github/smoketests/python
python3 MSL_NotchFilter.py
```

**Full Octave test suite:**
```bash
cd TESTSUITE
octave --no-gui run_testsuite.m          # add --list, --all-engines or a test name
```
It reports PASS/FAIL per test and exits non-zero on failure. See
[TESTSUITE/README.md](TESTSUITE/README.md).

GPU regression tests compare the basic, reference and device engines. A device
acceptance run must require its backend so a CPU fallback cannot pass:

```bash
python3 python/Tests/GPU_Engine.py --require-backend HIP
python3 python/Tests/GPU_Setup.py --require-backend HIP
python3 -m openEMS.compare run_a run_b --rtol 1e-4 --l2-rtol 1e-3
```

The comparator requires matching HDF5 frequency/mesh attributes and exact HDF5
or VTR coordinates. NF2FF radiated power/directivity use numerical tolerances.
Non-finite data fails both comparison gates.

GPU operator construction defaults to CPU. Select `--engine=gpu --gpu-setup=gpu`
(Python: `engine='gpu', gpu_setup='gpu'`; Octave/MATLAB: the same CLI options
string) to require HIP construction, or `gpu_setup='auto'` to rebuild on the
CPU when preparation is unsupported or runs out of construction resources.
HIP construction uses stock CSXCAD and fparser. It covers Cartesian and
cylindrical meshes, including nested cylindrical multigrid, with constant
material/source/dispersion inputs. Coordinate-independent expressions are
evaluated once with the host parser and uploaded as constants. Built-in geometry
includes Box, Sphere, Cylinder, LinPoly, Polygon, MultiBox, RotPoly,
SphericalShell, CylindricalShell, Wire and Point. Metal and excitation Curves use
ordered mesh paths. Transforms follow CSXCAD's full matrix inversion and
first-three-row evaluation without perspective division.
Gaussian, sinusoidal, Dirac and step waveforms, default UPML grading, sources,
RC/RLC, Mur/local absorbers, multiple TFSF sources, Lorentz/Drude, Debye,
conducting sheets, steady-state tables, cylindrical axis corrections and
multigrid interpolation are constructed on HIP. Their persistent results and
histories pass directly to the engine. Core/UPML palettes preserve the existing
compression threshold and 16/32-bit indexing.

Spatial weights, UserDefined geometry, native/imported polyhedra, discrete/voxel
materials, file-weighted sources, spatial/file-based mode matching, custom
waveforms and custom PML grading require CPU construction. `auto` reports the
unsupported input and rebuilds a fresh CPU operator while retaining HIP
stepping; strict `gpu` reports an error before device construction.
HIP construction checks its 32-bit sampling and extension indices before
allocation. The engine retains upstream's 32/64-bit stepping support; device
handoff widens construction indices where the stepping kernels require it.
Models outside the construction index range fail in strict mode and use fresh
CPU construction in `auto`.
Model generation, constant-expression parsing, scalar orchestration and file
encoding remain host work. Core/material getters materialize requested data.
Probe/box snapping, dump subsampling and resolution selection, E/H sampling maps,
constant-function mode matching and normalization use HIP. Sampling indices
pass directly to device E/H and SAR consumers; requested mesh coordinates are
read for file encoding. SAR cell density, conductivity, volume and E-field DFT
samples are prepared on HIP. Standalone SAR/NF2FF calculation remains host work.
CPU processing of requested J/D/B or custom SAR outputs uses explicit field and
bounded material reads. A late C++ material debug dump can evaluate requested
materials on the host if they were not retained. These explicit output paths are
reported by diagnostics.
Custom C++ primitive/property/extension subclasses require explicit adapters;
base-class compatibility is never inferred. Non-HIP backends retain CPU
construction. Runtime device faults and invalid numerical results remain errors.
The implementation and validation gaps are tracked in [GPU_SETUP_STATUS.md](GPU_SETUP_STATUS.md).
`OPENEMS_GPU_SETUP_MEMORY_LIMIT`
caps the construction budget in bytes (including coefficient packing, both
core field pairs and reserved headroom).
Use `numThreads=1` / `--numThreads=1` for serial CPU construction; `0` selects
available CPUs. Cartesian CPU setup parallelizes averaging, coefficients and PEC.
Verbose output (`verbose=1` / `-v`) reports setup phase times and process peak
resident memory (zero means unavailable). `OPENEMS_GPU_DIAGNOSTICS=1` reports
engine batches and HIP range/full-field
download counts, bytes and work-stream waits, including setup waits. Snapshot,
DFT and energy downloads are separate paths and are excluded from the field
transfer counters. Construction separately reports H2D/D2H bytes, owned setup
peak bytes, device/runtime/compiler information and reserved memory. Sampling
maps report their input transfers, output/scratch bytes and requested readbacks.
The whole-context setup report includes all HIP buffer allocations and transfers
on the shared hierarchy context through the first voltage update, including
requested time-zero output. Its device peak counts application buffers; driver
allocations are outside that counter. Pinned host peaks are reported separately.
Keep `OPENEMS_HIP_KERNEL_TIMES` disabled for throughput runs;
it synchronizes after each kernel.
Unprofiled HIP construction phase times include kernel work and readback.
Separate profiling runs with `OPENEMS_HIP_KERNEL_TIMES=1` report core and
extension construction kernels. Sparse extension outputs use measured counts;
resource shortages in `auto` rebuild the operator from the beginning.
`python/Tests/GPU_Setup_Benchmark.py --output <directory>` records repeated
process-cold/warm CPU/HIP setup and total wall times with output comparisons.
Run it again with `--diagnostics` in a separate directory for instrumented data.

An explicit Octave/Matlab HIP fixture is in `tests/octave/GPU_Setup.m`. Add that
directory and the installed scripting interfaces to the search path, then call
`GPU_Setup` with a new output directory. The regular CPU engine sweep does not
select HIP.

**Python unit tests:**
```bash
python3 -m unittest discover -s python/Tests -p "test_*.py" -v
```

Configure `-DOPENEMS_BUILD_TESTS=ON` and run `ctest --output-on-failure` for the
native geometry and curve-path comparisons. A HIP build also runs the device
coefficient-packing test, including bitwise decoding, index-width transitions
and dense fallback. The HIP tests require a working device; CPU tests do not
validate kernel compilation or execution.

---

## Contributing

Pull requests are welcome. [AGENTS.md](AGENTS.md) has the working rules —
architecture, build and test cycle, code style, changelog and commit
conventions — and applies to human and AI-assisted contributions alike.
Disclosure and sign-off requirements are in [AI_POLICY.md](AI_POLICY.md).

---

## License

openEMS is free software licensed under the **GNU General Public License v3**.
See [COPYING](COPYING) for the full text.

Copyright © 2010-2026 Thorsten Liebig and contributors.
