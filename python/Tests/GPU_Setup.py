# Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
"""HIP construction versus CPU construction with the same stepping backend."""
import argparse
import os
import re
import tempfile
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from GPU_Engine import (run_captured, compare_outputs, case_excitation, case_cpu_setup,
                        case_pml, case_mur, case_lumped, case_materials, case_tfsf,
                        case_absorbers, case_steady_state, case_dumps, case_dumps_snapshot,
                        case_cylinder_closed, case_multigrid)


def geometry_case(method=3, central=False, factor=1, forced=0):
    fdtd = openEMS(NrTS=600, EndCriteria=0, TimeStepMethod=method,
                   CellConstantMaterial=central, TimeStepFactor=factor)
    if forced:
        fdtd.SetTimeStep(forced)
    fdtd.SetGaussExcite(5e9, 4e9)
    fdtd.SetBoundaryCond(['PEC', 'PMC', 'PEC', 'PEC', 'PEC', 'PMC'])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    for axis in 'xyz':
        grid.AddLine(axis, np.unique(np.r_[np.linspace(-10, 10, 21), [-7.3, -2.4, 1.2, 6.1]]))
    mat = csx.AddMaterial('anisotropic', epsilon=[2, 3, 4], kappa=[0.01, 0.02, 0.03],
                          mue=[1.1, 1.2, 1.3], sigma=[0.001, 0.002, 0.003])
    mat.SetIsotropy(False)
    mat.AddBox([-8, -8, -8], [8, 8, 8], priority=2).AddTransform('RotateAxis', 'z', 0.2, deg=False)
    csx.AddMaterial('equal_priority', epsilon=4.4).AddBox([1, 1, 1], [7, 7, 7], priority=2)
    sphere = csx.AddMaterial('sphere', epsilon=1.5).AddSphere([3, 2, 1], 3, priority=5)
    sphere.AddTransform('Scale', [1.1, 0.9, 1])
    sphere.AddTransform('Translate', [0.2, -0.3, 0.1])
    cylinder = csx.AddMaterial('cylinder', epsilon=2.5).AddCylinder([-7, -4, -4], [-3, 2, 4], 1.2, priority=6)
    cylinder.AddTransform('RotateAxis', 'y', 0.1, deg=False)
    polygon = [[-6, -2, -2, -4, -4, -6], [2, 2, 6, 6, 4, 4]]
    csx.AddMaterial('concave', epsilon=3.2).AddLinPoly(polygon, 'z', -2, 4, priority=7)
    csx.AddMaterial('negative', epsilon=2.2).AddLinPoly(polygon, 'x', 7, -2, priority=8)
    csx.AddMetal('thin').AddLinPoly([[2, 6, 6, 2], [-7, -7, -4, -4]], 'z', 0, 0, priority=9)
    source = csx.AddExcitation('source', exc_type=0, exc_val=[0, 0, 1])
    source.SetDelay(2e-12)
    source.AddBox([0, 0, -1], [0, 0, 1])
    magnetic = csx.AddExcitation('magnetic', exc_type=2, exc_val=[1, 0, 0])
    magnetic.SetDelay(3e-12)
    magnetic.AddBox([4, -2, -2], [4, 2, 2])
    csx.AddProbe('ep', p_type=2).AddPoint([5, 2, 1])
    csx.AddDump('Et', dump_type=0, file_type=1).AddBox([-10, -10, 0], [10, 10, 0])
    return fdtd, csx


def hybrid_case():
    fdtd, csx = geometry_case()
    csx.AddLumpedElement('rc', ny='z', caps=False, R=200, C=1e-12, LEtype=0).AddBox([0, 0, 0], [0, 0, 1])
    csx.AddLumpedElement('rlc', ny='z', caps=False, R=20, L=1e-9, C=1e-12, LEtype=1).AddBox([1, 0, 0], [1, 0, 2])
    csx.AddConductingSheet('sheet', conductivity=5.8e7, thickness=1e-6).AddBox([-2, 5, -2], [2, 5, 2], priority=10)
    return fdtd, csx


def upml_material_case(central=False):
    fdtd, csx = geometry_case(central=central)
    fdtd.SetBoundaryCond(['PML_4'] * 6)
    return fdtd, csx


def rlc_case(overlap=False):
    fdtd, csx = geometry_case()
    csx.AddLumpedElement('series', ny='z', caps=True, R=10, L=1e-9, C=0, LEtype=1).AddBox([1, 1, 0], [2, 2, 2])
    start, stop = ([1, 1, 0], [2, 2, 2]) if overlap else ([-5, -3, 0], [-4, -2, 2])
    csx.AddLumpedElement('parallel', ny='z', caps=True, R=50, L=2e-9, C=2e-12, LEtype=0).AddBox(start, stop)
    csx.AddLumpedElement('short', ny='y', caps=True, R=0, C=0, LEtype=0).AddBox([3, -3, 2], [5, -1, 4])
    return fdtd, csx


def delayed_sources_case(constant=False):
    fdtd, csx = geometry_case()
    if constant:
        fdtd.SetCustomExcite('1', 1e9, 9e9)
    else:
        fdtd.SetStepExcite(9e9)
    hard_e = csx.AddExcitation('hard_e', exc_type=1, exc_val=[0, 0, 0.5], delay=4e-12)
    hard_e.AddBox([-4, 0, -1], [-4, 0, 1])
    hard_h = csx.AddExcitation('hard_h', exc_type=3, exc_val=[0.25, 0, 0], delay=5e-12)
    hard_h.AddBox([4, 0, 0], [4, 2, 0])
    return fdtd, csx


def weighted_source_case():
    fdtd, csx = geometry_case()
    csx.GetPropertiesByName('source')[0].SetWeightFunction(['1', '1', '1+x/100'])
    return fdtd, csx


def weighted_dispersion_case(debye=False):
    fdtd, csx = case_materials()
    name, property_name = ('debye', 'eps_delta') if debye else ('drude', 'eps_plasma')
    csx.GetPropertiesByName(name)[0].SetDispersiveMaterialWeight(0, **{property_name: '1+z/1000'})
    return fdtd, csx


def short_debye_case():
    fdtd, csx = case_materials()
    csx.GetPropertiesByName('debye')[0].SetDispersiveMaterialProperty(0, eps_relax=1e-18)
    return fdtd, csx


def compare_case(name, case, root, required=('Excitation Extension',), **options):
    cpu, gpu = os.path.join(root, name+'_cpu'), os.path.join(root, name+'_gpu')
    a = run_captured(case, cpu, 'gpu', gpu_setup='cpu', **options)
    b = run_captured(case, gpu, 'gpu', gpu_setup='gpu', **options)
    assert 'backend: HIP' in a and 'backend: HIP' in b, 'FAIL: HIP stepping was not selected'
    assert 'HIP construction active' in b and 'rebuilding a fresh CPU' not in b, 'FAIL: device construction did not activate'
    for extension in required:
        assert f'HIP extension construction active: {extension}' in b, f'FAIL [{name}]: {extension} construction did not activate'
    if name in ('hybrid', 'rlc_caps', 'rlc_overlap'):
        assert 'HIP RC construction active' in b, 'FAIL: RC construction did not activate'
    diff, _, _, _ = compare_outputs(cpu, gpu, rtol=1e-4)
    assert not diff, f'FAIL [{name}]: CPU/GPU construction differs: {diff}'
    unused = lambda text: re.findall(r'Warning: Unused primitive[^\n]*', text)
    assert unused(a) == unused(b), f'FAIL [{name}]: primitive-used flags differ'
    print('PASS', name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-backend', choices=('HIP',))
    args = parser.parse_args()
    root = os.path.join(tempfile.gettempdir(), 'GPU_Setup')
    os.makedirs(root, exist_ok=True)
    log = run_captured(case_excitation, os.path.join(root, 'detect'), 'gpu')
    if args.require_backend:
        assert 'backend: HIP' in log, 'FAIL: required HIP stepping backend unavailable'
    if 'backend: HIP' in log:
        for method in (1, 3):
            for central in (False, True):
                compare_case(f'geometry_{method}_{central}',
                             lambda: geometry_case(method, central), root)
        for factor, forced in ((0.8, 0), (1, 1e-13)):
            compare_case(f'timestep_{factor}_{forced}', lambda: geometry_case(factor=factor, forced=forced), root)
        for central in (False, True):
            compare_case(f'upml_material_{central}', lambda: upml_material_case(central), root,
                         required=('Uniaxial PML Extension', 'Excitation Extension'))
        for overlap in (False, True):
            compare_case('rlc_overlap' if overlap else 'rlc_caps', lambda: rlc_case(overlap), root,
                         required=('Excitation Extension', 'Series/Parallel Lumped RLC load'))
        for constant in (False, True):
            compare_case(f'delayed_sources_{constant}', lambda: delayed_sources_case(constant), root)
        expected = {
            'excitation': ('Excitation Extension',),
            'pml': ('Uniaxial PML Extension', 'Excitation Extension'),
            'mur': ('Mur ABC Extension', 'Excitation Extension'),
            'hybrid': ('Excitation Extension', 'Series/Parallel Lumped RLC load', 'Conducting Sheet Extension'),
            'materials': ('Uniaxial PML Extension', 'Excitation Extension', 'Drude/Lorentz Dispersive Material Extension',
                          'Debye Dispersive Material Extension', 'Conducting Sheet Extension'),
            'tfsf': ('Uniaxial PML Extension', 'Total-Field/Scattered-Field Extension'),
            'absorbers': ('Local absorbing boundary condition sheet', 'Excitation Extension'),
            'periodic': ('Uniaxial PML Extension', 'Excitation Extension'),
            'dumps': ('Uniaxial PML Extension', 'Mur ABC Extension', 'Excitation Extension'),
            'snapshots': ('Uniaxial PML Extension', 'Excitation Extension'),
        }
        for name, case in [('excitation', case_excitation), ('pml', case_pml), ('mur', case_mur),
                           ('hybrid', hybrid_case), ('materials', case_materials), ('tfsf', case_tfsf),
                           ('absorbers', case_absorbers), ('periodic', case_steady_state),
                           ('dumps', case_dumps), ('snapshots', case_dumps_snapshot)]:
            compare_case(name, case, root, required=expected[name])
        compare_case('short_debye', short_debye_case, root, required=expected['materials'])
        samples = np.loadtxt(os.path.join(root, 'geometry_3_False_gpu', 'ep'), comments='%')
        assert np.any(samples[:, 1:] != 0), 'FAIL: geometry excitation was not exercised'
    else:
        print('HIP unavailable: GPU construction comparisons skipped')
    # Weighted materials and unsupported backends must recover or fail explicitly.
    fallback_cases = [(case_cpu_setup, 'gpu', 'spatial material weights'),
                                 (case_lumped, 'gpu', 'unsupported material/metal primitive'),
                                 (weighted_source_case, 'gpu', 'spatial source weights'),
                                 (case_excitation, 'gpu-reference', 'does not support GPU operator construction')]
    if 'backend: HIP' in log:
        fallback_cases += [(weighted_dispersion_case, 'gpu', 'spatial dispersion weights'),
                           (lambda: weighted_dispersion_case(True), 'gpu', 'spatial dispersion weights')]
    for case, engine, reason in fallback_cases:
        cpu, auto = os.path.join(root, 'fallback_cpu'), os.path.join(root, 'fallback_auto')
        run_captured(case, cpu, engine, gpu_setup='cpu')
        log = run_captured(case, auto, engine, gpu_setup='auto')
        assert 'rebuilding a fresh CPU operator' in log and reason in log, log
        assert not compare_outputs(cpu, auto, rtol=1e-4 if engine=='gpu' else 0)[0], 'FAIL: automatic fallback differs from CPU setup'
        try:
            run_captured(case, os.path.join(root, 'strict'), engine, gpu_setup='gpu')
        except RuntimeError as error:
            assert reason in str(error), error
        else:
            raise AssertionError('FAIL: strict GPU setup silently fell back')
    for case in (case_cylinder_closed, case_multigrid):
        cpu, auto = os.path.join(root, 'cylinder_cpu'), os.path.join(root, 'cylinder_auto')
        run_captured(case, cpu, 'gpu', gpu_setup='cpu')
        log = run_captured(case, auto, 'gpu', gpu_setup='auto')
        assert 'CPU selected (cylindrical/multigrid setup)' in log and 'HIP construction active' not in log
        assert not compare_outputs(cpu, auto, rtol=1e-4)[0], 'FAIL: cylindrical CPU setup changed'
        try:
            run_captured(case, os.path.join(root, 'cylinder_strict'), 'gpu', gpu_setup='gpu')
        except RuntimeError as error:
            assert 'requires a Cartesian GPU engine' in str(error), error
        else:
            raise AssertionError('FAIL: strict GPU setup accepted cylindrical construction')
    if 'backend: HIP' in run_captured(case_excitation, os.path.join(root, 'memory_detect'), 'gpu'):
        fdtd, csx = geometry_case()
        same_model = lambda: (fdtd, csx)
        run_captured(same_model, os.path.join(root, 'reuse_cpu'), 'gpu', gpu_setup='cpu')
        previous_limit = os.environ.get('OPENEMS_GPU_SETUP_MEMORY_LIMIT')
        try:
            # Windows removes empty values from the C runtime environment.
            invalid_limits = ('-1', ' -1', 'garbage', '18446744073709551616')
            if os.name != 'nt':
                invalid_limits += ('',)
            for invalid in invalid_limits:
                os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT'] = invalid
                try:
                    run_captured(case_excitation, os.path.join(root, 'memory_invalid'), 'gpu', gpu_setup='auto')
                except ValueError as error:
                    assert 'must be an unsigned byte count' in str(error), error
                else:
                    raise AssertionError(f'FAIL: malformed construction budget {invalid!r} was accepted')
            os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT'] = '1'
            log = run_captured(case_excitation, os.path.join(root, 'memory_auto'), 'gpu', gpu_setup='auto')
            assert 'insufficient device memory budget' in log and 'rebuilding a fresh CPU operator' in log
            try:
                run_captured(same_model, os.path.join(root, 'memory_strict'), 'gpu', gpu_setup='gpu')
            except RuntimeError as error:
                assert 'insufficient device memory budget' in str(error)
            else:
                raise AssertionError('FAIL: strict GPU memory preflight silently fell back')
        finally:
            if previous_limit is None:
                del os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT']
            else:
                os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT'] = previous_limit
        for repeat in range(2):
            path = os.path.join(root, f'reuse_gpu_{repeat}')
            log = run_captured(same_model, path, 'gpu', gpu_setup='gpu')
            assert 'HIP construction active' in log
            assert not compare_outputs(os.path.join(root, 'reuse_cpu'), path, rtol=1e-4)[0]
    print('PASS GPU construction and fallback checks')
