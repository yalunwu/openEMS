# Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
"""HIP construction versus CPU construction with the same stepping backend."""
import argparse
import os
import re
import tempfile
import numpy as np
from CSXCAD import ContinuousStructure
from CSXCAD.CSProperties import CSPropLorentzMaterial, CSPropDebyeMaterial
from openEMS import openEMS
import h5py
from GPU_Engine import (run_captured, compare_outputs, case_excitation, case_cpu_setup,
                        case_pml, case_mur, case_lumped, case_materials, case_tfsf,
                        case_absorbers, case_steady_state, case_dumps, case_dumps_snapshot, case_tfsf_multiple,
                        case_cylinder_closed, case_cylinder_wedge, case_multigrid, case_multigrid2, case_multigrid_wedge)


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


def voxel_case(central=False, database_background=False):
    fdtd, csx = geometry_case(central=central)
    filename = os.path.join(tempfile.gettempdir(), 'gpu_setup_voxel.h5')
    with h5py.File(filename, 'w') as data:
        data.attrs['Version'] = 2.0
        for axis in 'xyz':
            data['mesh/'+axis] = np.array([-6, -2, 3, 6], dtype=np.float32)
        indices = data.create_dataset('DiscData', data=(np.arange(27).reshape(3, 3, 3) % 3).astype(np.uint8))
        indices.attrs['DB_Size'] = np.int32(3)
        for name, values in dict(epsR=[2.5, 4, 6], kappa=[.02, .01, .03], density=[900, 1000, 1200]).items():
            indices.attrs[name] = np.array(values, dtype=np.float32)
    material = csx.AddDiscMaterial('voxels', filename=filename, scale=1.1,
                                 use_db_background=database_background, epsilon=7, mue=1.2, kappa=.04)
    material.GetTransform().AddTransform('Translate', [.5, -.25, .75])
    assert material.ReadFile(), 'voxel fixture could not be decoded'
    material.AddBox([-8]*3, [8]*3, priority=20)
    return fdtd, csx


def file_source_case(axis=2):
    fdtd, csx = geometry_case()
    filename = os.path.join(tempfile.gettempdir(), f'gpu_setup_source_{axis}.h5')
    lines = np.array([-8, -2, 1, 5, 8], dtype=float)
    with h5py.File(filename, 'w') as data:
        data.attrs['Version'] = 1.0
        data['x'] = lines
        data['y'] = lines
        x, y = np.meshgrid(lines, lines)
        data['Vx'] = .2 + .01*x + .005*y
        data['Vy'] = -.1 + .004*x - .01*y
    direction = [0, 0, 0]
    direction[axis] = 1
    source = csx.AddExcitation('file_profile', exc_type=0, exc_val=[1, 2, 3], weight_file=filename)
    source.SetPropagationDir(direction)
    source.SetWeightOrigin([.3, -.2, .1])
    start, stop = [-7]*3, [7]*3
    start[axis] = stop[axis] = -4
    source.AddBox(start, stop, priority=30)
    return fdtd, csx


def mode_match_case(file=False, cylindrical=False, constant=False):
    fdtd, csx = cylinder_features_case() if cylindrical else geometry_case()
    start, stop = ([5, -1, 9], [26, 1, 9]) if cylindrical else ([-7, -6, 2], [7, 6, 2])
    for magnetic in (False, True):
        probe = csx.AddProbe('mode_h' if magnetic else 'mode_e', p_type=11 if magnetic else 10)
        probe.SetModeOrigin([.3, -.2, .1])
        if file:
            filename = os.path.join(tempfile.gettempdir(), 'gpu_setup_mode.h5')
            with h5py.File(filename, 'w') as data:
                data.attrs['Version'] = 1.0
                lines = np.array([-40, -3, 1, 12, 40], dtype=float)
                data['x'] = lines
                data['y'] = lines
                x, y = np.meshgrid(lines, lines)
                data['Vx'] = .2 + .01*x + .005*y
                data['Vy'] = -.1 + .004*x - .01*y
            probe.SetModeFile(filename)
        else:
            probe.SetModeFunction(['1/2', '.3', '0'] if constant else ['1+x/100+rho/1000', '.3+y/200+sin(a)/100', '0'])
        probe.AddBox(start, stop)
    return fdtd, csx


def material_dump_case(central=False, voxel=False, cylindrical=False):
    if cylindrical:
        fdtd, csx = cylinder_features_case(central=central)
        start, stop = [0, -np.pi, 9], [40, np.pi, 9]
    else:
        fdtd, csx = voxel_case(central) if voxel else geometry_case(central=central)
        start, stop = [-8, -8, 2], [8, 8, 2]
    fdtd.SetNumberOfTimeSteps(120)
    for kind in (2, 4, 5):
        for mode in (0, 2):
            csx.AddDump(f'material_{kind}_{mode}',dump_type=kind,dump_mode=mode,file_type=1).AddBox(start,stop)
        csx.AddDump(f'material_fd_{kind}',dump_type=10+kind,dump_mode=1,file_type=1,frequency=[3e9]).AddBox(start,stop)
    return fdtd, csx


def density_case(voxel=False, cylindrical=False, weighted=False):
    if cylindrical:
        fdtd, csx = cylinder_features_case()
        material = csx.AddMaterial('density_tissue',epsilon=2.4,kappa=.03,density=1000)
        if weighted: material.SetMaterialWeight(density='1+rho/1000+z/2000',kappa='1+rho/500')
        material.AddBox([3,-np.pi,-3],[33,np.pi,27],priority=20)
        start, stop = [5,-1,2],[26,1,20]
    elif voxel:
        fdtd, csx = voxel_case()
        material = csx.GetPropertiesByName('voxels')[0]
        material.SetMaterialProperty(density=950)
        if weighted: material.SetMaterialWeight(density='1+x/100+y/200+z/300')
        start, stop = [-7,-7,-5],[7,7,5]
    else:
        fdtd, csx = geometry_case()
        material = csx.AddMaterial('density_tissue',epsilon=2.4,kappa=.03,density=1000)
        if weighted: material.SetMaterialWeight(density='1+x/100+y/200+z/300',kappa='1+z/1000')
        material.AddBox([-8]*3,[8]*3,priority=20)
        start, stop = [-6,-6,-4],[6,6,4]
    fdtd.SetNumberOfTimeSteps(4800)
    for name, kind in (('density_raw',29),('density_sar',20)):
        csx.AddDump(name,dump_type=kind,dump_mode=2,file_type=1,frequency=[3e9]).AddBox(start,stop)
    return fdtd, csx


def processing_pressure_case():
    fdtd, csx = geometry_case()
    fdtd.SetNumberOfTimeSteps(2)
    for n in range(40):
        csx.AddDump(f'pressure_{n}', dump_type=0, file_type=1).AddBox([-10]*3, [10]*3)
    return fdtd, csx


def density_averages_case():
    fdtd,csx=density_case(cylindrical=True)
    for name,kind in (('density_1g',21),('density_10g',22)):
        csx.AddDump(name,dump_type=kind,dump_mode=2,file_type=1,frequency=[3e9]).AddBox([5,-1,2],[26,1,20])
    return fdtd,csx


def sampling_selection_case(cylindrical=False,optimal=False):
    fdtd,csx=cylinder_features_case() if cylindrical else geometry_case()
    start,stop=([3,-2,-2],[28,2,22]) if cylindrical else ([-7,-6,-5],[7,6,5])
    options={'opt_resolution':[5,.9,4] if cylindrical else [3,4,3]} if optimal else {'sub_sampling':[2,3,2]}
    for magnetic in (False,True):
        csx.AddDump('selected_'+str(magnetic),dump_type=int(magnetic),dump_mode=1,file_type=1,**options).AddBox(start,stop)
    return fdtd,csx


def hybrid_case():
    fdtd, csx = geometry_case()
    csx.AddLumpedElement('rc', ny='z', caps=False, R=200, C=1e-12, LEtype=0).AddBox([0, 0, 0], [0, 0, 1])
    csx.AddLumpedElement('rlc', ny='z', caps=False, R=20, L=1e-9, C=1e-12, LEtype=1).AddBox([1, 0, 0], [1, 0, 2])
    csx.AddConductingSheet('sheet', conductivity=5.8e7, thickness=1e-6).AddBox([-2, 5, -2], [2, 5, 2], priority=10)
    return fdtd, csx


def explicit_geometry_case(shape):
    fdtd, csx = geometry_case()
    material = csx.AddMaterial('explicit', epsilon=5.1)
    options = dict(priority=20)
    if shape == 'multibox':
        material.AddMultiBox([([-6, -2, -2], [-2, 2, 2]), ([2, -2, -2], [6, 2, 2])], **options)
    elif shape == 'polygon':
        csx.AddMetal('polygon').AddPolygon([[-4, 4, 4, -4], [-4, -4, 4, 4]], 'z', 2, **options)
    elif shape == 'rotpoly':
        material.AddRotPoly([[1, 4, 4, 1], [-3, -3, 3, 3]], 'y', 0, 'z', [5, 2], **options)
    elif shape == 'spherical_shell':
        material.AddSphericalShell([0, 0, 0], 5, 2, **options)
    elif shape == 'cylindrical_shell':
        material.AddCylindricalShell([-3, -1, -4], [2, 3, 4], 3, 1, **options)
    elif shape == 'wire':
        material.AddWire([[-6, 0, 0, 6], [-2, 0, 0, 3], [-3, 0, 0, 4]], 1.1, **options)
    elif shape == 'curve':
        csx.AddMetal('curve').AddCurve([[-12, -4, 0, 4, 12], [0, -3, 0, 3, 0], [0, -3, 0, 3, 0]], **options)
    elif shape == 'point':
        material.AddPoint([0, 0, 0], **options)
    elif shape == 'cylindrical_box':
        material.AddBox([2, -0.5, -3], [6, 4.5, 3], **options).SetCoordinateSystem(1)
    else:
        raise ValueError(shape)
    return fdtd, csx


def polyhedron_case():
    fdtd, csx = geometry_case()
    polyhedron = csx.AddMaterial('imported_solid', epsilon=3).AddPolyhedron(priority=20)
    for point in ([2.13, 2.13, 2.13], [4.13, 2.13, 2.13], [2.13, 4.13, 2.13], [2.13, 2.13, 4.13]):
        polyhedron.AddVertex(*point)
    for face in ([0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]):
        polyhedron.AddFace(face)
    return fdtd, csx


def imported_polyhedron_case(ply=False):
    fdtd, csx = geometry_case()
    vertices = [[2.13, 2.13, 2.13], [6.13, 2.13, 2.13], [2.13, 6.13, 2.13], [2.13, 2.13, 6.13]]
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    filename = os.path.join(tempfile.gettempdir(), 'openems-generated-solid.' + ('ply' if ply else 'stl'))
    with open(filename, 'w', encoding='ascii') as fixture:
        if ply:
            fixture.write('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n'
                          'element face 4\nproperty list uchar int vertex_indices\nend_header\n')
            for point in vertices:
                fixture.write(' '.join(map(str, point)) + '\n')
            for face in faces:
                fixture.write('3 ' + ' '.join(map(str, face)) + '\n')
        else:
            fixture.write('solid generated_openems_fixture\n')
            for face in faces:
                fixture.write('facet normal 0 0 0\nouter loop\n')
                for index in face:
                    fixture.write('vertex ' + ' '.join(map(str, vertices[index])) + '\n')
                fixture.write('endloop\nendfacet\n')
            fixture.write('endsolid generated_openems_fixture\n')
    csx.AddMaterial('imported', epsilon=4).AddPolyhedronReader(filename, priority=20)
    return fdtd, csx


def cylinder_features_case(kind='constant', central=False, closed=True, axis=True):
    fdtd = openEMS(CoordSystem=1, NrTS=900, EndCriteria=0, CellConstantMaterial=central)
    fdtd.SetGaussExcite(3e9, 2e9)
    fdtd.SetBoundaryCond(['PEC'] * 6)
    csx = ContinuousStructure(CoordSystem=1)
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    radii = np.array([0, 1.5, 3, 5, 8, 11, 15, 20, 26, 33, 40]) + (0 if axis else 2)
    angles = -np.pi + 2*np.pi*np.linspace(0, 1, 25)**1.2 if closed else np.linspace(-1, 1, 13)**3
    mesh.AddLine('r', radii)
    mesh.AddLine('a', angles)
    mesh.AddLine('z', [-8, -6, -3, 0, 2, 5, 9, 14, 20, 27, 35, 44])
    start, stop = [radii[0], angles[0], -3], [33, angles[-1], 27]
    if kind == 'lorentz':
        material = CSPropLorentzMaterial(csx.GetParameterSet(), order=2, epsilon=2.3)
        material.SetDispersiveMaterialProperty(0, eps_plasma=2e9, eps_pole_freq=5e9, eps_relax=8e-10,
                                               mue_plasma=5e8, mue_relax=6e-10)
        material.SetDispersiveMaterialProperty(1, eps_plasma=1e9, eps_relax=4e-10)

        csx.AddProperty(material)
    elif kind == 'debye':
        material = CSPropDebyeMaterial(csx.GetParameterSet(), order=2, epsilon=2.3)
        material.SetDispersiveMaterialProperty(0, eps_delta=0.3, eps_relax=1.2e-10)
        material.SetDispersiveMaterialProperty(1, eps_delta=0.5, eps_relax=4e-10)

        csx.AddProperty(material)
    else:
        material = csx.AddMaterial('weighted', epsilon=[2, 3, 4], mue=[1.1, 1.2, 1.3], kappa=0.001)
        material.SetIsotropy(False)
        if kind == 'weighted': material.SetMaterialWeight(epsilon='1+rho/1000+z/2000', mue='1+rho/2000')
    material.SetName('cylindrical_material')
    material.AddBox(start, stop, priority=2)
    csx.AddExcitation('source', exc_type=0, exc_val=[0, 0, 1]).AddBox([5, -0.3, 2], [15, 0.3, 14])
    if kind == 'curve':
        a0, a1 = (2.95, -2.95) if closed else (-0.7, 0.7)
        csx.AddMetal('seam_curve').AddCurve([[8.2, 13.2], [a0, a1], [3.5, 24.2]], priority=10)
        csx.AddExcitation('curve_source', exc_type=0, exc_val=[0.1, 0.2, 0.3]).AddCurve(
            [[11.2, 15.3, 11.2], [a0, a1, a0], [6.5, 17.3, 24.2]])
    elif kind == 'sheet':
        csx.AddConductingSheet('sheet', conductivity=1e5, thickness=2e-6).AddBox(
            [5, angles[0], 14], [26, angles[-1], 14], priority=10)
    elif kind == 'rlc':
        csx.AddLumpedElement('rc', ny='z', R=100, C=1e-12, caps=True).AddBox([5, -0.2, 2], [8, 0.2, 14])
        csx.AddLumpedElement('series', ny='z', R=20, L=1e-9, C=0, LEtype=1, caps=True).AddBox([11, -0.2, 2], [15, 0.2, 14])
    elif kind == 'angular_pml':
        fdtd.SetBoundaryCond(['PEC', 'PEC', 'PML_3', 'PML_3', 'PEC', 'PEC'])
    csx.AddProbe('ep', p_type=2).AddPoint([20, 0.2, 9])
    csx.AddDump('Et', dump_type=0, file_type=1).AddBox([radii[0], angles[0], 9], [40, angles[-1], 9])
    return fdtd, csx


def multigrid_timestep_case(periodic=False, forced=False):
    fdtd, csx = case_multigrid2()
    fdtd.SetTimeStepFactor(0.8)
    if periodic:
        fdtd.SetSinusExcite(3e9)
    if forced:
        fdtd.SetTimeStep(1e-8)
    return fdtd, csx


def constant_expression_case():
    fdtd, csx = geometry_case()
    csx.GetPropertiesByName('anisotropic')[0].SetMaterialWeight(epsilon='1/2', mue='sqrt(4)',
                                                              kappa='sin(pi/2)', sigma='if(0,1/0,1)')
    csx.GetPropertiesByName('source')[0].SetWeightFunction(['1/2', 'sin(pi/2)', '1/2'])
    return fdtd, csx


def curve_source_case():
    fdtd, csx = geometry_case()
    source = csx.AddExcitation('curve_source', exc_type=0, exc_val=[0, 0, 1])
    source.AddCurve([[4, 4, 4], [4, 4, 4], [-3, 0, 3]])
    source.AddCurve([[-4, -4], [4, 4], [-3, 3]])
    source.SetWeightFunction(['1', '1', '1/2'])
    disabled = csx.AddExcitation('disabled_curve', exc_type=0, exc_val=[0, 0, 10])
    disabled.SetEnabled(False)
    disabled.AddCurve([[0, 0], [4, 4], [-3, 3]])
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


def nf2ff_case(recorders):
    fdtd = openEMS(NrTS=1200, EndCriteria=0)
    fdtd.SetGaussExcite(1e9, .5e9)
    fdtd.SetBoundaryCond(['PML_6']*6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    for axis in 'xyz':
        grid.AddLine(axis, np.linspace(-300, 300, 41))
    csx.AddExcitation('source', exc_type=0, exc_val=[0,0,1]).AddBox([0,0,-15],[0,0,15])
    csx.AddProbe('ep', p_type=2).AddPoint([60,0,0])
    recorders.append(fdtd.CreateNF2FFBox(start=[-150]*3, stop=[150]*3, frequency=[1e9]))
    return fdtd, csx


def compare_nf2ff(root):
    recorders = []
    compare_case('nf2ff', lambda: nf2ff_case(recorders), root,
                 required=('Excitation Extension','Uniaxial PML Extension'))
    results = [box.CalcNF2FF(os.path.join(root,'nf2ff_'+mode), 1e9,
                            np.arange(0,181,15), np.arange(0,361,30), read_cached=False)
               for box, mode in zip(recorders, ('cpu','gpu'))]
    for result in results:
        assert np.all(result.Prad>0), 'FAIL: NF2FF radiated power is zero'
    for name in ('freq','theta','phi','r'):
        np.testing.assert_array_equal(getattr(results[0],name),getattr(results[1],name))
    for name in ('Prad','Dmax','E_theta','E_phi','P_rad'):
        a,b = (np.asarray(getattr(result,name)) for result in results)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), 'FAIL: non-finite NF2FF '+name
        peak = np.max(np.abs(a))
        assert np.max(np.abs(a-b)) <= 1e-4*peak, 'FAIL: NF2FF peak error '+name
        assert np.linalg.norm(a-b) <= 1e-3*np.linalg.norm(a), 'FAIL: NF2FF L2 error '+name
    print('PASS NF2FF far-field transformation', flush=True)


def compare_unsupported_case(name, case, root, reason):
    cpu, auto, strict = [os.path.join(root, name + '_' + mode) for mode in ('cpu', 'auto', 'gpu')]
    run_captured(case, cpu, 'gpu', gpu_setup='cpu')
    log = run_captured(case, auto, 'gpu', gpu_setup='auto')
    assert reason in log and 'rebuilding a fresh CPU operator' in log, log
    assert 'HIP construction active' not in log, 'FAIL: unsupported input reached device construction'
    assert not compare_outputs(cpu, auto, rtol=1e-4)[0], 'FAIL: unsupported-input fallback differs: ' + name
    try:
        run_captured(case, strict, 'gpu', gpu_setup='gpu')
    except RuntimeError as error:
        assert reason in str(error), error
        with open(strict + '.log') as stream:
            log = stream.read()
        assert 'HIP construction active' not in log and 'rebuilding a fresh CPU operator' not in log
    else:
        raise AssertionError('FAIL: strict GPU construction accepted unsupported input: ' + name)
    print('PASS unsupported input, fresh CPU fallback and strict rejection: ' + name, flush=True)


def compare_dump_paths(root):
    """Exercise prepared maps with upstream's forced snapshot and host paths."""
    previous = os.environ.get('OPENEMS_GPU_DUMPS')
    try:
        for mode, case in (('snapshot', case_dumps_snapshot), ('host', case_dumps)):
            os.environ['OPENEMS_GPU_DUMPS'] = mode
            name = 'dump_path_' + mode
            compare_case(name, case, root, required=('Excitation Extension', 'Uniaxial PML Extension'))
            with open(os.path.join(root, name + '_gpu.log')) as stream:
                log = stream.read()
            assert 'frequency domain dumps accumulated on the device' not in log
            if mode == 'snapshot':
                assert 'Engine_GPU: field dumps from snapshots' in log
                assert 'GPU requested host field storage:' not in log
                assert 'GPU requested sampling map readback:' not in log
            else:
                assert 'GPU requested host field storage:' in log
                assert 'GPU requested sampling map readback:' in log
    finally:
        if previous is None:
            os.environ.pop('OPENEMS_GPU_DUMPS', None)
        else:
            os.environ['OPENEMS_GPU_DUMPS'] = previous


def compare_case(name, case, root, required=('Excitation Extension',), **options):
    cpu, gpu = os.path.join(root, name+'_cpu'), os.path.join(root, name+'_gpu')
    previous = os.environ.get('OPENEMS_GPU_DIAGNOSTICS')
    os.environ['OPENEMS_GPU_DIAGNOSTICS'] = '1'
    try:
        a = run_captured(case, cpu, 'gpu', gpu_setup='cpu', **options)
        b = run_captured(case, gpu, 'gpu', gpu_setup='gpu', **options)
    finally:
        if previous is None:
            del os.environ['OPENEMS_GPU_DIAGNOSTICS']
        else:
            os.environ['OPENEMS_GPU_DIAGNOSTICS'] = previous
    assert 'backend: HIP' in a and 'backend: HIP' in b, 'FAIL: HIP stepping was not selected'
    assert 'HIP construction active' in b and 'rebuilding a fresh CPU' not in b, 'FAIL: device construction did not activate'
    assert 'HIP device construction result adopted' in b, 'FAIL: core construction result was not adopted'
    assert 'HIP primitive bounds construction active' in b, 'FAIL: primitive bounds were not constructed on HIP'
    if name.startswith('multigrid'):
        assert b.count('HIP multigrid split planning active') >= (2 if 'nested' in name else 1), 'FAIL: multigrid split selection did not execute on HIP'
    if name.startswith('geometry_') or name in ('excitation', 'dumps', 'snapshots', 'nf2ff'):
        assert 'GPU requested host field storage:' not in b, 'FAIL: E/H output materialized full host fields'
        assert 'GPU requested sampling map readback:' not in b, 'FAIL: E/H output materialized its sampling map'
    if name.startswith('sampling_'):
        assert b.count('HIP processing mesh selection active')>=2, 'FAIL: output mesh selection skipped HIP'
        assert 'GPU requested sampling map readback:' not in b and 'GPU requested host field storage:' not in b, 'FAIL: sampled E/H output requested full host state'
    if name.startswith('mode_'):
        assert b.count('HIP mode table construction active') == 2, 'FAIL: mode-match tables were not constructed on HIP'
    if name.startswith('density_'):
        assert b.count('HIP cell material sampling active') == (4 if name=='density_averages' else 2), 'FAIL: SAR cell materials were not prepared on HIP'
        assert b.count('HIP sampling map adopted by DFT:') >= (4 if name=='density_averages' else 2), 'FAIL: SAR E-field histories were not adopted on HIP'
        assert 'GPU requested host field storage:' not in b and 'GPU requested sampling map readback:' not in b, 'FAIL: SAR sampling requested full host fields/maps'
    for extension in required:
        assert f'HIP extension construction active: {extension}' in b, f'FAIL [{name}]: {extension} construction did not activate'
    if 'Uniaxial PML Extension' in required:
        assert 'HIP UPML construction result adopted' in b, 'FAIL: UPML construction tables were not adopted'
    if 'Excitation Extension' in required:
        assert 'HIP excitation construction result adopted' in b, 'FAIL: source construction tables were not adopted'
    if 'Total-Field/Scattered-Field Extension' in required:
        assert 'HIP TFSF construction result adopted' in b, 'FAIL: TFSF construction tables were not adopted'
        assert 'HIP TFSF numerical metadata active' in b, 'FAIL: TFSF normalization skipped HIP'
    if 'Mur ABC Extension' in required:
        assert 'HIP Mur construction result adopted' in b, 'FAIL: Mur construction tables were not adopted'
    if name == 'periodic' or 'Steady-State Detection Extension' in required:
        assert 'HIP steady-state construction result adopted' in b, 'FAIL: steady-state construction was not adopted'
    if any(extension in required for extension in ('Drude/Lorentz Dispersive Material Extension', 'Conducting Sheet Extension')):
        assert 'HIP Lorentz construction result adopted' in b, 'FAIL: dispersive construction tables were not adopted'
    if 'Debye Dispersive Material Extension' in required:
        assert 'HIP Debye construction result adopted' in b, 'FAIL: Debye construction tables were not adopted'
    if 'Series/Parallel Lumped RLC load' in required:
        assert 'HIP RLC construction result adopted' in b, 'FAIL: RLC construction tables were not adopted'
    if 'Local absorbing boundary condition sheet' in required:
        assert 'HIP local absorber construction result adopted' in b, 'FAIL: absorber construction tables were not adopted'
    if 'Extension for the Cylinder-Coords Operator' in required:
        assert 'HIP cylindrical axis/seam construction result adopted' in b, 'FAIL: cylindrical construction was not adopted'
    if name.startswith('multigrid'):
        levels=2 if 'nested' in name else 1
        assert b.count('HIP multigrid construction result adopted')==levels, 'FAIL: multigrid device interpolation was not adopted'
        assert b.count('HIP device construction result adopted')==levels+1, 'FAIL: some multigrid levels used CPU construction'
    if name in ('hybrid', 'rlc_caps', 'rlc_overlap'):
        assert 'HIP RC construction active' in b, 'FAIL: RC construction did not activate'
    diff, _, _, _ = compare_outputs(cpu, gpu, rtol=1e-4)
    assert not diff, f'FAIL [{name}]: CPU/GPU construction differs: {diff}'
    if name.startswith('density_'):
        with h5py.File(os.path.join(cpu,'density_raw.h5')) as expected, h5py.File(os.path.join(gpu,'density_raw.h5')) as actual:
            for channel in ('Density','Conductivity','Volume'):
                np.testing.assert_allclose(actual['CellData/'+channel][...],expected['CellData/'+channel][...],rtol=5e-7,atol=0)
        fields=['density_raw.h5','density_sar.h5']
        if name=='density_averages':
            fields.extend(('density_1g.h5','density_10g.h5'))
        for folder in (cpu,gpu):
            for filename in fields:
                with h5py.File(os.path.join(folder,filename)) as data:
                    assert np.max(np.abs(data['FieldData/FD/f0'][...]))>0, f'FAIL [{name}]: {filename} has no sampled field'
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
        for shape in ('multibox', 'polygon', 'rotpoly', 'spherical_shell', 'cylindrical_shell',
                      'wire', 'curve', 'point', 'cylindrical_box'):
            compare_case('explicit_'+shape, lambda shape=shape: explicit_geometry_case(shape), root)
        compare_case('constant_expressions', constant_expression_case, root)
        compare_case('curve_sources', curve_source_case, root)
        for cylindrical in (False, True):
            compare_case(f'mode_constant_{cylindrical}', lambda: mode_match_case(cylindrical=cylindrical, constant=True), root)
        for central in (False, True):
            compare_case(f'material_dump_{central}', lambda: material_dump_case(central), root)
            compare_case(f'cylinder_material_dump_{central}', lambda: material_dump_case(central, cylindrical=True), root,
                         required=('Excitation Extension', 'Extension for the Cylinder-Coords Operator'))
        compare_case('density_constant',density_case,root)
        compare_case('density_cylinder',lambda: density_case(cylindrical=True),root,
                     required=('Excitation Extension','Extension for the Cylinder-Coords Operator'))
        compare_case('density_averages',density_averages_case,root,
                     required=('Excitation Extension','Extension for the Cylinder-Coords Operator'))
        compare_nf2ff(root)
        for cylindrical in (False,True):
            for optimal in (False,True):
                compare_case(f'sampling_{cylindrical}_{optimal}',lambda: sampling_selection_case(cylindrical,optimal),root,
                             required=('Excitation Extension','Extension for the Cylinder-Coords Operator') if cylindrical else ('Excitation Extension',))
        compare_case('tfsf_multiple', case_tfsf_multiple, root,
                     required=('Total-Field/Scattered-Field Extension',))
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
        compare_case('delayed_step_sources', delayed_sources_case, root)
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
        compare_dump_paths(root)
        compare_case('cylinder_closed',case_cylinder_closed,root,
                     required=('Excitation Extension','Extension for the Cylinder-Coords Operator'))
        compare_case('cylinder_wedge',case_cylinder_wedge,root,
                     required=('Excitation Extension','Uniaxial PML Extension'))
        compare_case('multigrid',case_multigrid,root,required=('Excitation Extension','Extension for the Cylinder-Coords Operator'))
        compare_case('multigrid_nested',case_multigrid2,root,required=('Excitation Extension','Extension for the Cylinder-Coords Operator'))
        compare_case('multigrid_wedge',case_multigrid_wedge,root,required=('Excitation Extension','Uniaxial PML Extension'))
        for periodic in (False, True):
            for forced in (False, True):
                required=['Excitation Extension','Extension for the Cylinder-Coords Operator']
                if periodic:
                    required.append('Steady-State Detection Extension')
                compare_case(f'multigrid_nested_timestep_{periodic}_{forced}',lambda: multigrid_timestep_case(periodic,forced),root,required=required)
        for kind in ('constant', 'lorentz', 'debye', 'sheet', 'rlc', 'curve'):
            required = ['Excitation Extension', 'Extension for the Cylinder-Coords Operator']
            if kind == 'lorentz':
                required.append('Drude/Lorentz Dispersive Material Extension')
            elif kind == 'debye':
                required.append('Debye Dispersive Material Extension')
            elif kind == 'sheet':
                required.append('Conducting Sheet Extension')
            elif kind == 'rlc':
                required.append('Series/Parallel Lumped RLC load')
            for central in (False, True):
                compare_case(f'cylinder_{kind}_{central}',lambda: cylinder_features_case(kind,central),root,required=required)
        for axis in (False, True):
            compare_case(f'cylinder_angular_pml_{axis}',lambda: cylinder_features_case('angular_pml',closed=False,axis=axis),root,
                         required=('Excitation Extension','Uniaxial PML Extension'))
        unsupported = [
            ('weighted_materials', case_cpu_setup, 'spatial expressions'),
            ('weighted_sources', weighted_source_case, 'spatial expressions'),
            ('weighted_lorentz', weighted_dispersion_case, 'spatial expressions'),
            ('weighted_debye', lambda: weighted_dispersion_case(True), 'spatial expressions'),
            ('weighted_density', lambda: density_case(weighted=True), 'spatial expressions'),
            ('cylinder_weighted', lambda: cylinder_features_case('weighted'), 'spatial expressions'),
            ('custom_waveform', lambda: delayed_sources_case(True), 'custom waveforms'),
            ('polyhedron', polyhedron_case, 'polyhedron geometry'),
        ]
        for central in (False, True):
            for background in (False, True):
                unsupported.append((f'voxel_{central}_{background}', lambda central=central, background=background: voxel_case(central, background), 'discrete materials'))
        for axis in range(3):
            unsupported.append((f'file_source_{axis}', lambda axis=axis: file_source_case(axis), 'file-weighted sources'))
        for cylindrical in (False, True):
            for file in (False, True):
                unsupported.append((f'mode_{cylindrical}_{file}', lambda cylindrical=cylindrical, file=file: mode_match_case(file, cylindrical),
                                    'file-based mode matching' if file else 'spatial mode functions'))
        for ply in (False, True):
            unsupported.append(('imported_ply' if ply else 'imported_stl', lambda ply=ply: imported_polyhedron_case(ply), 'polyhedron geometry'))
        for name, case, reason in unsupported:
            compare_unsupported_case(name, case, root, reason)
        samples = np.loadtxt(os.path.join(root, 'geometry_3_False_gpu', 'ep'), comments='%')
        assert np.any(samples[:, 1:] != 0), 'FAIL: geometry excitation was not exercised'
    else:
        print('HIP unavailable: GPU construction comparisons skipped')
    # Unsupported stepping backends must recover or fail explicitly.
    fallback_cases = [(case_excitation, 'gpu-reference', 'does not support GPU operator construction')]
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
            # This budget fits the operator/fields but not all processing maps.
            # A late preparation failure must discard the engine and the
            # already-created processing objects before rebuilding on CPU.
            os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT'] = str(100*1024*1024)
            pressure_cpu = os.path.join(root, 'processing_memory_cpu')
            pressure_auto = os.path.join(root, 'processing_memory_auto')
            run_captured(processing_pressure_case, pressure_cpu, 'gpu', gpu_setup='cpu')
            log = run_captured(processing_pressure_case, pressure_auto, 'gpu', gpu_setup='auto')
            assert 'insufficient device memory for sampling maps' in log and 'rebuilding a fresh CPU operator' in log, log
            assert not compare_outputs(pressure_cpu,pressure_auto,rtol=1e-4)[0], 'FAIL: processing-memory fallback differs'
            try:
                run_captured(processing_pressure_case,os.path.join(root,'processing_memory_strict'),'gpu',gpu_setup='gpu')
            except RuntimeError as error:
                assert 'insufficient device memory for sampling maps' in str(error), error
            else:
                raise AssertionError('FAIL: strict processing-memory failure silently fell back')
            # Find a budget that allows a child result, then fails a later
            # level. Physical allocation granularity varies between drivers.
            hierarchy_fdtd,hierarchy_csx=case_multigrid2()
            hierarchy_fdtd.SetNumberOfTimeSteps(100)
            hierarchy=lambda: (hierarchy_fdtd,hierarchy_csx)
            hierarchy_cpu=os.path.join(root,'hierarchy_memory_cpu')
            run_captured(hierarchy,hierarchy_cpu,'gpu',gpu_setup='cpu')
            os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT']='1'
            path=os.path.join(root,'hierarchy_planning_auto')
            log=run_captured(hierarchy,path,'gpu',gpu_setup='auto')
            assert 'insufficient device memory for multigrid planning' in log and 'rebuilding a fresh CPU operator' in log,log
            assert not compare_outputs(hierarchy_cpu,path,rtol=1e-4)[0],'FAIL: multigrid planning fallback differs'
            try:
                run_captured(hierarchy,os.path.join(root,'hierarchy_planning_strict'),'gpu',gpu_setup='gpu')
            except RuntimeError as error:
                assert 'insufficient device memory for multigrid planning' in str(error),error
            else:
                raise AssertionError('FAIL: strict multigrid planning failure silently fell back')
            partial_limit=None
            for mib in range(128,63,-1):
                os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT']=str(mib*1024*1024)
                path=os.path.join(root,'hierarchy_memory_probe')
                try:
                    run_captured(hierarchy,path,'gpu',gpu_setup='gpu',setup_only=True)
                except RuntimeError as error:
                    assert 'insufficient device memory' in str(error),error
                    with open(path+'.log') as stream:
                        log=stream.read()
                    assert 'rebuilding a fresh CPU operator' not in log
                    if 'HIP operator construction completed' in log:
                        partial_limit=mib*1024*1024
                        break
            assert partial_limit is not None,'FAIL: no partial-hierarchy resource failure was exercised'
            os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT']=str(partial_limit)
            path=os.path.join(root,'hierarchy_memory_auto')
            log=run_captured(hierarchy,path,'gpu',gpu_setup='auto')
            assert 'HIP operator construction completed' in log and 'rebuilding a fresh CPU operator' in log,log
            assert not compare_outputs(hierarchy_cpu,path,rtol=1e-4)[0],'FAIL: partial-hierarchy fallback differs'
            os.environ['OPENEMS_GPU_SETUP_MEMORY_LIMIT']=str(1024*1024*1024)
            path=os.path.join(root,'hierarchy_memory_reuse')
            run_captured(hierarchy,path,'gpu',gpu_setup='gpu')
            assert not compare_outputs(hierarchy_cpu,path,rtol=1e-4)[0],'FAIL: hierarchy reuse after failure differs'
            print('PASS partial multigrid resource failure, fresh fallback and reuse')
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
        # A device evaluation failure is an error in both modes, never a
        # recoverable resource/adapter fallback. The same model remains usable.
        fdtd, csx = geometry_case()
        material = csx.GetPropertiesByName('anisotropic')[0]
        material.SetMaterialWeight(epsilon='1/0')
        same_model = lambda: (fdtd,csx)
        for mode in ('auto','gpu'):
            path = os.path.join(root,'numerical_failure_'+mode)
            try:
                run_captured(same_model,path,'gpu',gpu_setup=mode)
            except (ValueError, RuntimeError) as error:
                assert 'invalid constant' in str(error), error
                with open(path+'.log') as log:
                    assert 'rebuilding a fresh CPU operator' not in log.read(), 'FAIL: invalid numerical result fell back'
            else:
                raise AssertionError('FAIL: invalid device material expression was accepted')
        material.SetMaterialWeight(epsilon='1')
        a,b = os.path.join(root,'numerical_reuse_cpu'),os.path.join(root,'numerical_reuse_gpu')
        run_captured(same_model,a,'gpu',gpu_setup='cpu')
        run_captured(same_model,b,'gpu',gpu_setup='gpu')
        assert not compare_outputs(a,b,rtol=1e-4)[0], 'FAIL: reuse after a numerical failure differs'
        fdtd.SetAbort(True)
        try:
            run_captured(same_model,os.path.join(root,'cancelled'),'gpu',gpu_setup='auto')
        except RuntimeError as error:
            assert 'FDTD setup cancelled' in str(error),error
        else:
            raise AssertionError('FAIL: cancelled setup completed')
        finally:
            fdtd.SetAbort(False)
        path=os.path.join(root,'cancelled_reuse')
        run_captured(same_model,path,'gpu',gpu_setup='gpu')
        assert not compare_outputs(a,path,rtol=1e-4)[0],'FAIL: reuse after cancellation differs'
    print('PASS GPU construction and fallback checks')
