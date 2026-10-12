"""Uniform lossy TE10 guide against complex continuum impedance/propagation.

Use the antenna's isotropic RO3003 and FR4 epsilon/conductivity constants.
At the approximately lambda_material/35 mesh, require measured complex modal
impedance within 3%, transmission attenuation within 2%, and phase within
0.05 rad of beta=sqrt(omega^2*mu*epsilon-j*omega*mu*kappa-kc^2).
Both complete pulses decay to -60 dB; no second solver supplies the reference.
"""

import os
import tempfile
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, MUE0


def run_material(name, epsilon, kappa):
    unit = 1e-6
    a, b, length, resolution = 10700, 4300, 50000, 150
    near, measure = 18*resolution, 24*resolution
    fdtd = openEMS(NrTS=20000, EndCriteria=1e-6)
    fdtd.SetGaussExcite(23e9, 3e9)
    fdtd.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PML_12', 'PML_12'])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [0, a])
    mesh.AddLine('y', [0, b])
    mesh.AddLine('z', [0, near, measure, length-measure, length-near, length])
    material = csx.AddMaterial(name, epsilon=epsilon, kappa=kappa)
    material.AddBox([0, 0, 0], [a, b, length])
    ports = [fdtd.AddRectWaveGuidePort(1, [0, 0, near], [a, b, measure], 'z',
                                     a*unit, b*unit, 'TE10', excite=1),
             fdtd.AddRectWaveGuidePort(2, [0, 0, length-near], [a, b, length-measure], 'z',
                                     a*unit, b*unit, 'TE10', excite=0)]
    mesh.SmoothMeshLines('all', resolution, ratio=1.4)
    path = os.path.join(tempfile.gettempdir(), 'Dielectric_Waveguide_' + name)
    fdtd.Run(path, cleanup=True, exact_endcriteria=True)
    frequency = np.linspace(20e9, 26e9, 121)
    for port in ports:
        port.CalcPort(path, frequency)
    omega = 2*np.pi*frequency
    beta = np.sqrt((omega/C0)**2*epsilon - 1j*omega*MUE0*kappa - (np.pi/(a*unit))**2)
    impedance = omega*MUE0 / beta
    expected = np.exp(-1j*beta*(length-2*measure)*unit)
    measured = ports[1].uf_tot / ports[0].uf_tot
    z_measured = ports[0].uf_tot / ports[0].if_tot
    z_error = float(np.max(abs(z_measured/impedance-1)))
    amplitude_error = float(np.max(abs(abs(measured/expected)-1)))
    phase_error = float(np.max(abs(np.angle(measured/expected))))
    print(f'{name}: impedance error={z_error:.6g}; attenuation error={amplitude_error:.6g}; '
          f'phase error={phase_error:.6g} rad', flush=True)
    assert z_error < 0.03, f'{name}: complex impedance error {z_error:.6g} exceeds 3%'
    assert amplitude_error < 0.02, f'{name}: attenuation error {amplitude_error:.6g} exceeds 2%'
    assert phase_error < 0.05, f'{name}: phase error {phase_error:.6g} exceeds 0.05 rad'


run_material('RO3003', 3.0, 0.01285111)
run_material('FR4', 4.4, 0.3769658)
print('PASS')
