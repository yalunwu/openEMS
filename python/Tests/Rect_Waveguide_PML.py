"""WR-42 TE10 propagation with analytical ports and two PML terminations.

Check all propagation axes against continuum TE10 impedance and phase, with
3% impedance and 0.05-radian phase tolerances for this approximately lambda/30
mesh. Reflection must stay below -35 dB, transmission within 1% of unity, and
the lossless two-port power defect below 2%. The reference is an analytic
formula, not another solver backend. Full pulses decay to -60 dB.
"""

import os
import tempfile
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, Z0


def run_axis(direction):
    unit = 1e-6
    a, b, length = 10700.0, 4300.0, 50000.0
    resolution = C0 / 24e9 / unit / 30
    axis = 'xyz'.index(direction)
    transverse = ((axis + 1) % 3, (axis + 2) % 3)
    fdtd = openEMS(NrTS=20000, EndCriteria=1e-6)
    fdtd.SetGaussExcite(23e9, 3e9)
    boundary = ['PEC'] * 6
    boundary[2*axis:2*axis+2] = ['PML_8', 'PML_8']
    fdtd.SetBoundaryCond(boundary)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine(direction, [0, length])
    mesh.AddLine('xyz'[transverse[0]], [0, a])
    mesh.AddLine('xyz'[transverse[1]], [0, b])
    ports = []
    # Integer drawing coordinates survive standalone XML serialization exactly,
    # including the zero-thickness source planes and matching mesh anchors.
    near_plane, measurement_plane = round(10*resolution), round(15*resolution)
    for index, (near, far) in enumerate(((near_plane, measurement_plane),
                                       (length-near_plane, length-measurement_plane))):
        start = np.zeros(3)
        stop = np.zeros(3)
        start[axis], stop[axis] = near, far
        stop[transverse[0]], stop[transverse[1]] = a, b
        mesh.AddLine(direction, [near, far])
        ports.append(fdtd.AddRectWaveGuidePort(index+1, start, stop, direction,
                     a*unit, b*unit, 'TE10', excite=1 if index == 0 else 0))
    mesh.SmoothMeshLines('all', resolution, ratio=1.4)
    path = os.path.join(tempfile.gettempdir(), 'Rect_Waveguide_PML_' + direction)
    fdtd.Run(path, cleanup=True, exact_endcriteria=True)
    frequency = np.linspace(20e9, 26e9, 121)
    for port in ports:
        port.CalcPort(path, frequency)
    s11 = ports[0].uf_ref / ports[0].uf_inc
    s21 = ports[1].uf_ref / ports[0].uf_inc
    beta = np.sqrt((2*np.pi*frequency/C0)**2 - (np.pi/(a*unit))**2)
    expected_impedance = 2*np.pi*frequency/C0 * Z0 / beta
    measured_impedance = ports[0].uf_tot / ports[0].if_tot
    impedance_error = float(np.max(abs(measured_impedance / expected_impedance - 1)))
    distance = (length - 2*measurement_plane) * unit
    phase_error = float(np.max(abs(np.angle(s21 * np.exp(1j*beta*distance)))))
    reflection_db = float(np.max(20*np.log10(abs(s11))))
    transmission_error = float(np.max(abs(abs(s21)-1)))
    power_defect = float(np.max(abs(1-abs(s11)**2-abs(s21)**2)))
    print(f'{direction}: impedance error={impedance_error:.6g}; phase error={phase_error:.6g} rad; '
          f'S11 max={reflection_db:.3f} dB; transmission error={transmission_error:.6g}; '
          f'power defect={power_defect:.6g}', flush=True)
    assert impedance_error < 0.03, f'{direction}: impedance error {impedance_error:.6g} exceeds 3%'
    assert phase_error < 0.05, f'{direction}: phase error {phase_error:.6g} exceeds 0.05 rad'
    assert reflection_db < -35, f'{direction}: reflection {reflection_db:.3f} dB exceeds -35 dB'
    assert transmission_error < 0.01, f'{direction}: transmission error {transmission_error:.6g} exceeds 1%'
    assert power_defect < 0.02, f'{direction}: power defect {power_defect:.6g} exceeds 2%'


for direction in 'xyz':
    run_axis(direction)
print('PASS')
