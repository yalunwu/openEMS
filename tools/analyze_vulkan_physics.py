#!/usr/bin/env python3
"""Analyze full antenna physics checks without changing their outputs.

S-parameter differences use absolute complex amplitude, normalized to the
incident source wave, so a near-zero reflection does not inflate relative error.
Dielectric loss uses cell-center quadrature and is a diagnostic approximation.
"""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from validate_vulkan_physics import configure

TOLERANCES = dict(s_parameter_absolute=0.005, radiated_power_relative=0.01,
                  normalized_pattern_absolute=0.01, cpu_gpu_relative_l2=0.001,
                  approximate_power_balance_relative=0.05)


def spectrum(data, frequency):
    import numpy as np
    return 2 * np.mean(np.diff(data[:, 0])) * np.sum(
        data[:, 1] * np.exp(-2j * np.pi * frequency * data[:, 0]))


def ports(case, frequency=77e9):
    import numpy as np
    values = []
    for port in range(1, 5):
        voltage = spectrum(np.loadtxt(case / f'port_ut_{port}', comments='%'), frequency)
        current = spectrum(np.loadtxt(case / f'port_it_{port}', comments='%'), frequency)
        values.append((voltage, current))
    values = np.array(values)
    voltage, current = values[:, 0], values[:, 1]
    incident = (voltage + 50 * current) / 2
    reflected = (voltage - 50 * current) / 2
    return dict(s=reflected / incident[0], impedance=voltage[0] / current[0],
                incident_power=abs(incident[0])**2 / 100,
                accepted_power=float(0.5 * np.sum((voltage * current.conj()).real)))


def far_field(case):
    import numpy as np
    from openEMS import _nf2ff
    from openEMS.nf2ff import nf2ff_results
    output = case / 'physics_farfield.h5'
    if not output.exists():
        theta = np.deg2rad(np.arange(0, 181, 3))
        phi = np.deg2rad(np.arange(0, 361, 5))
        transform = _nf2ff._nf2ff([77e9], theta, phi, [0, 0, 0], numThreads=4)
        for face in range(6):
            if not transform.AnalyseFile(str(case / f'nf2ff_E_{face}.h5'),
                                         str(case / f'nf2ff_H_{face}.h5')):
                raise RuntimeError('Far-field transformation failed: ' + str(case))
        if not transform.Write2HDF5(str(output)):
            raise RuntimeError('Far-field output failed: ' + str(output))
    return nf2ff_results(str(output))


def dielectric_loss(case):
    import h5py
    import numpy as np
    description = case / 'loss-recorders.json'
    if not description.exists():
        return None
    tree = ET.parse(case / 'model.xml')
    grid = tree.getroot().find('.//RectilinearGrid')
    scale = float(grid.get('DeltaUnit'))
    lines = [np.fromstring(grid.find(axis + 'Lines').text, sep=',') * scale for axis in 'XYZ']
    result = []
    for recorder in json.loads(description.read_text()):
        with h5py.File(case / (recorder['name'] + '.h5')) as source:
            field = source['FieldData/FD/f0'][()].astype(np.complex128)
            axes = [source['Mesh/' + axis][()] for axis in 'xyz']
        widths = []
        for axis, mesh in enumerate(lines):
            centers = (mesh[:-1] + mesh[1:]) / 2
            positions = axes[axis]
            indices = np.argmin(abs(positions[:, None] - centers[None, :]), axis=1)
            assert np.max(abs(positions - centers[indices])) < 1e-5 * np.min(np.diff(mesh))
            lo, hi = sorted((recorder['start'][axis] * scale, recorder['stop'][axis] * scale))
            widths.append(np.maximum(0, np.minimum(mesh[indices+1], hi) - np.maximum(mesh[indices], lo)))
        assert field.shape == (3, len(axes[0]), len(axes[1]), len(axes[2])), field.shape
        volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
        power = 0.5 * recorder['kappa'] * np.sum(np.sum(abs(field)**2, axis=0) * volume)
        result.append(dict(material=recorder['material'], power=float(power)))
    return result


def metrics(case):
    import numpy as np
    data = ports(case)
    ff = far_field(case)
    pattern = ff.E_norm[0].astype(float)
    loss = dielectric_loss(case)
    info = dict(case=str(case), s_parameters=[dict(real=float(v.real), imag=float(v.imag),
                magnitude=float(abs(v))) for v in data['s']],
                input_impedance=dict(real=float(data['impedance'].real), imag=float(data['impedance'].imag)),
                incident_power=float(data['incident_power']), accepted_power=data['accepted_power'],
                radiated_power=float(ff.Prad[0]), directivity=float(ff.Dmax[0]),
                radiation_efficiency=float(ff.Prad[0] / data['accepted_power']),
                radiated_over_incident=float(ff.Prad[0] / data['incident_power']),
                dielectric_loss_cell_center=loss)
    info['power_normalization'] = 'Unnormalized pulse Fourier amplitudes; common spectral scaling cancels in power ratios.'
    if loss is not None:
        info['approximate_power_balance_residual'] = float((data['accepted_power'] - ff.Prad[0] -
            sum(row['power'] for row in loss)) / data['accepted_power'])
        info['approximate_power_balance_passed'] = abs(info['approximate_power_balance_residual']) <= \
            TOLERANCES['approximate_power_balance_relative']
    info['passive_radiation_check'] = 0 <= info['radiation_efficiency'] <= 1.05
    if (case / 'result.json').exists():
        info['run'] = json.loads((case / 'result.json').read_text())
    info['_s'] = data['s']
    info['_pattern'] = pattern / np.max(pattern)
    return info


def compare_metrics(reference, candidate):
    import numpy as np
    errors = dict(s_parameter_absolute=float(np.max(abs(candidate['_s'] - reference['_s']))),
        radiated_power_relative=abs(candidate['radiated_over_incident'] / reference['radiated_over_incident'] - 1),
        normalized_pattern_absolute=float(np.max(abs(candidate['_pattern'] - reference['_pattern']))))
    return dict(errors=errors, passed={key: error <= TOLERANCES[key] for key, error in errors.items()})


def cpu_comparison(reference, candidate, tolerance=None, include_loss=False):
    import h5py
    import numpy as np
    results = []
    for name in sorted(p.name for p in reference.glob('port_*')):
        expected, actual = [np.loadtxt(case / name, comments='%') for case in (reference, candidate)]
        assert expected.shape == actual.shape, (name, expected.shape, actual.shape)
        assert np.allclose(expected[:, 0], actual[:, 0], rtol=1e-12, atol=1e-25)
        error = float(np.linalg.norm(actual[:, 1:] - expected[:, 1:]) / np.linalg.norm(expected[:, 1:]))
        results.append(dict(output=name, relative_l2=error, identical_values=bool(np.array_equal(expected, actual))))
    fields = list(reference.glob('nf2ff_[EH]_*.h5'))
    assert len(fields) == 12, 'Expected all twelve near-field surfaces'
    if include_loss:
        losses = list(reference.glob('physics_loss_*.h5'))
        assert len(losses) == 2, 'Expected both dielectric-loss volumes'
        fields += losses
    for name in sorted(p.name for p in fields):
        with h5py.File(reference / name) as rf, h5py.File(candidate / name) as cf:
            expected = rf['FieldData/FD/f0'][()].astype(np.complex128)
            actual = cf['FieldData/FD/f0'][()].astype(np.complex128)
            assert expected.shape == actual.shape and np.isfinite(actual).all()
            for axis in 'xyz':
                assert np.array_equal(rf['Mesh/' + axis][()], cf['Mesh/' + axis][()])
            assert rf['FieldData/FD/f0'].attrs['frequency'] == cf['FieldData/FD/f0'].attrs['frequency']
            error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        results.append(dict(output=name, relative_l2=error, identical_values=bool(np.array_equal(expected, actual))))
    assert len(results) == 8 + len(fields)
    if tolerance is None:
        tolerance = TOLERANCES['cpu_gpu_relative_l2']
    return dict(checks=results, maximum_relative_l2=max(r['relative_l2'] for r in results),
                tolerance=tolerance, all_identical=all(row['identical_values'] for row in results),
                passed=all(np.isfinite(r['relative_l2']) and r['relative_l2'] <= tolerance for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('project', 'native', 'python-install', 'bindings', 'root', 'reference40'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    handles = configure(args)
    root = args.root.resolve()
    cases = {'reference40': args.reference40.resolve()}
    for name in ('antenna-minus60', 'antenna-pml16', 'antenna-clearance16', 'antenna-halfdt',
                 'antenna-refined', 'antenna-refined-cpu-setup'):
        if (root / name / 'result.json').exists():
            if json.loads((root / name / 'result.json').read_text())['returncode'] == 0:
                cases[name] = root / name
    values = {}
    for name, case in cases.items():
        print('Analyzing', name, flush=True)
        values[name] = metrics(case)
    comparisons = {}
    if 'antenna-minus60' in values:
        comparisons['stopping_criterion'] = compare_metrics(values['antenna-minus60'], values['reference40'])
        for name in values:
            if name not in ('reference40', 'antenna-minus60'):
                comparisons[name] = compare_metrics(values['antenna-minus60'], values[name])
    cpu = root / 'antenna-cpu-full'
    if (cpu / 'result.json').exists():
        comparisons['full_cpu_gpu'] = cpu_comparison(args.reference40.resolve(), cpu)
    refined_reference = root / 'antenna-refined-indexed'
    if (refined_reference / 'result.json').exists():
        refined_result = json.loads((refined_reference / 'result.json').read_text())
        if refined_result.get('returncode') == 0 and 'refinement' in refined_result:
            fused_result = json.loads((root / 'antenna-refined-cpu-setup/result.json').read_text())
            assert refined_result['steps'] == fused_result['steps']
            assert refined_result['stop'] == fused_result['stop'] and refined_result['stop']
            comparisons['refined_indexed_fused'] = cpu_comparison(
                refined_reference, root / 'antenna-refined-cpu-setup', tolerance=1e-7, include_loss=True)
    output = dict(tolerances=TOLERANCES, comparisons=comparisons,
                  metrics={name: {key: value for key, value in row.items() if not key.startswith('_')}
                           for name, row in values.items()})
    (root / 'analysis.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    if comparisons:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = {'stopping_criterion': '-40 vs -60 dB', 'antenna-pml16': 'Thicker PML',
                  'antenna-clearance16': 'More clearance', 'antenna-halfdt': 'Half timestep',
                  'antenna-refined-cpu-setup': 'Half cell size', 'antenna-refined': 'Half cell size'}
        rows = [(name, row) for name, row in comparisons.items() if 'errors' in row]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout='constrained')
        titles = ['S-parameter change\n(% of incident amplitude)', 'Radiated-power change (%)',
                  'Normalized-pattern change (%)']
        for axis, key, title in zip(axes, ('s_parameter_absolute', 'radiated_power_relative',
                                          'normalized_pattern_absolute'), titles):
            errors = [100 * row['errors'][key] for _, row in rows]
            colors = ['#b33b35' if row['errors'][key] > TOLERANCES[key] else '#287a71' for _, row in rows]
            bars = axis.barh([labels.get(name, name) for name, _ in rows], errors, color=colors)
            axis.axvline(100*TOLERANCES[key], color='#555555', linestyle='--', label='Target')
            axis.set_title(title)
            axis.grid(axis='x', alpha=0.2)
            axis.invert_yaxis()
            axis.bar_label(bars, fmt='%.3g', padding=3)
            axis.set_xlim(0, max([100*TOLERANCES[key]] + errors)*1.25)
            axis.legend(loc='lower right', fontsize=8)
        figure.suptitle('Full antenna convergence checks at 77 GHz')
        figure.savefig(root / 'physics-convergence.png', dpi=170)
        plt.close(figure)
        refined = values.get('antenna-refined-cpu-setup')
        if refined is not None:
            import numpy as np
            figure, axes = plt.subplots(1, 2, figsize=(10, 6.5), subplot_kw={'projection': 'polar'})
            figure.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.77, wspace=0.28)
            theta = np.deg2rad(np.arange(0, 181, 3))
            angles = np.concatenate((theta, 2*np.pi-theta[-2:0:-1], [2*np.pi]))
            for axis, phi in zip(axes, (0, 90)):
                for row, label, color in ((values['antenna-minus60'], '4.17M cells', '#23678d'),
                                          (refined, '33.02M cells', '#cc6c2c')):
                    pattern = row['_pattern']
                    cut = np.concatenate((pattern[:, phi//5], pattern[-2:0:-1, (phi+180)//5],
                                          [pattern[0, phi//5]]))
                    axis.plot(angles, cut, label=label, color=color, linewidth=2)
                axis.set_theta_zero_location('N')
                axis.set_theta_direction(-1)
                axis.set_ylim(0, 1.08)
                axis.set_rticks([0.25, 0.5, 0.75, 1])
                axis.set_title(f'Azimuth plane {phi}° / {phi+180}°', pad=18)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=2)
            figure.suptitle('Antenna mesh sensitivity at 77 GHz\nFar-field amplitudes normalized separately', y=0.98)
            figure.savefig(root / 'physics-pattern-cuts.png', dpi=170)
            plt.close(figure)
    print(json.dumps(comparisons, indent=2), flush=True)
    failed = any(not all(row['passed'].values()) if isinstance(row['passed'], dict)
                 else not row['passed'] for row in comparisons.values())
    failed = failed or any(not row['passive_radiation_check'] or
                          not row.get('approximate_power_balance_passed', True) for row in values.values())
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
