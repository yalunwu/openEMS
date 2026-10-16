# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Compare the results of two openEMS runs.

Typical use is checking one engine against another: run the same simulation
with, e.g., ``engine='multithreaded'`` and ``engine='gpu'`` into two
simulation folders, then::

    python -m openEMS.compare <sim_path_a> <sim_path_b>

Run both with ``exact_endcriteria=True``, otherwise they may stop after a
different number of timesteps and every probe differs in length.
"""

import os
import re
import sys
import argparse

import numpy as np
import h5py


def deviation(a, b, peak=None):
    """Maximum deviation of `b` from `a`, relative to `peak`.

    `peak` defaults to the peak absolute value of `a`.  Returns 0 for identical
    arrays and ``inf`` if the shapes differ.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.inf
    if np.array_equal(a, b):
        return 0.0
    if peak is None:
        peak = _peak(a)
    if not np.isfinite(peak):
        return np.inf
    error = np.max(np.abs(a - b)) / (peak if peak > 0 else 1.0)
    return float(error) if np.isfinite(error) else np.inf


def _peak(a):
    return np.max(np.abs(a)) if np.size(a) else 0.0


def _peak_group(name):
    """The datasets of a HDF5 file that are compared relative to a common peak.

    These are the timesteps or frequencies of a field dump, and both
    polarisations of a far field: the cross-polarisation is often only
    numerical noise, which has no meaningful peak of its own.
    """
    if name.startswith('FieldData'):
        return 'FieldData'
    if name.startswith(('nf2ff/E_theta', 'nf2ff/E_phi')):
        return 'nf2ff/E'
    if name.startswith('nf2ff/P_rad'):
        return 'nf2ff/P_rad'
    return None


def _read_probe(fn):
    """The axis (time or frequency) and the value columns of a probe file."""
    data = np.loadtxt(fn, comments='%', ndmin=2)
    return data[:, 0], data[:, 1:]


def _read_vtr(fn):
    """The point data arrays and coordinates of a VTK rectilinear grid dump."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(fn)
    reader.Update()
    grid = reader.GetOutput()
    pd = grid.GetPointData()
    if reader.GetErrorCode() or not grid.GetNumberOfPoints() or not pd.GetNumberOfArrays():
        raise ValueError(f'cannot read field data from {fn}')
    fields = {pd.GetArrayName(n): vtk_to_numpy(pd.GetArray(n)) for n in range(pd.GetNumberOfArrays())}
    mesh = tuple(vtk_to_numpy(coords) for coords in
                 (grid.GetXCoordinates(), grid.GetYCoordinates(), grid.GetZCoordinates()))
    return fields, mesh


def _hdf5_group_attributes(h):
    """Sampling/mesh metadata and scalar NF2FF results stored on groups."""
    result = {}
    def collect(name, obj):
        if isinstance(obj, h5py.Group):
            for key in ('time', 'frequency', 'Frequency', 'mesh_scaling', 'mesh_type',
                        'MeshType', 'dump_type', 'dual_mesh', 'Eps_r', 'Mue_r', 'Prad', 'Dmax'):
                if key in obj.attrs:
                    result[name, key] = obj.attrs[key]
    collect('/', h)
    h.visititems(collect)
    return result


def _compare(path_a, path_b):
    """List of ``(group, name, deviation)``, see `compare_sim_results`."""
    files = sorted(f for f in os.listdir(path_a) if os.path.isfile(os.path.join(path_a, f)))
    other_files = {f for f in os.listdir(path_b) if os.path.isfile(os.path.join(path_b, f))}
    result = []
    for f in sorted(other_files - set(files)):
        if f.endswith(('.h5', '.vtr')):
            result.append((f, f, np.inf))
        elif '.' not in f:
            try:
                _read_probe(os.path.join(path_b, f))
            except ValueError:
                continue
            result.append((f, f, np.inf))

    # the VTK dumps are one file per timestep, a series shares its peak
    series = {}
    for f in files:
        if f.endswith('.vtr'):
            series.setdefault(re.sub(r'_\d+\.vtr$', '', f), []).append(f)
    if series:
        try:
            import vtk  # noqa: F401
        except ImportError:
            print('compare: vtk not available, cannot compare {} *.vtr files'.format(sum(map(len, series.values()))),
                  file=sys.stderr)
            result += [(group + '_*.vtr', f, np.inf) for group, names in series.items() for f in names]
            series = {}
    for group, names in series.items():
        diffs = []
        peak = 0.0
        for f in names:
            fb = os.path.join(path_b, f)
            if not os.path.isfile(fb):
                diffs.append((f, np.inf))
                continue
            (a, ma), (b, mb) = _read_vtr(os.path.join(path_a, f)), _read_vtr(fb)
            if any(not np.isfinite(x).all() or not np.array_equal(x, y) for x, y in zip(ma, mb)):
                diffs.append((f, np.inf))
                continue
            for n in b.keys() - a.keys():
                diffs.append((f, np.inf))
            for n, arr in a.items():
                peak = max(peak, _peak(arr))
                diffs.append((f, deviation(arr, b[n], 1.0) if n in b else np.inf))
        result += [(group + '_*.vtr', f, d / (peak if peak > 0 else 1.0)
                    if np.isfinite(d) and np.isfinite(peak) else np.inf) for f, d in diffs]

    for f in files:
        fa, fb = os.path.join(path_a, f), os.path.join(path_b, f)
        if '.' not in f:
            if not os.path.isfile(fb):
                result.append((f, f, np.inf))
                continue
            try:
                xa, a = _read_probe(fa)
            except ValueError:
                continue    # not a probe file
            try:
                xb, b = _read_probe(fb)
            except ValueError:
                result.append((f, f, np.inf))
                continue
            # the time or frequency axis has to agree exactly
            result.append((f, f, deviation(a, b) if np.isfinite(xa).all() and np.array_equal(xa, xb) else np.inf))
        elif f.endswith('.h5'):
            if not os.path.isfile(fb):
                result.append((f, f, np.inf))
                continue
            with h5py.File(fa, 'r') as a, h5py.File(fb, 'r') as b:
                aa, ab = _hdf5_group_attributes(a), _hdf5_group_attributes(b)
                for obj, key in sorted(aa.keys() | ab.keys()):
                    attr = obj, key
                    dev = np.inf
                    if attr in aa and attr in ab:
                        dev = deviation(aa[attr], ab[attr])
                        if key not in ('Prad', 'Dmax') and not np.array_equal(aa[attr], ab[attr]):
                            dev = np.inf
                    result.append((f, f'{f}:{obj}@{key}', dev))
                names = []
                a.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
                other_names = []
                b.visititems(lambda name, obj: other_names.append(name) if isinstance(obj, h5py.Dataset) else None)
                for n in sorted(set(other_names) - set(names)):
                    result.append((f, f'{f}:{n}', np.inf))
                # datasets of one kind share their peak
                peak = {}
                for n in names:
                    key = _peak_group(n)
                    if key is not None:
                        peak[key] = max(peak.get(key, 0.0), _peak(a[n][()]))
                for n in names:
                    if n not in b:
                        result.append((f, f'{f}:{n}', np.inf))
                    else:
                        # Frame times and frequency axes are part of the sampling contract.
                        attrs_equal = all((key in a[n].attrs) == (key in b[n].attrs) and
                                          (key not in a[n].attrs or
                                           (np.isfinite(a[n].attrs[key]).all() and
                                            np.isfinite(b[n].attrs[key]).all() and
                                            np.array_equal(a[n].attrs[key], b[n].attrs[key])))
                                          for key in ('time', 'frequency'))
                        av, bv = a[n][()], b[n][()]
                        exact = n.startswith('Mesh/')
                        dev = deviation(av, bv, peak.get(_peak_group(n)))
                        result.append((f, f'{f}:{n}', dev if attrs_equal and
                                       (not exact or np.array_equal(av, bv)) else np.inf))
    return result


def compare_sim_results(path_a, path_b):
    """Compare all probe files and field dumps of two simulation folders.

    Probe and port files (the files without an extension, time and frequency
    domain) have to agree exactly in their time or frequency axis; the values
    are compared relative to their peak.  In a field dump, all field datasets
    are compared relative to the peak of the whole dump, so that late, decayed
    timesteps are not judged against their own tiny peak.  That holds for the
    datasets of a HDF5 dump and for the files of a VTK (``*.vtr``) dump
    series; unavailable VTK support reports those files as incomparable.
    In a far field result (NF2FF), both polarisations share their peak.

    Parameters
    ----------
    path_a, path_b : str
        The simulation folders; `path_a` is the reference.

    Returns
    -------
    list of (str, float)
        One ``(name, deviation)`` per probe file, per HDF5 dump dataset
        (named ``file.h5:dataset``) and per array of a VTK dump file.  A file
        or dataset missing in `path_b` has deviation ``inf``.
    """
    return [(name, dev) for _, name, dev in _compare(path_a, path_b)]


def compare_sim_metrics(path_a, path_b):
    """Return (file/series, peak-relative error, relative-L2 error).

    Norms cover all field frames or probe value columns in a file, excluding
    axes and meshes. Missing/extra outputs, changed sampling and non-finite
    values fail with infinite error. HDF5 frames are read one at a time.
    """
    peaks = {}
    for group, _, dev in _compare(path_a, path_b):
        peaks[group] = max(peaks.get(group, 0.0), dev if np.isfinite(dev) else np.inf)
    result = []
    for group, peak in peaks.items():
        norm = error = scale = 0.0
        def add(a, b):
            nonlocal norm, error, scale
            a = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
            b = np.asarray(b, dtype=np.complex128 if np.iscomplexobj(b) else np.float64)
            new_scale = max(scale, _peak(a), _peak(b))
            if new_scale:
                factor = (scale/new_scale)**2
                norm = norm*factor + float(np.sum(np.abs(a/new_scale)**2))
                error = error*factor + float(np.sum(np.abs(b/new_scale-a/new_scale)**2))
                scale = new_scale
        if not np.isfinite(peak):
            result.append((group, np.inf, np.inf))
            continue
        fa, fb = os.path.join(path_a, group), os.path.join(path_b, group)
        if group.endswith('.h5'):
            with h5py.File(fa, 'r') as a, h5py.File(fb, 'r') as b:
                names = []
                a.visititems(lambda n, obj: names.append(n) if isinstance(obj, h5py.Dataset) and
                             _peak_group(n) is not None else None)
                for n in names:
                    add(a[n][()], b[n][()])
                if 'nf2ff' in a:
                    for key in ('Prad', 'Dmax'):
                        if key in a['nf2ff'].attrs:
                            add(a['nf2ff'].attrs[key], b['nf2ff'].attrs[key])
        elif group.endswith('_*.vtr'):
            prefix = group[:-6]
            for f in sorted(os.listdir(path_a)):
                if f.endswith('.vtr') and re.sub(r'_\d+\.vtr$', '', f) == prefix:
                    (a, _), (b, _) = _read_vtr(os.path.join(path_a, f)), _read_vtr(os.path.join(path_b, f))
                    for n in a:
                        add(a[n], b[n])
        else:
            add(_read_probe(fa)[1], _read_probe(fb)[1])
        l2 = np.sqrt(error/norm) if norm else (np.inf if error else 0.0)
        result.append((group, peak, float(l2) if np.isfinite(l2) else np.inf))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m openEMS.compare',
                                     description='Compare the probe files and field dumps of two openEMS runs.')
    parser.add_argument('path_a', help='reference simulation folder')
    parser.add_argument('path_b', help='simulation folder to check')
    parser.add_argument('--rtol', type=float, default=1e-4,
                        help='max. deviation relative to the peak value (default: %(default)g)')
    parser.add_argument('--l2-rtol', type=float,
                        help='also require this relative-L2 limit per probe file or field dump')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='list every dump dataset and file, not only the worst per dump')
    args = parser.parse_args(argv)
    if not np.isfinite(args.rtol) or args.rtol < 0 or (args.l2_rtol is not None and
            (not np.isfinite(args.l2_rtol) or args.l2_rtol < 0)):
        parser.error('comparison tolerances must be finite and nonnegative')

    result = _compare(args.path_a, args.path_b)
    if not result:
        print(f'no probe files or field dumps in {args.path_a}')
        return 1

    if args.verbose:
        rows = [(name, dev) for _, name, dev in result]
    else:
        worst = {}
        for group, _, dev in result:
            worst[group] = max(worst.get(group, 0.0), dev if np.isfinite(dev) else np.inf)
        rows = list(worst.items())
    width = max(len(n) for n, _ in rows)
    for name, dev in rows:
        print(f'{name:{width}s}  {dev:.2e}{"  FAIL" if not np.isfinite(dev) or dev > args.rtol else ""}')

    failed = [n for _, n, d in result if not np.isfinite(d) or d > args.rtol]
    peak_failed = len(failed)
    if args.l2_rtol is not None:
        for name, _, l2 in compare_sim_metrics(args.path_a, args.path_b):
            l2_failed = not np.isfinite(l2) or l2 > args.l2_rtol
            print(f'{name}: relative L2 {l2:.2e}{"  FAIL" if l2_failed else ""}')
            if l2_failed:
                failed.append(name + ':L2')
    print(f'{len(result) - peak_failed}/{len(result)} within rtol={args.rtol:g}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
