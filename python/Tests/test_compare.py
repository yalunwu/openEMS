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
# What compare_sim_results() judges a deviation against. Each of these choices
# decides whether two engines are reported as equal, and none of them shows in
# a comparison that passes.

import os
import tempfile
import unittest
import contextlib
import io
import sys
from unittest.mock import patch
import numpy as np
import h5py

import openEMS.compare as compare
from openEMS.compare import compare_sim_results, compare_sim_metrics, deviation, main


def write_run(path, probe, time, dump, e_phi=0.0):
    """ A minimal simulation folder: one probe, one field dump with two
    timesteps and a far field result """
    os.mkdir(path)
    np.savetxt(os.path.join(path, 'ut'), np.column_stack([time, probe]), header='t/s\tvoltage', comments='% ')
    with h5py.File(os.path.join(path, 'Et.h5'), 'w') as h:
        h['FieldData/TD/00000000'] = dump
        h['FieldData/TD/00000010'] = dump*1e-6      # a decayed timestep
    with h5py.File(os.path.join(path, 'nf2ff.h5'), 'w') as h:
        h['nf2ff/E_theta/FD/f0'] = np.ones(4)
        h['nf2ff/E_phi/FD/f0'] = np.full(4, e_phi)  # cross polarisation: numerical noise only


class Test_Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a = os.path.join(self.tmp.name, 'a')
        self.b = os.path.join(self.tmp.name, 'b')
        self.time = np.arange(5)*1e-12
        self.probe = np.array([0, 1, 2, 1, 0], dtype=float)
        self.dump = np.linspace(-1, 1, 24).reshape(3, 2, 2, 2)
        write_run(self.a, self.probe, self.time, self.dump, e_phi=1e-9)

    def tearDown(self):
        self.tmp.cleanup()

    def worst(self):
        return dict(compare_sim_results(self.a, self.b))

    def test_deviations_are_relative_to_the_peak_of_their_kind(self):
        probe = self.probe.copy()
        probe[1] += 2e-3                            # 1e-3 of the probe peak, not of the time values
        dump = self.dump.copy()
        write_run(self.b, probe, self.time, dump, e_phi=3e-9)
        with h5py.File(os.path.join(self.b, 'Et.h5'), 'a') as h:
            h['FieldData/TD/00000010'][0, 0, 0, 0] += 1e-7
        res = self.worst()
        self.assertAlmostEqual(res['ut'], 1e-3, delta=1e-9)
        # 10 % of the decayed timestep itself, but 1e-7 of the whole dump
        self.assertAlmostEqual(res['Et.h5:FieldData/TD/00000010'], 1e-7, delta=1e-12)
        self.assertEqual(res['Et.h5:FieldData/TD/00000000'], 0)
        # three times its own value, but 2e-9 of the co-polarisation
        self.assertAlmostEqual(res['nf2ff.h5:nf2ff/E_phi/FD/f0'], 2e-9, delta=1e-14)

    def test_what_cannot_be_compared_is_infinitely_off(self):
        # a run that stopped at another timestep, and one without the dump
        write_run(self.b, self.probe[:4], self.time[:4], self.dump, e_phi=1e-9)
        os.remove(os.path.join(self.b, 'Et.h5'))
        res = self.worst()
        self.assertEqual(res['ut'], np.inf)
        self.assertEqual(res['Et.h5'], np.inf)
        # the same values at shifted times are no agreement either
        self.tmp.cleanup()
        self.setUp()
        write_run(self.b, self.probe, self.time + 1e-12, self.dump, e_phi=1e-9)
        self.assertEqual(self.worst()['ut'], np.inf)
        # nor is an output that only the second run wrote
        self.tmp.cleanup()
        self.setUp()
        write_run(self.b, self.probe, self.time, self.dump, e_phi=1e-9)
        np.savetxt(os.path.join(self.b, 'it'), np.column_stack([self.time, self.probe]), comments='% ')
        self.assertEqual(self.worst()['it'], np.inf)

    def test_nonfinite_values_never_pass_even_when_identical(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertEqual(deviation([value], [value]), np.inf)
                self.assertEqual(deviation([1], [value]), np.inf)
        self.assertEqual(deviation([1.0], [1.1], peak=np.inf), np.inf)
        write_run(self.b, self.probe, self.time, self.dump)
        with h5py.File(os.path.join(self.b, 'Et.h5'), 'a') as h:
            h['FieldData/TD/00000000'][0, 0, 0, 0] = np.nan
        self.assertEqual(self.worst()['Et.h5:FieldData/TD/00000000'], np.inf)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([self.a, self.b]), 1)
        with h5py.File(os.path.join(self.b, 'Et.h5'), 'a') as h:
            h['FieldData/TD/00000000'][...] = self.dump
        for path in (self.a, self.b):
            with h5py.File(os.path.join(path, 'Et.h5'), 'a') as h:
                h['FieldData/TD/00000000'].attrs['time'] = np.inf
        self.assertEqual(self.worst()['Et.h5:FieldData/TD/00000000'], np.inf)

    def test_extra_outputs_and_frame_times_fail(self):
        write_run(self.b, self.probe, self.time, self.dump)
        np.savetxt(os.path.join(self.b, 'extra_probe'), np.column_stack([self.time, self.probe]))
        with h5py.File(os.path.join(self.a, 'Et.h5'), 'a') as h:
            h['FieldData/TD/00000000'].attrs['time'] = 0.0
        with h5py.File(os.path.join(self.b, 'Et.h5'), 'a') as h:
            h['FieldData/TD/00000000'].attrs['time'] = 1e-12
            h['FieldData/TD/00000020'] = self.dump
        res = self.worst()
        self.assertEqual(res['extra_probe'], np.inf)
        self.assertEqual(res['Et.h5:FieldData/TD/00000000'], np.inf)
        self.assertEqual(res['Et.h5:FieldData/TD/00000020'], np.inf)

    def test_mesh_changes_are_exact_and_malformed_probes_fail(self):
        write_run(self.b, self.probe, self.time, self.dump)
        for path, delta in ((self.a, 0), (self.b, 1e-12)):
            with h5py.File(os.path.join(path, 'Et.h5'), 'a') as h:
                h['Mesh/x'] = np.array([0, 1, 2 + delta])
        self.assertEqual(self.worst()['Et.h5:Mesh/x'], np.inf)
        with open(os.path.join(self.b, 'ut'), 'w') as f:
            f.write('invalid probe\n')
        self.assertEqual(self.worst()['ut'], np.inf)

    def test_relative_l2_catches_error_hidden_by_a_single_peak(self):
        write_run(self.b, self.probe, self.time, self.dump, e_phi=1e-9)
        time = np.arange(1001, dtype=float)
        a = np.zeros(1001)
        a[0] = 1
        b = a.copy()
        b[1:] = 9e-5
        for path, values in ((self.a, a), (self.b, b)):
            np.savetxt(os.path.join(path, 'ut'), np.column_stack([time, values]))
        metrics = {name: (peak, l2) for name, peak, l2 in compare_sim_metrics(self.a, self.b)}
        self.assertLess(metrics['ut'][0], 1e-4)
        self.assertGreater(metrics['ut'][1], 1e-3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([self.a, self.b, '--l2-rtol', '1e-3']), 1)

    def test_group_sampling_and_mesh_attributes_are_exact(self):
        write_run(self.b, self.probe, self.time, self.dump, e_phi=1e-9)
        for filename, group, key, av, bv in (
                ('nf2ff.h5', 'nf2ff', 'Frequency', [2.4e9], [5.8e9]),
                ('Et.h5', 'FieldData/FD', 'frequency', [1e9], [1e9 + 1]),
                ('Et.h5', 'Mesh', 'mesh_scaling', 1e-3, 1e-3 + 1e-12),
                ('Et.h5', 'Mesh', 'mesh_type', 0, 1),
                ('nf2ff.h5', 'nf2ff', 'Frequency', [np.inf], [np.inf])):
            with self.subTest(group=group, key=key, value=bv):
                for path, value in ((self.a, av), (self.b, bv)):
                    with h5py.File(os.path.join(path, filename), 'a') as h:
                        h.require_group(group).attrs[key] = value
                self.assertEqual(self.worst()[f'{filename}:{group}@{key}'], np.inf)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([self.a, self.b, '--l2-rtol', '1e-3']), 1)
                with h5py.File(os.path.join(self.b, filename), 'a') as h:
                    del h[group].attrs[key]
                self.assertEqual(self.worst()[f'{filename}:{group}@{key}'], np.inf)
                with h5py.File(os.path.join(self.a, filename), 'a') as h:
                    del h[group].attrs[key]

    def test_nf2ff_scalar_results_use_numerical_tolerances(self):
        write_run(self.b, self.probe, self.time, self.dump, e_phi=1e-9)
        for path, value in ((self.a, [1., 2.]), (self.b, [1., 2.0005])):
            with h5py.File(os.path.join(path, 'nf2ff.h5'), 'a') as h:
                h['nf2ff'].attrs['Prad'] = value
                h['nf2ff'].attrs['Dmax'] = value
        self.assertAlmostEqual(self.worst()['nf2ff.h5:nf2ff@Prad'], 2.5e-4)
        metrics = {name: (peak, l2) for name, peak, l2 in compare_sim_metrics(self.a, self.b)}
        self.assertGreater(metrics['nf2ff.h5'][1], 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([self.a, self.b, '--rtol', '1e-3', '--l2-rtol', '1e-3']), 0)

    def test_vtr_nonfinite_fields_and_mesh_changes_fail(self):
        write_run(self.b, self.probe, self.time, self.dump, e_phi=1e-9)
        for path in (self.a, self.b):
            open(os.path.join(path, 'Et_0001.vtr'), 'w').close()
        mesh = (np.array([0., 1.]), np.array([0.]), np.array([0.]))
        # Isolate the comparator from the optional VTK reader dependency.
        with patch.dict(sys.modules, {'vtk': object()}):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(value=value), patch.object(compare, '_read_vtr',
                        return_value=({'E': np.array([value])}, mesh)):
                    self.assertEqual(self.worst()['Et_0001.vtr'], np.inf)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main([self.a, self.b]), 1)
            changed = (mesh[0] + 1e-12, mesh[1], mesh[2])
            def read(filename):
                return {'E': np.ones(2)}, mesh if os.path.dirname(filename) == self.a else changed
            with patch.object(compare, '_read_vtr', side_effect=read):
                self.assertEqual(self.worst()['Et_0001.vtr'], np.inf)

    def test_vtr_l2_excludes_mesh_and_detects_small_repeated_errors(self):
        write_run(self.b, self.probe, self.time, self.dump, e_phi=1e-9)
        for path in (self.a, self.b):
            open(os.path.join(path, 'Et_0001.vtr'), 'w').close()
        a = np.zeros(1001)
        a[0] = 1
        b = a.copy()
        b[1:] = 9e-5
        mesh = (np.array([1e20]),) * 3
        def read(filename):
            return {'E': a if os.path.dirname(filename) == self.a else b}, mesh
        with patch.dict(sys.modules, {'vtk': object()}), patch.object(compare, '_read_vtr', side_effect=read):
            metrics = {name: (peak, l2) for name, peak, l2 in compare_sim_metrics(self.a, self.b)}
            self.assertLess(metrics['Et_*.vtr'][0], 1e-4)
            self.assertGreater(metrics['Et_*.vtr'][1], 1e-3)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([self.a, self.b, '--l2-rtol', '1e-3']), 1)


if __name__ == '__main__':
    unittest.main()
