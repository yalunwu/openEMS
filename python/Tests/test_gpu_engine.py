# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 openEMS Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#

import os
import ctypes
import shutil
import sys
import re
import subprocess
import unittest
import uuid
import xml.etree.ElementTree as ET
import h5py
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Keep Windows DLL-directory handles alive for the duration of the test
# process. Test runners supply the build/install locations through PATH.
_dll_directory_handles = []
if hasattr(os, 'add_dll_directory'):
    candidate_paths = [
        os.path.join(repo_root, 'build'),
        os.path.join(os.path.dirname(repo_root), 'CSXCAD', 'build', 'src'),
        os.path.join(sys.prefix, 'bin'),
    ]
    for env_name in ('OPENEMS_INSTALL_PATH', 'CSXCAD_INSTALL_PATH'):
        install_path = os.environ.get(env_name)
        if install_path:
            candidate_paths.extend([install_path, os.path.join(install_path, 'bin')])
    candidate_paths.extend(os.environ.get('PATH', '').split(os.pathsep))
    for path_entry in dict.fromkeys(candidate_paths):
        if os.path.isdir(path_entry):
            try:
                _dll_directory_handles.append(os.add_dll_directory(path_entry))
            except OSError:
                pass

    # Resolve the local backend before the installed Python bindings can load
    # another libopenEMS.dll through their dependency search directories.
    _local_backend = os.path.join(repo_root, 'build', 'libopenEMS.dll')
    if os.path.isfile(_local_backend):
        _dll_directory_handles.append(ctypes.CDLL(_local_backend))

from CSXCAD import ContinuousStructure
from CSXCAD.CSProperties import CSPropLorentzMaterial, CSPropDebyeMaterial
from openEMS.openEMS import openEMS


def _make_grid():
    csx = ContinuousStructure()
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.SetLines('x', np.linspace(-50, 50, 11))
    grid.SetLines('y', np.linspace(-50, 50, 11))
    grid.SetLines('z', np.linspace(-5, 5, 5))
    return csx


def _run_disabled_output_case(kind, sim_dir, engine='vulkan'):
    csx = ContinuousStructure()
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    for axis in ('x', 'y', 'z'):
        grid.SetLines(axis, np.linspace(-2, 2, 5))
    # An active probe supplies sample boundaries even when dumps are disabled.
    probe = csx.AddProbe('voltage', p_type=0, over_sampling=2)
    probe.AddBox([0, 0, -1], [0, 0, 1])
    if kind != 'none':
        dump = csx.AddDump('fields', dump_type=10 if kind == 'fd' else 0,
                           file_type=1)
        dump.AddBox([-2, -2, -2], [2, 2, 2])
        if kind == 'fd':
            dump.AddFrequency(20e9)
    fdtd = openEMS(NrTS=64, EndCriteria=0)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(20e9, 10e9)
    fdtd.SetBoundaryCond(['PEC'] * 6)
    fdtd.Run(sim_dir, engine=engine, vulkan_profile=True,
             vulkan_batch_size=1, numThreads=1,
             disable_dumps=(kind != 'enabled'), cleanup=False)


class Test_GPUEngine(unittest.TestCase):
    def setUp(self):
        temp_root = os.environ.get('OPENEMS_TEST_TMPDIR', os.path.join(repo_root, 'build'))
        os.makedirs(temp_root, exist_ok=True)
        self._temp_dir = os.path.join(temp_root, 'openems_gpu_' + uuid.uuid4().hex)
        os.makedirs(self._temp_dir)
        self.sim_dir = os.path.join(self._temp_dir, 'default')
        self._orig_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _sim_path(self, name):
        return os.path.join(self._temp_dir, name)

    def test_gpu_command_line_option(self):
        """Verify that openEMS accepts engine='vulkan' and engine='gpu' keyword arguments."""
        csx = _make_grid()
        fdtd = openEMS(NrTS=15)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8'] * 6)
        ret_vulkan = fdtd.Run(self.sim_dir, setup_only=True, engine='vulkan')
        self.assertEqual(ret_vulkan, 0)

        fdtd2 = openEMS(NrTS=15)
        fdtd2.SetCSX(_make_grid())
        fdtd2.SetGaussExcite(1e9, 0.5e9)
        fdtd2.SetBoundaryCond(['PML_8'] * 6)
        ret_gpu = fdtd2.Run(self.sim_dir, setup_only=True, engine='gpu')
        self.assertEqual(ret_gpu, 0)

    def test_lorentz_material_setup(self):
        """Verify that a model with Lorentz dispersive material initializes and runs setup on GPU."""
        csx = _make_grid()
        lorentz = CSPropLorentzMaterial(csx.GetParameterSet(), epsilon=2.0, order=1)
        lorentz.SetName('DispersivePlasma')
        csx.AddProperty(lorentz)
        lorentz.AddBox(start=[-10, -10, -2], stop=[10, 10, 2])

        fdtd = openEMS(NrTS=15)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8'] * 6)

        ret = fdtd.Run(self.sim_dir, setup_only=True, engine='gpu')
        self.assertEqual(ret, 0)

    def test_standard_model_gpu_setup(self):
        """Verify that a standard model with UPML boundaries initializes on GPU engine."""
        csx = _make_grid()
        metal = csx.AddMetal('PEC_Sheet')
        metal.AddBox(start=[-5, -5, 0], stop=[5, 5, 0])

        fdtd = openEMS(NrTS=15)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8'] * 6)

        ret = fdtd.Run(self.sim_dir, setup_only=True, engine='vulkan')
        self.assertEqual(ret, 0)

    def test_different_grid_sizes(self):
        """Verify Vulkan engine on a range of grid sizes (sub-warp, asymmetric, thin slab)."""
        sizes = [
            (7, 9, 11),   # Sub-warp: dimZ < 32
            (17, 7, 33),  # Asymmetric non-multiples
            (25, 25, 3),  # Thin planar slab
        ]
        for nx, ny, nz in sizes:
            with self.subTest(grid_size=(nx, ny, nz)):
                csx = ContinuousStructure()
                grid = csx.GetGrid()
                grid.SetDeltaUnit(1e-3)
                grid.SetLines('x', np.linspace(-20, 20, nx))
                grid.SetLines('y', np.linspace(-20, 20, ny))
                grid.SetLines('z', np.linspace(-20, 20, nz))

                fdtd = openEMS(NrTS=15)
                fdtd.SetCSX(csx)
                fdtd.SetGaussExcite(1e9, 0.5e9)
                fdtd.SetBoundaryCond(['PEC'] * 6)

                sim_dir = self._sim_path(f'test_size_{nx}_{ny}_{nz}')
                ret = fdtd.Run(sim_dir, engine='vulkan', cleanup=True)
                self.assertIn(ret, [0, None])

    def test_multisource_excitation(self):
        """Verify Vulkan engine with multiple independent excitation sources."""
        csx = ContinuousStructure()
        grid = csx.GetGrid()
        grid.SetDeltaUnit(1e-3)
        grid.SetLines('x', np.linspace(-30, 30, 25))
        grid.SetLines('y', np.linspace(-30, 30, 25))
        grid.SetLines('z', np.linspace(-30, 30, 25))

        exc1 = csx.AddExcitation('excite_x', exc_type=0, exc_val=[1, 0, 0])
        exc1.AddBox([-15, -5, -2], [-5, 5, 2])

        exc2 = csx.AddExcitation('excite_z', exc_type=0, exc_val=[0, 0, 1])
        exc2.AddBox([5, -5, -2], [15, 5, 2])

        fdtd = openEMS(NrTS=20)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PEC'] * 6)

        sim_dir = self._sim_path('test_multisource_gpu')
        ret = fdtd.Run(sim_dir, engine='vulkan', cleanup=True)
        self.assertIn(ret, [0, None])

    def test_multi_axis_probes_fidelity(self):
        """Verify multi-axis voltage probes match CPU results to within 0.05%."""
        def build_setup():
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(-25, 25, 30))
            grid.SetLines('y', np.linspace(-25, 25, 30))
            grid.SetLines('z', np.linspace(-25, 25, 30))

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-2, -2, -2], [2, 2, 2])

            probe_x = csx.AddProbe('probe_x', p_type=0)
            probe_x.AddBox([5, 0, 0], [10, 0, 0])

            probe_y = csx.AddProbe('probe_y', p_type=0)
            probe_y.AddBox([0, 5, 0], [0, 10, 0])

            probe_z = csx.AddProbe('probe_z', p_type=0)
            probe_z.AddBox([0, 0, 5], [0, 0, 10])

            fdtd = openEMS(NrTS=60)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['PEC'] * 6)
            return fdtd

        sim_cpu = self._sim_path('test_probes_cpu')
        sim_gpu = self._sim_path('test_probes_gpu')

        fdtd_cpu = build_setup()
        fdtd_cpu.Run(sim_cpu, engine='multithreaded', cleanup=False)

        fdtd_gpu = build_setup()
        fdtd_gpu.Run(sim_gpu, engine='vulkan', cleanup=False)

        for p_name in ['probe_x', 'probe_y', 'probe_z']:
            cpu_file = os.path.join(sim_cpu, p_name)
            gpu_file = os.path.join(sim_gpu, p_name)
            self.assertTrue(os.path.exists(cpu_file), f"CPU {p_name} missing")
            self.assertTrue(os.path.exists(gpu_file), f"GPU {p_name} missing")
            data_cpu = np.loadtxt(cpu_file, comments='%')
            data_gpu = np.loadtxt(gpu_file, comments='%')
            diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
            peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
            rel_diff = diff / peak
            self.assertLess(rel_diff, 0.0005, f"{p_name} relative difference {rel_diff:.4%} exceeded 0.05%")

    def test_mixed_boundary_conditions(self):
        """Verify Vulkan engine with mixed boundary conditions (PEC, PMC, Mur)."""
        csx = _make_grid()
        fdtd = openEMS(NrTS=20)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PEC', 'PEC', 'PMC', 'PMC', 'MUR', 'MUR'])
        ret = fdtd.Run(self.sim_dir, engine='vulkan', cleanup=True)
        self.assertIn(ret, [0, None])

    def test_lumped_port(self):
        """Verify lumped port simulation on Vulkan matches multithreaded CPU."""
        def run_port(engine_name, sdir):
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(-25, 25, 30))
            grid.SetLines('y', np.linspace(-25, 25, 30))
            grid.SetLines('z', np.linspace(-25, 25, 30))

            fdtd = openEMS(NrTS=80)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['PEC'] * 6)
            fdtd.AddLumpedPort(port_nr=1, R=50, start=[-3, -3, -5], stop=[3, 3, 5], p_dir='z', excite=1.0)
            ret = fdtd.Run(sdir, engine=engine_name, cleanup=False)
            self.assertIn(ret, [0, None])

        sim_cpu = self._sim_path('test_lumped_cpu')
        sim_gpu = self._sim_path('test_lumped_gpu')

        run_port('multithreaded', sim_cpu)
        run_port('vulkan', sim_gpu)

        cpu_port = np.loadtxt(os.path.join(sim_cpu, 'port_ut_1'), comments='%')
        gpu_port = np.loadtxt(os.path.join(sim_gpu, 'port_ut_1'), comments='%')
        diff = np.max(np.abs(cpu_port[:, 1] - gpu_port[:, 1]))
        peak = np.max(np.abs(cpu_port[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.0005, f"Lumped port relative diff {rel_diff:.4%} exceeded 0.05%")

    def test_upml_boundary_fidelity(self):
        """Verify that 6-sided UPML boundaries on Vulkan match CPU within 0.05%."""
        def run_sim(eng, sdir):
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(-30, 30, 31))
            grid.SetLines('y', np.linspace(-30, 30, 31))
            grid.SetLines('z', np.linspace(-30, 30, 31))

            exc = csx.AddExcitation('excite_z', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-3, -3, -5], [3, 3, 5])

            p_center = csx.AddProbe('probe_center', p_type=0)
            p_center.AddBox([0, 0, -2], [0, 0, 2])

            p_edge = csx.AddProbe('probe_edge', p_type=0)
            p_edge.AddBox([0, 15, -2], [0, 15, 2])

            fdtd = openEMS(NrTS=100, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['PML_8'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_upml_cpu')
        sim_gpu = self._sim_path('test_upml_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_cpu, [0, None])
        self.assertIn(ret_gpu, [0, None])

        for pname in ['probe_center', 'probe_edge']:
            f_cpu = os.path.join(sim_cpu, pname)
            f_gpu = os.path.join(sim_gpu, pname)
            self.assertTrue(os.path.exists(f_cpu), f"CPU {pname} missing")
            self.assertTrue(os.path.exists(f_gpu), f"GPU {pname} missing")
            data_cpu = np.loadtxt(f_cpu, comments='%')
            data_gpu = np.loadtxt(f_gpu, comments='%')
            diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
            peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
            rel_diff = diff / peak
            self.assertLess(rel_diff, 0.0005, f"{pname} relative error {rel_diff:.4%} exceeded 0.05%")

    def test_mur_abc_boundary_fidelity(self):
        """Verify that 6-sided Mur ABC boundaries on Vulkan match CPU within 0.05%."""
        def run_sim(eng, sdir):
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(-25, 25, 26))
            grid.SetLines('y', np.linspace(-25, 25, 26))
            grid.SetLines('z', np.linspace(-25, 25, 26))

            exc = csx.AddExcitation('excite_z', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-3, -3, -5], [3, 3, 5])

            p_center = csx.AddProbe('probe_center', p_type=0)
            p_center.AddBox([0, 0, -2], [0, 0, 2])

            p_edge = csx.AddProbe('probe_edge', p_type=0)
            p_edge.AddBox([0, 15, -2], [0, 15, 2])

            fdtd = openEMS(NrTS=80, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_mur_cpu')
        sim_gpu = self._sim_path('test_mur_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_cpu, [0, None])
        self.assertIn(ret_gpu, [0, None])

        for pname in ['probe_center', 'probe_edge']:
            f_cpu = os.path.join(sim_cpu, pname)
            f_gpu = os.path.join(sim_gpu, pname)
            self.assertTrue(os.path.exists(f_cpu), f"CPU {pname} missing")
            self.assertTrue(os.path.exists(f_gpu), f"GPU {pname} missing")
            data_cpu = np.loadtxt(f_cpu, comments='%')
            data_gpu = np.loadtxt(f_gpu, comments='%')
            diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
            peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
            rel_diff = diff / peak
            self.assertLess(rel_diff, 0.0005, f"{pname} relative error {rel_diff:.4%} exceeded 0.05%")

    def test_steady_state_detection(self):
        """Verify that sinusoidal CW excitation with steady-state detection runs cleanly on Vulkan."""
        csx = ContinuousStructure()
        grid = csx.GetGrid()
        grid.SetDeltaUnit(1e-3)
        grid.SetLines('x', np.linspace(-20, 20, 21))
        grid.SetLines('y', np.linspace(-20, 20, 21))
        grid.SetLines('z', np.linspace(-20, 20, 21))

        exc = csx.AddExcitation('excite_sinus', exc_type=0, exc_val=[0, 0, 1])
        exc.AddBox([-3, -3, -3], [3, 3, 3])

        p_center = csx.AddProbe('probe_center', p_type=0)
        p_center.AddBox([0, 0, -2], [0, 0, 2])

        fdtd = openEMS(NrTS=160)
        fdtd.SetCSX(csx)
        fdtd.SetSinusExcite(10e9)
        fdtd.SetBoundaryCond(['MUR'] * 6)

        sim_gpu = self._sim_path('test_ss_gpu')
        ret_gpu = fdtd.Run(sim_gpu, engine='vulkan', cleanup=True)
        self.assertIn(ret_gpu, [0, None])

        probe_path = os.path.join(sim_gpu, 'probe_center')
        self.assertTrue(os.path.exists(probe_path), 'steady-state probe output missing')
        probe_data = np.atleast_2d(np.loadtxt(probe_path, comments='%'))
        self.assertGreater(probe_data.shape[0], 20)
        self.assertTrue(np.all(np.isfinite(probe_data[:, 1])))
        self.assertGreater(np.max(np.abs(probe_data[:, 1])), 0.0)

    def test_tfsf_plane_wave_execution(self):
        """Verify that TFSF plane-wave excitation runs on Vulkan and matches CPU fields."""
        def run_sim(eng, sdir):
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(-20, 20, 21))
            grid.SetLines('y', np.linspace(-20, 20, 21))
            grid.SetLines('z', np.linspace(-20, 20, 21))

            pw = csx.AddExcitation('PlaneWave', 10, [0, 0, 1])
            pw.SetPropagationDir([1, 0, 0])
            pw.AddBox([-12, -12, -12], [12, 12, 12])

            p_in = csx.AddProbe('probe_inside', p_type=0)
            p_in.AddBox([0, 0, -2], [0, 0, 2])

            fdtd = openEMS(NrTS=50, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_tfsf_cpu')
        sim_gpu = self._sim_path('test_tfsf_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_inside')
        f_gpu = os.path.join(sim_gpu, 'probe_inside')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_inside missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_inside missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"TFSF relative probe error {rel_diff:.4%} exceeded 0.1%")

    def test_lumped_rlc_execution(self):
        """Verify lumped RLC load runs on Vulkan and matches CPU reference."""
        def run_sim(eng, sdir):
            os.makedirs(sdir, exist_ok=True)
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            lines = np.linspace(-20.0, 20.0, 21)
            grid.SetLines('x', lines)
            grid.SetLines('y', lines)
            grid.SetLines('z', lines)

            rlc = csx.AddLumpedElement('rlc_parallel', ny=2, caps=False, R=50.0, L=1e-9, C=1e-12, LEtype=0)
            rlc.AddBox([-1, -1, 0], [1, 1, 2])

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-3, -3, -10], [3, 3, -6])

            p = csx.AddProbe('probe_rlc', p_type=0)
            p.AddBox([0, 0, 0], [0, 0, 2])

            fdtd = openEMS(NrTS=50, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_rlc_cpu')
        sim_gpu = self._sim_path('test_rlc_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_rlc')
        f_gpu = os.path.join(sim_gpu, 'probe_rlc')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_rlc missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_rlc missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"Lumped RLC relative probe error {rel_diff:.4%} exceeded 0.1%")

    def test_absorbing_bc_equivalence(self):
        """Verify Absorbing BC sheet execution and CPU vs GPU equivalence."""
        def run_sim(eng, sdir):
            os.makedirs(sdir, exist_ok=True)
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            lines = np.linspace(-20.0, 20.0, 21)
            grid.SetLines('x', lines)
            grid.SetLines('y', lines)
            grid.SetLines('z', lines)

            abc = csx.AddAbsorbingBC('abc_sheet', AbsorbingBoundaryType=2, NormalSignPositive=True, PhaseVelocity=3e8)
            abc.AddBox([-20, -20, 10], [20, 20, 10])

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-1, -1, -5], [1, 1, -3])

            p = csx.AddProbe('probe_abc', p_type=0)
            p.AddBox([0, 0, 4], [0, 0, 6])

            fdtd = openEMS(NrTS=50, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_abc_cpu')
        sim_gpu = self._sim_path('test_abc_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_abc')
        f_gpu = os.path.join(sim_gpu, 'probe_abc')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_abc missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_abc missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"Absorbing BC relative probe error {rel_diff:.4%} exceeded 0.1%")

    def test_lorentz_material_equivalence(self):
        """Verify Lorentz dispersive material simulation and CPU vs GPU equivalence."""
        def run_sim(eng, sdir):
            os.makedirs(sdir, exist_ok=True)
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            lines = np.linspace(-20.0, 20.0, 21)
            grid.SetLines('x', lines)
            grid.SetLines('y', lines)
            grid.SetLines('z', lines)

            lor = CSPropLorentzMaterial(csx.GetParameterSet(), epsilon=2.0, order=1)
            lor.SetName('LorMat')
            lor.SetDispersiveMaterialProperty(0, eps_plasma=2e9, eps_pole_freq=1e9, eps_relax=1e-9)
            csx.AddProperty(lor)
            lor.AddBox([-5, -5, -5], [5, 5, 5])

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-1, -1, -8], [1, 1, -6])

            p = csx.AddProbe('probe_lor', p_type=0)
            p.AddBox([0, 0, 2], [0, 0, 4])

            fdtd = openEMS(NrTS=30, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_lor_cpu')
        sim_gpu = self._sim_path('test_lor_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_lor')
        f_gpu = os.path.join(sim_gpu, 'probe_lor')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_lor missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_lor missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"Lorentz material relative probe error {rel_diff:.4%} exceeded 0.1%")

    def test_debye_material_equivalence(self):
        """Verify Debye dispersive material simulation and CPU vs GPU equivalence."""
        def run_sim(eng, sdir):
            os.makedirs(sdir, exist_ok=True)
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            lines = np.linspace(-20.0, 20.0, 21)
            grid.SetLines('x', lines)
            grid.SetLines('y', lines)
            grid.SetLines('z', lines)

            deb = CSPropDebyeMaterial(csx.GetParameterSet(), epsilon=2.0, order=1)
            deb.SetName('DebMat')
            deb.SetDispersiveMaterialProperty(0, eps_delta=3.0, eps_relax=1e-9)
            csx.AddProperty(deb)
            deb.AddBox([-5, -5, -5], [5, 5, 5])

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-1, -1, -8], [1, 1, -6])

            p = csx.AddProbe('probe_deb', p_type=0)
            p.AddBox([0, 0, 2], [0, 0, 4])

            fdtd = openEMS(NrTS=30, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_deb_cpu')
        sim_gpu = self._sim_path('test_deb_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_deb')
        f_gpu = os.path.join(sim_gpu, 'probe_deb')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_deb missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_deb missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"Debye material relative probe error {rel_diff:.4%} exceeded 0.1%")

    def test_conducting_sheet_equivalence(self):
        """Verify Conducting Sheet simulation and CPU vs GPU equivalence."""
        def run_sim(eng, sdir):
            os.makedirs(sdir, exist_ok=True)
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            lines = np.linspace(-20.0, 20.0, 21)
            grid.SetLines('x', lines)
            grid.SetLines('y', lines)
            grid.SetLines('z', lines)

            sheet = csx.AddConductingSheet('Sheet')
            sheet.SetConductivity(1e5)
            sheet.SetThickness(50e-6)
            sheet.AddBox([-10, -10, 0], [10, 10, 0])

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-1, -1, -8], [1, 1, -6])

            p = csx.AddProbe('probe_sheet', p_type=0)
            p.AddBox([0, 0, 2], [0, 0, 4])

            fdtd = openEMS(NrTS=30, EndCriteria=0.0)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_sheet_cpu')
        sim_gpu = self._sim_path('test_sheet_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_sheet')
        f_gpu = os.path.join(sim_gpu, 'probe_sheet')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_sheet missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_sheet missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"Conducting Sheet relative probe error {rel_diff:.4%} exceeded 0.1%")

    def test_cylinder_coords_equivalence(self):
        """Verify Cylindrical coordinates simulation and CPU vs GPU equivalence."""
        def run_sim(eng, sdir):
            os.makedirs(sdir, exist_ok=True)
            csx = ContinuousStructure(CoordSystem=1)
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(0.0, 50.0, 16))
            grid.SetLines('y', np.linspace(-np.pi, np.pi, 16))
            grid.SetLines('z', np.linspace(0.0, 50.0, 16))

            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([15, -0.5, 15], [35, 0.5, 35])

            p = csx.AddProbe('probe_cyl', p_type=0)
            p.AddBox([25, 0, 15], [25, 0, 35])

            fdtd = openEMS(NrTS=60, EndCriteria=0.0, CoordSystem=1, OverSampling=50)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(1e9, 0.5e9)
            fdtd.SetBoundaryCond(['PEC'] * 6)
            return fdtd.Run(sdir, engine=eng, cleanup=False)

        sim_cpu = self._sim_path('test_cyl_cpu')
        sim_gpu = self._sim_path('test_cyl_gpu')

        ret_cpu = run_sim('multithreaded', sim_cpu)
        self.assertIn(ret_cpu, [0, None])

        ret_gpu = run_sim('vulkan', sim_gpu)
        self.assertIn(ret_gpu, [0, None])

        f_cpu = os.path.join(sim_cpu, 'probe_cyl')
        f_gpu = os.path.join(sim_gpu, 'probe_cyl')
        self.assertTrue(os.path.exists(f_cpu), "CPU probe_cyl missing")
        self.assertTrue(os.path.exists(f_gpu), "GPU probe_cyl missing")
        data_cpu = np.loadtxt(f_cpu, comments='%')
        data_gpu = np.loadtxt(f_gpu, comments='%')
        if data_cpu.ndim == 1:
            data_cpu = data_cpu[np.newaxis, :]
            data_gpu = data_gpu[np.newaxis, :]
        diff = np.max(np.abs(data_cpu[:, 1] - data_gpu[:, 1]))
        peak = np.max(np.abs(data_cpu[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.001, f"Cylinder coordinates relative probe error {rel_diff:.4%} exceeded 0.1%")

    def _run_multigrid_pair(self, splits, name, field_dumps=False):
        """Compare probes in the refined and active regions against the CPU."""
        outputs = {}
        for engine in ('multithreaded', 'vulkan'):
            sdir = self._sim_path(name + '_' + engine)
            csx = ContinuousStructure(CoordSystem=1)
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(0.0, 40.0, 41))
            grid.SetLines('y', np.linspace(-np.pi, np.pi, 33))
            grid.SetLines('z', np.linspace(0.0, 20.0, 17))
            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([0, -0.4, 8], [36, 0.4, 12])
            for radius in (4, 18, 32):
                probe = csx.AddProbe('probe_' + str(radius), p_type=0)
                probe.AddBox([radius, 0, 5], [radius, 0, 15])
            if field_dumps:
                for level in range(len(splits) + 1):
                    dump = csx.AddDump('level_' + str(level), dump_type=0, dump_mode=0, file_type=1)
                    dump.AddBox([0, -0.4, 5], [40, 0.4, 15])
            fdtd = openEMS(NrTS=160, EndCriteria=0.0, CoordSystem=1, OverSampling=100)
            fdtd.SetMultiGrid(splits)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(20e9, 10e9)
            fdtd.SetBoundaryCond(['PEC'] * 6)
            if field_dumps:
                # MultiGridLevel is an existing XML setting, not exposed by the
                # CSXCAD Python dump wrapper. Exercise the public XML path.
                os.makedirs(sdir, exist_ok=True)
                model_path = os.path.join(sdir, 'model.xml')
                self.assertTrue(fdtd.Write2XML(model_path))
                model = ET.parse(model_path)
                for dump in model.findall('.//DumpBox'):
                    dump.set('MultiGridLevel', dump.get('Name').split('_')[-1])
                model.write(model_path, encoding='utf-8', xml_declaration=True)
                self.assertTrue(fdtd.ReadFromXML(model_path))
            ret = fdtd.Run(sdir, engine=engine, cleanup=False, numThreads=1)
            self.assertIn(ret, [0, None])
            outputs[engine] = {
                radius: np.atleast_2d(np.loadtxt(os.path.join(sdir, 'probe_' + str(radius)), comments='%'))
                for radius in (4, 18, 32)
            }
        for radius in (4, 18, 32):
            cpu = outputs['multithreaded'][radius]
            gpu = outputs['vulkan'][radius]
            np.testing.assert_array_equal(cpu[:, 0], gpu[:, 0])
            self.assertTrue(np.isfinite(gpu).all())
            peak = np.max(np.abs(cpu[:, 1]))
            self.assertGreater(peak, 0.0, 'Multigrid probe has zero fields')
            error = np.max(np.abs(cpu[:, 1] - gpu[:, 1])) / peak
            self.assertLess(error, 0.001, f'Multigrid probe at r={radius} relative error {error:.4%}')
        if field_dumps:
            for level in range(len(splits) + 1):
                cpu_path = os.path.join(self._sim_path(name + '_multithreaded'), 'level_' + str(level) + '.h5')
                gpu_path = os.path.join(self._sim_path(name + '_vulkan'), 'level_' + str(level) + '.h5')
                with h5py.File(cpu_path, 'r') as cpu, h5py.File(gpu_path, 'r') as gpu:
                    self.assertEqual(set(cpu['FieldData/TD']), set(gpu['FieldData/TD']))
                    peak = 0.0
                    error = 0.0
                    for timestep in cpu['FieldData/TD']:
                        expected = cpu['FieldData/TD/' + timestep][...]
                        actual = gpu['FieldData/TD/' + timestep][...]
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_array_equal(cpu['FieldData/TD/' + timestep].attrs['time'],
                                                      gpu['FieldData/TD/' + timestep].attrs['time'])
                        peak = max(peak, np.max(np.abs(expected)))
                        error = max(error, np.max(np.abs(expected - actual)))
                    self.assertGreater(peak, 0.0, 'Child-level dump contains zero fields')
                    self.assertLess(error, 1e-4)
                    self.assertLess(error / peak, 0.001)

    def test_cylindrical_multigrid_equivalence(self):
        self._run_multigrid_pair([24.0], 'cyl_mg')

    def test_nested_cylindrical_multigrid_equivalence(self):
        self._run_multigrid_pair([12.0, 24.0], 'cyl_mg_nested', field_dumps=True)

    def test_mixed_probe_sampling_equivalence(self):
        """Fused gathers preserve V/I/E/H samples and derived impedance within 0.1%."""
        outputs = {}
        for engine in ('basic', 'vulkan'):
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            for axis, count in zip('xyz', (25, 17, 33)):
                grid.SetLines(axis, np.linspace(-20, 20, count))
            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-2, -2, -3], [2, 2, 3])
            for name, kind, sampling, start, stop in (
                ('voltage', 0, 1, [5, 0, -5], [5, 0, 5]),
                ('current', 1, 2, [2, -5, 0], [8, 5, 0]),
                ('electric', 2, 3, [5, 0, 0], [5, 0, 0]),
                ('magnetic', 3, 4, [5, 0, 0], [5, 0, 0]),
            ):
                probe = csx.AddProbe(name, p_type=kind, over_sampling=sampling,
                                     frequency=[20e9], norm_dir=2)
                probe.AddBox(start, stop)
            fdtd = openEMS(NrTS=321, EndCriteria=0, OverSampling=1)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(20e9, 10e9)
            fdtd.SetBoundaryCond(['MUR'] * 6)
            sdir = self._sim_path('mixed_' + engine)
            fdtd.Run(sdir, engine=engine, exact_endcriteria=True, numThreads=1, cleanup=False)
            outputs[engine] = {
                name: np.atleast_2d(np.loadtxt(os.path.join(sdir, name), comments='%'))
                for name in ('voltage', 'current', 'electric', 'magnetic')
            }
        for name in outputs['basic']:
            expected, actual = outputs['basic'][name], outputs['vulkan'][name]
            np.testing.assert_array_equal(expected[:, 0], actual[:, 0])
            self.assertTrue(np.isfinite(actual).all())
            peak = np.max(np.abs(expected[:, 1:]))
            self.assertGreater(peak, 0, name + ' has no signal')
            self.assertLess(np.max(np.abs(actual[:, 1:] - expected[:, 1:])) / peak, 0.001)
        # Compare a derived Fourier voltage/current ratio with each sampling interval.
        impedances = {}
        for engine, probes in outputs.items():
            voltage, current = probes['voltage'], probes['current']
            u = np.sum(voltage[:, 1] * np.exp(-2j * np.pi * 20e9 * voltage[:, 0])) * (voltage[1, 0] - voltage[0, 0])
            i = np.sum(current[:, 1] * np.exp(-2j * np.pi * 20e9 * current[:, 0])) * (current[1, 0] - current[0, 0])
            self.assertGreater(abs(i), 0)
            impedances[engine] = u / i
        self.assertLess(abs(impedances['vulkan'] - impedances['basic']) / abs(impedances['basic']), 0.001)

    def test_disabled_dumps_preserve_readback_volume(self):
        """Disabled TD/FD dumps transfer no more full fields than a probe-only run."""
        downloads = {}
        for kind in ('none', 'td', 'fd', 'enabled'):
            sdir = self._sim_path('disabled_' + kind)
            # Capture native C++ stdout in a child process on every platform.
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 '--disabled-output-case', kind, sdir],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, kind + ': ' + result.stdout + result.stderr)
            profile = re.search(r'VULKAN_PROFILE .*?downloaded_bytes=(\d+)', result.stdout)
            self.assertIsNotNone(profile, result.stdout + result.stderr)
            downloads[kind] = int(profile.group(1))
            if kind == 'enabled':
                self.assertTrue(os.path.isfile(os.path.join(sdir, 'fields.h5')))
            else:
                self.assertFalse(os.path.exists(os.path.join(sdir, 'fields.h5')))
        self.assertEqual(downloads['td'], downloads['none'])
        self.assertEqual(downloads['fd'], downloads['none'])
        self.assertGreater(downloads['enabled'], downloads['none'])
        # Disabled FD post-processing also skipped initialization on the CPU.
        sdir = self._sim_path('disabled_fd_basic')
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             '--disabled-output-case', 'fd', sdir, 'basic'],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(os.path.exists(os.path.join(sdir, 'fields.h5')))

    def test_vulkan_batch_sizes_preserve_samples(self):
        """Batching preserves probe/dump times and fields within 1e-4 / 0.1%."""
        outputs = {}
        for batch in (1, 32, 64):
            csx = ContinuousStructure()
            grid = csx.GetGrid()
            grid.SetDeltaUnit(1e-3)
            grid.SetLines('x', np.linspace(-20, 20, 25))
            grid.SetLines('y', np.linspace(-20, 20, 17))
            grid.SetLines('z', np.linspace(-20, 20, 33))
            exc = csx.AddExcitation('excite', exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([-2, -2, -2], [2, 2, 2])
            probe = csx.AddProbe('probe', p_type=0)
            probe.AddBox([5, 0, -5], [5, 0, 5])
            dump = csx.AddDump('fields', dump_type=0, file_type=1)
            dump.AddBox([-20, -20, 0], [20, 20, 0])
            fdtd = openEMS(NrTS=97, EndCriteria=0, OverSampling=1)
            fdtd.SetCSX(csx)
            fdtd.SetGaussExcite(20e9, 10e9)
            fdtd.SetBoundaryCond(['MUR', 'PEC', 'PEC', 'PML_4', 'PEC', 'PEC'])
            sdir = self._sim_path('batch_' + str(batch))
            fdtd.Run(sdir, engine='vulkan', vulkan_batch_size=batch,
                     vulkan_profile=(batch == 32), numThreads=1, cleanup=False)
            outputs[batch] = (np.loadtxt(os.path.join(sdir, 'probe'), comments='%'),
                              os.path.join(sdir, 'fields.h5'))
        expected, expected_dump = outputs[1]
        peak = np.max(np.abs(expected[:, 1]))
        self.assertGreater(peak, 0)
        for batch in (32, 64):
            actual, actual_dump = outputs[batch]
            np.testing.assert_array_equal(actual[:, 0], expected[:, 0])
            self.assertLess(np.max(np.abs(actual[:, 1] - expected[:, 1])), peak * 0.001)
            with h5py.File(expected_dump) as reference, h5py.File(actual_dump) as result:
                self.assertEqual(set(reference['FieldData/TD']), set(result['FieldData/TD']))
                for timestep in reference['FieldData/TD']:
                    a = reference['FieldData/TD/' + timestep]
                    b = result['FieldData/TD/' + timestep]
                    np.testing.assert_array_equal(a.attrs['time'], b.attrs['time'])
                    np.testing.assert_allclose(a[...], b[...], rtol=0.001, atol=1e-4)

if __name__ == '__main__':
    if len(sys.argv) in (4, 5) and sys.argv[1] == '--disabled-output-case':
        _run_disabled_output_case(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else 'vulkan')
    else:
        unittest.main()
