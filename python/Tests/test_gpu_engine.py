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
import tempfile
import unittest
import numpy as np

from CSXCAD import ContinuousStructure
from CSXCAD.CSProperties import CSPropLorentzMaterial
from openEMS.openEMS import openEMS


def _make_grid():
    csx = ContinuousStructure()
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.SetLines('x', np.linspace(-50, 50, 11))
    grid.SetLines('y', np.linspace(-50, 50, 11))
    grid.SetLines('z', np.linspace(-5, 5, 5))
    return csx


class Test_GPUEngine(unittest.TestCase):
    def setUp(self):
        self.sim_dir = os.path.join(tempfile.gettempdir(), 'test_gpu_openems')
        self._orig_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._orig_cwd)

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

    def test_graceful_fallback_lorentz_material(self):
        """Verify that a model with unsupported Lorentz material falls back gracefully."""
        csx = _make_grid()
        lorentz = CSPropLorentzMaterial(csx.GetParameterSet(), epsilon=2.0, order=1)
        lorentz.SetName('DispersivePlasma')
        csx.AddProperty(lorentz)
        lorentz.AddBox(start=[-10, -10, -2], stop=[10, 10, 2])

        fdtd = openEMS(NrTS=15)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8'] * 6)

        # Run with engine='gpu' should detect Lorentz and fall back to multithreaded CPU
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

                sim_dir = os.path.join(tempfile.gettempdir(), f'test_size_{nx}_{ny}_{nz}')
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

        sim_dir = os.path.join(tempfile.gettempdir(), 'test_multisource_gpu')
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

        sim_cpu = os.path.join(tempfile.gettempdir(), 'test_probes_cpu')
        sim_gpu = os.path.join(tempfile.gettempdir(), 'test_probes_gpu')

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

        sim_cpu = os.path.join(tempfile.gettempdir(), 'test_lumped_cpu')
        sim_gpu = os.path.join(tempfile.gettempdir(), 'test_lumped_gpu')

        run_port('multithreaded', sim_cpu)
        run_port('vulkan', sim_gpu)

        cpu_port = np.loadtxt(os.path.join(sim_cpu, 'port_ut_1'), comments='%')
        gpu_port = np.loadtxt(os.path.join(sim_gpu, 'port_ut_1'), comments='%')
        diff = np.max(np.abs(cpu_port[:, 1] - gpu_port[:, 1]))
        peak = np.max(np.abs(cpu_port[:, 1])) + 1e-12
        rel_diff = diff / peak
        self.assertLess(rel_diff, 0.0005, f"Lumped port relative diff {rel_diff:.4%} exceeded 0.05%")


if __name__ == '__main__':
    unittest.main()
