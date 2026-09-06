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

    def test_gpu_command_line_option(self):
        """Verify that openEMS accepts engine='webgpu' and engine='gpu' keyword arguments."""
        csx = _make_grid()
        fdtd = openEMS(NrTS=15)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8'] * 6)
        ret_webgpu = fdtd.Run(self.sim_dir, setup_only=True, engine='webgpu')
        self.assertEqual(ret_webgpu, 0)

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

        ret = fdtd.Run(self.sim_dir, setup_only=True, engine='webgpu')
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()
