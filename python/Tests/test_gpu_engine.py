# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 openEMS Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#

import unittest
import numpy as np

from CSXCAD import ContinuousStructure
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
    def test_gpu_command_line_option(self):
        """Verify that openEMS accepts --engine=webgpu and --engine=gpu arguments."""
        fdtd = openEMS()
        self.assertIsNotNone(fdtd)
        # Passing GPU options should be accepted cleanly
        fdtd.SetLibraryArguments(["--engine=webgpu"])
        fdtd_gpu = openEMS()
        fdtd_gpu.SetLibraryArguments(["--engine=gpu"])

    def test_graceful_fallback_lorentz_material(self):
        """Verify that a model with unsupported Lorentz material falls back gracefully."""
        csx = _make_grid()
        # Add a Lorentz dispersive material
        mat = csx.AddLorentzMaterial('DispersivePlasma')
        mat.SetParams(eps_inf=1.0, f0=1e9, gamma=1e6)
        start = [-10, -10, -2]
        stop = [10, 10, 2]
        mat.AddBox(start, stop)

        fdtd = openEMS(NrTS=50)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        fdtd.SetLibraryArguments(["--engine=gpu"])

        # SetupFDTD should detect Lorentz material and fall back to CPU without throwing an exception
        ret = fdtd.SetupFDTD()
        self.assertEqual(ret, 0)

    def test_standard_model_gpu_setup(self):
        """Verify that a standard antenna/microwave model with UPML boundaries initializes."""
        csx = _make_grid()
        metal = csx.AddMetal('PEC_Sheet')
        start = [-5, -5, 0]
        stop = [5, 5, 0]
        metal.AddBox(start, stop)

        fdtd = openEMS(NrTS=50)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(1e9, 0.5e9)
        fdtd.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        fdtd.SetLibraryArguments(["--engine=gpu"])

        ret = fdtd.SetupFDTD()
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()
