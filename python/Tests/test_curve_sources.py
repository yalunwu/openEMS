# Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
"""Curve sources retain each primitive and honor their enabled state."""
import os
import re
import tempfile
import unittest

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from GPU_Engine import run_captured


class CurveSourceTests(unittest.TestCase):
    def test_multiple_primitives_and_disabled_source(self):
        def case():
            fdtd = openEMS(NrTS=8, EndCriteria=0)
            fdtd.SetDiracExcite(5e9)
            fdtd.SetBoundaryCond(['PEC'] * 6)
            csx = ContinuousStructure()
            fdtd.SetCSX(csx)
            mesh = csx.GetGrid()
            mesh.SetDeltaUnit(1e-3)
            for axis in 'xyz':
                mesh.AddLine(axis, [-3, -2, -1, 0, 1, 2, 3])
            source = csx.AddExcitation('curves', exc_type=0, exc_val=[0, 0, 1])
            source.AddCurve([[-1, -1], [0, 0], [-1, 1]])
            source.AddCurve([[1, 1], [0, 0], [-1, 1]])
            disabled = csx.AddExcitation('disabled', exc_type=0, exc_val=[0, 0, 10])
            disabled.SetEnabled(False)
            disabled.AddCurve([[0, 0], [0, 0], [-1, 1]])
            return fdtd, csx

        with tempfile.TemporaryDirectory(prefix='openems_curve_sources_') as root:
            previous = os.getcwd()
            try:
                log = run_captured(case, os.path.join(root, 'run'), 'basic', verbose=1)
            finally:
                os.chdir(previous)
        counts = re.findall(r'Voltage excitations\s*:\s*(\d+)\s*\((\d+),\s*(\d+),\s*(\d+)\)', log)
        self.assertTrue(counts, log)
        self.assertEqual(tuple(map(int, counts[-1])), (4, 0, 0, 4))


if __name__ == '__main__':
    unittest.main()
