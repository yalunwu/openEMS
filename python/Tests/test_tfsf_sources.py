# Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
"""Independent TFSF source tables superpose across engines."""
import os
import tempfile
import unittest
import numpy as np
from GPU_Engine import case_tfsf_multiple, run_captured


class TFSFSourceTests(unittest.TestCase):
    def test_superposition_and_enabled_state(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory(prefix='openems_tfsf_sources_') as root:
            try:
                cases = (('a', ('a',), False, 'basic'), ('b', ('b',), False, 'basic'),
                         ('ab', ('a', 'b'), False, 'basic'), ('disabled', ('a', 'b'), True, 'basic'),
                         ('reference', ('a', 'b'), False, 'gpu-reference'))
                for name, sources, disabled, engine in cases:
                    run_captured(lambda: case_tfsf_multiple(sources, disabled), os.path.join(root, name), engine)
                for probe in ('e', 'h'):
                    values = {name: np.loadtxt(os.path.join(root, name, probe), comments='%') for name, _, _, _ in cases}
                    for name in values:
                        np.testing.assert_array_equal(values[name][:, 0], values['ab'][:, 0])
                    expected = values['a'][:, 1:] + values['b'][:, 1:]
                    peak = np.max(np.abs(expected))
                    self.assertGreater(np.max(np.abs(values['a'][:, 1:])), 0)
                    self.assertGreater(np.max(np.abs(values['b'][:, 1:])), 0)
                    self.assertLessEqual(np.max(np.abs(values['ab'][:, 1:] - expected)), 2e-5 * peak)
                    np.testing.assert_array_equal(values['ab'], values['disabled'])
                    np.testing.assert_array_equal(values['ab'], values['reference'])
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
