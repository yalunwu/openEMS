"""Vulkan FD/TD output equivalence: fields, schedules and NF2FF."""
import os
import base64
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib
import h5py
import numpy as np
import test_gpu_engine as gpu


class Test_VulkanFD(unittest.TestCase):
    def setUp(self):
        self.original = os.getcwd()
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('OPENEMS_TEST_TMPDIR'))

    def tearDown(self):
        os.chdir(self.original)
        self.temp.cleanup()

    def run_case(self, name, mode=0, accumulation='gpu', batch=32, engine='vulkan',
                 steps=129, mixed=False, nf2ff=False, legacy=False, end=0,
                 windows=True, directions=None, mirror=None, disabled=False, vtk=False,
                 td_accumulation=None, td_only=False):
        path = os.path.join(self.temp.name, name)
        csx = gpu.ContinuousStructure()
        grid = csx.GetGrid()
        grid.SetDeltaUnit(1e-3)
        grid.SetLines('x', [-20, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 20])
        grid.SetLines('y', np.linspace(-20, 20, 13))
        grid.SetLines('z', np.linspace(-20, 20, 17))
        material = csx.AddMaterial('lossy', epsilon=[2, 3, 4], kappa=[.01, .02, .03])
        material.SetIsotropy(False)
        material.AddBox([-5, -5, -5], [5, 5, 5])
        source = csx.AddExcitation('source', exc_type=0, exc_val=[0, 0, 1])
        source.AddBox([0, 0, -5], [0, 0, -2.5])
        for kind in ((0, 1) if td_only else (10, 11)):
            field = 'e' if kind in (0, 10) else 'h'
            for region, start, stop, sampling in (
                    ('volume', [-20]*3, [20]*3, {'sub_sampling': [2, 3, 2]}),
                    ('plane', [-17, -20, 0], [17, 20, 0], {'opt_resolution': [5, 6, 1]}),
                    ('line', [0, 0, -20], [0, 0, 20], {})):
                dump = csx.AddDump(field + '_' + region, dump_type=kind, dump_mode=mode,
                                   file_type=0 if vtk else 1, frequency=[10e9, 20e9], over_sampling=3, **sampling)
                dump.AddBox(start, stop)
        if mixed:
            td = csx.AddDump('td', dump_type=0, dump_mode=1, file_type=1, over_sampling=1)
            td.AddBox([-9, -10, 0], [9, 10, 0])
            probe = csx.AddProbe('voltage', p_type=0, over_sampling=100)
            probe.AddBox([5, 0, -5], [5, 0, 5])
            unsupported = csx.AddDump('d', dump_type=14, file_type=1, frequency=[20e9])
            unsupported.AddBox([-5, -5, 0], [5, 5, 0])
            unsupported_td = csx.AddDump('d_td', dump_type=4, file_type=1)
            unsupported_td.AddBox([-5, -5, 0], [5, 5, 0])
        fdtd = gpu.openEMS(NrTS=steps, EndCriteria=end)
        fdtd.SetCSX(csx)
        fdtd.SetGaussExcite(20e9, 10e9)
        fdtd.SetBoundaryCond(['MUR']*6)
        recorder = None
        if nf2ff:
            recorder = fdtd.CreateNF2FFBox(start=[-13, -13, -12.5], stop=[13, 13, 12.5],
                                         frequency=[10e9, 20e9], over_sampling=2,
                                         directions=directions or [True]*6, mirror=mirror or [0]*6)
        # CSXCAD's Python probe binding does not expose the existing XML windows.
        xml = os.path.join(self.temp.name, name + '.xml')
        csx.Write2XML(xml)
        tree = ET.parse(xml)
        for dump in tree.iter('DumpBox'):
            if windows and dump.get('Name', '').startswith(('e_', 'h_')):
                dump.set('StartTime', '2e-11')
                dump.set('StopTime', '3e-10')
        tree.write(xml)
        self.assertEqual(csx.ReadFromXML(xml), '')
        options = {}
        if accumulation is not None:
            options['vulkan_fd'] = accumulation
        if td_accumulation is not None:
            options['vulkan_td'] = td_accumulation
        fdtd.Run(path, engine=engine, vulkan_batch_size=batch, **options,
                 vulkan_profile=True, exact_endcriteria=True, numThreads=1,
                 legacyHDF5Dumps=legacy, disable_dumps=disabled,
                 dump_statistics=end>0, cleanup=False)
        return path, recorder

    def test_automatic_fd_selection(self):
        reference, _ = self.run_case('auto_reference', accumulation='cpu')
        for mode in (None, 'auto'):
            actual, _ = self.run_case('automatic_' + str(mode), accumulation=mode)
            self.compare_files(reference, actual)

    def test_time_domain_gathering(self):
        for mode in (0, 1, 2):
            reference, _ = self.run_case('td_reference_' + str(mode), mode=mode,
                                         td_only=True, td_accumulation='cpu', steps=65)
            for selection in (None, 'gpu'):
                actual, _ = self.run_case('td_' + str(mode) + str(selection), mode=mode,
                                          td_only=True, td_accumulation=selection, steps=65, batch=64)
                self.compare_files(reference, actual)

    def compare_files(self, expected, actual):
        names = sorted(name for name in os.listdir(expected) if name.endswith('.h5'))
        self.assertEqual(names, sorted(name for name in os.listdir(actual) if name.endswith('.h5')))
        for name in names:
            with h5py.File(os.path.join(expected, name)) as a, h5py.File(os.path.join(actual, name)) as b:
                paths = []
                a.visititems(lambda path, item: paths.append(path) if isinstance(item, h5py.Dataset) else None)
                other = []
                b.visititems(lambda path, item: other.append(path) if isinstance(item, h5py.Dataset) else None)
                self.assertEqual(paths, other)
                # CPU interpolation uses double intermediates, while the GPU
                # uses FP32. Bound cancellation noise by eight FP32 epsilons
                # of this dump's peak across time, including near-zero frames.
                td_peak = max((np.max(np.abs(a[path][...]), initial=0) for path in paths
                               if path.startswith('FieldData/TD/')), default=0)
                td_floor = td_peak * (8*np.finfo(np.float32).eps)
                for path in paths:
                    x, y = a[path][...], b[path][...]
                    self.assertEqual(x.shape, y.shape)
                    self.assertEqual(set(a[path].attrs), set(b[path].attrs))
                    for attr in a[path].attrs:
                        np.testing.assert_array_equal(a[path].attrs[attr], b[path].attrs[attr])
                    if path.startswith('Mesh'):
                        np.testing.assert_array_equal(x, y)
                    else:
                        peak = np.max(np.abs(x), initial=0)
                        error = np.max(np.abs(x-y), initial=0)
                        floor = td_floor if path.startswith('FieldData/TD/') else 0
                        self.assertLessEqual(error, peak*.001 + floor + 1e-18, (name, path, peak, error))
                self.assertEqual(set(a.attrs), set(b.attrs))
                for attr in a.attrs:
                    np.testing.assert_array_equal(a.attrs[attr], b.attrs[attr])

    def test_native_batches_and_cpu_engine(self):
        reference, _ = self.run_case('reference', accumulation='cpu', batch=1)
        for batch in (1, 32, 64):
            actual, _ = self.run_case('gpu' + str(batch), batch=batch)
            self.compare_files(reference, actual)
        basic, _ = self.run_case('basic', accumulation='cpu', engine='basic')
        self.compare_files(reference, basic)

    def test_node_and_cell(self):
        for mode in (1, 2):
            reference, _ = self.run_case('reference' + str(mode), mode=mode, accumulation='cpu')
            actual, _ = self.run_case('gpu' + str(mode), mode=mode, batch=64)
            self.compare_files(reference, actual)

    def test_mixed_output_and_long_decay(self):
        reference, _ = self.run_case('mixed_reference', mode=1, accumulation='cpu', td_accumulation='cpu', mixed=True, steps=2049, windows=False)
        actual, _ = self.run_case('mixed_gpu', mode=1, mixed=True, steps=2049, batch=64, windows=False)
        self.compare_files(reference, actual)
        np.testing.assert_allclose(np.loadtxt(os.path.join(reference, 'voltage'), comments='%'),
                                   np.loadtxt(os.path.join(actual, 'voltage'), comments='%'), rtol=.001, atol=1e-8)

    def test_disabled_dumps(self):
        reference, _ = self.run_case('disabled_reference', accumulation='cpu', disabled=True)
        actual, _ = self.run_case('disabled_gpu', disabled=True)
        self.compare_files(reference, actual)
        self.assertFalse(any(name.endswith('.h5') for name in os.listdir(actual)))

    def test_legacy_and_early_stopping(self):
        reference, _ = self.run_case('early_reference', accumulation='cpu', legacy=True, steps=2049, end=1e-3, windows=False)
        actual, _ = self.run_case('early_gpu', legacy=True, steps=2049, end=1e-3, batch=64, windows=False)
        self.compare_files(reference, actual)
        def stopped_at(folder):
            with open(os.path.join(folder, 'openEMS_stats.txt')) as stats:
                return next(int(line.split()[0]) for line in stats if '% number of iterations' in line)
        self.assertEqual(stopped_at(reference), stopped_at(actual))
        self.assertLess(stopped_at(actual), 2049)

    def test_vtk_output(self):
        reference, _ = self.run_case('vtk_reference', mode=1, accumulation='cpu', vtk=True)
        actual, _ = self.run_case('vtk_gpu', mode=1, vtk=True)
        self.compare_vtk_files(reference, actual)

    def test_time_domain_vtk_output(self):
        reference, _ = self.run_case('td_vtk_reference', mode=0, td_only=True,
                                     td_accumulation='cpu', vtk=True, steps=65)
        actual, _ = self.run_case('td_vtk_auto', mode=0, td_only=True, vtk=True, steps=65)
        self.compare_vtk_files(reference, actual)

    def compare_vtk_files(self, reference, actual):
        names = sorted(name for name in os.listdir(reference) if name.endswith('.vtr'))
        self.assertTrue(names)
        self.assertEqual(names, sorted(name for name in os.listdir(actual) if name.endswith('.vtr')))
        for name in names:
            a = ET.parse(os.path.join(reference, name)).getroot()
            b = ET.parse(os.path.join(actual, name)).getroot()
            self.assertEqual(a.attrib, b.attrib)
            x, y = list(a.iter('DataArray')), list(b.iter('DataArray'))
            self.assertEqual(len(x), len(y))
            magnitude = None
            if name.endswith('_arg.vtr'):
                magnitude_root = ET.parse(os.path.join(reference, name.replace('_arg.vtr', '_abs.vtr'))).getroot()
                magnitude = list(magnitude_root.iter('DataArray'))
            for index, (p, q) in enumerate(zip(x, y)):
                # Coordinate arrays have VTK-generated names containing object addresses.
                def metadata(array):
                    return {k: v for k, v in array.attrib.items() if not k.startswith('Range')
                            and not (k == 'Name' and v.startswith('Array 0x'))}
                self.assertEqual(metadata(p), metadata(q))
                u, v = self.read_vtk_array(a, p), self.read_vtk_array(b, q)
                self.assertEqual(u.shape, v.shape)
                if magnitude is not None and p.get('NumberOfComponents') == '3':
                    amplitude = self.read_vtk_array(magnitude_root, magnitude[index])
                    floor = max(np.max(np.abs(amplitude), initial=0)*1e-6, 1e-18)
                    mask = np.abs(amplitude)>floor
                    # Phase is circular; it is undefined where the spectrum vanishes.
                    error = np.abs(np.angle(np.exp(1j*(u[mask]-v[mask]))))
                    self.assertLessEqual(np.max(error, initial=0), .001, name)
                    continue
                peak = np.max(np.abs(u), initial=0)
                self.assertLessEqual(np.max(np.abs(u-v), initial=0), peak*.001+1e-18, name)

    @staticmethod
    def read_vtk_array(root, array):
        # VTK XML inline binary uses a separately base64-encoded zlib block header.
        header_type = '<u8' if root.get('header_type') == 'UInt64' else '<u4'
        width = np.dtype(header_type).itemsize
        encoded = ''.join(array.text.split())
        first = base64.b64decode(encoded[:4*((width+2)//3)])
        blocks = int(np.frombuffer(first[:width], dtype=header_type)[0])
        header_chars = 4*((width*(3+blocks)+2)//3)
        header = np.frombuffer(base64.b64decode(encoded[:header_chars]), dtype=header_type)
        compressed = base64.b64decode(encoded[header_chars:])
        offset, pieces = 0, []
        for length in header[3:]:
            length = int(length)
            pieces.append(zlib.decompress(compressed[offset:offset+length]))
            offset += length
        dtype = '<f4' if array.get('type') == 'Float32' else '<f8'
        return np.frombuffer(b''.join(pieces), dtype=dtype)

    def test_nf2ff_surfaces_and_far_field(self):
        for variant, directions, mirror in (
                ('all', [True]*6, [0]*6),
                ('mirror', [False, True, True, True, True, False], [1, 0, 0, 0, 0, 0])):
            reference, a = self.run_case('nf_reference_' + variant, mode=1, accumulation='cpu', nf2ff=True,
                                         steps=257, directions=directions, mirror=mirror)
            actual, b = self.run_case('nf_gpu_' + variant, mode=1, nf2ff=True, steps=257,
                                      batch=64, directions=directions, mirror=mirror)
            self.compare_far_field(reference, a, actual, b)

    def compare_far_field(self, reference, a, actual, b):
        self.compare_files(reference, actual)
        x = a.CalcNF2FF(reference, [10e9, 20e9], [0, 30, 60, 90, 120, 150, 180], [0, 45, 90], read_cached=False)
        y = b.CalcNF2FF(actual, [10e9, 20e9], [0, 30, 60, 90, 120, 150, 180], [0, 45, 90], read_cached=False)
        for name in ('E_theta', 'E_phi', 'E_cprh', 'E_cplh', 'Prad', 'Dmax'):
            p, q = np.asarray(getattr(x, name)), np.asarray(getattr(y, name))
            np.testing.assert_allclose(p, q, rtol=.001, atol=np.max(np.abs(p), initial=0)*1e-6 + 1e-20)


if __name__ == '__main__':
    unittest.main()
