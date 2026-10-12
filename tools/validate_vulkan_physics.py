#!/usr/bin/env python3
"""Run independent physics benchmarks through the native memory-test harness.

Existing benchmark scripts and their assertions are executed unchanged. A small
Python proxy exports each setup to XML and launches either the CPU executable or
an explicitly selected Vulkan PML candidate. Results and backend selection are
retained beside the XML; a silent candidate fallback fails the run.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_export(directory, xml):
    tree = ET.parse(directory / xml)
    grid = tree.getroot().find('.//RectilinearGrid')
    lines = [[float(v) for v in grid.find(axis + 'Lines').text.split(',')] for axis in 'XYZ']
    issues = []
    # Tangential electric sources on zero-thickness planes require a primary
    # mesh line. XML geometry coordinates can lose more digits than grid lines.
    for source in tree.getroot().findall('.//Properties/Excitation'):
        if source.get('Type') not in ('0', '1'):
            continue
        excite = [float(v) for v in source.get('Excite').split(',')]
        for box in source.findall('Primitives/Box'):
            for axis, name in enumerate('XYZ'):
                lo, hi = (float(box.find(point).get(name)) for point in ('P1', 'P2'))
                if lo != hi or excite[axis] != 0:
                    continue
                nearest = min(lines[axis], key=lambda v: abs(v-lo))
                if abs(nearest-lo) > 1e-12 * max(1, abs(nearest)):
                    issues.append(dict(source=source.get('Name'), axis=name,
                                       plane=lo, nearest_mesh_line=nearest,
                                       mismatch=abs(nearest-lo)))
    (directory / 'export-alignment.json').write_text(json.dumps(issues, indent=2))
    if issues:
        raise ValueError('Exported source plane misses the mesh; check XML coordinate precision: ' + str(issues))


def configure(args):
    project = args.project.resolve()
    native = args.native.resolve()
    dependencies = [native, project / 'build/msvc-release/install/bin',
                    project / 'build/msvc-release/dependencies/x64-windows-release/bin']
    os.environ['PATH'] = os.pathsep.join(map(str, dependencies)) + os.pathsep + os.environ['PATH']
    os.environ['OPENEMS_INSTALL_PATH'] = str(args.python_install.resolve())
    os.environ['CSXCAD_INSTALL_PATH'] = str(project / 'build/msvc-release/install')
    os.environ['MPLBACKEND'] = 'Agg'
    for key in ('VK_INSTANCE_LAYERS', 'VK_LAYER_PATH', 'VK_LAYER_SETTINGS_PATH'):
        os.environ.pop(key, None)
    # Load the exact solver before h5py can load a different HDF5 runtime.
    handles = []
    if os.name == 'nt':
        handles.extend(os.add_dll_directory(str(p)) for p in dependencies)
        handles.append(os.add_dll_directory(str(args.python_install.resolve() / 'bin')))
        handles.append(ctypes.CDLL(str(native / 'openEMS.dll')))
    sys.path.insert(0, str(args.bindings.resolve()))
    return handles


def run_model(args, directory, engine, xml='model.xml', timeout=3600):
    directory = Path(directory).resolve()
    check_export(directory, xml)
    native = args.native.resolve()
    common = ['--dump-statistics', '--exact-endcriteria', '-v']
    if engine == 'cpu':
        command = [str(native / 'openEMS.exe'), xml, '--engine=multithreaded',
                   '--numThreads=8'] + common
    else:
        command = [str(native / 'test_backend.exe'), '--vulkan-memory-model', xml,
                   '--pml=' + engine, '--vulkan-setup=' + args.setup,
                   '--numThreads=' + str(args.setup_threads), '--vulkan-fd=gpu', '--vulkan-td=gpu',
                   '--vulkan-coefficients=dense', '--vulkan-batch-size=32'] + common
    info = dict(command=command, engine=engine, dll_sha256=sha256(native / 'openEMS.dll'),
                executable_sha256=sha256(command[0]), xml_sha256=sha256(directory / xml))
    print('Starting', directory.name, engine, flush=True)
    start = time.perf_counter()
    with (directory / 'console.log').open('w', encoding='utf-8') as out, \
            (directory / 'stderr.log').open('w', encoding='utf-8') as err:
        result = subprocess.run(command, cwd=directory, stdout=out, stderr=err,
                                timeout=timeout)
    info.update(returncode=result.returncode, elapsed_s=time.perf_counter() - start)
    log = (directory / 'console.log').read_text(encoding='utf-8', errors='replace')
    match = re.search(r'Time for (\d+) iterations with ([\d.]+) cells', log)
    if match:
        info.update(steps=int(match[1]), cells=int(float(match[2])))
    info['stop'] = re.findall(r'end-criteria of ([-\d.]+)dB reached after (\d+) timesteps \((.*?)dB\)', log)
    if engine != 'cpu':
        selection = re.search(r'VULKAN_MEMORY_RESULT requested=(\S+) selected=(\S+)', log)
        info['selected'] = selection[2] if selection else None
    (directory / 'result.json').write_text(json.dumps(info, indent=2))
    if result.returncode or not match:
        raise RuntimeError('Native simulation failed: ' + str(directory))
    if engine != 'cpu' and info['selected'] != engine:
        raise RuntimeError('Candidate fallback: ' + str(info))
    print('Completed', directory.name, info['steps'], 'steps in', round(info['elapsed_s'], 2), 's', flush=True)
    return info


def benchmark(args):
    import tempfile
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tempfile.tempdir = str(root)
    import openEMS as package
    original = package.openEMS
    records = []

    class ExportedFDTD:
        def __init__(self, *positional, **keywords):
            self.fdtd = original(*positional, **keywords)

        def __getattr__(self, name):
            return getattr(self.fdtd, name)

        def Run(self, sim_path, cleanup=False, setup_only=False, **keywords):
            if setup_only:
                raise ValueError('Physics benchmark requires time stepping')
            directory = Path(sim_path)
            directory.mkdir(parents=True, exist_ok=True)
            self.fdtd.Write2XML(str(directory / 'model.xml'))
            tree = ET.parse(directory / 'model.xml')
            fdtd = tree.getroot().find('FDTD')
            requested = int(fdtd.get('NumberOfTimesteps'))
            capped = requested > args.benchmark_max_steps
            if capped:
                fdtd.set('NumberOfTimesteps', str(args.benchmark_max_steps))
                tree.write(directory / 'model.xml', encoding='utf-8', xml_declaration=True)
            info = run_model(args, directory, args.engine)
            info.update(original_maximum_steps=requested, benchmark_maximum_steps=args.benchmark_max_steps,
                        benchmark_script_sha256=sha256(args.script))
            records.append(info)
            (directory / 'result.json').write_text(json.dumps(info, indent=2))
            if capped and not info['stop']:
                raise RuntimeError('Benchmark did not converge within its safety limit')
            return None

    package.openEMS = ExportedFDTD
    try:
        runpy.run_path(str(args.script.resolve()), run_name='__main__')
    finally:
        package.openEMS = original
        (root / 'runs.json').write_text(json.dumps(records, indent=2))


def antenna(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(args.source)
    fdtd = tree.getroot().find('FDTD')
    fdtd.set('endCriteria', str(args.criterion))
    fdtd.set('NumberOfTimesteps', str(args.steps))
    if args.timestep_factor != 1:
        fdtd.set('TimeStepFactor', str(args.timestep_factor))
    if args.refine:
        import numpy as np
        grid = tree.getroot().find('.//RectilinearGrid')
        for axis in ('XLines', 'YLines', 'ZLines'):
            node = grid.find(axis)
            old = np.fromstring(node.text, sep=',')
            new = np.sort(np.concatenate((old, (old[1:] + old[:-1]) / 2)))
            node.text = ','.join(format(float(v), '.17g') for v in new)
            node.set('Qty', str(len(new)))
        # Preserve the original physical PML thickness after halving every cell.
        bc = fdtd.find('BoundaryCond')
        for key, value in list(bc.attrib.items()):
            if value.startswith('PML_'):
                bc.set(key, 'PML_' + str(2 * int(value[4:])))
    if args.pml:
        bc = fdtd.find('BoundaryCond')
        for key, value in list(bc.attrib.items()):
            if value.startswith('PML_'):
                bc.set(key, 'PML_' + str(args.pml))
    if args.clearance:
        import numpy as np
        grid = tree.getroot().find('.//RectilinearGrid')
        for axis in ('XLines', 'YLines', 'ZLines'):
            node = grid.find(axis)
            old = np.fromstring(node.text, sep=',')
            n = args.clearance
            new = np.concatenate((old[0] - (old[1] - old[0]) * np.arange(n, 0, -1),
                                  old, old[-1] + (old[-1] - old[-2]) * np.arange(1, n+1)))
            node.text = ','.join(format(float(v), '.17g') for v in new)
            node.set('Qty', str(len(new)))
    if args.record_loss:
        import copy
        properties = tree.getroot().find('.//ContinuousStructure/Properties')
        losses = []
        for material in list(properties.findall('Material')):
            values = material.find('Property')
            kappa = float(values.get('Kappa', '0').split(',')[0])
            if kappa == 0:
                continue
            if material.get('Isotropy') != '1':
                raise ValueError('Loss diagnostic currently requires isotropic materials')
            for number, primitive in enumerate(material.find('Primitives')):
                if primitive.tag != 'Box':
                    raise ValueError('Loss diagnostic currently requires material boxes')
                name = 'physics_loss_' + material.get('ID') + '_' + str(number)
                dump = ET.SubElement(properties, 'DumpBox', dict(
                    ID=str(1000 + len(losses)), Name=name, Type='0', Weight='1',
                    NormDir='-1', StartTime='0', StopTime='0', OverSampling='4',
                    DumpType='10', DumpMode='2', FileType='1', MultiGridLevel='0'))
                ET.SubElement(dump, 'Primitives').append(copy.deepcopy(primitive))
                ET.SubElement(dump, 'FD_Samples').text = '77000000000'
                losses.append(dict(name=name, kappa=kappa, material=material.get('Name'),
                                   start=[float(primitive.find('P1').get(a)) for a in 'XYZ'],
                                   stop=[float(primitive.find('P2').get(a)) for a in 'XYZ']))
        (root / 'loss-recorders.json').write_text(json.dumps(losses, indent=2))
    tree.write(root / 'model.xml', encoding='utf-8', xml_declaration=True)
    result = run_model(args, root, args.engine)
    result.update(source_xml_sha256=sha256(args.source), criterion=args.criterion,
                  refinement=2 if args.refine else 1, pml_override=args.pml,
                  extra_clearance_cells=args.clearance, timestep_factor=args.timestep_factor)
    (root / 'result.json').write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--python-install', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', choices=['cpu', 'indexed', 'direct', 'regions', 'fused', 'fused-boxes'], required=True)
    parser.add_argument('--setup', choices=['cpu', 'gpu', 'auto'], default='cpu')
    parser.add_argument('--setup-threads', type=int, default=1)
    commands = parser.add_subparsers(dest='action', required=True)
    tests = commands.add_parser('benchmark')
    tests.add_argument('script', type=Path)
    tests.add_argument('--benchmark-max-steps', type=int, default=100000)
    model = commands.add_parser('antenna')
    model.add_argument('source', type=Path)
    model.add_argument('--criterion', type=float, default=1e-6)
    model.add_argument('--steps', type=int, default=100000)
    model.add_argument('--refine', action='store_true')
    model.add_argument('--pml', type=int)
    model.add_argument('--clearance', type=int, default=0)
    model.add_argument('--timestep-factor', type=float, default=1)
    model.add_argument('--record-loss', action='store_true',
                       help='Add cell-center FD E dumps for dielectric-loss diagnostics')
    args = parser.parse_args()
    handles = configure(args)
    if args.action == 'benchmark':
        benchmark(args)
    else:
        antenna(args)
    return handles


if __name__ == '__main__':
    main()
