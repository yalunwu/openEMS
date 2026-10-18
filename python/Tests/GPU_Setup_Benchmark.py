# Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later
"""Repeated setup/total timings with identical CPU/HIP inputs and fresh processes.

Writes JSON records and complete logs. Each worker's first run is process-cold;
later runs are warm. This does not claim to flush OS file or driver disk caches.
Use --diagnostics separately from the default throughput measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import numpy as np
import GPU_Setup as setup


def many_primitives():
    fdtd, csx = setup.geometry_case()
    material = csx.AddMaterial('many_spheres', epsilon=3.5)
    for x in np.linspace(-7, 7, 8):
        for y in np.linspace(-7, 7, 8):
            for z in np.linspace(-7, 7, 8):
                material.AddSphere([x, y, z], .8, priority=20)
    return fdtd, csx


CASES = dict(analytic=setup.case_excitation, many=many_primitives,
             cylinder=setup.case_cylinder_closed, multigrid=setup.case_multigrid2)


def source_state(repository):
    try:
        command = ['git', '-C', str(repository)]
        commit = subprocess.check_output(command+['rev-parse', 'HEAD'], text=True,stderr=subprocess.DEVNULL).strip()
        difference = subprocess.check_output(command+['diff', 'HEAD'])
        names = subprocess.check_output(command+['ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0')
        untracked = {name: hashlib.sha256((repository/name).read_bytes()).hexdigest()
                     for name in names if name and (repository/name).is_file()}
        return dict(commit=commit, diff_sha256=hashlib.sha256(difference).hexdigest(), untracked=untracked)
    except (OSError, subprocess.CalledProcessError):
        return None


def worker(args):
    records = []
    for repeat in range(args.repeats):
        generated = []
        def factory():
            start = time.perf_counter()
            fdtd, csx = CASES[args.case]()
            fdtd.SetNumberOfTimeSteps(args.timesteps)
            generated.append(time.perf_counter()-start)
            return fdtd, csx
        output = args.output / f'{args.case}_{args.mode}_{repeat}'
        start = time.perf_counter()
        log = setup.run_captured(factory, str(output), 'gpu', gpu_setup=args.mode, verbose=1)
        elapsed = time.perf_counter()-start
        assert 'backend: HIP' in log, 'benchmark requires the HIP stepping backend'
        if args.mode == 'gpu':
            assert 'HIP construction active' in log and 'rebuilding a fresh CPU operator' not in log
        phases = [{'owner': owner, 'phase': name, 'seconds': float(value)} for owner, name, value in
                  re.findall(r'^(FDTD setup|CPU processing setup): (.*?) time: ([0-9.eE+-]+) s', log, re.M)]
        record = dict(case=args.case, mode=args.mode, process=args.process_index, repeat=repeat, state='process-cold' if repeat == 0 else 'warm',
                      total_seconds=elapsed, model_seconds=generated[0], setup_seconds=sum(p['seconds'] for p in phases), phases=phases,
                      host_peak_bytes=max(map(int, re.findall(r'host_peak_bytes=(\d+)', log)), default=None),
                      diagnostic_lines=[line for line in log.splitlines() if
                                        any(key in line for key in ('GPU setup environment:', 'GPU setup diagnostics', 'GPU waveform', 'GPU mode preparation:',
                                                                   'HIP sampling map', 'GPU requested', 'GPU diagnostics', 'GPU initialization:',
                                                                   'GPU whole-context setup diagnostics:'))],
                      log=str(output)+'.log')
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / f'{args.case}_{args.mode}.json').write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--mode', choices=('cpu', 'gpu'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--processes', type=int, default=3, help='fresh worker processes per case and setup mode')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--process-index', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.timesteps < 1 or args.processes < 1:
        parser.error('processes, repeats and timesteps must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if not (args.case and args.mode):
            parser.error('a worker requires a case and setup mode')
        worker(args)
        return
    environment = os.environ.copy()
    environment['OPENEMS_GPU_DIAGNOSTICS'] = '1' if args.diagnostics else '0'
    environment['OPENEMS_HIP_KERNEL_TIMES'] = '1' if args.diagnostics else '0'
    records = []
    for case in ([args.case] if args.case else CASES):
        for process in range(args.processes):
            output = args.output / f'process_{process}'
            modes=('cpu','gpu') if process%2==0 else ('gpu','cpu')
            for mode in ([args.mode] if args.mode else modes):
                command = [sys.executable, __file__, '--worker', '--process-index', str(process), '--case', case, '--mode', mode,
                           '--repeats', str(args.repeats), '--timesteps', str(args.timesteps), '--output', str(output)]
                subprocess.run(command, env=environment, check=True)
                records.extend(json.loads((output / f'{case}_{mode}.json').read_text()))
            for repeat in range(args.repeats):
                if args.mode is None:
                    differences = setup.compare_outputs(str(output / f'{case}_cpu_{repeat}'), str(output / f'{case}_gpu_{repeat}'), rtol=1e-4)[0]
                    assert not differences, f'benchmark output mismatch: {case}: {differences}'
    report = dict(python=sys.version, platform=platform.platform(), diagnostics=args.diagnostics, processes=args.processes, repeats=args.repeats,
                  cold_definition='fresh worker process; OS file/driver disk caches are not flushed',
                  mode_order='alternating CPU/GPU by worker-process index',records=records)
    source = Path(__file__).resolve().parents[3]
    report['source_revisions'] = {name: source_state(source/name) for name in ('openEMS', 'CSXCAD', 'fparser')}
    report['host_peak_definition'] = 'process lifetime peak resident memory, including library/driver initialization'
    binary = Path(os.environ.get('OPENEMS_INSTALL_PATH', '')) / 'openEMS.dll'
    if binary.is_file():
        report['binary'] = dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        report['binaries'] = {name: dict(path=str(binary.parent/name), sha256=hashlib.sha256((binary.parent/name).read_bytes()).hexdigest())
                              for name in ('openEMS.dll','CSXCAD.dll','fparser.dll') if (binary.parent/name).is_file()}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
