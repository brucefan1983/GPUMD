#!/usr/bin/env python3
# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD. If not, see <https://www.gnu.org/licenses/>.
# Major contributors: Zheyong Fan (NEP multi-GPU release tests).


"""Opt-in NEP89 release correctness and performance tests (no pytest/NumPy)."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

import cases
from checks import (CheckError, check_liquid, check_run, compare_runs, exact,
                    load_tolerances, sha256, thermo_rows, write_json)

HERE = Path(__file__).resolve().parent


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def device_environment(backend, devices):
    env = os.environ.copy()
    # HIP accepts CUDA_VISIBLE_DEVICES as an alias. Avoid conflicting aliases.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('HIP_VISIBLE_DEVICES', None)
    env['HIP_VISIBLE_DEVICES' if backend == 'hip' else 'CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    env['OMP_NUM_THREADS'] = '1'
    # An inherited ROCR mask is deliberately retained (e.g. scheduler allocation).
    return env


def log_rates(path, expected_gpus, steps, expected_partitions=None):
    if expected_partitions is None:
        expected_partitions = expected_gpus
    log = Path(path).read_text(encoding='utf-8', errors='replace')
    counts = re.findall(r'number of GPUs = (\d+)', log)
    exact(counts == [str(expected_gpus)], 'Log does not confirm the requested visible GPU count')
    multi = re.findall(r'Try to use (\d+) GPUs for the NEP part\.', log)
    exact(multi == ([] if expected_partitions == 1 else [str(expected_partitions)]),
          'Log does not confirm the expected NEP partition count: {}'.format(expected_partitions))
    actual_steps = [int(x) for x in re.findall(r'^Run (\d+) steps\.', log, re.M)]
    exact(actual_steps == steps, 'Completed input run sequence differs: ' + str(actual_steps))
    rates = [float(x) for x in re.findall(r'Speed of this run = (\S+) atom\*step/second\.', log)]
    elapsed = [float(x) for x in re.findall(r'Time used for this run = (\S+) second\.', log)]
    exact(len(rates) == len(steps) == len(elapsed), 'Missing run timing/completion records')
    exact(all(math.isfinite(x) and x > 0 for x in rates + elapsed), 'Invalid run timings')
    return rates, elapsed


def execute(root, report, binary, build, gpus, model, text, spec, relative, partitions=None):
    if partitions is None:
        partitions = gpus
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model, directory / 'model.xyz')
    (directory / 'potential.txt').symlink_to(report['potential']['path'])
    (directory / 'run.in').write_text(text, encoding='utf-8')
    inputs = {name: sha256(directory / name) for name in ('model.xyz', 'potential.txt', 'run.in')}
    record = {'path': relative, 'build': build, 'gpus': gpus, 'partitions': partitions, 'spec': spec,
              'input_hashes': inputs, 'process_status': 'RUNNING', 'logs': []}
    report['runs'].append(record)
    write_json(root / 'report.json', report)
    env = device_environment(report['backend'], report['devices'][:gpus])
    record['environment'] = {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                            'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'OMP_NUM_THREADS') if k in env}
    try:
        binary_hash = report['binaries'][build.removesuffix('_repeat')]['sha256']
        exact(sha256(binary) == binary_hash, 'Executable changed after the test started')
        exact(inputs['potential.txt'] == report['potential']['sha256'],
              'NEP89 changed after the test started')
        for execution in range(spec.get('executions', 1)):
            log_name = 'run-{:02d}.log'.format(execution + 1)
            print('RUN {} [{} physical GPU(s), {} NEP partition(s), execution {}]'.format(
                relative, gpus, partitions, execution + 1), flush=True)
            start = time.monotonic()
            with (directory / log_name).open('w', encoding='utf-8') as log:
                process = subprocess.Popen([str(binary)], cwd=directory, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=report['timeout_seconds'])
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            record['logs'].append({'file': log_name, 'returncode': returncode,
                                   'wall_seconds': time.monotonic() - start})
            exact(returncode == 0, 'Exit code {}; see {}'.format(returncode, directory / log_name))
            log_rates(directory / log_name, gpus, spec['steps'], partitions)
        for name, digest in inputs.items():
            exact(sha256(directory / name) == digest, 'Input changed during execution: ' + name)
        exact(sha256(binary) == binary_hash, 'Executable changed during execution')
        record['process_status'] = 'COMPLETE'
    except (OSError, CheckError, ValueError, subprocess.TimeoutExpired) as error:
        record.update(process_status='ERROR', error=str(error))
        print('  ERROR: ' + str(error), flush=True)
    finally:
        record['output_hashes'] = {p.name: sha256(p) for p in directory.iterdir()
                                   if p.is_file() and p.name not in inputs}
        write_json(root / 'report.json', report)
    return record


def result(name, function):
    try:
        metrics = function()
        failed = [k for k, v in metrics.items() if v.get('exceedances', 0)]
        return {'name': name, 'status': 'FAIL' if failed else 'PASS',
                'failed_metrics': failed, 'metrics': metrics}
    except (OSError, CheckError, ValueError, KeyError, OverflowError) as error:
        return {'name': name, 'status': 'ERROR', 'error': str(error)}


def verify_record(root, record):
    exact(record['process_status'] == 'COMPLETE', record.get('error', 'Process did not complete'))
    directory = root / record['path']
    for name, digest in {**record['input_hashes'], **record['output_hashes']}.items():
        exact(sha256(directory / name) == digest, 'Retained file changed: ' + str(directory / name))
    for log in record['logs']:
        log_rates(directory / log['file'], record['gpus'], record['spec']['steps'],
                  record.get('partitions', record['gpus']))


def assess(root, report, tolerances):
    checks, by_case = [], defaultdict(dict)
    for run in report['runs']:
        spec = run['spec']
        if spec['kind'] not in ('short', 'energy'):
            continue

        def check(run=run, spec=spec):
            verify_record(root, run)
            initial = cases.only_frame(root / run['path'] / 'model.xyz')
            return check_run(root / run['path'], initial, spec, tolerances)

        entry = result(run['path'], check)
        checks.append(entry)
        run['validation_status'] = entry['status']
        if spec['name'] != 'preflight':
            by_case[spec['name']][(run['build'], run['gpus'])] = run
    for name, runs in by_case.items():
        pairs = []
        for key in runs:
            build, count = key
            if count > 1:
                pairs.append((key, (build, 1)))
            if build == 'new' and ('old', count) in runs:
                pairs.append((key, ('old', count)))
            if build == 'zheyong':
                pairs.append((key, ('new', 1)))
            if build.endswith('_repeat'):
                pairs.append((key, (build.removesuffix('_repeat'), count)))
        for left_key, right_key in pairs:
            left, right = runs.get(left_key), runs.get(right_key)
            label = '{}: {}_{} / {}_{}'.format(name, *left_key, *right_key)
            if left is None or right is None:
                checks.append({'name': label, 'status': 'ERROR', 'error': 'Comparison run missing'})
                continue
            if left.get('validation_status') != 'PASS' or right.get('validation_status') != 'PASS':
                checks.append({'name': label, 'status': 'SKIP', 'error': 'A prerequisite run did not pass'})
                continue
            if left['input_hashes'] != right['input_hashes']:
                checks.append({'name': label, 'status': 'ERROR', 'error': 'Comparison inputs differ'})
                continue
            initial = cases.only_frame(root / left['path'] / 'model.xyz')
            checks.append(result(label, lambda left=left, right=right, initial=initial:
                                 compare_runs(root / left['path'], root / right['path'],
                                              left['spec'], tolerances, initial['n'])))
    report['correctness_checks'] = checks
    expected_cases = set(report['selected_cases']) if report['mode'] != 'performance' else set()
    completed_cases = set(by_case)
    statuses = Counter(c['status'] for c in checks)
    if not expected_cases:
        status = 'NOT_RUN'
    elif report.get('fatal_error') or completed_cases != expected_cases or statuses['ERROR'] or statuses['SKIP']:
        status = 'ERROR'
    elif statuses['FAIL']:
        status = 'FAIL'
    else:
        status = 'PASS'
    report['correctness_status'] = status
    report['real_multigpu_correctness_status'] = (
        'NOT_RUN' if report.get('execution_mode') == 'zheyong' else status)
    report['full_correctness_scope'] = expected_cases == set(cases.NAMES)
    report['tolerances'] = tolerances
    return checks


def performance_summary(root, report):
    is_zheyong = report.get('execution_mode') == 'zheyong'
    groups = defaultdict(list)
    for run in report['runs']:
        spec = run['spec']
        if spec['kind'] != 'performance':
            continue
        key = (spec['phase'], spec['size'], spec['profile'], run['build'], run['gpus'])
        sample = {'path': run['path'], 'repeat': spec['repeat'], 'atoms': spec['atoms'],
                  'partitions': run.get('partitions', run['gpus'])}
        try:
            verify_record(root, run)
            rates, elapsed = log_rates(root / run['path'] / 'run-01.log', run['gpus'],
                                       spec['steps'], sample['partitions'])
            rows = thermo_rows(root / run['path'] / 'thermo.out', spec['thermo_rows'])
            exact(all(0 <= row[0] < 10000 and row[1] >= 0 for row in rows),
                  'Unphysical benchmark temperature or kinetic energy')
            trajectory = root / run['path'] / 'trajectory.xyz'
            if spec['profile'] == 'output':
                frame = cases.only_frame(trajectory)
                exact(frame['n'] == spec['atoms'] and frame['schema'] ==
                      {'species': ('S', 1), 'pos': ('R', 3)}, 'Invalid benchmark trajectory')
            else:
                exact(not trajectory.exists(), 'Unexpected trajectory in minimal benchmark')
            # The second run is measured; initialization, warmup and the final
            # finite-output probe are excluded. Check the public timing formula.
            expected_rate = spec['atoms'] * spec['steps'][1] / elapsed[1]
            exact(abs(rates[1] / expected_rate - 1) < 5e-5, 'Inconsistent atom-step/s timing')
            sample.update(status='PASS', rate=rates[1], elapsed_seconds=elapsed[1])
        except (OSError, ValueError, CheckError) as error:
            sample.update(status='ERROR', error=str(error))
        groups[key].append(sample)
    table = []
    for key, samples in sorted(groups.items()):
        row = dict(zip(('phase', 'size', 'profile', 'build', 'gpus'), key))
        row['samples'] = samples
        row.update(atoms=samples[0]['atoms'], partitions=samples[0]['partitions'])
        if (len(samples) != report['repeats'] or any(s['status'] != 'PASS' for s in samples) or
                any((s['atoms'], s['partitions']) != (row['atoms'], row['partitions']) for s in samples)):
            row['status'] = 'ERROR'
        else:
            rates = [s['rate'] for s in samples]
            mean = statistics.mean(rates)
            row.update(status='COMPLETE', mean_rate=mean,
                       cv_percent=100 * statistics.stdev(rates) / mean if len(rates) > 1 else None)
        table.append(row)
    index = {(r['phase'], r['size'], r['profile'], r['build'], r['gpus']): r for r in table}
    for row in table:
        if row['status'] != 'COMPLETE':
            continue
        prefix = (row['phase'], row['size'], row['profile'])
        if is_zheyong:
            normal = index.get(prefix + ('new', 1), {})
            if normal.get('status') == 'COMPLETE' and normal['atoms'] == row['atoms']:
                row['relative_to_normal'] = row['mean_rate'] / normal['mean_rate']
            continue
        one = index.get(prefix + (row['build'], 1), {})
        old = index.get(prefix + ('old', row['gpus']), {})
        if one.get('status') == 'COMPLETE':
            row['scaling'] = row['mean_rate'] / one['mean_rate']
            row['efficiency'] = row['scaling'] / row['gpus']
        if row['build'] == 'new' and old.get('status') == 'COMPLETE':
            row['new_over_old'] = row['mean_rate'] / old['mean_rate']
    report['performance'] = table
    if report['mode'] == 'correctness' or not groups:
        report['performance_status'] = 'NOT_RUN'
    elif (report.get('fatal_error') or len(table) != report['expected_performance_groups'] or
          any(r['status'] != 'COMPLETE' for r in table)):
        report['performance_status'] = 'ERROR'
    else:
        report['performance_status'] = 'COMPLETE'
    report['real_multigpu_performance_status'] = 'NOT_RUN' if is_zheyong else report['performance_status']


def save_report(destination, report):
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'report.json', report)
    is_zheyong = report.get('execution_mode') == 'zheyong'
    fields = ('phase', 'size', 'profile', 'build', 'gpus', 'partitions', 'atoms', 'status', 'mean_rate',
              'relative_to_normal', 'new_over_old', 'scaling', 'efficiency', 'cv_percent')
    with (destination / 'performance.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report.get('performance', []))
    counts = Counter(c['status'] for c in report.get('correctness_checks', []))
    lines = ['# NEP89 multi-GPU release tests', '',
             'Correctness: **{}**. Performance: **{}**.'.format(
                 report.get('correctness_status', 'NOT_RUN'), report.get('performance_status', 'NOT_RUN')),
             '', 'Scope: {}.'.format(', '.join(report['selected_cases']) if report['mode'] != 'performance' else 'performance only'),
             '', 'Checks: ' + ', '.join('{} {}'.format(counts[s], s) for s in ('PASS', 'FAIL', 'ERROR', 'SKIP')), '']
    if is_zheyong:
        lines += ['Execution: **ZHEYONG single-physical-GPU partition tests**.',
                  'Both builds use one physical GPU: new = ordinary NEP; zheyong = three NEP partitions.',
                  '**Real multi-GPU correctness: NOT_RUN. Real multi-GPU performance: NOT_RUN.**',
                  'Performance compares the two paths on one physical GPU; it does not measure multi-GPU scaling or efficiency.', '']
    if report.get('fatal_error'):
        lines += ['Run stopped: ' + report['fatal_error'], '']
    if report.get('performance_blocked'):
        lines += ['Performance was not started: ' + report['performance_blocked'], '']
    if report['mode'] != 'performance' and not report.get('full_correctness_scope'):
        lines += ['**Focused diagnostics: this is not the full release correctness suite.**', '']
    if report.get('liquid'):
        lines += ['Liquid preparation: ' + json.dumps(report['liquid'], ensure_ascii=False), '']
    if is_zheyong:
        lines += ['| Phase / size / profile | Build | Physical GPUs | NEP partitions | Atoms | Mean atom-step/s | Rate / ordinary | CV (%) |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
    else:
        lines += ['| Phase / size / profile | Build | GPUs | Mean atom-step/s | New / old | Scaling | Efficiency | CV (%) |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in report.get('performance', []):
        columns = (('mean_rate', 'relative_to_normal', 'cv_percent') if is_zheyong else
                   ('mean_rate', 'new_over_old', 'scaling', 'efficiency', 'cv_percent'))
        values = [format(row[k], '.6g') if row.get(k) is not None else '—' for k in columns]
        if row['status'] != 'COMPLETE':
            values[0] = row['status']
        hardware = str(row['gpus'])
        if is_zheyong:
            hardware += ' | {} | {}'.format(row['partitions'], row['atoms'])
        lines.append('| {} / {} / {} | {} | {} | {} |'.format(row['phase'], row['size'], row['profile'],
                     row['build'], hardware, ' | '.join(values)))
    lines += ['', 'Rates are arithmetic means of per-repeat atom-step/s; ratios use those means.',
              'Performance completion is not a speedup acceptance gate.',
              'Inspect CV and individual samples; CV is undefined for one repeat.', '']
    lines += ['Bounds are initial NEP89 test bounds and require calibration on the release hardware.',
              'Correctness NOT_RUN is not a correctness pass.', '', '## Check results', '']
    for check in report.get('correctness_checks', []):
        lines.append('- **{}** {}{}'.format(check['status'], check['name'],
                     ': ' + check['error'] if 'error' in check else
                     (': ' + ', '.join(check.get('failed_metrics', [])) if check.get('failed_metrics') else '')))
    lines += ['', 'Inputs, original logs, hashes, per-field errors and per-repeat rates are in report.json and the run directories.', '']
    (destination / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', '--candidate', type=Path, help='candidate gpumd executable')
    parser.add_argument('--old', '--baseline', type=Path, help='optional accepted gpumd executable')
    parser.add_argument('--zheyong', type=Path,
                        help='same-source -DZHEYONG executable; compare three partitions on one physical GPU')
    parser.add_argument('--backend', choices=('cuda', 'hip'))
    parser.add_argument('--devices', '--device', help='visible GPU indices; default: 0,1 (0 with --zheyong)')
    parser.add_argument('--counts', help='GPU counts, starting at 1; default: 1 and all selected GPUs')
    parser.add_argument('--mode', choices=('correctness', 'performance', 'all'), default='correctness')
    parser.add_argument('--cases', default=','.join(cases.NAMES), help='comma-separated correctness case names')
    parser.add_argument('--list', action='store_true', help='list correctness cases and exit without a GPU')
    parser.add_argument('--repo-root', type=Path, default=HERE.parents[1])
    parser.add_argument('--output', type=Path, default=HERE / 'reports' / stamp())
    parser.add_argument('--tolerances', type=Path, default=HERE / 'tolerances.json')
    parser.add_argument('--timeout', type=float, default=7200, help='seconds per process (default: 7200)')
    parser.add_argument('--sizes', default='small,medium,large')
    parser.add_argument('--profiles', default='minimal', help='minimal,output')
    parser.add_argument('--phases', default='solid,liquid', help='solid,liquid')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--build-note', default='', help='record compiler, flags, driver, and GPU setup')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(cases.NAMES))
        return args
    if args.new is None or args.backend is None:
        parser.error('--new and --backend are required')
    if args.zheyong and (args.old is not None or args.counts is not None):
        parser.error('--zheyong cannot be combined with --old or --counts')
    args.devices = (args.devices if args.devices is not None else ('0' if args.zheyong else '0,1')).split(',')
    if (len(set(args.devices)) != len(args.devices) or
            any(not re.fullmatch(r'0|[1-9]\d*', x) for x in args.devices)):
        parser.error('--devices must contain distinct nonnegative integer indices')
    if args.zheyong:
        if len(args.devices) != 1:
            parser.error('--zheyong requires exactly one visible physical GPU')
        args.counts = [1]
    else:
        if not 2 <= len(args.devices) <= 16:
            parser.error('--devices must contain 2 to 16 GPU indices without --zheyong')
        try:
            args.counts = [int(x) for x in args.counts.split(',')] if args.counts else [1, len(args.devices)]
        except ValueError:
            parser.error('--counts must contain integer GPU counts')
        if (args.counts != sorted(set(args.counts)) or args.counts[0] != 1 or len(args.counts) < 2 or
                args.counts[-1] > len(args.devices)):
            parser.error('--counts must be increasing, include 1 and a multi-GPU count, and fit --devices')
    for key, allowed in (('cases', cases.NAMES), ('sizes', cases.TILES),
                         ('profiles', ('minimal', 'output')), ('phases', ('solid', 'liquid'))):
        value = getattr(args, key).split(',')
        if not value or len(value) != len(set(value)) or any(x not in allowed for x in value):
            parser.error('Invalid --' + key)
        setattr(args, key, value)
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or args.repeats < 1 or
            args.warmup < 1 or args.steps < 1):
        parser.error('Timeout, repeats, warmup and steps must be positive')
    for key in ('new', 'old', 'zheyong'):
        value = getattr(args, key)
        if value is not None:
            value = value.resolve()
            if not value.is_file() or not os.access(value, os.X_OK):
                parser.error('--{} is not an executable file: {}'.format(key, value))
            setattr(args, key, value)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('--output already exists; choose a fresh directory')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.list:
        return 0
    tolerances = load_tolerances(args.tolerances)
    potential = (args.repo_root / cases.POTENTIAL).resolve()
    with potential.open() as stream:
        first = stream.readline().split()
        exact(first[:2] == ['nep', '89'], 'Expected the repository NEP89 model in current format')
        header_lines = [stream.readline().strip() for _ in range(4)]
        exact(header_lines[-1].split()[:3] == ['cutoff', '6', '5'], 'NEP89 cutoff differs from fixture geometry')
    root = args.output
    root.mkdir(parents=True)
    (root / 'inputs').mkdir()
    (root / 'code').mkdir()
    for source in HERE.iterdir():
        if source.suffix in ('.py', '.json', '.md'):
            shutil.copyfile(source, root / 'code' / source.name)
    binaries = {'new': args.new}
    if args.old:
        binaries['old'] = args.old
    if args.zheyong:
        binaries['zheyong'] = args.zheyong
    configs = ([('new', 1, 1), ('zheyong', 1, 3)] if args.zheyong else
               [(build, count, count) for build in binaries for count in args.counts])
    max_partitions = max(partitions for _, _, partitions in configs)
    try:
        revision = subprocess.run(['git', '-C', str(args.repo_root), 'rev-parse', 'HEAD'],
                                  text=True, capture_output=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    report = {'schema_version': 1, 'created_utc': stamp(), 'command': sys.argv,
              'backend': args.backend, 'devices': args.devices, 'counts': args.counts,
              'execution_mode': 'zheyong' if args.zheyong else 'multigpu',
              'max_partitions': max_partitions,
              'mode': args.mode, 'selected_cases': args.cases, 'timeout_seconds': args.timeout,
              'repeats': args.repeats, 'build_note': args.build_note,
              'host': platform.platform(), 'python': sys.version,
              'repository_revision': revision,
              'potential': {'path': str(potential), 'sha256': sha256(potential)},
              'binaries': {k: {'path': str(v), 'sha256': sha256(v)} for k, v in binaries.items()},
              'package_hashes': {p.name: sha256(p) for p in (root / 'code').iterdir()},
              'expected_performance_groups': len(args.phases) * len(args.sizes) * len(args.profiles) * len(configs),
              'runs': [], 'pairs_order_seed': 1906, 'tolerances': tolerances,
              'correctness_status': 'NOT_RUN', 'performance_status': 'NOT_RUN'}
    try:
        solid = root / 'inputs' / 'solid.xyz'
        cases.write_state(solid, cases.crystal(max_partitions))
        # Fail early for a wrong backend, missing loader libraries or invisible GPUs.
        text, spec = cases.correctness('axis_x')
        spec.update(name='preflight', max_partitions=max_partitions)
        for build, count, partitions in configs:
            run = execute(root, report, binaries[build], build, count, solid, text, spec,
                          'preflight/{}_{}'.format(build, count), partitions=partitions)
            verify_record(root, run)
            preflight = check_run(root / run['path'], cases.only_frame(solid), spec, tolerances)
            exact(not any(m.get('exceedances', 0) for m in preflight.values()),
                  'Preflight independent output check failed: ' + run['path'])

        need_liquid = ((args.mode != 'performance' and any('liquid' in x for x in args.cases)) or
                       (args.mode != 'correctness' and 'liquid' in args.phases))
        liquid = root / 'inputs' / 'prepared_liquid.xyz'
        if need_liquid:
            seed = root / 'inputs' / 'liquid_seed.xyz'
            cases.write_state(seed, cases.crystal(max_partitions, 'liquid_seed'))
            build = 'old' if args.old else 'new'
            prep_spec = {'kind': 'preparation', 'name': 'liquid', 'steps': [5000, 2000, 3000]}
            run = execute(root, report, binaries[build], build, 1, seed,
                          cases.liquid_preparation(), prep_spec, 'preparation/liquid')
            verify_record(root, run)
            report['liquid'] = check_liquid(root / run['path'])
            cases.snapshot_to_input(root / run['path'] / 'restart.xyz', liquid)

        if args.mode != 'performance':
            for name in args.cases:
                text, spec = cases.correctness(name)
                spec['max_partitions'] = max_partitions
                model = root / 'inputs' / (name + '.xyz')
                if 'liquid' in name:
                    shutil.copyfile(liquid, model)
                elif name == 'restart':
                    build = 'old' if args.old else 'new'
                    cases.snapshot_to_input(root / 'preflight' / (build + '_1') / 'restart.xyz', model)
                else:
                    cases.write_state(model, cases.crystal(max_partitions, name, spec['axis']))
                for build, count, partitions in configs:
                    execute(root, report, binaries[build], build, count, model, text, spec,
                            'correctness/{}/{}_{}'.format(name, build, count), partitions=partitions)
                if name == 'axis_x':
                    # Both the single- and multi-GPU paths have a repeatability control.
                    for build, count, partitions in configs:
                        execute(root, report, binaries[build], build + '_repeat', count, model, text, spec,
                                'correctness/{}/{}_{}_repeat'.format(name, build, count), partitions=partitions)
            assess(root, report, tolerances)
            save_report(root, report)
            for check in report['correctness_checks']:
                print('{} {}'.format(check['status'], check['name']), flush=True)

        if args.mode == 'all' and report['correctness_status'] != 'PASS':
            report['performance_blocked'] = 'Correctness did not pass'
        if args.mode != 'correctness' and not report.get('performance_blocked'):
            rng = random.Random(report['pairs_order_seed'])
            for phase in args.phases:
                model = liquid if phase == 'liquid' else solid
                base_atoms = cases.only_frame(model)['n']
                for size in args.sizes:
                    for profile in args.profiles:
                        for repeat in range(args.repeats):
                            order = list(configs)
                            rng.shuffle(order)  # Interleave builds/counts within each repeat.
                            for build, count, partitions in order:
                                spec = {'kind': 'performance', 'phase': phase, 'size': size,
                                        'profile': profile, 'repeat': repeat + 1,
                                        'steps': [args.warmup, args.steps, 1],
                                        'atoms': base_atoms * math.prod(cases.TILES[size]),
                                        'thermo_rows': (args.steps // max(1, args.steps // 5) if profile == 'output' else 0) + 1}
                                relative = 'performance/{}/{}/{}/repeat-{:02d}/{}_{}'.format(
                                    phase, size, profile, repeat + 1, build, count)
                                execute(root, report, binaries[build], build, count, model,
                                        cases.performance(size, profile, args.warmup, args.steps), spec, relative,
                                        partitions=partitions)
                                performance_summary(root, report)
                                save_report(root, report)
    except (OSError, CheckError, ValueError, KeyboardInterrupt) as error:
        report['fatal_error'] = str(error) or 'Interrupted by user'
        print('ERROR: ' + report['fatal_error'], file=sys.stderr)
    finally:
        assess(root, report, tolerances)
        performance_summary(root, report)
        save_report(root, report)
    print('Results: ' + str(root / 'summary.md'))
    return int(bool(report.get('fatal_error')) or report['correctness_status'] in ('FAIL', 'ERROR') or
               report['performance_status'] == 'ERROR')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, CheckError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
