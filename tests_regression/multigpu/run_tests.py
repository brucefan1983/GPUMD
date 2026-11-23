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


"""Compare ordinary NEP with three ZHEYONG partitions on one physical GPU."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
import cases
from checks import CheckError, check_run, compare_runs, exact, sha256, thermo_rows

REPO = Path(__file__).resolve().parents[2]
BUILDS = ('single', 'zheyong')
ERRORS = (OSError, ValueError, OverflowError, CheckError)


def device_environment(backend, device):
    env = os.environ.copy()
    key = 'HIP_VISIBLE_DEVICES' if backend == 'hip' else 'CUDA_VISIBLE_DEVICES'
    if device is None:
        mask = env.get(key, env.get('CUDA_VISIBLE_DEVICES', '0'))
        device = mask.split(',')[0].strip()
    exact(bool(device) and ',' not in device and not device.startswith('-'),
          'Select one visible device with --device (default: first inherited device, or 0)')
    # HIP accepts the CUDA alias too; avoid conflicting masks. Retain a scheduler's ROCR mask.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('HIP_VISIBLE_DEVICES', None)
    env[key] = device
    env['OMP_NUM_THREADS'] = '1'
    return env


def log_rates(log, build, steps):
    exact(re.findall(r'number of GPUs = (\d+)', log) == ['1'],
          'Log must confirm exactly one visible physical GPU')
    partitions = re.findall(r'Try to use (\d+) GPUs for the NEP part\.', log)
    exact(partitions == (['3'] if build == 'zheyong' else []),
          build + ': unexpected NEP partition count; check the executable and -DZHEYONG')
    actual = [int(x) for x in re.findall(r'^Run (\d+) steps\.', log, re.M)]
    exact(actual == steps, 'Run sequence differs: ' + str(actual))
    rates = [float(x) for x in re.findall(r'Speed of this run = (\S+) atom\*step/second\.', log)]
    elapsed = [float(x) for x in re.findall(r'Time used for this run = (\S+) second\.', log)]
    exact(len(rates) == len(steps) == len(elapsed), 'Missing run timing/completion records')
    exact(all(math.isfinite(x) and x > 0 for x in rates + elapsed), 'Invalid run timings')
    return rates, elapsed


def assess(function, *args):
    try:
        metrics = function(*args)
        exact(bool(metrics), 'No numerical checks were performed')
        failed = [(name, m) for name, m in metrics.items() if m['exceedances']]
        details = ['{}: max_abs={:.6g}, RMS={:.6g}, atol={:.6g}, rtol={:.6g}, '
                   'outside={}/{}'.format(name, m['max_abs'], m['rms'], m['atol'],
                                         m['rtol'], m['exceedances'], m['values'])
                   for name, m in failed[:8]]
        if len(failed) > 8:
            details.append('{} more failed metrics'.format(len(failed) - 8))
        return ('FAIL' if failed else 'PASS'), details
    except CheckError as error:
        return 'FAIL', [str(error)]
    except ERRORS as error:
        return 'ERROR', [str(error)]


def show_row(name, results):
    print('{:<24} {:>10} {:>10} {:>12}'.format(name, *(r[0] for r in results)), flush=True)
    for label, (_, details) in zip(('single', 'zheyong', 'comparison'), results):
        for detail in details:
            print('  {}: {}'.format(label, detail), flush=True)
    return all(r[0] in ('PASS', '-') for r in results)


class Runner:
    def __init__(self, root, args):
        self.root = Path(root)
        self.binaries = {'single': args.single, 'zheyong': args.zheyong}
        self.hashes = {build: sha256(path) for build, path in self.binaries.items()}
        self.env = device_environment(args.backend, args.device)
        self.timeout = args.timeout
        self.potential = REPO / cases.POTENTIAL
        with self.potential.open() as stream:
            exact(stream.readline().split()[:2] == ['nep', '89'], 'Expected the repository NEP89 model')
            lines = [stream.readline() for _ in range(4)]
            exact(lines[-1].split()[:3] == ['cutoff', '6', '5'], 'NEP89 cutoff differs from test geometry')
        print('NEP89: ordinary NEP vs three ZHEYONG partitions; one physical GPU')
        key = 'HIP_VISIBLE_DEVICES' if args.backend == 'hip' else 'CUDA_VISIBLE_DEVICES'
        print('{}={}\nAll run files are temporary and automatically removed.'.format(key, self.env[key]))
        for build, path in self.binaries.items():
            print('{}: {}\n  SHA256 {}'.format(build, path, self.hashes[build]), flush=True)

    def execute(self, label, build, model, text, spec):
        directory = self.root / label / build
        directory.mkdir(parents=True)
        shutil.copyfile(model, directory / 'model.xyz')
        (directory / 'potential.txt').symlink_to(self.potential)
        (directory / 'run.in').write_text(text, encoding='utf-8')
        log = ''
        try:
            exact(sha256(self.binaries[build]) == self.hashes[build], 'Executable changed during tests')
            for execution in range(spec.get('executions', 1)):
                with (directory / 'run.log').open('w', encoding='utf-8') as output:
                    process = subprocess.Popen([str(self.binaries[build])], cwd=directory, env=self.env,
                                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=self.timeout)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        raise
                log = (directory / 'run.log').read_text(encoding='utf-8', errors='replace')
                exact(code == 0, 'Execution {} exited with code {}'.format(execution + 1, code))
                timing = log_rates(log, build, spec['steps'])
            exact(sha256(self.binaries[build]) == self.hashes[build], 'Executable changed during tests')
        except (subprocess.TimeoutExpired, *ERRORS) as error:
            if (directory / 'run.log').exists():
                log = (directory / 'run.log').read_text(encoding='utf-8', errors='replace')
            tail = '\n'.join(log.splitlines()[-25:])
            raise CheckError('{} / {}: {}\n{}'.format(label, build, error, tail)) from error
        return directory, timing

    def run_checked(self, label, build, model, text, spec, initial):
        try:
            directory, _ = self.execute(label, build, model, text, spec)
        except ERRORS as error:
            return 'ERROR', [str(error)]
        return assess(check_run, directory, initial, spec)

    def pair(self, name, model, text, spec):
        initial = cases.only_frame(model)
        results = [self.run_checked(name, build, model, text, spec, initial) for build in BUILDS]
        comparison = (assess(compare_runs, self.root / name / 'zheyong', self.root / name / 'single',
                             spec, initial['n']) if all(r[0] == 'PASS' for r in results)
                      else ('SKIP', ['Both executions must pass before comparison']))
        return show_row(name, results + [comparison]), results

    def correctness(self):
        print('\nCORRECTNESS (18 cases + 2 repeatability checks; at most 100 steps per case/build)')
        print('{:<24} {:>10} {:>10} {:>12}'.format('Case', 'Single', 'ZHEYONG', 'Comparison'))
        passed = 0
        total = len(cases.NAMES) + 2
        axis_results = []
        for name in cases.NAMES:
            text, spec = cases.correctness(name)
            model = self.root / (name + '.xyz')
            try:
                if name == 'restart':
                    if not axis_results or axis_results[0][0] != 'PASS':
                        show_row(name, [('SKIP', ['Valid single axis_x restart required']),
                                        ('SKIP', []), ('SKIP', [])])
                        continue
                    cases.snapshot_to_input(self.root / 'axis_x' / 'single' / 'restart.xyz', model)
                else:
                    state = cases.water(REPO) if 'water' in name else cases.crystal(name, spec['axis'])
                    cases.write_state(model, state)
                ok, results = self.pair(name, model, text, spec)
            except ERRORS as error:
                results = [('ERROR', [str(error)]), ('ERROR', [])]
                ok = show_row(name, results + [('SKIP', [])])
            passed += ok
            if name == 'axis_x':
                axis_results = results
                if any(result[0] == 'ERROR' for result in results):
                    print('Stopped after the first case: execution/output error; remaining checks NOT_RUN.')
                    break
                initial = cases.only_frame(model)
                for build in BUILDS:
                    result = (self.run_checked('repeat', build, model, text, spec, initial)
                              if axis_results[BUILDS.index(build)][0] == 'PASS'
                              else ('SKIP', ['Original {} run did not pass'.format(build)]))
                    if result[0] == 'PASS':
                        result = assess(compare_runs, self.root / 'repeat' / build,
                                        self.root / name / build, spec, initial['n'])
                    columns = [('-', []), ('-', []), result]
                    columns[BUILDS.index(build)] = result
                    # Print diagnostics once, with the repeated build's column.
                    columns[2] = (result[0], [])
                    passed += show_row('repeat_' + build, columns)
        print('Correctness: {} ({}/{} checks passed)'.format('PASS' if passed == total else 'FAIL',
                                                            passed, total), flush=True)
        return passed == total, passed, total

    def preflight(self):
        print('\nPREFLIGHT (full correctness suite NOT_RUN)')
        print('{:<24} {:>10} {:>10} {:>12}'.format('Case', 'Single', 'ZHEYONG', 'Comparison'))
        model = self.root / 'preflight.xyz'
        cases.write_state(model, cases.crystal())
        text, spec = cases.correctness('axis_x')
        return self.pair('preflight', model, text, spec)[0]

    def performance(self):
        print('\nPERFORMANCE: 10 warmup + 100 measured steps; 3 repeats; atom-step/s', flush=True)
        rows = []
        try:
            for phase, tile in cases.PERFORMANCE.items():
                model = self.root / ('benchmark_' + phase + '.xyz')
                state = cases.water(REPO, (1, 1, 1)) if phase == 'water' else cases.crystal()
                cases.write_state(model, state)
                atoms = len(state['atoms']) * math.prod(tile)
                spec = {'steps': [cases.WARMUP, cases.STEPS]}
                samples = {build: [] for build in BUILDS}
                for repeat in range(cases.REPEATS):
                    order = BUILDS if (repeat + len(rows)) % 2 == 0 else BUILDS[::-1]
                    for build in order:
                        print('  {} {:,} atoms, repeat {}/{}, {}'.format(
                            phase, atoms, repeat + 1, cases.REPEATS, build), flush=True)
                        directory, timing = self.execute('performance/{}/{}'.format(phase, repeat), build,
                                                         model, cases.performance(phase), spec)
                        samples[build].append(performance_rate(directory, timing, atoms))
                a, acv = rate_stats(samples['single'])
                b, bcv = rate_stats(samples['zheyong'])
                rows.append((phase, atoms, a, b, b / a, acv, bcv))
        finally:
            if rows:
                print('\nPERFORMANCE RESULTS (atom-step/s)')
                print('{:<8} {:>10} {:>14} {:>14} {:>9} {:>12} {:>12}'.format(
                    'System', 'Atoms', 'Single', 'ZHEYONG', 'Ratio', 'Single CV%', 'ZHEYONG CV%'))
                for row in rows:
                    print('{:<8} {:>10,} {:>14.6e} {:>14.6e} {:>8.3f}x {:>12.3f} {:>12.3f}'.format(*row))
                print('Rates: arithmetic means; ratio: ZHEYONG / single; CV: sample standard deviation / mean.')
                print('One physical GPU: these results do not measure multi-GPU scaling or efficiency.', flush=True)


def performance_rate(directory, timing, atoms):
    rows = thermo_rows(Path(directory) / 'thermo.out', 1)
    exact(all(0 <= row[0] < 10000 and row[1] >= 0 for row in rows),
          'Unphysical benchmark temperature or kinetic energy')
    exact(not (Path(directory) / 'state.xyz').exists(), 'Unexpected benchmark trajectory')
    rates, elapsed = timing
    expected = atoms * cases.STEPS / elapsed[1]
    exact(abs(rates[1] / expected - 1) < 5e-5, 'Inconsistent atom-step/s timing')
    return rates[1]


def rate_stats(rates):
    exact(len(rates) == cases.REPEATS and all(math.isfinite(r) and r > 0 for r in rates),
          'Expected three finite, positive throughput samples')
    mean = statistics.mean(rates)
    return mean, 100 * statistics.stdev(rates) / mean


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', type=Path, default=REPO / 'src/gpumd', help='ordinary NEP executable')
    parser.add_argument('--zheyong', type=Path, default=REPO / 'src_multigpu/gpumd', help='-DZHEYONG executable')
    parser.add_argument('--mode', choices=('all', 'correctness', 'performance'), default='all')
    parser.add_argument('--backend', choices=('cuda', 'hip'), default='cuda')
    parser.add_argument('--device', help='one visible device; default: first inherited device, or 0')
    parser.add_argument('--timeout', type=float, default=900, help='timeout per process in seconds (default: 900)')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    for build in BUILDS:
        path = getattr(args, build).resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error('--{} is not an executable file: {}'.format(build, path))
        setattr(args, build, path)
    return args


def main(argv=None):
    args = arguments(argv)
    correctness = performance = 'NOT_RUN'
    failed = False
    try:
        with TemporaryDirectory(prefix='gpumd-nep-partitions-') as temporary:
            runner = Runner(temporary, args)
            ready = True
            if args.mode != 'performance':
                correctness = 'ERROR'
                ready, passed, total = runner.correctness()
                correctness = '{} ({}/{})'.format('PASS' if ready else 'FAIL', passed, total)
            else:
                ready = runner.preflight()
            failed = not ready
            if args.mode != 'correctness' and ready:
                performance = 'ERROR'
                runner.performance()
                performance = 'COMPLETE'
            elif args.mode != 'correctness':
                print('Performance blocked because correctness/preflight did not pass.')
    except (KeyboardInterrupt, *ERRORS) as error:
        failed = True
        print('ERROR: ' + (str(error) or 'Interrupted by user'), flush=True)
    print('\nCorrectness: {}. Performance: {}.'.format(correctness, performance))
    print('Temporary run files removed.')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
