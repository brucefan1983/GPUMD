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


"""CPU-only tests of parsing, failure reporting and benchmark arithmetic."""

import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import cases
import checks
import run_tests as runner
import recheck_results


# This deliberately simple zero-force executable tests the *test controller*,
# never GPUMD physics. Small stationary fixtures make its output independent of
# the production parsers/comparators. It is created only in a temporary directory.
EMULATOR = r'''
import os, re
from pathlib import Path
if os.environ.get('NEP_TEST_EMULATOR_FAIL'):
    raise SystemExit(127)
gpus = len(os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(','))
print('number of GPUs =', gpus)
if gpus > 1:
    print('Try to use %d GPUs for the NEP part.' % gpus)
lines = Path('model.xyz').read_text().splitlines()
box = re.search(r'Lattice="([^"]+)"', lines[1]).group(1)
atoms = [line.split() for line in lines[2:]]
dt, t = .05, 0.
dumps, thermo, restart = [], 0, 0

def xyz(path, chosen, time, is_restart=False, positions_only=False):
    schema = 'species:S:1:pos:R:3'
    if not positions_only:
        schema += ':mass:R:1:vel:R:3:group:I:1' if is_restart else (
            ':mass:R:1:vel:R:3:forces:R:3:energy_atom:R:1:virial:R:9:group:I:1:unwrapped_position:R:3')
    meta = 'pbc="T T T" Lattice="' + box + '" Properties=' + schema
    if not is_restart:
        meta += ' Time=%.8f energy=0 virial="0 0 0 0 0 0 0 0 0" stress="0 0 0 0 0 0 0 0 0"' % time
    with open(path, 'w' if is_restart else 'a') as out:
        out.write(str(len(chosen)) + '\n' + meta + '\n')
        for a in chosen:
            # Input schema: species pos vel mass group. Restart order differs.
            mass, group = a[7], a[8]
            row = a[:4]
            if not positions_only:
                row += [mass] + ['0'] * 3
                if not is_restart:
                    row += ['0'] * 13
                row += [group]
                if not is_restart:
                    row += a[1:4]
            out.write(' '.join(row) + '\n')

for line in Path('run.in').read_text().splitlines():
    p = line.split()
    if not p:
        continue
    if p[0] == 'replicate':
        factor = int(p[1]) * int(p[2]) * int(p[3])
        atoms *= factor
    elif p[0] == 'time_step':
        dt = float(p[1])
    elif p[0] == 'dump_xyz':
        dumps.append(p)
    elif p[0] == 'dump_thermo':
        thermo = int(p[1])
    elif p[0] == 'dump_restart':
        restart = int(p[1])
    elif p[0] == 'run':
        steps = int(p[1])
        for step in range(1, steps + 1):
            t += dt
            if thermo and step % thermo == 0:
                with open('thermo.out', 'a') as out:
                    out.write(' '.join(['0'] * 9) + ' ' + box + '\n')
            for dump in dumps:
                if step % int(dump[1]) == 0:
                    chosen = [a for a in atoms if a[8] == '1'] if 'group' in dump else atoms
                    xyz(dump[2], chosen, t, positions_only=len(dump) == 3)
            if restart and step % restart == 0:
                xyz('restart.xyz', atoms, t, is_restart=True)
        print('Run %d steps.' % steps)
        print('Time used for this run = 0.01 second.')
        print('Speed of this run = %g atom*step/second.' % (len(atoms) * steps / .01))
        dumps, thermo, restart = [], 0, 0
'''


def timing(gpus, rates=(20, 100, 10), steps=(2, 50, 1), atoms=10):
    text = 'number of GPUs = {}\n'.format(gpus)
    if gpus > 1:
        text += 'Try to use {} GPUs for the NEP part.\n'.format(gpus)
    for steps_i, rate in zip(steps, rates):
        text += ('Run {} steps.\nTime used for this run = {} second.\n'
                 'Speed of this run = {} atom*step/second.\n').format(steps_i, atoms * steps_i / rate, rate)
    return text


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def xyz(self, text):
        path = self.root / 'sample.xyz'
        path.write_text(text)
        return path

    def test_xyz_aliases_and_integer_groups(self):
        frame = cases.only_frame(self.xyz('1\nProperties=species:S:1:forces:R:3:energy_atom:R:1:group:I:1\nCu 1 2 3 4 0\n'))
        self.assertEqual(list(frame['data']['force']), [1, 2, 3])
        self.assertEqual(list(frame['data']['potential']), [4])

    def test_xyz_rejects_bad_records(self):
        for body in ('1 2', '1 2 nan', '1 2 inf', '1 2 1e999'):
            with self.subTest(body=body), self.assertRaises(checks.CheckError):
                list(checks.frames(self.xyz('1\nProperties=pos:R:3\n' + body + '\n')))
        with self.assertRaises(checks.CheckError):
            list(checks.frames(self.xyz('1\nProperties=group:I:1\n0.5\n')))

    def test_xyz_duplicate_properties_rejected(self):
        with self.assertRaises(checks.CheckError):
            list(checks.frames(self.xyz('1\nProperties=pos:R:1:pos:R:1\n1 2\n')))

    def test_extra_frame_rejected_for_restart(self):
        with self.assertRaises(checks.CheckError):
            cases.only_frame(self.xyz('1\nProperties=pos:R:1\n1\n' * 2))

    def test_thermo_comments_and_count(self):
        path = self.root / 'thermo.out'
        path.write_text('# dump_thermo 1\n# format_version 1\n' + ' '.join(['1'] * 18) + '\n')
        self.assertEqual(len(checks.thermo_rows(path, 1)), 1)
        with self.assertRaises(checks.CheckError):
            checks.thermo_rows(path, 2)

    def test_metric_threshold_and_nonfinite(self):
        result = checks.metric([1, 2.1, -3.2], [1, 2, -3], .05)
        self.assertEqual(result['exceedances'], 2)
        self.assertAlmostEqual(result['max_abs'], .2)
        self.assertAlmostEqual(result['rms'], (.05 / 3) ** .5)
        with self.assertRaises(checks.CheckError):
            checks.metric([float('nan')], [0], 1)

    def test_bad_tolerances_rejected(self):
        path = self.root / 'tol.json'
        tol = checks.load_tolerances(runner.HERE / 'tolerances.json')
        for value in ([float('inf'), 0], [-1, 0], [1], [True, 0]):
            tol['force'] = value
            path.write_text(json.dumps(tol))
            with self.assertRaises(checks.CheckError):
                checks.load_tolerances(path)

    def test_log_requires_multi_gpu_execution(self):
        path = self.root / 'run.log'
        path.write_text(timing(2))
        self.assertEqual(runner.log_rates(path, 2, [2, 50, 1])[0], [20, 100, 10])
        for bad in (timing(1), timing(2).replace('Try to use 2 GPUs for the NEP part.\n', ''),
                    timing(2).replace('Run 50 steps.', 'Run 40 steps.')):
            path.write_text(bad)
            with self.assertRaises(checks.CheckError):
                runner.log_rates(path, 2, [2, 50, 1])

    def test_device_aliases_and_scheduler_mask(self):
        with patch.dict(os.environ, {'ROCR_VISIBLE_DEVICES': '2,3', 'CUDA_VISIBLE_DEVICES': '9'}):
            env = runner.device_environment('hip', ['0', '1'])
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'], '2,3')
        self.assertNotIn('CUDA_VISIBLE_DEVICES', env)

    def test_crystal_identity_and_legal_partition_lengths(self):
        for axis in range(3):
            state = cases.crystal(2, axis=axis)
            self.assertGreaterEqual(int(state['box'][axis][axis] / 3) // 2, 10)
            path = self.root / 'model.xyz'
            cases.write_state(path, state)
            frame = cases.only_frame(path)
            self.assertEqual(frame['n'], 6144)
            self.assertEqual(len(set(frame['data']['mass'])), frame['n'])

    def test_wrong_atom_identity_rejected(self):
        state = cases.crystal(2)
        path = self.root / 'model.xyz'
        cases.write_state(path, state)
        initial = cases.only_frame(path)
        frame = cases.only_frame(path)
        frame['data']['mass'][0], frame['data']['mass'][1] = frame['data']['mass'][1], frame['data']['mass'][0]
        with self.assertRaises(checks.CheckError):
            checks.check_identity(frame, initial, list(range(initial['n'])), initial['schema'])

    def test_performance_ratio_of_means_and_cv_percent(self):
        report = {'runs': [], 'repeats': 2, 'mode': 'performance', 'expected_performance_groups': 4}
        for build, count, values in (('old', 1, (80, 120)), ('old', 2, (140, 180)),
                                     ('new', 1, (90, 110)), ('new', 2, (180, 220))):
            for repeat, rate in enumerate(values):
                path = '{}-{}-{}'.format(build, count, repeat)
                directory = self.root / path
                directory.mkdir()
                (directory / 'run-01.log').write_text(timing(count, (20, rate, 10)))
                (directory / 'thermo.out').write_text(' '.join(['1'] * 18) + '\n')
                report['runs'].append({'path': path, 'build': build, 'gpus': count,
                    'process_status': 'COMPLETE', 'input_hashes': {}, 'output_hashes': {},
                    'logs': [{'file': 'run-01.log'}], 'spec': {'kind': 'performance', 'phase': 'solid',
                    'size': 'small', 'profile': 'minimal', 'repeat': repeat, 'steps': [2, 50, 1],
                    'atoms': 10, 'thermo_rows': 1}})
        runner.performance_summary(self.root, report)
        row = next(r for r in report['performance'] if r['build'] == 'new' and r['gpus'] == 2)
        self.assertEqual(row['mean_rate'], 200)
        self.assertEqual(row['new_over_old'], 1.25)
        self.assertEqual(row['scaling'], 2)
        self.assertEqual(row['efficiency'], 1)
        self.assertAlmostEqual(row['cv_percent'], 14.1421356237)
        self.assertEqual(report['performance_status'], 'COMPLETE')
        report['runs'].pop()
        runner.performance_summary(self.root, report)
        self.assertEqual(report['performance_status'], 'ERROR')

    def test_failed_process_cannot_be_a_comparison_pass(self):
        report = {'runs': [{'path': 'bad', 'build': 'new', 'gpus': 1,
                           'spec': {'name': 'axis_x', 'kind': 'short'},
                           'process_status': 'ERROR', 'error': 'Exit code 127'}],
                  'selected_cases': ['axis_x'], 'mode': 'correctness'}
        runner.assess(self.root, report, {})
        self.assertEqual(report['correctness_status'], 'ERROR')
        self.assertIn('127', report['correctness_checks'][0]['error'])

    def test_input_mutation_is_detected(self):
        path = self.root / 'model.xyz'
        path.write_text('original')
        digest = checks.sha256(path)
        path.write_text('changed')
        with self.assertRaises(checks.CheckError):
            runner.verify_record(self.root, {'path': '.', 'process_status': 'COMPLETE',
                                 'input_hashes': {'model.xyz': digest}, 'output_hashes': {}})

    def controller(self, output, extra):
        binary = self.root / 'emulator'
        binary.write_text('#!' + sys.executable + '\n' + EMULATOR)
        binary.chmod(0o755)
        potential = self.root / cases.POTENTIAL
        potential.parent.mkdir(parents=True, exist_ok=True)
        potential.write_text('nep 89 Cu\nformat_version 1\nmodel nep\nzbl 1 2\ncutoff 6 5 274 158\n')
        tiny = {'box': [[12, 0, 0], [0, 12, 0], [0, 0, 12]], 'pbc': [True] * 3,
                'atoms': [['Cu', [1, 2, 3], [0, 0, 0], 63.546],
                          ['Cu', [5, 6, 7], [0, 0, 0], 63.546001]]}
        with patch.object(cases, 'crystal', return_value=tiny), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(['--new', str(binary), '--old', str(binary), '--backend', 'hip',
                                '--repo-root', str(self.root), '--output', str(output),
                                '--cases', 'axis_x,append', '--repeats', '2',
                                '--sizes', 'small', '--phases', 'solid', '--steps', '5',
                                '--warmup', '2'] + extra)
        return code, json.loads((output / 'report.json').read_text())

    def test_controller_end_to_end_and_recheck_inputs(self):
        output = self.root / 'results'
        code, report = self.controller(output, [])
        self.assertEqual(code, 0, report.get('fatal_error') or report['correctness_checks'])
        self.assertEqual(report['correctness_status'], 'PASS')
        self.assertEqual(report['performance_status'], 'NOT_RUN')
        runner.assess(output, report, report['tolerances'])
        self.assertEqual(report['correctness_status'], 'PASS')
        before = checks.sha256(output / 'report.json')
        (self.root / 'emulator').unlink()
        with patch.object(sys, 'argv', ['recheck_results.py', str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recheck_results.main(), 0)
        self.assertEqual(checks.sha256(output / 'report.json'), before)
        self.assertEqual(len(list(output.glob('recheck-*/summary.md'))), 1)
        path = output / 'correctness/append/new_2/state.xyz'
        path.write_text(path.read_text().replace('63.546', '63.646', 1))
        runner.assess(output, report, report['tolerances'])
        self.assertEqual(report['correctness_status'], 'ERROR')

    def test_controller_missing_library_fails_before_workloads(self):
        with patch.dict(os.environ, {'NEP_TEST_EMULATOR_FAIL': '1'}):
            code, report = self.controller(self.root / 'failed', [])
        self.assertEqual(code, 1)
        self.assertEqual(report['correctness_status'], 'ERROR')
        self.assertEqual(len(report['runs']), 1)
        self.assertIn('127', report['fatal_error'])

    def test_performance_only_does_not_claim_correctness(self):
        code, report = self.controller(self.root / 'performance', ['--mode', 'performance',
                                                                 '--profiles', 'minimal,output'])
        self.assertEqual(code, 0, report.get('fatal_error') or report['performance'])
        self.assertEqual(report['correctness_status'], 'NOT_RUN')
        self.assertEqual(report['performance_status'], 'COMPLETE')
        self.assertEqual(len(report['performance']), 8)

    def test_berendsen_post_rescaling_velocities(self):
        output = self.root / 'berendsen'
        _, report = self.controller(output, [])
        directory = output / 'correctness/axis_x/new_1'
        for name in ('state.xyz', 'selected.xyz', 'restart.xyz'):
            parsed = list(checks.frames(directory / name))
            with (directory / name).open('w') as out:
                for frame in parsed:
                    data = frame['data']
                    for i, mass in enumerate(data['mass']):
                        # T_before=200, target=300, tau=100 gives T_after=201.
                        speed = math.sqrt(3 * 8.617343e-5 * 201 / mass) / 10.18051
                        data['vel'][3 * i] = speed if data['group'][i] == 0 else -speed
                    out.write(str(frame['n']) + '\n' + ' '.join(
                        '{}="{}"'.format(k, v) for k, v in frame['meta'].items()) + '\n')
                    for i in range(frame['n']):
                        values = [str(data[k][i * width + j]) for k, (_, width) in frame['schema'].items()
                                  for j in range(width)]
                        out.write(' '.join(values) + '\n')
        rows = checks.thermo_rows(directory / 'thermo.out')
        for row in rows:
            row[0], row[1] = 200, 1.5 * 2 * 8.617343e-5 * 200
        (directory / 'thermo.out').write_text('\n'.join(' '.join(map(str, r)) for r in rows) + '\n')
        _, spec = cases.correctness('nvt')
        metrics = checks.check_run(directory, cases.only_frame(directory / 'model.xyz'), spec, report['tolerances'])
        self.assertFalse([k for k, v in metrics.items() if v['exceedances']])

    def test_liquid_requires_growth_and_removes_rigid_translation(self):
        state = {'box': [[12, 0, 0], [0, 12, 0], [0, 0, 12]], 'pbc': [True] * 3,
                 'atoms': [['Cu', [1, 2, 3], [0, 0, 0], 63.546],
                           ['Cu', [5, 6, 7], [0, 0, 0], 63.546001]]}
        cases.write_state(self.root / 'model.xyz', state)
        (self.root / 'thermo.out').write_text(('2000 ' + ' '.join(['1'] * 17) + '\n') * 30)
        for translation_only in (False, True):
            with (self.root / 'diffusion.xyz').open('w') as out:
                for step in range(3):
                    out.write('2\npbc="T T T" Lattice="12 0 0 0 12 0 0 0 12" Time={} '
                              'Properties=species:S:1:pos:R:3:mass:R:1:group:I:1:unwrapped_position:R:3\n'.format(8000 + step * 1000))
                    for i, atom in enumerate(state['atoms']):
                        x, y, z = atom[1]
                        shift = step if translation_only or i == 0 else -step
                        out.write('Cu {} {} {} {} {} {} {} {}\n'.format(x, y, z, atom[3], i, x + shift, y, z))
            if translation_only:
                with self.assertRaises(checks.CheckError):
                    checks.check_liquid(self.root)
            else:
                self.assertEqual(checks.check_liquid(self.root)['msd_A2_at_1ps_2ps'], [1, 4])


if __name__ == '__main__':
    unittest.main()
